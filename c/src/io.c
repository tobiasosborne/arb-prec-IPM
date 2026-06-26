/*
 * src/io.c -- SDPA-sparse (.dat-s) reader + the precision-preserving
 * arbsdp_problem data model.  See include/arbsdp/problem.h for the binding
 * design decisions: values are stored as raw decimal STRINGS (no premature
 * double truncation) and materialized to arb / arb_mat at a caller-given prec
 * via arb_set_str; the off-diagonal "2v" doubling lives in the Frobenius inner
 * product, NOT in the stored matrix (the materializer writes A_{ij}=A_{ji}=v).
 *
 * Parsing semantics match SdpaSparse.ts exactly (so golden-master comparison
 * is meaningful):
 *   - lines starting with '*' or '"' (after trimming) are comments / skipped;
 *   - a trailing "= ..." on any line is stripped (treated as a comment);
 *   - the first four NON-comment lines are: m, nblocks, block sizes, b vector;
 *   - remaining non-empty lines are entries "<matno> <blk> <k> <l> <v>".
 *
 * Differences from the TS parser are deliberate and in the FAIL-LOUD direction
 * (CLAUDE.md rule 5): where SdpaSparse.ts silently `continue`s past a malformed
 * or out-of-range entry, this reader returns nonzero.  A corrupted problem is
 * worse than a clear error (CLAUDE.md rule 5: a wrong "certificate" is the worst
 * possible output, and that starts with silently mis-read data).
 *
 * Arb memory discipline (CLAUDE.md rule 7): all heap (block_sizes, the b
 * strings, every entry string, the mats grid) is freed in arbsdp_problem_clear;
 * on a parse failure the partially-built problem is cleared before return.
 * Temporary arb_t's in the materializers are init'd and cleared in scope.
 */

#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* mpfr.h MUST precede the FLINT headers: flint/arf.h declares the mpfr-interop
 * functions (arf_get_mpfr) only inside `#ifdef __MPFR_H`, i.e. only if mpfr.h's
 * include guard is already defined.  Including it after arf.h yields an implicit
 * declaration (FLINT_NOTES.md / CLAUDE.md rule 12: confirm symbol availability). */
#include <mpfr.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>
#include <flint/arf.h>

#include "arbsdp/api.h"
#include "arbsdp/io.h"
#include "arbsdp/problem.h"

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

void
arbsdp_problem_init(arbsdp_problem *p)
{
    assert(p != NULL);
    p->m = 0;
    p->nblocks = 0;
    p->block_sizes = NULL;
    p->b = NULL;
    p->mats = NULL;
    p->maximize = 1; /* SDPA convention; .dat-s problems are maximizations */
}

void
arbsdp_problem_clear(arbsdp_problem *p)
{
    assert(p != NULL);

    if (p->b != NULL) {
        for (int i = 0; i < p->m; i++)
            free(p->b[i]);
        free(p->b);
    }

    if (p->mats != NULL) {
        slong ncells = (slong) (p->m + 1) * p->nblocks;
        for (slong c = 0; c < ncells; c++) {
            arbsdp_matblock *mb = &p->mats[c];
            for (int e = 0; e < mb->count; e++)
                free(mb->entries[e].value);
            free(mb->entries);
        }
        free(p->mats);
    }

    free(p->block_sizes);

    arbsdp_problem_init(p);
}

/* -------------------------------------------------------------------------
 * Sparse-entry append
 * ---------------------------------------------------------------------- */

/* Append (i, j, value-string) to a matblock, growing as needed.  Takes
 * OWNERSHIP of `value` (a heap string).  Returns 0 on success, nonzero if the
 * allocation failed (in which case `value` is freed). */
static int
matblock_push(arbsdp_matblock *mb, int i, int j, char *value)
{
    if (mb->count == mb->capacity) {
        int newcap = mb->capacity == 0 ? 4 : mb->capacity * 2;
        arbsdp_entry *grown =
            realloc(mb->entries, (size_t) newcap * sizeof(*grown));
        if (grown == NULL) {
            free(value);
            return 1;
        }
        mb->entries = grown;
        mb->capacity = newcap;
    }
    mb->entries[mb->count].i = i;
    mb->entries[mb->count].j = j;
    mb->entries[mb->count].value = value;
    mb->count++;
    return 0;
}

/* -------------------------------------------------------------------------
 * Lexing helpers
 * ---------------------------------------------------------------------- */

/* In place: cut the line at the first '=' (SdpaSparse.ts stripTrailingComment
 * treats "= ..." as a comment) and at the first '\r' or '\n'. */
static void
strip_comment_and_eol(char *line)
{
    for (char *q = line; *q != '\0'; q++) {
        if (*q == '=' || *q == '\r' || *q == '\n') {
            *q = '\0';
            return;
        }
    }
}

/* True if the trimmed line is blank or a '*' / '"' comment line. */
static int
is_skippable(const char *line)
{
    const char *q = line;
    while (*q != '\0' && isspace((unsigned char) *q))
        q++;
    return (*q == '\0' || *q == '*' || *q == '"');
}

/* Parse a whitespace/comma/brace-delimited integer token.  Advances *cursor
 * past the token.  Returns 0 on success and writes the value to *out; returns
 * nonzero if no valid integer token is present (fail loud). */
static int
next_int(char **cursor, long *out)
{
    char *s = *cursor;
    char *end;
    long v;

    /* skip SDPA delimiters: whitespace, comma, braces, parens. */
    while (*s != '\0' && (isspace((unsigned char) *s) || *s == ',' ||
                          *s == '{' || *s == '}' || *s == '(' || *s == ')'))
        s++;
    if (*s == '\0')
        return 1;

    v = strtol(s, &end, 10);
    if (end == s)
        return 1; /* no digits consumed */
    /* The token must be wholly numeric: the next char is a delimiter or end. */
    if (*end != '\0' && !isspace((unsigned char) *end) && *end != ',' &&
        *end != '{' && *end != '}' && *end != '(' && *end != ')')
        return 1;

    *out = v;
    *cursor = end;
    return 0;
}

/* Extract the next whitespace/comma/brace-delimited token as a freshly
 * allocated decimal string (the precision-preserving representation).  Advances
 * *cursor.  Returns the owned string, or NULL if no token remains.  The caller
 * validates parseability separately (via arb_set_str) so the digits are kept
 * verbatim and never routed through double. */
static char *
next_token_dup(char **cursor)
{
    char *s = *cursor;
    char *start, *end;
    size_t len;
    char *out;

    while (*s != '\0' && (isspace((unsigned char) *s) || *s == ',' ||
                          *s == '{' || *s == '}' || *s == '(' || *s == ')'))
        s++;
    if (*s == '\0')
        return NULL;

    start = s;
    while (*s != '\0' && !isspace((unsigned char) *s) && *s != ',' &&
           *s != '{' && *s != '}' && *s != '(' && *s != ')')
        s++;
    end = s;

    len = (size_t) (end - start);
    out = malloc(len + 1);
    if (out == NULL)
        return NULL;
    memcpy(out, start, len);
    out[len] = '\0';

    *cursor = end;
    return out;
}

/* Validate that a decimal string parses as an arb (no garbage).  Returns 0 if
 * it parses cleanly at a probe precision, nonzero otherwise.  arb_set_str
 * returns nonzero on a malformed string. */
static int
decimal_is_valid(const char *s)
{
    arb_t probe;
    int bad;
    arb_init(probe);
    bad = arb_set_str(probe, s, 64);
    arb_clear(probe);
    return bad ? 1 : 0;
}

/* -------------------------------------------------------------------------
 * Reader
 * ---------------------------------------------------------------------- */

/* Read the next non-skippable, comment-stripped logical line from `f` into the
 * caller's growable buffer (*buf / *cap).  Returns 1 if a line was produced
 * (NUL-terminated, comment/EOL stripped, possibly with leading/trailing
 * whitespace), 0 on EOF with no more content, -1 on read/alloc error. */
static int
next_data_line(FILE *f, char **buf, size_t *cap)
{
    for (;;) {
        size_t len = 0;
        int c;

        for (;;) {
            c = fgetc(f);
            if (c == EOF)
                break;
            if (len + 1 >= *cap) {
                size_t newcap = (*cap == 0) ? 256 : *cap * 2;
                char *grown = realloc(*buf, newcap);
                if (grown == NULL)
                    return -1;
                *buf = grown;
                *cap = newcap;
            }
            (*buf)[len++] = (char) c;
            if (c == '\n')
                break;
        }

        if (len == 0 && c == EOF)
            return 0; /* genuine EOF, nothing buffered */

        (*buf)[len] = '\0';
        strip_comment_and_eol(*buf);
        if (!is_skippable(*buf))
            return 1;

        if (c == EOF)
            return 0; /* trailing skippable line at EOF */
    }
}

int
arbsdp_read_sdpa(arbsdp_problem *p, const char *path)
{
    FILE *f = NULL;
    char *buf = NULL;
    size_t cap = 0;
    char *cursor;
    long m_l, nb_l;
    int rc = 1; /* default: failure */

    assert(p != NULL);
    /* Start from a clean state WITHOUT dereferencing *p's prior contents: the
     * caller may pass an uninitialized struct (garbage pointers).  Clearing here
     * would free() that garbage -> heap corruption (bead b29; CLAUDE.md rule 5: a
     * corrupted heap is the worst possible output).  arbsdp_problem_init only
     * writes zero/NULL, so it is foolproof on garbage AND on a fresh struct.
     * CONTRACT (see problem.h): *p's prior heap is NOT freed -- to RE-read into a
     * used struct, arbsdp_problem_clear(p) first, else its prior contents leak. */
    arbsdp_problem_init(p);

    f = fopen(path, "r");
    if (f == NULL)
        goto done; /* unreadable file: fail loud */

    /* Line 1: m. */
    if (next_data_line(f, &buf, &cap) != 1)
        goto done;
    cursor = buf;
    if (next_int(&cursor, &m_l) != 0 || m_l < 0)
        goto done;
    p->m = (int) m_l;

    /* Line 2: nblocks. */
    if (next_data_line(f, &buf, &cap) != 1)
        goto done;
    cursor = buf;
    if (next_int(&cursor, &nb_l) != 0 || nb_l <= 0)
        goto done;
    p->nblocks = (int) nb_l;

    /* Line 3: block sizes (exactly nblocks of them). */
    p->block_sizes = malloc((size_t) p->nblocks * sizeof(*p->block_sizes));
    if (p->block_sizes == NULL)
        goto done;
    if (next_data_line(f, &buf, &cap) != 1)
        goto done;
    cursor = buf;
    for (int b = 0; b < p->nblocks; b++) {
        long sz;
        if (next_int(&cursor, &sz) != 0 || sz == 0)
            goto done; /* missing size, or zero (degenerate) */
        p->block_sizes[b] = (int) sz;
    }
    {
        /* Reject a trailing extra block size (count mismatch, fail loud). */
        long extra;
        if (next_int(&cursor, &extra) == 0)
            goto done;
    }

    /* Allocate the (m+1) x nblocks matrix grid, zeroed. */
    {
        size_t ncells = (size_t) (p->m + 1) * p->nblocks;
        p->mats = calloc(ncells, sizeof(*p->mats));
        if (p->mats == NULL)
            goto done;
    }

    /* Line 4: b vector (exactly m decimal strings). */
    p->b = calloc((size_t) (p->m > 0 ? p->m : 1), sizeof(*p->b));
    if (p->b == NULL)
        goto done;
    if (p->m > 0) {
        if (next_data_line(f, &buf, &cap) != 1)
            goto done;
        cursor = buf;
        for (int i = 0; i < p->m; i++) {
            char *tok = next_token_dup(&cursor);
            if (tok == NULL || decimal_is_valid(tok) != 0) {
                free(tok);
                goto done; /* missing or malformed b value */
            }
            p->b[i] = tok;
        }
        {
            /* Reject a trailing extra b value (count mismatch). */
            char *extra = next_token_dup(&cursor);
            if (extra != NULL) {
                free(extra);
                goto done;
            }
        }
    } else {
        /* m == 0: the b line, if present, must be empty after stripping. */
        int got = next_data_line(f, &buf, &cap);
        if (got < 0)
            goto done;
        if (got == 1) {
            cursor = buf;
            char *extra = next_token_dup(&cursor);
            if (extra != NULL) {
                free(extra);
                goto done;
            }
        }
    }

    /* Remaining lines: entries "<matno> <blk> <k> <l> <v>". */
    for (;;) {
        long matno, blk, k, l;
        char *val;
        int got = next_data_line(f, &buf, &cap);
        int nsz;
        slong cell;

        if (got < 0)
            goto done; /* read/alloc error */
        if (got == 0)
            break; /* EOF, all entries consumed */

        cursor = buf;
        if (next_int(&cursor, &matno) != 0)
            goto done;
        if (next_int(&cursor, &blk) != 0)
            goto done;
        if (next_int(&cursor, &k) != 0)
            goto done;
        if (next_int(&cursor, &l) != 0)
            goto done;
        val = next_token_dup(&cursor);
        if (val == NULL || decimal_is_valid(val) != 0) {
            free(val);
            goto done;
        }

        /* Range / well-formedness checks (fail loud; SdpaSparse.ts silently
         * skips these, but a wrong matrix is worse than a clear error). */
        if (matno < 0 || matno > p->m) {
            free(val);
            goto done;
        }
        if (blk < 1 || blk > p->nblocks) {
            free(val);
            goto done;
        }
        nsz = p->block_sizes[blk - 1];
        if (nsz < 0)
            nsz = -nsz; /* diagonal block side = |size| */
        if (k < 1 || l < 1 || k > nsz || l > nsz) {
            free(val);
            goto done;
        }
        if (k > l) { /* upper-triangle-only contract (i <= j) */
            free(val);
            goto done;
        }
        if (p->block_sizes[blk - 1] < 0 && k != l) {
            /* Diagonal (LP) block: only i==j entries are meaningful. */
            free(val);
            goto done;
        }

        cell = (slong) matno * p->nblocks + (blk - 1);
        if (matblock_push(&p->mats[cell], (int) (k - 1), (int) (l - 1), val)
            != 0)
            goto done; /* alloc failure; val already freed by matblock_push */
    }

    rc = 0; /* success */

done:
    if (f != NULL)
        fclose(f);
    free(buf);
    if (rc != 0)
        arbsdp_problem_clear(p); /* leave p clean on failure */
    return rc;
}

/* -------------------------------------------------------------------------
 * Materialization
 * ---------------------------------------------------------------------- */

void
arbsdp_problem_block_mat(arb_mat_t out, const arbsdp_problem *p, int matno,
                         int block, slong prec)
{
    const arbsdp_matblock *mb;
    slong n;
    arb_t v;

    assert(p != NULL);
    assert(matno >= 0 && matno <= p->m);
    assert(block >= 0 && block < p->nblocks);

    n = p->block_sizes[block] < 0 ? -p->block_sizes[block]
                                  : p->block_sizes[block];
    assert(arb_mat_nrows(out) == n && arb_mat_ncols(out) == n);

    arb_mat_zero(out);
    arb_init(v);

    mb = &p->mats[(slong) matno * p->nblocks + block];
    for (int e = 0; e < mb->count; e++) {
        int i = mb->entries[e].i;
        int j = mb->entries[e].j;
        /* arb_set_str yields a correctly-rounded enclosure of the exact decimal
         * at `prec` bits -- a true ball, so rigor is preserved.  The string was
         * validated at parse time, so parse failure here is unexpected; assert
         * it (fail loud rather than silently zero). */
        int bad = arb_set_str(v, mb->entries[e].value, prec);
        assert(bad == 0);
        (void) bad;

        /* Duplicates accumulate (matches SdpProblem.ts `M += v`). */
        arb_add(arb_mat_entry(out, i, j), arb_mat_entry(out, i, j), v, prec);
        if (i != j) {
            /* Symmetric off-diagonal: A_{ji} = A_{ij} = v.  NO factor of 2 here;
             * the 2v shows up only in the Frobenius inner product (golden/
             * README.md, problem.h banner). */
            arb_add(arb_mat_entry(out, j, i), arb_mat_entry(out, j, i), v, prec);
        }
    }

    arb_clear(v);
}

int
arbsdp_problem_b(arb_t out, const arbsdp_problem *p, int i, slong prec)
{
    assert(p != NULL);
    assert(i >= 0 && i < p->m);
    return arb_set_str(out, p->b[i], prec) ? 1 : 0;
}

/* =========================================================================
 * JSON result serializer + rigorous directed decimal (bead arb-prec-IPM-b21)
 * =========================================================================
 * This is the testable core of the CLI (the CLI driver itself is a later step).
 * arbsdp_result_to_json hand-rolls a single JSON object (no JSON library) from a
 * full solve+certify result; arbsdp_arb_decimal is the rigour-critical helper it
 * leans on so the printed [obj_lb, obj_ub] still ENCLOSES the optimum after
 * decimal rounding (CLAUDE.md rule 2).
 *
 * SCHEMA (keys in this exact order; valid JSON, no trailing commas):
 *   {
 *     "arbsdp_version": "<ARBSDP_VERSION>",
 *     "problem": { "m": <int>, "nblocks": <int>,
 *                  "block_sizes": [<signed ints>], "maximize": <true|false> },
 *     "solve": { "adaptive_status": "<optimal|limit|infeasible>",
 *                "value": "<directed decimal of ar->res.value, round_up=1>",
 *                "final_prec": <slong>, "iters": <int>,
 *                "prec_history": [<prec bits per escalation step>] },
 *     "certificate": { "status": "<optimal|inconclusive|primal_infeasible|
 *                                  dual_infeasible>",
 *                      "prec_c": <slong>,
 *                      "obj_lb": "<directed decimal of lb, round_up=0>",
 *                      "obj_ub": "<directed decimal of ub, round_up=1>",
 *                      "trace_bounds": { "<b>": "<decimal of xbar[b], round_up=1>",
 *                                        ... for each b with xbar_set[b] } }
 *   }
 *
 * RIGOR (CLAUDE.md rule 2): obj_lb is rounded toward -inf and obj_ub toward +inf
 * (arbsdp_arb_decimal), so the printed interval rigorously brackets the optimum
 * even after the decimal rounding.  +/-inf endpoints print as "-inf"/"+inf".
 *
 * Memory (CLAUDE.md rule 7): both functions return a malloc'd string the CALLER
 * frees.  arbsdp_result_to_json frees every arbsdp_arb_decimal result it embeds;
 * arbsdp_arb_decimal clears its arf_t / mpfr_t temporaries.
 * ====================================================================== */

/* dup_cstr -- malloc a copy of `s` (so the result is free()-able by the caller,
 * NOT mpfr_free_str-able).  strdup is POSIX, not C11, so we avoid it here. */
static char *
dup_cstr(const char *s)
{
    size_t n = strlen(s) + 1;
    char *out = malloc(n);
    assert(out != NULL);
    memcpy(out, s, n);
    return out;
}

char *
arbsdp_arb_decimal(const arb_t x, int round_up, slong digits)
{
    arf_t endpoint;
    slong mant_prec;
    char *out;

    if (digits < 1)
        digits = 1;

    /* mantissa precision >= ceil(digits*log2(10)) + 8 bits (log2(10) ~ 3.3219).
     * Computed without <math.h> (manual ceil). */
    {
        double bits = (double) digits * 3.321928094887362;
        mant_prec = (slong) bits;
        if ((double) mant_prec < bits)
            mant_prec += 1;
        mant_prec += 8;
    }

    /* Take x's DIRECTED endpoint: ubound (>= true x) for an upper bound, lbound
     * (<= true x) for a lower bound.  arb_get_ubound_arf rounds UP and
     * arb_get_lbound_arf rounds DOWN, so the directedness holds at any prec. */
    arf_init(endpoint);
    if (round_up)
        arb_get_ubound_arf(endpoint, x, mant_prec);
    else
        arb_get_lbound_arf(endpoint, x, mant_prec);

    if (arf_is_nan(endpoint)) {
        out = dup_cstr("nan");
    } else if (arf_is_inf(endpoint)) {
        out = dup_cstr(arf_sgn(endpoint) > 0 ? "+inf" : "-inf");
    } else {
        mpfr_t m;
        char *mstr = NULL;
        /* arf -> mpfr with the SAME directed rounding; endpoint already has
         * <= mant_prec bits, so this conversion is exact (no further rounding).
         * Then format with the SAME directed rounding so every step preserves the
         * bound: round_up=1 -> value >= true x; round_up=0 -> value <= true x. */
        mpfr_init2(m, mant_prec);
        arf_get_mpfr(m, endpoint, round_up ? MPFR_RNDU : MPFR_RNDD);
        mpfr_asprintf(&mstr, round_up ? "%.*RUe" : "%.*RDe",
                      (int) (digits - 1), m);
        assert(mstr != NULL);
        out = dup_cstr(mstr);
        mpfr_free_str(mstr);
        mpfr_clear(m);
    }

    arf_clear(endpoint);
    return out;
}

/* ----- enum -> string mappers ------------------------------------------- */

static const char *
adaptive_status_str(arbsdp_adaptive_status s)
{
    switch (s) {
    case ARBSDP_ADAPTIVE_OPTIMAL:    return "optimal";
    case ARBSDP_ADAPTIVE_LIMIT:      return "limit";
    case ARBSDP_ADAPTIVE_INFEASIBLE: return "infeasible";
    }
    return "unknown";
}

static const char *
certify_status_str(arbsdp_certify_status s)
{
    switch (s) {
    case ARBSDP_CERTIFY_OPTIMAL_BRACKET:   return "optimal";
    case ARBSDP_CERTIFY_INCONCLUSIVE:      return "inconclusive";
    case ARBSDP_CERTIFY_PRIMAL_INFEASIBLE: return "primal_infeasible";
    case ARBSDP_CERTIFY_DUAL_INFEASIBLE:   return "dual_infeasible";
    }
    return "unknown";
}

/* significant decimal digits implied by `prec` bits: floor(prec*log10(2))+1,
 * capped to [1, 1000] (avoids absurd field widths from a huge prec). */
static slong
digits_from_prec(slong prec)
{
    slong d = (slong) ((double) prec * 0.301) + 1;
    if (d < 1)
        d = 1;
    if (d > 1000)
        d = 1000;
    return d;
}

/* ----- a tiny growable string buffer (asserts on OOM; CLAUDE.md rule 5) -- */

typedef struct {
    char  *p;
    size_t len;
    size_t cap;
} sbuf;

static void
sbuf_init(sbuf *s)
{
    s->cap = 256;
    s->p = malloc(s->cap);
    assert(s->p != NULL);
    s->p[0] = '\0';
    s->len = 0;
}

static void
sbuf_reserve(sbuf *s, size_t extra)
{
    if (s->len + extra + 1 > s->cap) {
        size_t newcap = s->cap;
        while (s->len + extra + 1 > newcap)
            newcap *= 2;
        char *grown = realloc(s->p, newcap);
        assert(grown != NULL);
        s->p = grown;
        s->cap = newcap;
    }
}

static void
sbuf_puts(sbuf *s, const char *str)
{
    size_t n = strlen(str);
    sbuf_reserve(s, n);
    memcpy(s->p + s->len, str, n + 1);
    s->len += n;
}

#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
static void
sbuf_printf(sbuf *s, const char *fmt, ...)
{
    va_list ap, ap2;
    int n;

    va_start(ap, fmt);
    va_copy(ap2, ap);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    assert(n >= 0);
    sbuf_reserve(s, (size_t) n);
    vsnprintf(s->p + s->len, (size_t) n + 1, fmt, ap2);
    va_end(ap2);
    s->len += (size_t) n;
}

char *
arbsdp_result_to_json(const arbsdp_problem *p,
                      const arbsdp_adaptive_result *ar,
                      arbsdp_certify_status cstatus,
                      const arb_t lb, const arb_t ub,
                      slong prec_c, const arbsdp_apriori *ab)
{
    assert(p != NULL);
    assert(ar != NULL); /* NOT NULL-safe (io.h contract) */

    slong digits_val = digits_from_prec(ar->final_prec);
    slong digits_c   = digits_from_prec(prec_c);

    sbuf s;
    sbuf_init(&s);

    sbuf_puts(&s, "{");

    /* version */
    sbuf_printf(&s, "\"arbsdp_version\": \"%s\", ", ARBSDP_VERSION);

    /* problem */
    sbuf_printf(&s, "\"problem\": {\"m\": %d, \"nblocks\": %d, \"block_sizes\": [",
                p->m, p->nblocks);
    for (int b = 0; b < p->nblocks; b++)
        sbuf_printf(&s, "%s%d", b ? ", " : "", p->block_sizes[b]);
    sbuf_printf(&s, "], \"maximize\": %s}, ", p->maximize ? "true" : "false");

    /* solve (point-mode recovered objective; round_up=1 acceptable, io.h note) */
    {
        char *val = arbsdp_arb_decimal(ar->res.value, 1, digits_val);
        sbuf_printf(&s,
                    "\"solve\": {\"adaptive_status\": \"%s\", \"value\": \"%s\", "
                    "\"final_prec\": %lld, \"iters\": %d, \"prec_history\": [",
                    adaptive_status_str(ar->status), val,
                    (long long) ar->final_prec, ar->res.iters);
        free(val);
        for (int k = 0; k < ar->prec_history_len; k++)
            sbuf_printf(&s, "%s%lld", k ? ", " : "",
                        (long long) ar->prec_history[k].prec);
        sbuf_puts(&s, "]}, ");
    }

    /* certificate (obj_lb toward -inf, obj_ub toward +inf -> rigorous enclosure) */
    {
        char *slb = arbsdp_arb_decimal(lb, 0, digits_c);
        char *sub = arbsdp_arb_decimal(ub, 1, digits_c);
        sbuf_printf(&s,
                    "\"certificate\": {\"status\": \"%s\", \"prec_c\": %lld, "
                    "\"obj_lb\": \"%s\", \"obj_ub\": \"%s\", \"trace_bounds\": {",
                    certify_status_str(cstatus), (long long) prec_c, slb, sub);
        free(slb);
        free(sub);

        if (ab != NULL) {
            int first = 1;
            for (int b = 0; b < ab->nblocks; b++) {
                if (ab->xbar_set[b]) {
                    char *xb = arbsdp_arb_decimal(&ab->xbar[b], 1, digits_c);
                    sbuf_printf(&s, "%s\"%d\": \"%s\"", first ? "" : ", ", b, xb);
                    free(xb);
                    first = 0;
                }
            }
        }
        sbuf_puts(&s, "}}");
    }

    sbuf_puts(&s, "}");

    return s.p; /* caller frees */
}
