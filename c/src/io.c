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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

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
    arbsdp_problem_clear(p); /* start from a clean, init'd state */

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
