/* golden_provenance.c — focused scanner for golden/<name>/provenance.json (schema v2).
 * Parses OUR OWN file format (CLAUDE.md: not a general JSON library). Fails loud on
 * malformed/missing-required input (rule 5). All reads bounded to the buffer (rule 7). */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <assert.h>
#include <sys/stat.h>
#include "golden_provenance.h"

/* Locate the value position for a top-level key: the char after the ':' that follows
 * the quoted token "key". Returns NULL if not present. Bounded to [buf, buf+len). */
static const char *find_value(const char *buf, size_t len, const char *key) {
    char tok[160];
    int n = snprintf(tok, sizeof tok, "\"%s\"", key);
    if (n <= 0 || (size_t)n >= sizeof tok) return NULL;
    size_t tl = (size_t)n;
    for (size_t i = 0; i + tl <= len; i++) {
        if (memcmp(buf + i, tok, tl) != 0) continue;
        size_t j = i + tl;
        while (j < len && isspace((unsigned char)buf[j])) j++;
        if (j < len && buf[j] == ':') {
            j++;
            while (j < len && isspace((unsigned char)buf[j])) j++;
            return buf + j;
        }
    }
    return NULL;
}

/* Copy a quoted string value at p (which must point at the opening '"') into dst.
 * Our strings contain no escaped quotes; a backslash is treated literally. Always
 * NUL-terminates; truncates safely. Returns 1 on success, 0 if p is not a string. */
static int copy_string(const char *p, const char *end, char *dst, size_t cap) {
    if (p >= end || *p != '"') return 0;
    p++;
    size_t o = 0;
    while (p < end && *p != '"') {
        if (o + 1 < cap) dst[o++] = *p;
        p++;
    }
    dst[o < cap ? o : cap - 1] = '\0';
    return 1;
}

/* Read a numeric token at p into dst (bounded). Stops at , } ] whitespace or end. */
static void copy_number(const char *p, const char *end, char *dst, size_t cap) {
    size_t o = 0;
    while (p < end && (isdigit((unsigned char)*p) || *p == '-' || *p == '+'
                       || *p == '.' || *p == 'e' || *p == 'E')) {
        if (o + 1 < cap) dst[o++] = *p;
        p++;
    }
    dst[o] = '\0';
}

/* Optional flat string key -> bounded field. */
static void opt_string(const char *buf, size_t len, const char *key,
                       char *dst, size_t cap) {
    const char *p = find_value(buf, len, key);
    if (p) copy_string(p, buf + len, dst, cap);
}

/* Optional flat numeric key. Returns 1 if present (writes *out via strtod path). */
static int opt_number(const char *buf, size_t len, const char *key, char *num, size_t ncap) {
    const char *p = find_value(buf, len, key);
    if (!p) return 0;
    copy_number(p, buf + len, num, ncap);
    return num[0] != '\0';
}

/* Parse the known_gaps array: scan each {...} object for kind/reason/bead. */
static void parse_gaps(const char *buf, size_t len, golden_case *c) {
    const char *p = find_value(buf, len, "known_gaps");
    const char *end = buf + len;
    if (!p || p >= end || *p != '[') return;
    p++;
    const char *abeg = p;
    while (p < end && *p != ']') p++;
    const char *aend = p; /* array span [abeg, aend) */
    int first = 1;
    const char *q = abeg;
    while (q < aend) {
        if (*q != '{') { q++; continue; }
        const char *obeg = q + 1;
        const char *oe = obeg;
        while (oe < aend && *oe != '}') oe++;
        size_t olen = (size_t)(oe - obeg);
        char kind[32] = "";
        const char *kp = find_value(obeg, olen, "kind");
        if (kp) copy_string(kp, oe, kind, sizeof kind);
        if (strcmp(kind, "status") == 0) c->gap_status = 1;
        else if (strcmp(kind, "correctness") == 0) c->gap_correctness = 1;
        else if (strcmp(kind, "efficiency") == 0) c->gap_efficiency = 1;
        if (first) {
            const char *rp = find_value(obeg, olen, "reason");
            if (rp) copy_string(rp, oe, c->gap_reason, sizeof c->gap_reason);
            const char *bp = find_value(obeg, olen, "bead");
            if (bp) copy_string(bp, oe, c->gap_bead, sizeof c->gap_bead);
            first = 0;
        }
        q = (oe < aend) ? oe + 1 : aend;
    }
}

/* Parse one provenance file at path into *c (dats_path is set by caller). Returns
 * 1 on success, 0 on hard error (already reported to stderr). */
static int parse_provenance(const char *path, golden_case *c) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: %s: cannot open\n", path); return 0; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) { fclose(f); fprintf(stderr, "FATAL: %s: ftell\n", path); return 0; }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); fprintf(stderr, "FATAL: %s: oom\n", path); return 0; }
    size_t len = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[len] = '\0';

    /* defaults */
    memset(c, 0, sizeof *c);
    snprintf(c->expected_status, sizeof c->expected_status, "optimal");
    c->target_digits = 15;
    c->prec_cap = 8192;
    c->correctness_tol_digits = 0;
    c->expected_max_bits_per_digit = 1e9;
    c->expected_max_iters = INT_MAX;

    char num[64];
    const char *np;
    if (!(np = find_value(buf, len, "name")) || !copy_string(np, buf + len, c->name, sizeof c->name)) {
        fprintf(stderr, "FATAL: %s: missing required key \"name\"\n", path); free(buf); return 0;
    }
    if (!(np = find_value(buf, len, "optimal_value")) ||
        !copy_string(np, buf + len, c->optimal_value, sizeof c->optimal_value)) {
        fprintf(stderr, "FATAL: %s: missing required key \"optimal_value\"\n", path); free(buf); return 0;
    }
    if (opt_number(buf, len, "m", num, sizeof num)) c->m = (int)strtol(num, NULL, 10);
    if (opt_number(buf, len, "nblocks", num, sizeof num)) c->nblocks = (int)strtol(num, NULL, 10);
    if (opt_number(buf, len, "optimal_value_digits", num, sizeof num)) c->optimal_value_digits = (int)strtol(num, NULL, 10);
    if (opt_number(buf, len, "target_digits", num, sizeof num)) c->target_digits = (int)strtol(num, NULL, 10);
    if (opt_number(buf, len, "prec_cap", num, sizeof num)) c->prec_cap = strtol(num, NULL, 10);
    if (opt_number(buf, len, "correctness_tol_digits", num, sizeof num)) c->correctness_tol_digits = (int)strtol(num, NULL, 10);
    if (opt_number(buf, len, "expected_max_bits_per_digit", num, sizeof num)) c->expected_max_bits_per_digit = strtod(num, NULL);
    if (opt_number(buf, len, "expected_max_iters", num, sizeof num)) c->expected_max_iters = (int)strtol(num, NULL, 10);
    opt_string(buf, len, "expected_status", c->expected_status, sizeof c->expected_status);
    parse_gaps(buf, len, c);
    free(buf);
    return 1;
}

static int by_name(const void *a, const void *b) {
    return strcmp(((const golden_case *)a)->name, ((const golden_case *)b)->name);
}

int golden_discover(const char *golden_dir, golden_case *cases, int max) {
    DIR *d = opendir(golden_dir);
    if (!d) { fprintf(stderr, "FATAL: %s: cannot open dir\n", golden_dir); return -1; }
    int count = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        /* Generous locals (golden_dir up to PATH_MAX, name twice) so bounded snprintf
         * has no truncation risk; the final dats_path[512] copy is bounded separately. */
        char prov[PATH_MAX + NAME_MAX + 32], dats[PATH_MAX + 2 * NAME_MAX + 32];
        snprintf(prov, sizeof prov, "%s/%s/provenance.json", golden_dir, e->d_name);
        snprintf(dats, sizeof dats, "%s/%s/%s.dat-s", golden_dir, e->d_name, e->d_name);
        struct stat sp, sd;
        if (stat(prov, &sp) != 0 || !S_ISREG(sp.st_mode)) continue;
        if (stat(dats, &sd) != 0 || !S_ISREG(sd.st_mode)) continue;
        if (count >= max) {
            fprintf(stderr, "FATAL: %s: more cases than max=%d\n", golden_dir, max);
            closedir(d); return -1;
        }
        if (!parse_provenance(prov, &cases[count])) { closedir(d); return -1; }
        int dn = snprintf(cases[count].dats_path, sizeof cases[count].dats_path, "%s", dats);
        if (dn < 0 || (size_t)dn >= sizeof cases[count].dats_path) {
            fprintf(stderr, "FATAL: %s: .dat-s path too long\n", dats);
            closedir(d); return -1;
        }
        count++;
    }
    closedir(d);
    qsort(cases, (size_t)count, sizeof *cases, by_name);
    return count;
}
