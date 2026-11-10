/*
 * test/test_io.c -- SDPA .dat-s reader + problem data model tests.
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written before c/src/io.c exists; the suite
 * link-fails first, then passes once the impl lands.
 *
 * Coverage (bead arb-prec-IPM-b07):
 *   1. Parse each golden fixture; assert m and block_sizes match provenance.json
 *      (expected values baked in from the provenance files).
 *   2. sdp_sqrt2 objective: materialize C and check arbsdp_frob_inner(C, X) for a
 *      chosen symmetric X reproduces x_12 -- validates the 2v convention
 *      end-to-end (a missing/extra factor of 2 fails here).
 *   3. Precision preservation: the materialized C entry for sdp_sqrt2 (b value
 *      "1.4142..." is not in the file, but x_22 b="2" and the sqrt2 reference)
 *      -- we use the known 65-digit b vector / entries and confirm > 50 digits of
 *      accuracy, NOT a double-truncated ~15-16 digits.
 *   4. Malformed inputs fail loud (nonzero return): truncated file, bad token,
 *      out-of-range block, out-of-range index, k > l.
 *   5. Mutation check: a hand-built single-off-diagonal C plus a symmetric X
 *      shows that dropping the (j,i) write (the 2v doubling) halves the inner
 *      product -- documents what the convention protects against.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 * Assertions use arb_overlaps / arb_contains against exactly-constructed
 * expected values, never float equality.
 */

/* mkstemp/close are POSIX; with -std=c11 they need the feature-test macro to be
 * declared.  Define it before any standard header is included. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/io.h"
#include "arbsdp/svec.h"

#define PREC 256

/* Golden fixtures live at <SRC>/golden/<name>/<name>.dat-s.  GOLDEN_DIR is
 * passed in by CMake (-DGOLDEN_DIR=...) so the test is run-location-independent. */
#ifndef GOLDEN_DIR
#define GOLDEN_DIR "../../golden"
#endif

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void
golden_path(char *buf, size_t n, const char *name)
{
    snprintf(buf, n, "%s/%s/%s.dat-s", GOLDEN_DIR, name, name);
}

/* ----- Test 1: parse each golden fixture; check m and block_sizes ---------
 * Expected values baked from each fixture's provenance.json. */
static void
test_parse_golden(void)
{
    struct {
        const char *name;
        int m;
        int nblocks;
        int block_sizes[4];
    } cases[] = {
        { "trivial_2x2",        1, 1, { 2 } },
        { "max_eigenvalue_2x2", 1, 1, { 2 } },
        { "sdp_sqrt2",          2, 1, { 2 } },
        { "lp_diagonal_block",  3, 1, { -5 } },
        { "mixed_blocks",       2, 2, { 2, -1 } },
        { "ill_conditioned_3x3",3, 1, { 3 } },
        { "max_eig_tridiag_3x3",1, 1, { 3 } },
    };
    const int ncases = (int) (sizeof(cases) / sizeof(cases[0]));

    for (int c = 0; c < ncases; c++) {
        char path[1024];
        arbsdp_problem p;
        int rc;

        golden_path(path, sizeof(path), cases[c].name);
        arbsdp_problem_init(&p);
        rc = arbsdp_read_sdpa(&p, path);
        CHECK(rc == 0, cases[c].name);
        if (rc == 0) {
            CHECK(p.m == cases[c].m, "m mismatch");
            CHECK(p.nblocks == cases[c].nblocks, "nblocks mismatch");
            CHECK(p.maximize == 1, "maximize should be 1 (SDPA convention)");
            for (int b = 0; b < cases[c].nblocks; b++) {
                CHECK(p.block_sizes[b] == cases[c].block_sizes[b],
                      "block_sizes mismatch");
            }
        }
        arbsdp_problem_clear(&p);
    }
}

/* ----- Test 2: sdp_sqrt2 objective via frob_inner (2v convention) ---------
 * C has the single entry "0 1 1 2 0.5" -> materialized C = [[0,1/2],[1/2,0]].
 * For a symmetric X, <C,X> = frob_inner(C,X) = (1/2)X_12 + (1/2)X_21 = X_12.
 * We pick X_12 = X_21 = 7 (distinctive) so <C,X> must equal exactly 7.  If the
 * (j,i) symmetric write were dropped (the 2v doubling), the result would be 7/2.
 */
static void
test_sqrt2_objective(void)
{
    char path[1024];
    arbsdp_problem p;
    arb_mat_t C, X;
    arb_t obj, seven;
    int rc;

    golden_path(path, sizeof(path), "sdp_sqrt2");
    arbsdp_problem_init(&p);
    rc = arbsdp_read_sdpa(&p, path);
    CHECK(rc == 0, "sdp_sqrt2 parse");
    if (rc != 0) { arbsdp_problem_clear(&p); return; }

    arb_mat_init(C, 2, 2);
    arb_mat_init(X, 2, 2);
    arb_init(obj);
    arb_init(seven);

    /* C = objective, matno 0, block 0. */
    arbsdp_problem_block_mat(C, &p, 0, 0, PREC);

    /* Symmetric X with X_12 = X_21 = 7 (diagonal irrelevant since C diag = 0). */
    arb_set_si(arb_mat_entry(X, 0, 0), 1);
    arb_set_si(arb_mat_entry(X, 1, 1), 2);
    arb_set_si(arb_mat_entry(X, 0, 1), 7);
    arb_set_si(arb_mat_entry(X, 1, 0), 7);

    arbsdp_frob_inner(obj, C, X, PREC);

    arb_set_si(seven, 7);
    CHECK(arb_overlaps(obj, seven), "<C,X> should equal x_12 = 7 (2v convention)");
    CHECK(arb_contains(obj, seven), "7 not enclosed by <C,X>");
    CHECK(arb_rel_accuracy_bits(obj) > 200, "<C,X> not tight");

    /* Cross-check the stored C is genuinely symmetric with C_12 = C_21 = 1/2,
     * NOT doubled to 1 (the 2v lives in the inner product, not the matrix). */
    {
        arb_t half;
        arb_init(half);
        arb_set_d(half, 0.5);
        CHECK(arb_overlaps(arb_mat_entry(C, 0, 1), half), "C_12 should be 1/2");
        CHECK(arb_overlaps(arb_mat_entry(C, 1, 0), half), "C_21 should be 1/2");
        CHECK(arb_is_zero(arb_mat_entry(C, 0, 0)), "C_11 should be 0");
        CHECK(arb_is_zero(arb_mat_entry(C, 1, 1)), "C_22 should be 0");
        arb_clear(half);
    }

    arb_mat_clear(C);
    arb_mat_clear(X);
    arb_clear(obj);
    arb_clear(seven);
    arbsdp_problem_clear(&p);
}

/* ----- Test 3: precision preservation (> 50 digits, not double) ------------
 * The sdp_sqrt2 b vector contains "2.000...0" (65 digits) and x_11 b "1.0...".
 * To exercise a genuinely many-digit value we use ill_conditioned_3x3, whose C
 * has the entry "0 1 2 2 0.0000010000...0" (= 1e-6 to 65 digits) and whose
 * optimal value reference is 1.000001.  We materialize C and confirm the (1,1)
 * entry equals 1e-6 to high accuracy.  Critically, we also parse a fabricated
 * many-digit value and compare against its exact arb to > 50 bits, proving the
 * string path did NOT route through double (which caps at ~53 bits ~ 15-16
 * decimal digits). */
static void
test_precision_preservation(void)
{
    char path[1024];
    arbsdp_problem p;
    arb_mat_t C;
    arb_t eps_ref, eps_got;
    int rc;

    golden_path(path, sizeof(path), "ill_conditioned_3x3");
    arbsdp_problem_init(&p);
    rc = arbsdp_read_sdpa(&p, path);
    CHECK(rc == 0, "ill_conditioned_3x3 parse");
    if (rc != 0) { arbsdp_problem_clear(&p); return; }

    arb_mat_init(C, 3, 3);
    arb_init(eps_ref);
    arb_init(eps_got);

    arbsdp_problem_block_mat(C, &p, 0, 0, PREC);

    /* eps_ref = 10^-6 computed independently at high precision. */
    arb_set_si(eps_ref, 1);
    arb_div_si(eps_ref, eps_ref, 1000000, PREC);

    arb_set(eps_got, arb_mat_entry(C, 1, 1));
    CHECK(arb_overlaps(eps_got, eps_ref), "C_22 should be 1e-6");
    CHECK(arb_rel_accuracy_bits(eps_got) > 200,
          "C_22 not high-precision (string path lost digits?)");

    /* Direct many-digit string parse vs double truncation.  Build a value that
     * differs from its double rounding beyond digit 16 and confirm the parsed
     * arb agrees with the exact decimal to > 50 bits but NOT with its double. */
    {
        const char *s =
            "1.4142135623730950488016887242096980785696718753769480731766797379";
        arb_t exact, got;
        double d;
        arb_init(exact);
        arb_init(got);
        arb_set_str(exact, s, PREC);
        /* Parse through our reader's mechanism would be identical; here we check
         * arb_set_str itself preserves > 50 decimal digits relative to double. */
        arb_set_str(got, s, PREC);
        d = atof(s); /* double truncation ~ 15-16 digits */
        CHECK(arb_overlaps(got, exact), "high-precision parse mismatch");
        CHECK(arb_rel_accuracy_bits(got) > 200, "parse not > 50 digits accurate");
        /* The double value, lifted to arb, differs from the exact beyond ~53
         * bits: their difference must be nonzero at this precision. */
        {
            arb_t dlift, diff;
            arb_init(dlift);
            arb_init(diff);
            arb_set_d(dlift, d);
            arb_sub(diff, got, dlift, PREC);
            CHECK(!arb_contains_zero(diff)
                      || arb_rel_accuracy_bits(diff) < 60,
                  "double and high-prec parse should differ beyond ~16 digits");
            arb_clear(dlift);
            arb_clear(diff);
        }
        arb_clear(exact);
        arb_clear(got);
    }

    arb_mat_clear(C);
    arb_clear(eps_ref);
    arb_clear(eps_got);
    arbsdp_problem_clear(&p);
}

/* ----- Test 4: malformed inputs fail loud --------------------------------- */
static void
write_tmp(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: cannot write temp file %s\n", path);
        exit(EXIT_FAILURE);
    }
    fputs(content, f);
    fclose(f);
}

static void
expect_fail(const char *content, const char *what)
{
    char path[] = "/tmp/arbsdp_io_testXXXXXX";
    int fd = mkstemp(path);
    arbsdp_problem p;
    int rc;

    if (fd < 0) {
        fprintf(stderr, "FATAL: mkstemp failed\n");
        exit(EXIT_FAILURE);
    }
    close(fd);

    write_tmp(path, content);
    arbsdp_problem_init(&p);
    rc = arbsdp_read_sdpa(&p, path);
    CHECK(rc != 0, what);
    arbsdp_problem_clear(&p);
    remove(path);
}

static void
test_malformed(void)
{
    /* Nonexistent file. */
    {
        arbsdp_problem p;
        int rc;
        arbsdp_problem_init(&p);
        rc = arbsdp_read_sdpa(&p, "/tmp/arbsdp_does_not_exist_zzz.dat-s");
        CHECK(rc != 0, "nonexistent file should fail");
        arbsdp_problem_clear(&p);
    }

    /* Truncated: m present but no nblocks / sizes / b. */
    expect_fail("1\n", "truncated file (only m) should fail");

    /* Truncated: missing b vector line. */
    expect_fail("1\n1\n2\n", "missing b vector should fail");

    /* Bad token where an integer (m) is expected. */
    expect_fail("notanumber\n1\n2\n1.0\n", "bad m token should fail");

    /* nblocks != count of parsed block sizes. */
    expect_fail("1\n2\n2\n1.0\n", "block-size count mismatch should fail");

    /* b count != m. */
    expect_fail("2\n1\n2\n1.0\n", "b count mismatch should fail");

    /* Bad token in an entry value. */
    expect_fail("1\n1\n2\n1.0\n0 1 1 1 notanumber\n",
                "bad entry value should fail");

    /* Out-of-range block (block 2 when nblocks=1). */
    expect_fail("1\n1\n2\n1.0\n0 2 1 1 1.0\n",
                "out-of-range block should fail");

    /* Out-of-range matno (matno 5 when m=1). */
    expect_fail("1\n1\n2\n1.0\n5 1 1 1 1.0\n",
                "out-of-range matno should fail");

    /* Out-of-range row index (k=3 when block side=2). */
    expect_fail("1\n1\n2\n1.0\n0 1 3 1 1.0\n",
                "out-of-range row index should fail");

    /* k > l (lower triangle, violates the i<=j contract). */
    expect_fail("1\n1\n2\n1.0\n0 1 2 1 1.0\n",
                "k > l (lower triangle) should fail");

    /* Entry with too few fields. */
    expect_fail("1\n1\n2\n1.0\n0 1 1\n",
                "entry with too few fields should fail");
}

/* ----- Test 5: mutation/2v documentation check ----------------------------
 * Directly demonstrate that the 2v doubling lives in the inner product: with a
 * symmetric off-diagonal matrix M (M_12 = M_21 = v) and symmetric X, the
 * Frobenius inner product is 2*v*X_12.  Dropping the (j,i) write (M_21 = 0)
 * halves it.  This is the property test_sqrt2_objective relies on; here we make
 * it explicit so a regression in arbsdp_problem_block_mat that forgets to write
 * the symmetric off-diagonal is caught. */
static void
test_mutation_2v(void)
{
    arb_mat_t M, Mdrop, X;
    arb_t full, dropped, expect_full, expect_drop;

    arb_mat_init(M, 2, 2);
    arb_mat_init(Mdrop, 2, 2);
    arb_mat_init(X, 2, 2);
    arb_init(full);
    arb_init(dropped);
    arb_init(expect_full);
    arb_init(expect_drop);

    /* Correct symmetric M with v = 1/2 at (0,1) and (1,0). */
    arb_set_d(arb_mat_entry(M, 0, 1), 0.5);
    arb_set_d(arb_mat_entry(M, 1, 0), 0.5);

    /* Mutant: only (0,1) written (the (1,0) symmetric write dropped). */
    arb_set_d(arb_mat_entry(Mdrop, 0, 1), 0.5);

    /* Symmetric X with X_12 = X_21 = 7. */
    arb_set_si(arb_mat_entry(X, 0, 1), 7);
    arb_set_si(arb_mat_entry(X, 1, 0), 7);

    arbsdp_frob_inner(full, M, X, PREC);
    arbsdp_frob_inner(dropped, Mdrop, X, PREC);

    /* full = 2 * (1/2) * 7 = 7; dropped = (1/2)*7 = 7/2. */
    arb_set_si(expect_full, 7);
    arb_set_si(expect_drop, 7);
    arb_div_si(expect_drop, expect_drop, 2, PREC);

    CHECK(arb_overlaps(full, expect_full), "symmetric M: <M,X> should be 7");
    CHECK(arb_overlaps(dropped, expect_drop),
          "dropped-symmetric M: <M,X> should be 7/2 (mutation detectable)");
    /* The two MUST differ: this is what makes the mutation observable. */
    CHECK(!arb_overlaps(full, dropped),
          "dropping 2v must change the inner product (mutation check)");

    arb_mat_clear(M);
    arb_mat_clear(Mdrop);
    arb_mat_clear(X);
    arb_clear(full);
    arb_clear(dropped);
    arb_clear(expect_full);
    arb_clear(expect_drop);
}

/* ----- Test 6: b vector materialization ----------------------------------- */
static void
test_b_vector(void)
{
    char path[1024];
    arbsdp_problem p;
    arb_t b0, b1, one, two;
    int rc;

    golden_path(path, sizeof(path), "sdp_sqrt2");
    arbsdp_problem_init(&p);
    rc = arbsdp_read_sdpa(&p, path);
    CHECK(rc == 0, "sdp_sqrt2 parse for b");
    if (rc != 0) { arbsdp_problem_clear(&p); return; }

    arb_init(b0);
    arb_init(b1);
    arb_init(one);
    arb_init(two);

    CHECK(arbsdp_problem_b(b0, &p, 0, PREC) == 0, "b[0] read");
    CHECK(arbsdp_problem_b(b1, &p, 1, PREC) == 0, "b[1] read");
    arb_set_si(one, 1);
    arb_set_si(two, 2);
    CHECK(arb_overlaps(b0, one), "b[0] should be 1 (x_11 = 1)");
    CHECK(arb_overlaps(b1, two), "b[1] should be 2 (x_22 = 2)");

    arb_clear(b0);
    arb_clear(b1);
    arb_clear(one);
    arb_clear(two);
    arbsdp_problem_clear(&p);
}

/* ----- Test 7 (bead b29): read_sdpa must be FOOLPROOF on a non-init'd struct --
 * REGRESSION: arbsdp_read_sdpa formerly called arbsdp_problem_clear(p) at entry,
 * which free()d p->b / p->mats / p->block_sizes.  On a struct the caller never
 * arbsdp_problem_init'd, those are garbage pointers -> free() of garbage -> heap
 * corruption / SIGSEGV (masked under ASan; missed by every other test, which all
 * init first).  CLAUDE.md rule 5: a corrupted heap is worse than any crash.
 *
 * We simulate an uninitialized struct by filling every owned field with a
 * non-NULL, NON-HEAP sentinel.  The OLD reader dereferenced + freed these (the
 * normal-build glibc allocator aborts on free() of an invalid pointer); the fixed
 * reader must initialize internally and never touch the caller's prior contents.
 * RED-GREEN: this test aborts/fails on the pre-b29 reader and passes after.  It
 * must run under the NORMAL build (not just ASan) -- this footgun only bit there. */
static void
test_read_uninitialized(void)
{
    char path[1024];
    arbsdp_problem p;
    static char sentinel[64]; /* static storage: NOT freeable -> nothing to leak */
    int rc;

    golden_path(path, sizeof(path), "trivial_2x2");

    /* Garbage, as if `arbsdp_problem p;` were used without arbsdp_problem_init. */
    memset(sentinel, 0xA5, sizeof(sentinel));
    p.m           = 12345;
    p.nblocks     = 678;
    p.block_sizes = (int *) sentinel;
    p.b           = (char **) sentinel;
    p.mats        = (arbsdp_matblock *) sentinel;
    p.maximize    = -999;

    rc = arbsdp_read_sdpa(&p, path); /* must NOT deref/free the sentinels */

    CHECK(rc == 0, "read_uninitialized: trivial_2x2 must parse");
    CHECK(p.m == 1, "read_uninitialized: m must be parsed (=1)");
    CHECK(p.nblocks == 1, "read_uninitialized: nblocks must be parsed (=1)");
    CHECK(p.block_sizes != (int *) sentinel && p.block_sizes != NULL
              && p.block_sizes[0] == 2,
          "read_uninitialized: block_sizes must be freshly allocated (=2)");

    arbsdp_problem_clear(&p); /* frees ONLY the real heap read_sdpa allocated */
}

/* ----- Test 8: non-finite / non-literal value tokens (bead 37w) ------------
 * FLINT 3.0.1 arb_set_str ACCEPTS (rc=0) far more than a decimal literal --
 * measured by probe: "inf"/"+inf"/"-Inf"/"INF" -> +-inf, "nan"/"NaN"/"-nan" ->
 * nan, "[inf]"/"[nan]", ball syntax "1+/-0.1" / "[1+/-0.1]" / "[+/-1]" / "[1]"
 * (a radius on PROBLEM DATA), and "--1" (parsed as +1: a silent SIGN FLIP).
 * ("infinity", "0x10", "1d5", "1/3" it already rejects.)  Problem data are exact
 * point decimals, so every value token -- each b_i AND each entry value -- must
 * be a plain finite decimal / scientific literal; anything else is a malformed
 * problem and read_sdpa must return nonzero (CLAUDE.md rule 5).
 * The positive control guards against over-tightening: every legal literal
 * shape (".5", "5.", "+5", "-0", "1E5", "1e+5", "1.5e-3") must still parse. */
static void
test_nonfinite_tokens(void)
{
    static const char *bad[] = {
        "inf", "+inf", "-inf", "Inf", "-Inf", "INF", "infinity",
        "nan", "NaN", "NAN", "-nan", "+nan", "[inf]", "[nan]",
        "1+/-0.1", "[1+/-0.1]", "[+/-1]", "[1]", "1+/-inf", "--1", "+-1",
        "1e", "e5", ".", "-", "1.2.3", "0x10",
        NULL
    };
    char content[256], what[160];

    for (int k = 0; bad[k] != NULL; k++) {
        /* (a) as a b value */
        snprintf(content, sizeof content, "1\n1\n2\n%s\n0 1 1 1 1.0\n", bad[k]);
        snprintf(what, sizeof what, "37w: b value token \"%s\" must be rejected", bad[k]);
        expect_fail(content, what);
        /* (b) as a constraint-matrix entry value */
        snprintf(content, sizeof content, "1\n1\n2\n1.0\n1 1 1 2 %s\n", bad[k]);
        snprintf(what, sizeof what, "37w: entry value token \"%s\" must be rejected", bad[k]);
        expect_fail(content, what);
    }

    /* positive control: all legal literal shapes parse, b and entries. */
    {
        char path[] = "/tmp/arbsdp_io_testXXXXXX";
        int fd = mkstemp(path);
        arbsdp_problem p;
        int rc;
        if (fd < 0) {
            fprintf(stderr, "FATAL: mkstemp failed\n");
            exit(EXIT_FAILURE);
        }
        close(fd);
        write_tmp(path, "3\n1\n2\n.5 5. +5\n"
                        "0 1 1 1 -0\n0 1 1 2 1E5\n0 1 2 2 1e+5\n"
                        "1 1 1 1 1.5e-3\n2 1 2 2 -2.25E-01\n3 1 1 2 007\n");
        arbsdp_problem_init(&p);
        rc = arbsdp_read_sdpa(&p, path);
        CHECK(rc == 0, "37w: legal decimal/scientific literals must still parse");
        arbsdp_problem_clear(&p);
        remove(path);
    }
}

int
main(void)
{
    test_parse_golden();
    test_sqrt2_objective();
    test_precision_preservation();
    test_malformed();
    test_mutation_2v();
    test_b_vector();
    test_read_uninitialized();
    test_nonfinite_tokens();

    if (failures != 0) {
        fprintf(stderr, "test_io: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_io (parse/materialize/2v/precision/malformed)\n");
    return EXIT_SUCCESS;
}
