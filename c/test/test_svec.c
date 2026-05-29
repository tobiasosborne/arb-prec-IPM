/*
 * test/test_svec.c -- svec / smat / symmetrize / frob_inner unit tests.
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): these are written before c/src/svec.c
 * exists; they link-fail first, then pass once the impl lands.
 *
 * Property coverage (bead arb-prec-IPM-b04):
 *   1. arbsdp_svec_len(n) == n(n+1)/2 for n = 1..6.
 *   2. svec/smat roundtrip on exact rational symmetric A: smat(svec(A)) == A.
 *   3. inner-product identity: svec(A).svec(B) == frob_inner(A,B), incl. a case
 *      ([[0,1],[1,0]]) where a missing sqrt(2) would be caught (<A,A> = 2).
 *   4. diagonal NOT scaled, off-diagonal IS: explicit [[0,1],[1,0]] check that
 *      the packed off-diagonal entry equals sqrt(2).
 *   5. symmetrize correctness on a nonsymmetric input.
 *
 * Assertions use arb_overlaps / arb_contains against an exactly-constructed
 * expected value (NOT float equality), per the bead instructions.
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/svec.h"

#define PREC 256

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* Set arb_mat entry (i,j) to an exact rational p/q (q != 0). */
static void
mat_set_q(arb_mat_t A, slong i, slong j, slong p, slong q)
{
    arb_set_si(arb_mat_entry(A, i, j), p);
    arb_div_si(arb_mat_entry(A, i, j), arb_mat_entry(A, i, j), q, PREC);
}

/* ----- Test 1: svec length ------------------------------------------------ */
static void
test_len(void)
{
    for (slong n = 1; n <= 6; n++) {
        slong expect = n * (n + 1) / 2;
        CHECK(arbsdp_svec_len(n) == expect, "svec_len mismatch");
    }
}

/* ----- Test 2: svec/smat roundtrip on exact symmetric A ------------------- */
static void
test_roundtrip(void)
{
    const slong n = 4;
    slong d = arbsdp_svec_len(n);
    arb_mat_t A, B;
    arb_ptr v;

    arb_mat_init(A, n, n);
    arb_mat_init(B, n, n);
    v = _arb_vec_init(d);

    /* Build an exact-rational symmetric A_{ij} = (i+1)/(j+2) symmetrized. */
    for (slong i = 0; i < n; i++) {
        for (slong j = i; j < n; j++) {
            mat_set_q(A, i, j, (i + 1) * (j + 3), (i + j + 2) * 5);
            arb_set(arb_mat_entry(A, j, i), arb_mat_entry(A, i, j));
        }
    }

    arbsdp_svec(v, A, PREC);
    arbsdp_smat(B, v, n, PREC);

    for (slong i = 0; i < n; i++) {
        for (slong j = 0; j < n; j++) {
            CHECK(arb_overlaps(arb_mat_entry(A, i, j),
                               arb_mat_entry(B, i, j)),
                  "roundtrip entry mismatch");
            /* Tight: B must be accurate to many bits. */
            CHECK(arb_rel_accuracy_bits(arb_mat_entry(B, i, j)) > 200
                      || arb_is_zero(arb_mat_entry(B, i, j)),
                  "roundtrip entry not tight");
        }
    }

    _arb_vec_clear(v, d);
    arb_mat_clear(A);
    arb_mat_clear(B);
}

/* ----- Test 3: inner-product identity svec(A).svec(B) == <A,B> ------------ */
static void
test_inner_identity(void)
{
    const slong n = 5;
    slong d = arbsdp_svec_len(n);
    arb_mat_t A, B;
    arb_ptr va, vb;
    arb_t dot, frob;

    arb_mat_init(A, n, n);
    arb_mat_init(B, n, n);
    va = _arb_vec_init(d);
    vb = _arb_vec_init(d);
    arb_init(dot);
    arb_init(frob);

    /* Two distinct exact-rational symmetric matrices with nonzero off-diags. */
    for (slong i = 0; i < n; i++) {
        for (slong j = i; j < n; j++) {
            mat_set_q(A, i, j, 2 * i - j + 1, 3);
            arb_set(arb_mat_entry(A, j, i), arb_mat_entry(A, i, j));
            mat_set_q(B, i, j, i + 2 * j - 4, 7);
            arb_set(arb_mat_entry(B, j, i), arb_mat_entry(B, i, j));
        }
    }

    arbsdp_svec(va, A, PREC);
    arbsdp_svec(vb, B, PREC);

    /* dot = sum_k va[k] vb[k]. */
    arb_dot(dot, NULL, 0, va, 1, vb, 1, d, PREC);

    /* frob = <A,B> dense. */
    arbsdp_frob_inner(frob, A, B, PREC);

    CHECK(arb_overlaps(dot, frob), "svec dot != frob inner");
    CHECK(arb_rel_accuracy_bits(dot) > 200, "inner-product dot not tight");

    _arb_vec_clear(va, d);
    _arb_vec_clear(vb, d);
    arb_mat_clear(A);
    arb_mat_clear(B);
    arb_clear(dot);
    arb_clear(frob);
}

/* ----- Test 3b: the sqrt(2)-sensitive case A = B = [[0,1],[1,0]] ----------
 * <A,A> = 2.  A missing sqrt(2) would give svec(A).svec(A) = 1 (off-diagonal
 * stored unscaled) -- this catches it. */
static void
test_offdiag_inner(void)
{
    const slong n = 2;
    slong d = arbsdp_svec_len(n);
    arb_mat_t A;
    arb_ptr v;
    arb_t dot, two;

    arb_mat_init(A, n, n);
    v = _arb_vec_init(d);
    arb_init(dot);
    arb_init(two);

    /* A = [[0,1],[1,0]]. */
    arb_set_si(arb_mat_entry(A, 0, 1), 1);
    arb_set_si(arb_mat_entry(A, 1, 0), 1);

    arbsdp_svec(v, A, PREC);
    arb_dot(dot, NULL, 0, v, 1, v, 1, d, PREC);

    arb_set_si(two, 2);
    CHECK(arb_overlaps(dot, two), "svec(A).svec(A) != 2 for [[0,1],[1,0]]");
    CHECK(arb_contains(dot, two), "<A,A>=2 not enclosed");

    _arb_vec_clear(v, d);
    arb_mat_clear(A);
    arb_clear(dot);
    arb_clear(two);
}

/* ----- Test 4: diagonal unscaled, off-diagonal scaled by sqrt(2) ----------
 * For [[0,1],[1,0]] (n=2): k order is (0,0)->0, (0,1)->1, (1,1)->2.
 *   v[0] = A_00 = 0  (diagonal, unscaled)
 *   v[1] = sqrt(2)*A_01 = sqrt(2)
 *   v[2] = A_11 = 0
 * Also a diagonal-entry check: [[3,0],[0,5]] -> v = [3, 0, 5] unscaled. */
static void
test_scaling_explicit(void)
{
    const slong n = 2;
    slong d = arbsdp_svec_len(n);
    arb_mat_t A;
    arb_ptr v;
    arb_t sqrt2, three, five;

    arb_mat_init(A, n, n);
    v = _arb_vec_init(d);
    arb_init(sqrt2);
    arb_init(three);
    arb_init(five);
    arb_sqrt_ui(sqrt2, 2, PREC);
    arb_set_si(three, 3);
    arb_set_si(five, 5);

    /* Off-diagonal case [[0,1],[1,0]]. */
    arb_set_si(arb_mat_entry(A, 0, 1), 1);
    arb_set_si(arb_mat_entry(A, 1, 0), 1);
    arbsdp_svec(v, A, PREC);

    CHECK(arb_is_zero(v + 0), "v[0] (diag A_00) should be 0");
    CHECK(arb_overlaps(v + 1, sqrt2), "v[1] off-diag should be sqrt(2)");
    CHECK(arb_rel_accuracy_bits(v + 1) > 200, "off-diag entry not tight");
    CHECK(arb_is_zero(v + 2), "v[2] (diag A_11) should be 0");

    /* Diagonal case [[3,0],[0,5]] -- diagonals NOT scaled. */
    arb_mat_zero(A);
    arb_set_si(arb_mat_entry(A, 0, 0), 3);
    arb_set_si(arb_mat_entry(A, 1, 1), 5);
    arbsdp_svec(v, A, PREC);

    CHECK(arb_equal(v + 0, three), "v[0] diag should equal 3 (unscaled)");
    CHECK(arb_is_zero(v + 1), "v[1] off-diag should be 0");
    CHECK(arb_equal(v + 2, five), "v[2] diag should equal 5 (unscaled)");

    _arb_vec_clear(v, d);
    arb_mat_clear(A);
    arb_clear(sqrt2);
    arb_clear(three);
    arb_clear(five);
}

/* ----- Test 5: symmetrize on a nonsymmetric input ------------------------- */
static void
test_symmetrize(void)
{
    const slong n = 3;
    arb_mat_t A;
    arb_t expect;

    arb_mat_init(A, n, n);
    arb_init(expect);

    /* A_{ij} = i*n + j + 1, clearly nonsymmetric. */
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++)
            arb_set_si(arb_mat_entry(A, i, j), i * n + j + 1);

    arbsdp_symmetrize(A, PREC);

    /* After symmetrize, entry (i,j) = (orig_ij + orig_ji)/2.
     * orig_ij = i*n+j+1, orig_ji = j*n+i+1; sum = (i+j)*n + i + j + 2;
     * average = ((i+j)*(n+1))/2 + 1. */
    for (slong i = 0; i < n; i++) {
        for (slong j = 0; j < n; j++) {
            slong num = (i + j) * (n + 1) + 2; /* 2*avg */
            arb_set_si(expect, num);
            arb_div_si(expect, expect, 2, PREC);
            CHECK(arb_overlaps(arb_mat_entry(A, i, j), expect),
                  "symmetrize entry mismatch");
            /* Symmetric: (i,j) == (j,i). */
            CHECK(arb_overlaps(arb_mat_entry(A, i, j),
                               arb_mat_entry(A, j, i)),
                  "symmetrize result not symmetric");
        }
    }

    arb_mat_clear(A);
    arb_clear(expect);
}

int
main(void)
{
    test_len();
    test_roundtrip();
    test_inner_identity();
    test_offdiag_inner();
    test_scaling_explicit();
    test_symmetrize();

    if (failures != 0) {
        fprintf(stderr, "test_svec: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_svec (all svec/smat/symmetrize/frob checks)\n");
    return EXIT_SUCCESS;
}
