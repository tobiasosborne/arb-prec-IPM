/*
 * test/test_linalg.c -- POINT-MODE spectral kernel tests for linalg.c:
 * arbsdp_eigh / arbsdp_psd_sqrt / arbsdp_psd_invsqrt.
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written before c/src/linalg.c exists; the
 * target link-fails first (no arbsdp_eigh symbol), then passes once the impl
 * lands.
 *
 * POINT MODE (CLAUDE.md rule 1): these are Layer-0 kernels.  Outputs are used as
 * POINTS; their ball radii are NOT trusted.  Assertions therefore compare
 * MIDPOINTS within a justified absolute tolerance rather than demanding tight
 * rigorous enclosures (overlaps/contains would not test point accuracy here --
 * a wide ball trivially overlaps).  TOL = 1e-50 at prec=256: Jacobi delivers
 * ~prec-bit (~1e-77) midpoint accuracy (measured in docs/probes/eig_audition.c),
 * so 1e-50 is far above the achieved accuracy yet far below any wrong-result
 * error (a transposed/scaled/mis-sorted result is off by O(1)).
 *
 * Property coverage (bead arb-prec-IPM-b05):
 *   1. eigh on diag(3,1,2) -> eigenvalues {1,2,3} ascending; Q a permutation.
 *   2. eigh on [[2,1],[1,2]] -> {1,3}; and 1D Laplacian 3x3 -> {2-sqrt2,2,2+sqrt2};
 *      check A*Q == Q*diag(lambda).
 *   3. psd_sqrt: sqrt(A)*sqrt(A) == A on PD fixtures (n=2,3).
 *   4. psd_invsqrt: invsqrt(A)*A*invsqrt(A) == I (n=2,3).
 *   5. orthogonality: Q^T*Q == I.
 *   6. MUTATION sanity (CLAUDE.md rule 8): the same checker, fed a deliberately
 *      WRONG matrix, must FAIL -- proving the test discriminates.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/linalg.h"

#define PREC 256

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* Point-mode closeness: |mid(a) - mid(b)| <= tol.  Radii are not trusted. */
static int
close_tol(const arb_t a, const arb_t b, double tol)
{
    arb_t d, t;
    int ok;
    arb_init(d);
    arb_init(t);
    arb_get_mid_arb(d, a);
    arb_get_mid_arb(t, b);
    arb_sub(d, d, t, PREC);
    arb_abs(d, d);
    arb_set_d(t, tol);
    ok = arb_le(d, t); /* compares midpoints (both exact here) */
    arb_clear(d);
    arb_clear(t);
    return ok;
}

#define TOL 1e-50

/* max|mid(A) - mid(B)| <= tol over all entries; returns 1 if close. */
static int
mat_close(const arb_mat_t A, const arb_mat_t B, double tol)
{
    slong n = arb_mat_nrows(A), m = arb_mat_ncols(A);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < m; j++)
            if (!close_tol(arb_mat_entry(A, i, j), arb_mat_entry(B, i, j), tol))
                return 0;
    return 1;
}

static void
set_sym(arb_mat_t A, slong i, slong j, slong v)
{
    arb_set_si(arb_mat_entry(A, i, j), v);
    arb_set_si(arb_mat_entry(A, j, i), v);
}

/* ----- Test 1: eigh on a diagonal matrix --------------------------------- */
static void
test_eigh_diag(void)
{
    const slong n = 3;
    arb_mat_t A, Q, QtQ, Qt;
    arb_ptr lam;
    arb_t e;

    arb_mat_init(A, n, n);
    arb_mat_init(Q, n, n);
    arb_mat_init(QtQ, n, n);
    arb_mat_init(Qt, n, n);
    lam = _arb_vec_init(n);
    arb_init(e);

    /* diag(3,1,2): eigenvalues sorted ascending must be {1,2,3}. */
    arb_set_si(arb_mat_entry(A, 0, 0), 3);
    arb_set_si(arb_mat_entry(A, 1, 1), 1);
    arb_set_si(arb_mat_entry(A, 2, 2), 2);

    arbsdp_eigh(lam, Q, A, PREC);

    arb_set_si(e, 1);
    CHECK(close_tol(lam + 0, e, TOL), "diag eigval[0] != 1");
    arb_set_si(e, 2);
    CHECK(close_tol(lam + 1, e, TOL), "diag eigval[1] != 2");
    arb_set_si(e, 3);
    CHECK(close_tol(lam + 2, e, TOL), "diag eigval[2] != 3");

    /* Q must be orthogonal (a signed permutation for distinct diag entries). */
    arb_mat_transpose(Qt, Q);
    arb_mat_mul(QtQ, Qt, Q, PREC);
    {
        arb_mat_t I;
        arb_mat_init(I, n, n);
        arb_mat_one(I);
        CHECK(mat_close(QtQ, I, TOL), "diag: Q^T Q != I");
        arb_mat_clear(I);
    }

    _arb_vec_clear(lam, n);
    arb_clear(e);
    arb_mat_clear(A);
    arb_mat_clear(Q);
    arb_mat_clear(QtQ);
    arb_mat_clear(Qt);
}

/* Helper: check A*Q == Q*diag(lam) (the eigen relation), point-mode. */
static int
check_eigrelation(const arb_mat_t A, const arb_mat_t Q, arb_srcptr lam, slong n)
{
    arb_mat_t AQ, QD, D;
    int ok;
    arb_mat_init(AQ, n, n);
    arb_mat_init(QD, n, n);
    arb_mat_init(D, n, n);
    for (slong i = 0; i < n; i++)
        arb_set(arb_mat_entry(D, i, i), lam + i);
    arb_mat_mul(AQ, A, Q, PREC);
    arb_mat_mul(QD, Q, D, PREC);
    ok = mat_close(AQ, QD, TOL);
    arb_mat_clear(AQ);
    arb_mat_clear(QD);
    arb_mat_clear(D);
    return ok;
}

/* ----- Test 2: eigh on KNOWN closed-form spectra ------------------------- */
static void
test_eigh_known(void)
{
    arb_t e, two, r2;
    arb_init(e);
    arb_init(two);
    arb_init(r2);
    arb_set_si(two, 2);
    arb_sqrt_ui(r2, 2, PREC);

    /* [[2,1],[1,2]] -> {1,3}. */
    {
        const slong n = 2;
        arb_mat_t A, Q;
        arb_ptr lam;
        arb_mat_init(A, n, n);
        arb_mat_init(Q, n, n);
        lam = _arb_vec_init(n);

        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        set_sym(A, 0, 1, 1);

        arbsdp_eigh(lam, Q, A, PREC);

        arb_set_si(e, 1);
        CHECK(close_tol(lam + 0, e, TOL), "[[2,1],[1,2]] eig[0] != 1");
        arb_set_si(e, 3);
        CHECK(close_tol(lam + 1, e, TOL), "[[2,1],[1,2]] eig[1] != 3");
        CHECK(check_eigrelation(A, Q, lam, n), "[[2,1],[1,2]] A*Q != Q*diag");

        _arb_vec_clear(lam, n);
        arb_mat_clear(A);
        arb_mat_clear(Q);
    }

    /* 1D Laplacian 3x3 [[2,-1,0],[-1,2,-1],[0,-1,2]] -> {2-sqrt2, 2, 2+sqrt2}. */
    {
        const slong n = 3;
        arb_mat_t A, Q;
        arb_ptr lam;
        arb_mat_init(A, n, n);
        arb_mat_init(Q, n, n);
        lam = _arb_vec_init(n);

        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        arb_set_si(arb_mat_entry(A, 2, 2), 2);
        set_sym(A, 0, 1, -1);
        set_sym(A, 1, 2, -1);

        arbsdp_eigh(lam, Q, A, PREC);

        arb_sub(e, two, r2, PREC); /* 2 - sqrt2 */
        CHECK(close_tol(lam + 0, e, TOL), "laplacian eig[0] != 2-sqrt2");
        CHECK(close_tol(lam + 1, two, TOL), "laplacian eig[1] != 2");
        arb_add(e, two, r2, PREC); /* 2 + sqrt2 */
        CHECK(close_tol(lam + 2, e, TOL), "laplacian eig[2] != 2+sqrt2");
        CHECK(check_eigrelation(A, Q, lam, n), "laplacian A*Q != Q*diag");

        _arb_vec_clear(lam, n);
        arb_mat_clear(A);
        arb_mat_clear(Q);
    }

    arb_clear(e);
    arb_clear(two);
    arb_clear(r2);
}

/* Build a PD fixture into A (caller-initialized n x n). */
static void
make_pd(arb_mat_t A, slong n)
{
    if (n == 2) {
        /* [[4,1],[1,3]] : eigenvalues (7 +/- sqrt5)/2 > 0. */
        arb_set_si(arb_mat_entry(A, 0, 0), 4);
        arb_set_si(arb_mat_entry(A, 1, 1), 3);
        set_sym(A, 0, 1, 1);
    } else { /* n == 3 */
        /* [[5,1,2],[1,4,1],[2,1,6]] : diagonally dominant, PD. */
        arb_set_si(arb_mat_entry(A, 0, 0), 5);
        arb_set_si(arb_mat_entry(A, 1, 1), 4);
        arb_set_si(arb_mat_entry(A, 2, 2), 6);
        set_sym(A, 0, 1, 1);
        set_sym(A, 0, 2, 2);
        set_sym(A, 1, 2, 1);
    }
}

/* ----- Test 3: psd_sqrt -- sqrt(A)^2 == A -------------------------------- */
static void
test_psd_sqrt(void)
{
    for (slong n = 2; n <= 3; n++) {
        arb_mat_t A, R, R2;
        int st;
        arb_mat_init(A, n, n);
        arb_mat_init(R, n, n);
        arb_mat_init(R2, n, n);
        make_pd(A, n);

        st = arbsdp_psd_sqrt(R, A, PREC);
        CHECK(st == 0, "psd_sqrt: well-conditioned PD must return 0 (strictly PD)");
        arb_mat_mul(R2, R, R, PREC);

        CHECK(mat_close(R2, A, TOL),
              n == 2 ? "psd_sqrt n=2: sqrt(A)^2 != A"
                     : "psd_sqrt n=3: sqrt(A)^2 != A");

        arb_mat_clear(A);
        arb_mat_clear(R);
        arb_mat_clear(R2);
    }
}

/* ----- Test 4: psd_invsqrt -- invsqrt(A)*A*invsqrt(A) == I ---------------- */
static void
test_psd_invsqrt(void)
{
    for (slong n = 2; n <= 3; n++) {
        arb_mat_t A, R, tmp, prod, I;
        int st;
        arb_mat_init(A, n, n);
        arb_mat_init(R, n, n);
        arb_mat_init(tmp, n, n);
        arb_mat_init(prod, n, n);
        arb_mat_init(I, n, n);
        arb_mat_one(I);
        make_pd(A, n);

        st = arbsdp_psd_invsqrt(R, A, PREC);
        CHECK(st == 0, "psd_invsqrt: well-conditioned PD must return 0 (strictly PD)");
        arb_mat_mul(tmp, R, A, PREC);
        arb_mat_mul(prod, tmp, R, PREC);

        CHECK(mat_close(prod, I, TOL),
              n == 2 ? "psd_invsqrt n=2: invsqrt A invsqrt != I"
                     : "psd_invsqrt n=3: invsqrt A invsqrt != I");

        arb_mat_clear(A);
        arb_mat_clear(R);
        arb_mat_clear(tmp);
        arb_mat_clear(prod);
        arb_mat_clear(I);
    }
}

/* ----- Test 4b: STRICT-PD GATE (bead arb-prec-IPM-b30, CLAUDE.md rule 5) ----
 * The cone ops must SIGNAL failure (nonzero return) on a non-PD / singular input
 * instead of silently flooring it into a huge-but-finite garbage inverse-sqrt.
 *
 * RED-GREEN (rule 9): on the old 1e-300-floored implementation arbsdp_psd_invsqrt
 * returned void and produced a finite garbage matrix, so this test would not even
 * compile / would never see a nonzero status -> RED.  With the gate it returns
 * nonzero on a singular or rank-deficient input -> GREEN.
 *
 * MUTATION (rule 8): reverting psd_pow_half to the eps=1e-300 floor (so a singular
 * input is clamped rather than detected) makes the diag(1,0) / rank-1 cases below
 * return 0, FAILING this test -- proving the singular-detection bites.
 *
 * The positive control (a strictly-PD diag(4,9) returning 0 with the exact
 * inverse-sqrt diag(1/2,1/3)) guards against a gate that rejects everything. */
static void
test_strict_pd_gate(void)
{
    /* (a) well-conditioned PD diag(4,9): returns 0 and the exact invsqrt
     * diag(1/2, 1/3). */
    {
        arb_mat_t A, R, E;
        int st;
        arb_mat_init(A, 2, 2);
        arb_mat_init(R, 2, 2);
        arb_mat_init(E, 2, 2);
        arb_set_si(arb_mat_entry(A, 0, 0), 4);
        arb_set_si(arb_mat_entry(A, 1, 1), 9);
        st = arbsdp_psd_invsqrt(R, A, PREC);
        CHECK(st == 0, "gate: diag(4,9) is strictly PD -> return 0");
        arb_set_d(arb_mat_entry(E, 0, 0), 0.5);            /* 1/sqrt(4) = 1/2 */
        arb_set_str(arb_mat_entry(E, 1, 1), "0.333333333333333333333333333333333333333333333333333333333333333", PREC);
        CHECK(mat_close(R, E, TOL), "gate: invsqrt(diag(4,9)) != diag(1/2,1/3)");
        arb_mat_clear(A);
        arb_mat_clear(R);
        arb_mat_clear(E);
    }

    /* (b) SINGULAR diag(1,0): lambda_min = 0, not PD -> must SIGNAL (nonzero). */
    {
        arb_mat_t A, R;
        int st;
        arb_mat_init(A, 2, 2);
        arb_mat_init(R, 2, 2);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        /* (1,1) left at 0 */
        st = arbsdp_psd_invsqrt(R, A, PREC);
        CHECK(st != 0, "gate: SINGULAR diag(1,0) must SIGNAL failure (nonzero)");
        arb_mat_clear(A);
        arb_mat_clear(R);
    }

    /* (c) RANK-1 [[1,1],[1,1]]: eigenvalues {0, 2}, not PD -> must SIGNAL. */
    {
        arb_mat_t A, R;
        int st;
        arb_mat_init(A, 2, 2);
        arb_mat_init(R, 2, 2);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        arb_set_si(arb_mat_entry(A, 1, 1), 1);
        set_sym(A, 0, 1, 1);
        st = arbsdp_psd_invsqrt(R, A, PREC);
        CHECK(st != 0, "gate: RANK-1 [[1,1],[1,1]] must SIGNAL failure (nonzero)");
        arb_mat_clear(A);
        arb_mat_clear(R);
    }

    /* (d) psd_sqrt of the same singular diag(1,0) also signals (the gate is shared
     * by both powers; nt_scaling needs sqrt(X) to fail loud too). */
    {
        arb_mat_t A, R;
        int st;
        arb_mat_init(A, 2, 2);
        arb_mat_init(R, 2, 2);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        st = arbsdp_psd_sqrt(R, A, PREC);
        CHECK(st != 0, "gate: psd_sqrt of SINGULAR diag(1,0) must SIGNAL failure");
        arb_mat_clear(A);
        arb_mat_clear(R);
    }
}

/* ----- Test 5: orthogonality Q^T Q == I on a dense symmetric A ------------ */
static void
test_orthogonality(void)
{
    const slong n = 3;
    arb_mat_t A, Q, Qt, QtQ, I;
    arb_ptr lam;
    arb_mat_init(A, n, n);
    arb_mat_init(Q, n, n);
    arb_mat_init(Qt, n, n);
    arb_mat_init(QtQ, n, n);
    arb_mat_init(I, n, n);
    arb_mat_one(I);
    lam = _arb_vec_init(n);

    make_pd(A, n);
    arbsdp_eigh(lam, Q, A, PREC);

    arb_mat_transpose(Qt, Q);
    arb_mat_mul(QtQ, Qt, Q, PREC);
    CHECK(mat_close(QtQ, I, TOL), "orthogonality: Q^T Q != I");

    _arb_vec_clear(lam, n);
    arb_mat_clear(A);
    arb_mat_clear(Q);
    arb_mat_clear(Qt);
    arb_mat_clear(QtQ);
    arb_mat_clear(I);
}

/* ----- Test 6: MUTATION sanity (CLAUDE.md rule 8) -------------------------
 * The eigen-relation checker, fed a deliberately WRONG eigenvalue, MUST fail.
 * If it "passes" a wrong answer, the real tests would not catch a bug. */
static void
test_mutation_sanity(void)
{
    const slong n = 2;
    arb_mat_t A, Q;
    arb_ptr lam;
    arb_mat_init(A, n, n);
    arb_mat_init(Q, n, n);
    lam = _arb_vec_init(n);

    arb_set_si(arb_mat_entry(A, 0, 0), 2);
    arb_set_si(arb_mat_entry(A, 1, 1), 2);
    set_sym(A, 0, 1, 1);
    arbsdp_eigh(lam, Q, A, PREC);

    /* Corrupt eigenvalue[0]: now A*Q != Q*diag(lam_wrong). */
    arb_add_si(lam + 0, lam + 0, 1, PREC);
    CHECK(!check_eigrelation(A, Q, lam, n),
          "MUTATION: wrong eigenvalue must break A*Q == Q*diag (test is blind)");

    /* And a wrong-tolerance positive control: 1.0 is NOT within TOL of 3.0. */
    {
        arb_t a, b;
        arb_init(a);
        arb_init(b);
        arb_set_si(a, 1);
        arb_set_si(b, 3);
        CHECK(!close_tol(a, b, TOL),
              "MUTATION: close_tol must reject 1 vs 3 (checker is blind)");
        arb_clear(a);
        arb_clear(b);
    }

    _arb_vec_clear(lam, n);
    arb_mat_clear(A);
    arb_mat_clear(Q);
}

int
main(void)
{
    test_eigh_diag();
    test_eigh_known();
    test_psd_sqrt();
    test_psd_invsqrt();
    test_strict_pd_gate();
    test_orthogonality();
    test_mutation_sanity();

    if (failures != 0) {
        fprintf(stderr, "test_linalg: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_linalg (eigh/psd_sqrt/psd_invsqrt, point-mode)\n");
    return EXIT_SUCCESS;
}
