/*
 * test/test_ntscaling.c -- POINT-MODE tests for the Nesterov-Todd scaling
 * arbsdp_nt_scaling (c/src/ntscaling.c).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written before c/src/ntscaling.c exists; the
 * target link-fails first (undefined symbol arbsdp_nt_scaling), then passes once
 * the impl lands.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §3.3: NT scaling W^b per block satisfies W S W = X
 *     (Todd-Toh-Tutuncu 1998).
 *   - PsdCone.ts ntScaling (lines 169-183):
 *       W = X^{1/2} (X^{1/2} S X^{1/2})^{-1/2} X^{1/2}, with the intermediate
 *       `inner` and the final W both symmetrized.
 *
 * POINT MODE (CLAUDE.md rule 1): arbsdp_nt_scaling is a Layer-0 kernel built on
 * the point-mode cyclic-Jacobi eig (linalg.c).  Its output is consumed as a
 * POINT; ball radii are NOT trusted.  Assertions compare MIDPOINTS within a
 * justified absolute tolerance (as in b05 / test_linalg.c) rather than demanding
 * rigorous enclosures -- a wide ball trivially "overlaps" and would not test
 * point accuracy.  TOL = 1e-50 at prec=256: the Jacobi eig delivers ~prec-bit
 * (~1e-77) midpoint accuracy (measured in docs/probes/eig_audition.c), and
 * nt_scaling composes a handful of such operations, so 1e-50 sits far above the
 * achieved accuracy yet far below any wrong-result error (a swapped X/S, dropped
 * symmetrize, or missing sqrt is off by O(1)).
 *
 * Property coverage (bead arb-prec-IPM-b06):
 *   1. DEFINING IDENTITY  W S W == X  on PD (X,S) for n = 2,3,4.
 *   2. W SYMMETRIC        W == W^T.
 *   3. SANITY / KNOWN CASE:
 *        (a) X = S = I  =>  W = I.
 *        (b) X, S diagonal => W diagonal with W_ii = sqrt(X_ii / S_ii)
 *            (closed form; catches a swapped X/S or a missing sqrt).
 *   4. MUTATION CHECK (CLAUDE.md rule 8): the identity checker, fed the result of
 *      a deliberately WRONG scaling (X and S swapped), must FAIL -- proving the
 *      test discriminates.  Then the correct call is re-run to confirm green.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/ntscaling.h"

#define PREC 256
#define TOL 1e-50

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
    ok = arb_le(d, t);
    arb_clear(d);
    arb_clear(t);
    return ok;
}

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

/* W S W == X (the defining NT identity, point-mode). */
static int
check_wsw_eq_x(const arb_mat_t W, const arb_mat_t S, const arb_mat_t X, slong n)
{
    arb_mat_t WS, WSW;
    int ok;
    arb_mat_init(WS, n, n);
    arb_mat_init(WSW, n, n);
    arb_mat_mul(WS, W, S, PREC);
    arb_mat_mul(WSW, WS, W, PREC);
    ok = mat_close(WSW, X, TOL);
    arb_mat_clear(WS);
    arb_mat_clear(WSW);
    return ok;
}

/* PD fixtures for X and S (distinct, so a swapped X/S is detectable). */
static void
make_pd_X(arb_mat_t A, slong n)
{
    if (n == 2) {
        /* [[4,1],[1,3]] PD. */
        arb_set_si(arb_mat_entry(A, 0, 0), 4);
        arb_set_si(arb_mat_entry(A, 1, 1), 3);
        set_sym(A, 0, 1, 1);
    } else if (n == 3) {
        /* [[5,1,2],[1,4,1],[2,1,6]] diagonally dominant, PD. */
        arb_set_si(arb_mat_entry(A, 0, 0), 5);
        arb_set_si(arb_mat_entry(A, 1, 1), 4);
        arb_set_si(arb_mat_entry(A, 2, 2), 6);
        set_sym(A, 0, 1, 1);
        set_sym(A, 0, 2, 2);
        set_sym(A, 1, 2, 1);
    } else { /* n == 4 */
        /* [[7,1,2,0],[1,6,1,2],[2,1,8,1],[0,2,1,5]] diagonally dominant, PD. */
        arb_set_si(arb_mat_entry(A, 0, 0), 7);
        arb_set_si(arb_mat_entry(A, 1, 1), 6);
        arb_set_si(arb_mat_entry(A, 2, 2), 8);
        arb_set_si(arb_mat_entry(A, 3, 3), 5);
        set_sym(A, 0, 1, 1);
        set_sym(A, 0, 2, 2);
        set_sym(A, 1, 2, 1);
        set_sym(A, 1, 3, 2);
        set_sym(A, 2, 3, 1);
    }
}

static void
make_pd_S(arb_mat_t A, slong n)
{
    if (n == 2) {
        /* [[3,-1],[-1,5]] PD, distinct from X. */
        arb_set_si(arb_mat_entry(A, 0, 0), 3);
        arb_set_si(arb_mat_entry(A, 1, 1), 5);
        set_sym(A, 0, 1, -1);
    } else if (n == 3) {
        /* [[6,2,1],[2,7,-1],[1,-1,4]] diagonally dominant, PD. */
        arb_set_si(arb_mat_entry(A, 0, 0), 6);
        arb_set_si(arb_mat_entry(A, 1, 1), 7);
        arb_set_si(arb_mat_entry(A, 2, 2), 4);
        set_sym(A, 0, 1, 2);
        set_sym(A, 0, 2, 1);
        set_sym(A, 1, 2, -1);
    } else { /* n == 4 */
        /* [[9,2,-1,1],[2,8,1,0],[-1,1,7,2],[1,0,2,6]] diag-dominant, PD. */
        arb_set_si(arb_mat_entry(A, 0, 0), 9);
        arb_set_si(arb_mat_entry(A, 1, 1), 8);
        arb_set_si(arb_mat_entry(A, 2, 2), 7);
        arb_set_si(arb_mat_entry(A, 3, 3), 6);
        set_sym(A, 0, 1, 2);
        set_sym(A, 0, 2, -1);
        set_sym(A, 0, 3, 1);
        set_sym(A, 1, 2, 1);
        set_sym(A, 2, 3, 2);
    }
}

/* ----- Test 1: defining identity W S W == X for n = 2,3,4 ----------------- */
static void
test_wsw_identity(void)
{
    for (slong n = 2; n <= 4; n++) {
        arb_mat_t X, S, W;
        arb_mat_init(X, n, n);
        arb_mat_init(S, n, n);
        arb_mat_init(W, n, n);
        make_pd_X(X, n);
        make_pd_S(S, n);

        arbsdp_nt_scaling(W, X, S, PREC);

        CHECK(check_wsw_eq_x(W, S, X, n),
              n == 2 ? "nt_scaling n=2: W S W != X"
                     : (n == 3 ? "nt_scaling n=3: W S W != X"
                               : "nt_scaling n=4: W S W != X"));

        arb_mat_clear(X);
        arb_mat_clear(S);
        arb_mat_clear(W);
    }
}

/* ----- Test 2: W symmetric (W == W^T) for n = 2,3,4 ----------------------- */
static void
test_w_symmetric(void)
{
    for (slong n = 2; n <= 4; n++) {
        arb_mat_t X, S, W, Wt;
        arb_mat_init(X, n, n);
        arb_mat_init(S, n, n);
        arb_mat_init(W, n, n);
        arb_mat_init(Wt, n, n);
        make_pd_X(X, n);
        make_pd_S(S, n);

        arbsdp_nt_scaling(W, X, S, PREC);
        arb_mat_transpose(Wt, W);

        CHECK(mat_close(W, Wt, TOL),
              n == 2 ? "nt_scaling n=2: W != W^T"
                     : (n == 3 ? "nt_scaling n=3: W != W^T"
                               : "nt_scaling n=4: W != W^T"));

        arb_mat_clear(X);
        arb_mat_clear(S);
        arb_mat_clear(W);
        arb_mat_clear(Wt);
    }
}

/* ----- Test 3a: X = S = I  =>  W = I -------------------------------------- */
static void
test_identity_case(void)
{
    const slong n = 3;
    arb_mat_t X, S, W, I;
    arb_mat_init(X, n, n);
    arb_mat_init(S, n, n);
    arb_mat_init(W, n, n);
    arb_mat_init(I, n, n);
    arb_mat_one(X);
    arb_mat_one(S);
    arb_mat_one(I);

    arbsdp_nt_scaling(W, X, S, PREC);
    CHECK(mat_close(W, I, TOL), "nt_scaling X=S=I: W != I");

    arb_mat_clear(X);
    arb_mat_clear(S);
    arb_mat_clear(W);
    arb_mat_clear(I);
}

/* ----- Test 3b: diagonal X,S => W diagonal, W_ii = sqrt(X_ii / S_ii) ------
 * Closed-form check.  W S W = X for diagonal data forces W_ii^2 S_ii = X_ii,
 * hence W_ii = sqrt(X_ii / S_ii) and W off-diagonal = 0.  This single check
 * catches a swapped X/S (would give sqrt(S/X)) or a missing sqrt. */
static void
test_diagonal_closed_form(void)
{
    const slong n = 4;
    arb_mat_t X, S, W, Wexp;
    arb_t r;
    /* X = diag(4, 9, 16, 1); S = diag(1, 4, 4, 25). */
    const slong xd[4] = { 4, 9, 16, 1 };
    const slong sd[4] = { 1, 4, 4, 25 };

    arb_mat_init(X, n, n);
    arb_mat_init(S, n, n);
    arb_mat_init(W, n, n);
    arb_mat_init(Wexp, n, n);
    arb_init(r);

    for (slong i = 0; i < n; i++) {
        arb_set_si(arb_mat_entry(X, i, i), xd[i]);
        arb_set_si(arb_mat_entry(S, i, i), sd[i]);
        /* Wexp_ii = sqrt(X_ii / S_ii). */
        arb_set_si(r, xd[i]);
        arb_div_si(r, r, sd[i], PREC);
        arb_sqrt(r, r, PREC);
        arb_set(arb_mat_entry(Wexp, i, i), r);
    }

    arbsdp_nt_scaling(W, X, S, PREC);
    CHECK(mat_close(W, Wexp, TOL),
          "nt_scaling diagonal: W != diag(sqrt(X_ii/S_ii))");

    arb_mat_clear(X);
    arb_mat_clear(S);
    arb_mat_clear(W);
    arb_mat_clear(Wexp);
    arb_clear(r);
}

/* ----- Test 4: MUTATION sanity (CLAUDE.md rule 8) ------------------------- *
 * Compute the scaling with X and S SWAPPED.  The resulting W' satisfies
 * W' X W' == S, NOT W' S W' == X (since X != S here), so the defining-identity
 * checker MUST FAIL on it.  Then the CORRECT call is re-run and must PASS,
 * confirming the failure was the mutation and not a flaw in the checker. */
static void
test_mutation_sanity(void)
{
    const slong n = 3;
    arb_mat_t X, S, Wbad, Wgood;
    arb_mat_init(X, n, n);
    arb_mat_init(S, n, n);
    arb_mat_init(Wbad, n, n);
    arb_mat_init(Wgood, n, n);
    make_pd_X(X, n);
    make_pd_S(S, n);

    /* MUTATION: swap X and S. */
    arbsdp_nt_scaling(Wbad, S, X, PREC);
    CHECK(!check_wsw_eq_x(Wbad, S, X, n),
          "MUTATION: swapped X/S must break W S W == X (test is blind)");

    /* RESTORE: correct order must satisfy the identity. */
    arbsdp_nt_scaling(Wgood, X, S, PREC);
    CHECK(check_wsw_eq_x(Wgood, S, X, n),
          "MUTATION restore: correct W S W == X must hold");

    arb_mat_clear(X);
    arb_mat_clear(S);
    arb_mat_clear(Wbad);
    arb_mat_clear(Wgood);
}

int
main(void)
{
    test_wsw_identity();
    test_w_symmetric();
    test_identity_case();
    test_diagonal_closed_form();
    test_mutation_sanity();

    if (failures != 0) {
        fprintf(stderr, "test_ntscaling: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_ntscaling (W S W == X, point-mode)\n");
    return EXIT_SUCCESS;
}
