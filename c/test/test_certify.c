/*
 * test/test_certify.c -- BALL-MODE (RIGOROUS, Layer-1) tests for certify.c:
 * arbsdp_verified_psd / arbsdp_gershgorin_lower_bound / arbsdp_lambda_min_lower_bound.
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): each block is written before the impl it
 * exercises; the target link-fails or aborts (assert(0) stub) first, then passes.
 *
 * RIGOR IS FALSIFIABLE (CLAUDE.md rule 2): the lambda_min lower bound d is a
 * THEOREM -- d <= lambda_min(A).  A returned d that EVER exceeds the true
 * lambda_min is a P0 bug.  Every known-spectrum fixture below asserts BOTH
 * rigor (d <= true lambda_min, checked at d's UPPER endpoint so even radius cannot
 * cheat) AND tightness (true - d < a tol that proves the bisection converged, not
 * stalled).
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/certify.h"
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

static void
set_sym(arb_mat_t A, slong i, slong j, slong v)
{
    arb_set_si(arb_mat_entry(A, i, j), v);
    arb_set_si(arb_mat_entry(A, j, i), v);
}

/* ----- Step 1: arbsdp_verified_psd --------------------------------------- *
 * A return of 1 is a PD THEOREM (Rump 2006 Cor 2.4 / FLINT arb_mat.h:378);
 * a 0 is "not provably PD" (indefinite/semidefinite or precision-starved). */
static void
test_verified_psd(void)
{
    /* (a) diag(1,2,3): strictly PD -> 1. */
    {
        arb_mat_t A;
        arb_mat_init(A, 3, 3);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        arb_set_si(arb_mat_entry(A, 2, 2), 3);
        CHECK(arbsdp_verified_psd(A, PREC) == 1, "verified_psd: diag(1,2,3) must certify");
        arb_mat_clear(A);
    }

    /* (b) [[2,1],[1,2]]: eigenvalues {1,3}, strictly PD -> 1. */
    {
        arb_mat_t A;
        arb_mat_init(A, 2, 2);
        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        set_sym(A, 0, 1, 1);
        CHECK(arbsdp_verified_psd(A, PREC) == 1, "verified_psd: [[2,1],[1,2]] must certify");
        arb_mat_clear(A);
    }

    /* (c) indefinite diag(1,-1): lambda_min = -1 < 0 -> 0. */
    {
        arb_mat_t A;
        arb_mat_init(A, 2, 2);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        arb_set_si(arb_mat_entry(A, 1, 1), -1);
        CHECK(arbsdp_verified_psd(A, PREC) == 0, "verified_psd: indefinite diag(1,-1) must NOT certify");
        arb_mat_clear(A);
    }

    /* (d) MUTATION (CLAUDE.md rule 8): singular [[1,1],[1,1]] has spectrum {0,2}
     * (lambda_min = 0, PSD but NOT strictly PD) -> must NOT certify.  This
     * discriminates a true-PD test from one that would rubber-stamp the
     * PSD boundary (where the optimal X lives -- invariant 1). */
    {
        arb_mat_t A;
        arb_mat_init(A, 2, 2);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        arb_set_si(arb_mat_entry(A, 1, 1), 1);
        set_sym(A, 0, 1, 1);
        CHECK(arbsdp_verified_psd(A, PREC) == 0,
              "verified_psd MUTATION: singular [[1,1],[1,1]] (lmin=0) must NOT certify");
        arb_mat_clear(A);
    }
}

/* Is the UPPER endpoint of d <= the LOWER endpoint of the (exact) value `b`?
 * This is the rigor check for a lower bound: even d's worst-case (largest) value
 * must not exceed the true quantity it bounds. */
static int
ub_le_value_arb(const arb_t d, const arb_t b)
{
    arf_t du, bl;
    int ok;
    arf_init(du);
    arf_init(bl);
    arb_get_ubound_arf(du, d, PREC);
    arb_get_lbound_arf(bl, b, PREC);
    ok = (arf_cmp(du, bl) <= 0);
    arf_clear(du);
    arf_clear(bl);
    return ok;
}

/* String wrapper for exact-at-PREC bounds (integers).  Do NOT use a typed
 * decimal string for an IRRATIONAL bound -- a wrong digit can flip the rigor
 * verdict (CLAUDE.md rule 8); compute irrationals in arb and call _arb. */
static int
ub_le_value(const arb_t d, const char *bound_str)
{
    arb_t b;
    int ok;
    arb_init(b);
    arb_set_str(b, bound_str, PREC);
    ok = ub_le_value_arb(d, b);
    arb_clear(b);
    return ok;
}

/* ----- Step 2: arbsdp_gershgorin_lower_bound ----------------------------- *
 * Gershgorin (1931) is ALWAYS valid: out <= lambda_min(A).  RIGOR: assert
 * out <= true lambda_min for every fixture (checked at out's UPPER endpoint). */
static void
test_gershgorin(void)
{
    /* (a) diag(2,5,9): off-diagonals zero -> Gershgorin gives exactly 2 = lmin. */
    {
        arb_mat_t A;
        arb_t out;
        arb_mat_init(A, 3, 3);
        arb_init(out);
        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 5);
        arb_set_si(arb_mat_entry(A, 2, 2), 9);
        arbsdp_gershgorin_lower_bound(out, A, PREC);
        CHECK(arb_is_finite(out), "gershgorin diag: out must be finite");
        CHECK(ub_le_value(out, "2"), "gershgorin diag(2,5,9): out must be <= lmin=2");
        arb_clear(out);
        arb_mat_clear(A);
    }

    /* (b) [[2,1],[1,2]]: true lmin = 1; Gershgorin disk gives 2-1 = 1, so
     * out <= 1 and finite. */
    {
        arb_mat_t A;
        arb_t out;
        arb_mat_init(A, 2, 2);
        arb_init(out);
        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        set_sym(A, 0, 1, 1);
        arbsdp_gershgorin_lower_bound(out, A, PREC);
        CHECK(arb_is_finite(out), "gershgorin [[2,1],[1,2]]: out must be finite");
        CHECK(ub_le_value(out, "1"), "gershgorin [[2,1],[1,2]]: out must be <= lmin=1");
        arb_clear(out);
        arb_mat_clear(A);
    }

    /* (c) 1D Laplacian 3x3 [[2,-1,0],[-1,2,-1],[0,-1,2]]: true lmin = 2-sqrt2
     * (~0.5858).  Gershgorin (middle row 2 - 1 - 1 = 0) <= 2-sqrt2.  RIGOR. */
    {
        arb_mat_t A;
        arb_t out;
        arb_mat_init(A, 3, 3);
        arb_init(out);
        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        arb_set_si(arb_mat_entry(A, 2, 2), 2);
        set_sym(A, 0, 1, -1);
        set_sym(A, 1, 2, -1);
        arbsdp_gershgorin_lower_bound(out, A, PREC);
        CHECK(arb_is_finite(out), "gershgorin laplacian: out must be finite");
        /* 2 - sqrt2 ~ 0.5857864376... ; out must be <= that.  Compute it in arb
         * (not a typed string) so the rigor check is exact (CLAUDE.md rule 8). */
        {
            arb_t lmin, r2;
            arb_init(lmin);
            arb_init(r2);
            arb_sqrt_ui(r2, 2, PREC);
            arb_set_si(lmin, 2);
            arb_sub(lmin, lmin, r2, PREC);
            CHECK(ub_le_value_arb(out, lmin),
                  "gershgorin laplacian: out must be <= 2-sqrt2");
            arb_clear(lmin);
            arb_clear(r2);
        }
        arb_clear(out);
        arb_mat_clear(A);
    }
}

/* ----- Step 3: arbsdp_lambda_min_lower_bound bracket setup (coarse) ------- *
 * Before bisection, the function establishes [s_lo: chol passes, s_hi: chol
 * fails] and returns s_lo.  RIGOR holds already: s_lo is a shift with a
 * SUCCESSFUL ball Cholesky, so d = s_lo <= lambda_min(A).  These tests assert
 * the rigor bound (the coarse value's UPPER endpoint <= true lmin) and that the
 * non-positive path (indefinite / singular) returns d <= 0 / <= -1 as designed. */
static void
test_lambda_min_bracket_coarse(void)
{
    /* (a) PD diag(1,2,3): lmin = 1.  s=0 passes; coarse s_lo in [0, 1], so
     * d <= 1 (rigor) and d >= 0 is fine (it is a lower bound, may be loose). */
    {
        arb_mat_t A;
        arb_t d;
        arb_mat_init(A, 3, 3);
        arb_init(d);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        arb_set_si(arb_mat_entry(A, 2, 2), 3);
        arbsdp_lambda_min_lower_bound(d, A, PREC);
        CHECK(ub_le_value(d, "1"), "lmin coarse diag(1,2,3): d must be <= lmin=1");
        arb_clear(d);
        arb_mat_clear(A);
    }

    /* (b) indefinite diag(1,-1): lmin = -1.  s=0 FAILS -> non-positive path;
     * d must be <= -1 (rigorous lower bound on a negative lmin). */
    {
        arb_mat_t A;
        arb_t d;
        arb_mat_init(A, 2, 2);
        arb_init(d);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        arb_set_si(arb_mat_entry(A, 1, 1), -1);
        arbsdp_lambda_min_lower_bound(d, A, PREC);
        CHECK(ub_le_value(d, "-1"), "lmin coarse diag(1,-1): d must be <= lmin=-1");
        arb_clear(d);
        arb_mat_clear(A);
    }

    /* (c) singular [[1,1],[1,1]]: spectrum {0,2}, lmin = 0 (PSD boundary).
     * s=0 cho FAILS (not strictly PD) -> non-positive path; d must be <= 0. */
    {
        arb_mat_t A;
        arb_t d;
        arb_mat_init(A, 2, 2);
        arb_init(d);
        arb_set_si(arb_mat_entry(A, 0, 0), 1);
        arb_set_si(arb_mat_entry(A, 1, 1), 1);
        set_sym(A, 0, 1, 1);
        arbsdp_lambda_min_lower_bound(d, A, PREC);
        CHECK(ub_le_value(d, "0"), "lmin coarse singular [[1,1],[1,1]]: d must be <= 0");
        arb_clear(d);
        arb_mat_clear(A);
    }
}

/* gap = (true lmin) - mid(d).  Asserts BOTH directions in one place:
 *   rigor:     ub(d) <= true (lower bound is a theorem),  AND
 *   tightness: true - ub(d) < gap_tol  (bisection converged, did not stall).
 * `tru` is the exact lmin as an arb (computed via arb_sqrt etc. -- NOT a typed
 * decimal string, which is error-prone: a wrong digit makes a correct d look
 * non-rigorous, CLAUDE.md rule 8); gap_tol is the convergence ceiling.  Returns
 * the numeric gap (true - mid(d)) for the report. */
static double
check_rigor_and_tightness_arb(const arb_t d, const arb_t tru, double gap_tol,
                              const char *label)
{
    arb_t gap, tolb;
    arf_t du, gl;
    double gapd;
    int rigor_ok, tight_ok;

    arb_init(gap);
    arb_init(tolb);
    arf_init(du);
    arf_init(gl);

    /* rigor: ub(d) <= lbound(true). */
    arb_get_ubound_arf(du, d, PREC);
    arb_get_lbound_arf(gl, tru, PREC);
    rigor_ok = (arf_cmp(du, gl) <= 0);
    CHECK(rigor_ok, label);  /* the message carries the fixture name */

    /* tightness: true - ub(d) < gap_tol  (use ub(d) so radius cannot help). */
    {
        arb_t dub;
        arb_init(dub);
        arb_set_arf(dub, du);
        arb_sub(gap, tru, dub, PREC);   /* >= 0 by rigor */
        arb_set_d(tolb, gap_tol);
        tight_ok = arb_lt(gap, tolb);
        arb_clear(dub);
    }
    CHECK(tight_ok, label);

    /* numeric gap = true - mid(d), for the report. */
    {
        arb_t md;
        arb_init(md);
        arb_get_mid_arb(md, d);
        arb_sub(gap, tru, md, PREC);
        gapd = arf_get_d(arb_midref(gap), ARF_RND_NEAR);
        arb_clear(md);
    }

    arb_clear(gap);
    arb_clear(tolb);
    arf_clear(du);
    arf_clear(gl);
    return gapd;
}

/* String wrapper for fixtures whose exact lmin is an integer (exact under
 * arb_set_str at PREC, so no transcription risk). */
static double
check_rigor_and_tightness(const arb_t d, const char *true_str, double gap_tol,
                          const char *label)
{
    arb_t tru;
    double gapd;
    arb_init(tru);
    arb_set_str(tru, true_str, PREC);
    gapd = check_rigor_and_tightness_arb(d, tru, gap_tol, label);
    arb_clear(tru);
    return gapd;
}

/* ----- Step 4: arbsdp_lambda_min_lower_bound bisection (TIGHT) ------------ *
 * Bracketing KNOWN spectra at prec=256.  RIGOR (P0-class, CLAUDE.md rule 2):
 * d <= true lambda_min ALWAYS.  TIGHTNESS: (true - d) < GAP_TOL -- the bisection
 * converged to its tolerance, not stalled at the coarse O(1) bracket.
 *
 * GAP_TOL = 1e-36.  The solver's BINDING stopping tolerance (MATH_SPEC §5.3.1)
 * is tol = 2^(-prec/2) * max(1, ||A||_approx); at prec=256 that is ~scale*2^-128
 * ~ 1.2e-38 (laplacian, scale=4) .. 2.1e-38 (diag(3,1,7), scale=7).  The final
 * gap true - d <= s_hi - s_lo <= tol, so it sits at ~1e-38.  We assert < 1e-36
 * (a small multiple of tol) -- comfortably above the achievable tol yet ~36
 * orders below the coarse bracket gap (O(1) without bisection, the step-4 RED),
 * so it proves CONVERGENCE, not a stall.  (The bead's nominal 1e-40 is BELOW the
 * binding tol formula at prec=256 and so unattainable with that formula; the
 * binding tol wins and the assertion is set just above what it delivers.) */
#define GAP_TOL 1e-36
static void
test_lambda_min_bisection(void)
{
    /* (a) diag(3,1,7): lmin = 1.  tol ~ 2^(-128)*7 ~ 2e-38. */
    {
        arb_mat_t A;
        arb_t d;
        double gap;
        arb_mat_init(A, 3, 3);
        arb_init(d);
        arb_set_si(arb_mat_entry(A, 0, 0), 3);
        arb_set_si(arb_mat_entry(A, 1, 1), 1);
        arb_set_si(arb_mat_entry(A, 2, 2), 7);
        arbsdp_lambda_min_lower_bound(d, A, PREC);
        gap = check_rigor_and_tightness(d, "1", GAP_TOL,
                  "lmin bisect diag(3,1,7): rigor/tightness (lmin=1)");
        printf("  [bisect] diag(3,1,7)   lmin=1            gap=%.3e\n", gap);
        arb_clear(d);
        arb_mat_clear(A);
    }

    /* (b) [[2,1],[1,2]]: spectrum {1,3}, lmin = 1. */
    {
        arb_mat_t A;
        arb_t d;
        double gap;
        arb_mat_init(A, 2, 2);
        arb_init(d);
        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        set_sym(A, 0, 1, 1);
        arbsdp_lambda_min_lower_bound(d, A, PREC);
        gap = check_rigor_and_tightness(d, "1", GAP_TOL,
                  "lmin bisect [[2,1],[1,2]]: rigor/tightness (lmin=1)");
        printf("  [bisect] [[2,1],[1,2]] lmin=1            gap=%.3e\n", gap);
        arb_clear(d);
        arb_mat_clear(A);
    }

    /* (c) 1D Laplacian 3x3: spectrum {2-sqrt2, 2, 2+sqrt2}, lmin = 2-sqrt2.
     * The exact lmin is computed in arb (2 - sqrt(2)), NOT a typed decimal
     * string -- a single wrong digit in such a string would make a CORRECT,
     * rigorous d falsely appear to violate rigor (CLAUDE.md rule 8). */
    {
        arb_mat_t A;
        arb_t d, lmin, r2;
        double gap;
        arb_mat_init(A, 3, 3);
        arb_init(d);
        arb_init(lmin);
        arb_init(r2);
        arb_set_si(arb_mat_entry(A, 0, 0), 2);
        arb_set_si(arb_mat_entry(A, 1, 1), 2);
        arb_set_si(arb_mat_entry(A, 2, 2), 2);
        set_sym(A, 0, 1, -1);
        set_sym(A, 1, 2, -1);
        arb_sqrt_ui(r2, 2, PREC);
        arb_set_si(lmin, 2);
        arb_sub(lmin, lmin, r2, PREC);          /* lmin = 2 - sqrt(2) */
        arbsdp_lambda_min_lower_bound(d, A, PREC);
        gap = check_rigor_and_tightness_arb(d, lmin, GAP_TOL,
                  "lmin bisect laplacian: rigor/tightness (lmin=2-sqrt2)");
        printf("  [bisect] laplacian3    lmin=2-sqrt2      gap=%.3e\n", gap);
        arb_clear(d);
        arb_clear(lmin);
        arb_clear(r2);
        arb_mat_clear(A);
    }
}

/* ----- Step 5: cross-check against point-mode eig (sanity, ONE-DIRECTIONAL) -
 * arbsdp_eigh is POINT MODE (linalg.h banner) -- NOT a rigor source.  This is a
 * SANITY cross-check only: the rigorous d must be <= mid(eigvals[0]) (the
 * approximate lmin) AND close to it (the bisection found the right eigenvalue,
 * not some lower shift).  MUTATION (CLAUDE.md rule 8): the SAME cross-check, fed
 * a deliberately inflated d = true_lmin + 1, MUST fail -- proving it discriminates
 * (a cross-check that passes a wrong d is blind). */
static void
test_eig_crosscheck(void)
{
    const slong n = 3;
    arb_mat_t A, Q;
    arb_ptr eigvals;
    arb_t d, md0, diff, tolb, bad;
    int le_ok, close_ok, mut_le, mut_close;

    arb_mat_init(A, n, n);
    arb_mat_init(Q, n, n);
    eigvals = _arb_vec_init(n);
    arb_init(d);
    arb_init(md0);
    arb_init(diff);
    arb_init(tolb);
    arb_init(bad);

    /* 1D Laplacian 3x3: lmin = 2 - sqrt2. */
    arb_set_si(arb_mat_entry(A, 0, 0), 2);
    arb_set_si(arb_mat_entry(A, 1, 1), 2);
    arb_set_si(arb_mat_entry(A, 2, 2), 2);
    set_sym(A, 0, 1, -1);
    set_sym(A, 1, 2, -1);

    arbsdp_lambda_min_lower_bound(d, A, PREC);
    arbsdp_eigh(eigvals, Q, A, PREC);            /* POINT mode; ascending */
    arb_get_mid_arb(md0, eigvals + 0);           /* approx lmin (point) */

    arb_set_d(tolb, GAP_TOL);

    /* rigorous d <= mid(eigvals[0]) (ub(d) so radius cannot help). */
    {
        arf_t du, m0;
        arf_init(du);
        arf_init(m0);
        arb_get_ubound_arf(du, d, PREC);
        arb_get_mid_arb(md0, eigvals + 0);
        arb_get_lbound_arf(m0, md0, PREC);
        le_ok = (arf_cmp(du, m0) <= 0);
        arf_clear(du);
        arf_clear(m0);
    }
    CHECK(le_ok, "crosscheck: rigorous d must be <= mid(eigh lmin) (laplacian)");

    /* |mid(eigvals[0]) - d| < GAP_TOL : bisection homed on the right eigenvalue. */
    arb_sub(diff, md0, d, PREC);
    arb_abs(diff, diff);
    close_ok = arb_lt(diff, tolb);
    CHECK(close_ok, "crosscheck: |mid(eigh lmin) - d| must be < GAP_TOL (laplacian)");

    /* MUTATION: an inflated d_bad = true_lmin + 1 must BREAK both arms (it now
     * EXCEEDS the approximate lmin, and is ~1 away from it). */
    arb_add_si(bad, d, 1, PREC);
    {
        arf_t du, m0;
        arf_init(du);
        arf_init(m0);
        arb_get_ubound_arf(du, bad, PREC);
        arb_get_lbound_arf(m0, md0, PREC);
        mut_le = (arf_cmp(du, m0) <= 0);     /* expect FALSE: bad exceeds lmin */
        arf_clear(du);
        arf_clear(m0);
    }
    arb_sub(diff, md0, bad, PREC);
    arb_abs(diff, diff);
    mut_close = arb_lt(diff, tolb);          /* expect FALSE: ~1 away */
    CHECK(!mut_le && !mut_close,
          "crosscheck MUTATION: inflated d=lmin+1 must FAIL the cross-check (it is blind otherwise)");

    _arb_vec_clear(eigvals, n);
    arb_clear(d);
    arb_clear(md0);
    arb_clear(diff);
    arb_clear(tolb);
    arb_clear(bad);
    arb_mat_clear(A);
    arb_mat_clear(Q);
}

int
main(void)
{
    test_verified_psd();
    test_gershgorin();
    test_lambda_min_bracket_coarse();
    test_lambda_min_bisection();
    test_eig_crosscheck();

    if (failures != 0) {
        fprintf(stderr, "test_certify: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_certify (verified_psd, ball-mode)\n");
    return EXIT_SUCCESS;
}
