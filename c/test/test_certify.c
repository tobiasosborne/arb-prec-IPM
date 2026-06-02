/*
 * test/test_certify.c -- BALL-MODE (RIGOROUS, Layer-1) tests for certify.c:
 * arbsdp_verified_psd / arbsdp_gershgorin_lower_bound / arbsdp_lambda_min_lower_bound
 * / arbsdp_dual_residual / the arbsdp_apriori bound API.
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): each block is written before the impl it
 * exercises; the target link-fails or aborts (assert(0) stub) first, then passes.
 *
 * RIGOR IS FALSIFIABLE (CLAUDE.md rule 2): the lambda_min lower bound d is a
 * THEOREM -- d <= lambda_min(A).  A returned d that EVER exceeds the true
 * lambda_min is a P0 bug.  Every known-spectrum fixture below asserts BOTH
 * rigor (d <= true lambda_min, checked at d's UPPER endpoint so even radius cannot
 * cheat) AND tightness (true - d < a tol that proves the bisection converged, not
 * stalled).  The dual residual Z must be a true ball ENCLOSURE: each returned
 * Z[b] entry's ball must CONTAIN the exact C[b] - sum_i y_i A_i[b] (rule 2).
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

/* mkstemp/close are POSIX; with -std=c11 they need the feature-test macro to be
 * declared before any standard header (matches test_io.c). */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/certify.h"
#include "arbsdp/linalg.h"
#include "arbsdp/svec.h"

#define PREC 256
/* Certification precision: the caller passes prec_c = prec + 128 (MATH_SPEC §10,
 * decision 2; bead notes).  The dual-residual tests run at this prec_c. */
#define PREC_C (PREC + 128)

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

/* ----- Step 6: arbsdp_dual_residual ENCLOSURE (bead b18) ------------------ *
 * Z[b] = C[b] - sum_i y_i A_i[b] must be a true ball ENCLOSURE (CLAUDE.md rule
 * 2): each returned entry's ball must CONTAIN the independently-computed exact
 * value.  We build a 1-block 2x2 problem with KNOWN integer C, A_1, A_2 via a
 * temp .dat-s parsed through the real materialization path (arbsdp_read_sdpa ->
 * arbsdp_problem_block_mat), then check the enclosure entry by entry.
 *
 * Problem (file/SDPA sign, matno 0 = C, 1..m = A_i; off-diag stored verbatim,
 * block_mat writes A_{ij}=A_{ji}=v):
 *   C   = [[5, 2],[2, 7]]
 *   A_1 = [[1, 0],[0, 0]]
 *   A_2 = [[0, 1],[1, 3]]
 * With y = (3, 1/2)  (both exact in binary):
 *   Z = C - 3*A_1 - (1/2)*A_2
 *     = [[5-3, 2-1/2],[2-1/2, 7-3/2]] = [[2, 3/2],[3/2, 11/2]].
 */

/* Write `content` to a fresh temp .dat-s; return the owned path (caller frees) or
 * NULL.  Mirrors test_io.c's temp-file pattern. */
static char *
write_temp_dats(const char *content)
{
    char tmpl[] = "/tmp/arbsdp_certify_XXXXXX";
    int fd = mkstemp(tmpl);
    FILE *f;
    char *path;
    if (fd < 0)
        return NULL;
    f = fdopen(fd, "w");
    if (f == NULL) {
        close(fd);
        return NULL;
    }
    fputs(content, f);
    fclose(f);
    path = malloc(strlen(tmpl) + 1);
    if (path != NULL)
        strcpy(path, tmpl);
    return path;
}

/* Assert Z[b]_entry encloses the exact value `exact_str` (CLAUDE.md rule 2). */
static void
check_encloses(const arb_mat_t Z, slong r, slong c, const char *exact_str,
               const char *label)
{
    arb_t e;
    arb_init(e);
    arb_set_str(e, exact_str, PREC_C);
    CHECK(arb_contains(arb_mat_entry(Z, r, c), e), label);
    arb_clear(e);
}

static void
test_dual_residual(void)
{
    /* (1) full enclosure with y = (3, 1/2). */
    {
        const char *dats =
            "* test problem b18 dual residual\n"
            "2\n"          /* m = 2 constraints */
            "1\n"          /* nblocks = 1 */
            "2\n"          /* block 0 is dense PSD 2x2 */
            "1.0 2.0\n"    /* b vector (unused by dual_residual) */
            "0 1 1 1 5\n"  /* C_11 = 5 */
            "0 1 1 2 2\n"  /* C_12 = C_21 = 2 */
            "0 1 2 2 7\n"  /* C_22 = 7 */
            "1 1 1 1 1\n"  /* A_1: (1,1) = 1 */
            "2 1 1 2 1\n"  /* A_2: (1,2)=(2,1) = 1 */
            "2 1 2 2 3\n"; /* A_2: (2,2) = 3 */
        char *path = write_temp_dats(dats);
        arbsdp_problem p;
        arb_mat_t Z[1];
        arb_ptr y;
        double maxrad = 0.0;

        CHECK(path != NULL, "dual_residual: temp file creation");
        arbsdp_problem_init(&p);
        CHECK(arbsdp_read_sdpa(&p, path) == 0, "dual_residual: parse temp .dat-s");

        arb_mat_init(Z[0], 2, 2);
        y = _arb_vec_init(p.m);
        arb_set_si(&y[0], 3);                 /* y_1 = 3 */
        arb_set_d(&y[1], 0.5);                /* y_2 = 1/2 (exact binary) */

        arbsdp_dual_residual(Z, &p, y, PREC_C);

        /* Exact Z = [[2, 3/2],[3/2, 11/2]] -- assert each ball ENCLOSES it. */
        check_encloses(Z[0], 0, 0, "2",   "dual_residual Z_00 encloses 2");
        check_encloses(Z[0], 0, 1, "1.5", "dual_residual Z_01 encloses 3/2");
        check_encloses(Z[0], 1, 0, "1.5", "dual_residual Z_10 encloses 3/2");
        check_encloses(Z[0], 1, 1, "5.5", "dual_residual Z_11 encloses 11/2");

        /* Symmetry: Z_01 == Z_10 (downstream Cholesky reads the lower triangle). */
        CHECK(arb_equal(arb_mat_entry(Z[0], 0, 1), arb_mat_entry(Z[0], 1, 0)),
              "dual_residual: Z must be symmetric (Z_01 == Z_10)");

        /* Report the enclosure radius magnitude (rule 11: concrete numbers). */
        for (slong i = 0; i < 2; i++)
            for (slong j = 0; j < 2; j++) {
                double rad = mag_get_d(arb_radref(arb_mat_entry(Z[0], i, j)));
                if (rad > maxrad)
                    maxrad = rad;
            }
        printf("  [dual_res] 2x2 y=(3,1/2)   max entry radius = %.3e (prec_c=%d)\n",
               maxrad, PREC_C);

        _arb_vec_clear(y, p.m);
        arb_mat_clear(Z[0]);
        arbsdp_problem_clear(&p);
        free(path);
    }

    /* (1b) NON-binary-exact y forces a strictly-positive radius, so the
     * "ball CONTAINS the exact value" claim actually exercises rounding (rule 8:
     * the exact-input case above has radius 0 and cannot catch a rounding bug).
     * y = (1/10, 1/3) (neither binary-exact).  Exact Z:
     *   Z_00 = 5 - (1/10)*1               = 49/10
     *   Z_01 = 2 - (1/3)*1                = 5/3
     *   Z_11 = 7 - (1/3)*3               = 6
     * The exact targets are recomputed INDEPENDENTLY in arb at a much HIGHER
     * precision (2*PREC_C) so the reference ball is ~2^-PREC_C tighter than Z's,
     * making arb_contains a genuine enclosure test (CLAUDE.md rule 8). */
    {
        const char *dats =
            "* test problem b18 dual residual non-exact y\n"
            "2\n"
            "1\n"
            "2\n"
            "1.0 2.0\n"
            "0 1 1 1 5\n"
            "0 1 1 2 2\n"
            "0 1 2 2 7\n"
            "1 1 1 1 1\n"
            "2 1 1 2 1\n"
            "2 1 2 2 3\n";
        char *path = write_temp_dats(dats);
        arbsdp_problem p;
        arb_mat_t Z[1];
        arb_ptr y;
        arb_t e, third, tenth;
        double maxrad = 0.0;

        CHECK(path != NULL, "dual_residual non-exact: temp file creation");
        arbsdp_problem_init(&p);
        CHECK(arbsdp_read_sdpa(&p, path) == 0,
              "dual_residual non-exact: parse .dat-s");

        arb_mat_init(Z[0], 2, 2);
        y = _arb_vec_init(p.m);
        arb_init(e);
        arb_init(third);
        arb_init(tenth);

        /* y_1 = 1/10, y_2 = 1/3 at prec_c (NOT exact). */
        arb_set_si(tenth, 1);
        arb_div_si(tenth, tenth, 10, PREC_C);
        arb_set(&y[0], tenth);
        arb_set_si(third, 1);
        arb_div_si(third, third, 3, PREC_C);
        arb_set(&y[1], third);

        arbsdp_dual_residual(Z, &p, y, PREC_C);

        /* Reference exacts at 2*PREC_C (much tighter than Z's enclosure). */
        arb_set_si(e, 49);                          /* Z_00 = 49/10 */
        arb_div_si(e, e, 10, 2 * PREC_C);
        CHECK(arb_contains(arb_mat_entry(Z[0], 0, 0), e),
              "dual_residual non-exact: Z_00 encloses 49/10");

        arb_set_si(e, 5);                            /* Z_01 = 5/3 */
        arb_div_si(e, e, 3, 2 * PREC_C);
        CHECK(arb_contains(arb_mat_entry(Z[0], 0, 1), e),
              "dual_residual non-exact: Z_01 encloses 5/3");

        arb_set_si(e, 6);                            /* Z_11 = 7 - 3*(1/3) = 6 */
        CHECK(arb_contains(arb_mat_entry(Z[0], 1, 1), e),
              "dual_residual non-exact: Z_11 encloses 6");

        for (slong i = 0; i < 2; i++)
            for (slong j = 0; j < 2; j++) {
                double rad = mag_get_d(arb_radref(arb_mat_entry(Z[0], i, j)));
                if (rad > maxrad)
                    maxrad = rad;
            }
        CHECK(maxrad > 0.0,
              "dual_residual non-exact: radius is strictly positive (rounding exercised)");
        printf("  [dual_res] 2x2 y=(1/10,1/3) max entry radius = %.3e (prec_c=%d)\n",
               maxrad, PREC_C);

        arb_clear(e);
        arb_clear(third);
        arb_clear(tenth);
        _arb_vec_clear(y, p.m);
        arb_mat_clear(Z[0]);
        arbsdp_problem_clear(&p);
        free(path);
    }

    /* (2) y = 0 -> Z[b] == C[b] (enclosure contains C). */
    {
        const char *dats =
            "* test problem b18 dual residual y=0\n"
            "2\n"
            "1\n"
            "2\n"
            "1.0 2.0\n"
            "0 1 1 1 5\n"
            "0 1 1 2 2\n"
            "0 1 2 2 7\n"
            "1 1 1 1 1\n"
            "2 1 1 2 1\n"
            "2 1 2 2 3\n";
        char *path = write_temp_dats(dats);
        arbsdp_problem p;
        arb_mat_t Z[1];
        arb_ptr y;

        CHECK(path != NULL, "dual_residual y=0: temp file creation");
        arbsdp_problem_init(&p);
        CHECK(arbsdp_read_sdpa(&p, path) == 0, "dual_residual y=0: parse .dat-s");

        arb_mat_init(Z[0], 2, 2);
        y = _arb_vec_init(p.m);   /* all zero from _arb_vec_init */

        arbsdp_dual_residual(Z, &p, y, PREC_C);

        check_encloses(Z[0], 0, 0, "5", "dual_residual y=0: Z_00 == C_00 = 5");
        check_encloses(Z[0], 0, 1, "2", "dual_residual y=0: Z_01 == C_01 = 2");
        check_encloses(Z[0], 1, 1, "7", "dual_residual y=0: Z_11 == C_11 = 7");

        _arb_vec_clear(y, p.m);
        arb_mat_clear(Z[0]);
        arbsdp_problem_clear(&p);
        free(path);
    }
}

/* ----- Step 7: arbsdp_apriori bound API (bead b18) ------------------------ *
 * CLAUDE.md invariant 5 (NO silent boundedness): the *_set FLAG -- not a
 * sentinel value -- is the source of truth for "supplied".  MUTATION (rule 8):
 * an UNSET xbar[b] is distinguishable from a SET xbar[b] that holds a huge
 * (near-+inf) value; the flag, never the value, decides. */
static void
test_apriori(void)
{
    arbsdp_apriori ab;
    arb_t v, huge;

    arb_init(v);
    arb_init(huge);

    arbsdp_apriori_init(&ab, 3);

    /* Init: all bounds UNSET. */
    CHECK(ab.nblocks == 3, "apriori: nblocks recorded");
    CHECK(ab.xbar_set[0] == 0 && ab.xbar_set[1] == 0 && ab.xbar_set[2] == 0,
          "apriori init: all xbar UNSET");
    CHECK(ab.ybar_set == 0, "apriori init: ybar UNSET");

    /* Set xbar[1] = 4.25; only block 1 toggles. */
    arb_set_d(v, 4.25);
    arbsdp_apriori_set_xbar(&ab, 1, v);
    CHECK(ab.xbar_set[1] == 1, "apriori: set_xbar toggles xbar_set[1]");
    CHECK(ab.xbar_set[0] == 0 && ab.xbar_set[2] == 0,
          "apriori: set_xbar[1] leaves other blocks UNSET");
    CHECK(arb_equal(&ab.xbar[1], v), "apriori: xbar[1] stores the value");

    /* MUTATION (invariant 5): a SET huge xbar[2] is a FINITE bound, still
     * distinguishable from the UNSET xbar[0] by the FLAG (not by value). */
    arb_set_d(huge, 1e300);
    arbsdp_apriori_set_xbar(&ab, 2, huge);
    CHECK(ab.xbar_set[2] == 1, "apriori MUTATION: set huge xbar[2] is SET");
    CHECK(ab.xbar_set[0] == 0, "apriori MUTATION: unset xbar[0] stays UNSET");
    CHECK(arb_is_finite(&ab.xbar[2]),
          "apriori MUTATION: a set huge xbar[2] is FINITE, not +inf");
    /* The discriminator is the flag: block 0 (unset) and block 2 (set huge) carry
     * DIFFERENT flags even though neither value is +inf. */
    CHECK(ab.xbar_set[0] != ab.xbar_set[2],
          "apriori MUTATION: flag (not value) distinguishes unset from set-huge");

    /* ybar set/unset. */
    arb_set_si(v, 9);
    arbsdp_apriori_set_ybar(&ab, v);
    CHECK(ab.ybar_set == 1, "apriori: set_ybar toggles ybar_set");
    CHECK(arb_equal(ab.ybar, v), "apriori: ybar stores the value");

    arbsdp_apriori_clear(&ab);
    /* After clear: re-zeroed (idempotent, rule 7). */
    CHECK(ab.nblocks == 0 && ab.xbar == NULL && ab.xbar_set == NULL
              && ab.ybar_set == 0,
          "apriori clear: struct re-zeroed");

    /* nblocks == 0 edge case: init/clear must not allocate/free garbage. */
    arbsdp_apriori_init(&ab, 0);
    CHECK(ab.xbar == NULL && ab.xbar_set == NULL && ab.ybar_set == 0,
          "apriori init(0): no per-block allocation, ybar unset");
    arbsdp_apriori_clear(&ab);

    arb_clear(v);
    arb_clear(huge);
}

int
main(void)
{
    test_verified_psd();
    test_gershgorin();
    test_lambda_min_bracket_coarse();
    test_lambda_min_bisection();
    test_eig_crosscheck();
    test_dual_residual();
    test_apriori();

    if (failures != 0) {
        fprintf(stderr, "test_certify: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_certify (verified_psd, ball-mode)\n");
    return EXIT_SUCCESS;
}
