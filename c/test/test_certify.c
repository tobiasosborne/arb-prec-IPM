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
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <flint/fmpq_vec.h>

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
 * GAP_TOL = 1e-70.  Since bead ml6 the bound is the seeded verified-Cholesky
 * shift (MATH_SPEC §5.3.1, ml6 addendum): d = sigma - delta with
 * delta = 2^(clog2(n) + 6 - prec) * scale, so at prec=256 the gap is
 * ~2^-246 * scale ~ 2e-74 .. 9e-75 (measured below).  1e-70 sits ~4 orders above
 * that and ~70 orders below the coarse O(1) bracket, so it proves convergence.
 * (History: the pre-ml6 bisection tolerance 2^(-prec/2) * ||A|| gave ~1e-38 and
 * this gate was 1e-36.) */
#define GAP_TOL 1e-70
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

/* ----- Step 5b: PD-growth exhaustion (bead arb-prec-IPM-jhw) -------------- *
 * The PD branch of arbsdp_lambda_min_lower_bound doubles s from max(1,|g|)
 * (g = Gershgorin floor) for a bounded number of steps looking for a failing
 * shift.  If lambda_min exceeds 2^cap * max(1,|g|) no failure appears.  This
 * used to be guarded ONLY by assert(found_hi): under NDEBUG the bisection then
 * ran on an uninitialized s_hi.  The control flow must now handle it and still
 * return a THEOREM d <= lambda_min.
 *
 * Fixture: A = [[a, a],[a, c]], a = 2^300, c = 2^700.  Gershgorin floor
 * g = min(a - a, c - a) = 0 (so growth starts at 1), while
 *   lambda_min = (a+c)/2 - sqrt(((c-a)/2)^2 + a^2)  ~= a (1 - a/c) ~ 2^300,
 * far above 2^200: the growth loop exhausts.  Checks: the call returns (no
 * abort), d <= lambda_min (computed in arb, CLAUDE rule 8), and d >= 2^199 (the
 * last proven growth shift; never regress below a shift already proven). */
static void
test_lambda_min_growth_exhausted(void)
{
    const slong prec = 1024;
    arb_mat_t A;
    arb_t d, a, c, t, u, lmin, floor199;
    arf_t du, ll;

    arb_mat_init(A, 2, 2);
    arb_init(d); arb_init(a); arb_init(c); arb_init(t); arb_init(u);
    arb_init(lmin); arb_init(floor199);
    arf_init(du); arf_init(ll);

    arb_one(a); arb_mul_2exp_si(a, a, 300);
    arb_one(c); arb_mul_2exp_si(c, c, 700);
    arb_set(arb_mat_entry(A, 0, 0), a);
    arb_set(arb_mat_entry(A, 0, 1), a);
    arb_set(arb_mat_entry(A, 1, 0), a);
    arb_set(arb_mat_entry(A, 1, 1), c);

    /* exact lmin enclosure at high prec */
    arb_sub(t, c, a, 4 * prec);
    arb_mul_2exp_si(t, t, -1);
    arb_sqr(t, t, 4 * prec);
    arb_sqr(u, a, 4 * prec);
    arb_add(t, t, u, 4 * prec);
    arb_sqrt(t, t, 4 * prec);
    arb_add(lmin, a, c, 4 * prec);
    arb_mul_2exp_si(lmin, lmin, -1);
    arb_sub(lmin, lmin, t, 4 * prec);

    /* pass 0: default (seeded) path; pass 1: NaN seed forces the bisection
     * fallback, whose PD-growth loop is the branch this test was written for
     * (bead ml6 made the seeded path the default, so it must be forced). */
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 0) {
            arbsdp_lambda_min_lower_bound(d, A, prec);
        } else {
            arf_t nanseed;
            arf_init(nanseed);
            arf_nan(nanseed);
            arbsdp_lambda_min_lower_bound_seeded(d, A, nanseed, prec);
            arf_clear(nanseed);
        }

        arb_get_ubound_arf(du, d, prec);
        arb_get_lbound_arf(ll, lmin, 4 * prec);
        CHECK(arf_is_finite(du) && arf_cmp(du, ll) <= 0,
              "growth-exhausted: d must be finite and <= lambda_min (rigor)");
        arb_one(floor199); arb_mul_2exp_si(floor199, floor199, 199);
        arb_get_lbound_arf(du, d, prec);
        CHECK(arf_cmp(du, arb_midref(floor199)) >= 0,
              "growth-exhausted: d must not drop below the last proven shift 2^199");
    }

    arf_clear(du); arf_clear(ll);
    arb_clear(d); arb_clear(a); arb_clear(c); arb_clear(t); arb_clear(u);
    arb_clear(lmin); arb_clear(floor199);
    arb_mat_clear(A);
}

/* ----- Step 5c: seeded lambda_min -- tightness / probes / rigor (bead ml6) - *
 * Fixtures with EXACTLY known spectrum: A = Q D Q^T with Q a product of three
 * rational Householder reflectors H = I - 2 v v^T / (v^T v) (exactly orthogonal),
 * built in fmpq_mat and only then rounded into arb balls.  lambda_min = min D is
 * an exact rational, so the rigor check d <= lambda_min is decided exactly (no
 * typed decimals, CLAUDE rule 8). */
static void
rational_householder(fmpq_mat_t H, slong n, unsigned *st)
{
    fmpz_t vv, x;
    slong *v = flint_malloc(sizeof(slong) * n);
    int nz = 0;
    fmpz_init(vv);
    fmpz_init(x);
    while (!nz) {
        for (slong i = 0; i < n; i++) {
            *st = *st * 1103515245u + 12345u;
            v[i] = (slong) ((*st >> 16) % 11) - 5;
            if (v[i] != 0) nz = 1;
        }
    }
    fmpz_zero(vv);
    for (slong i = 0; i < n; i++)
        fmpz_add_si(vv, vv, v[i] * v[i]);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++) {
            fmpq_t t;
            fmpq_init(t);
            fmpz_set_si(x, -2 * v[i] * v[j]);
            fmpq_set_fmpz_frac(t, x, vv);
            if (i == j)
                fmpq_add_si(t, t, 1);
            fmpq_set(fmpq_mat_entry(H, i, j), t);
            fmpq_clear(t);
        }
    fmpz_clear(vv);
    fmpz_clear(x);
    flint_free(v);
}

/* A = Q diag(D) Q^T (exact), rounded to arb at prec.  lmin <- min D (exact). */
static void
spectral_fixture(arb_mat_t A, fmpq_t lmin, const fmpq *D, slong n,
                 unsigned seed, slong prec)
{
    fmpq_mat_t Q, H, T, Dm, Qt, R;
    unsigned st = seed;
    fmpq_mat_init(Q, n, n); fmpq_mat_init(H, n, n); fmpq_mat_init(T, n, n);
    fmpq_mat_init(Dm, n, n); fmpq_mat_init(Qt, n, n); fmpq_mat_init(R, n, n);
    fmpq_mat_one(Q);
    for (int k = 0; k < 3; k++) {
        rational_householder(H, n, &st);
        fmpq_mat_mul(T, Q, H);
        fmpq_mat_set(Q, T);
    }
    fmpq_set(lmin, D + 0);
    for (slong i = 0; i < n; i++) {
        fmpq_set(fmpq_mat_entry(Dm, i, i), D + i);
        if (fmpq_cmp(D + i, lmin) < 0)
            fmpq_set(lmin, D + i);
    }
    fmpq_mat_mul(T, Q, Dm);
    fmpq_mat_transpose(Qt, Q);
    fmpq_mat_mul(R, T, Qt);
    arb_mat_set_fmpq_mat(A, R, prec);
    fmpq_mat_clear(Q); fmpq_mat_clear(H); fmpq_mat_clear(T);
    fmpq_mat_clear(Dm); fmpq_mat_clear(Qt); fmpq_mat_clear(R);
}

/* Exact rigor check: ubound(d) <= lmin (lmin an exact rational). */
static int
d_le_exact(const arb_t d, const fmpq_t lmin)
{
    arf_t du;
    arb_t t;
    int ok;
    arf_init(du);
    arb_init(t);
    /* EXACT upper endpoint mid + rad (no rounding: a rounded-up ubound at a low
     * precision can step above an exact rational lmin and fake a violation). */
    arf_set_mag(du, arb_radref(d));
    arf_add(du, arb_midref(d), du, ARF_PREC_EXACT, ARF_RND_DOWN);
    if (!arf_is_finite(du)) {
        ok = arf_is_neg_inf(du);
    } else {
        /* exact: du is a dyadic, compare as fmpq */
        fmpq_t q;
        fmpq_init(q);
        arf_get_fmpq(q, du);
        ok = (fmpq_cmp(q, lmin) <= 0);
        fmpq_clear(q);
    }
    arb_clear(t);
    arf_clear(du);
    return ok;
}

/* log2((lmin - d)/scale) + prec, exactly (scale = Gershgorin |.| bound >= rho(A)).
 * The tightness gate requires this to be <= 64 (slack <= 2^-(prec-64) * scale). */
static double
slack_bits(const arb_t d, const fmpq_t lmin, const arb_mat_t A, slong prec)
{
    arb_t L, g;
    double r;
    arb_init(L);
    arb_init(g);
    arb_set_fmpq(L, lmin, 4 * prec);
    arb_sub(L, L, d, 4 * prec);
    arbsdp_gershgorin_upper_bound(g, A, prec);
    arb_abs(g, g);
    {
        arb_t gl;
        arb_init(gl);
        arbsdp_gershgorin_lower_bound(gl, A, prec);
        arb_abs(gl, gl);
        arb_max(g, g, gl, prec);
        arb_clear(gl);
    }
    if (arb_is_zero(L))
        r = -1e9;
    else
        r = (double) arf_abs_bound_lt_2exp_si(arb_midref(L))
            - (double) arf_abs_bound_lt_2exp_si(arb_midref(g)) + (double) prec;
    arb_clear(L);
    arb_clear(g);
    return r;
}

/* Spectrum kinds for the property sweep: 0 singular PSD (lmin = 0),
 * 1 tiny (2^-100), 2 clustered (1, 1+2^-60, 1+2^-30, ...), 3 indefinite (-1/3),
 * 4 wide (1 .. 2^60).  Remaining entries are small integers from the seed. */
static void
fill_spectrum(fmpq *D, slong n, int kind, unsigned seed)
{
    unsigned st = seed;
    for (slong i = 0; i < n; i++) {
        st = st * 1103515245u + 12345u;
        fmpq_set_si(D + i, 2 + (slong) ((st >> 16) % 9), 1);
    }
    switch (kind) {
    case 0: fmpq_zero(D + 0); break;
    case 1: fmpq_one(D + 0); fmpq_div_2exp(D + 0, D + 0, 100); break;
    case 2:
        fmpq_one(D + 0);
        if (n > 1) { fmpq_one(D + 1); fmpq_div_2exp(D + 1, D + 1, 60); fmpq_add_si(D + 1, D + 1, 1); }
        if (n > 2) { fmpq_one(D + 2); fmpq_div_2exp(D + 2, D + 2, 30); fmpq_add_si(D + 2, D + 2, 1); }
        break;
    case 3: fmpq_set_si(D + 0, -1, 3); break;
    default:
        fmpq_one(D + 0);
        if (n > 1) { fmpq_one(D + n - 1); fmpq_mul_2exp(D + n - 1, D + n - 1, 60); }
        break;
    }
}

/* TIGHTNESS + PROBE-COUNT gates (bead ml6).  Before ml6 the bisection stopped at
 * 2^(-prec/2)*||A||, i.e. slack_bits ~ prec/2 = 256 at prec 512, with ~prec/2
 * probes.  Gates: slack_bits <= 64 for every kind at n in {2,8,16}, prec 512,
 * via BOTH the seeded path (default) and the fallback path (NaN seed forces the
 * bisection fallback, so its tolerance is gated too); and <= 8 cho probes on the
 * default (seeded) path. */
static void
test_lambda_min_tight_and_cheap(void)
{
    const slong prec = 512;
    const slong ns[] = { 2, 8, 16 };
    double worst_seeded = -1e9, worst_fallback = -1e9;
    slong worst_probes = 0;
    for (int ni = 0; ni < 3; ni++) {
        for (int kind = 0; kind < 5; kind++) {
            slong n = ns[ni];
            fmpq *D = _fmpq_vec_init(n);
            fmpq_t lmin;
            arb_mat_t A;
            arb_t d;
            arf_t nanseed;
            slong probes;
            double sb;
            fmpq_init(lmin);
            arb_mat_init(A, n, n);
            arb_init(d);
            arf_init(nanseed);
            fill_spectrum(D, n, kind, 77u + (unsigned) n);
            spectral_fixture(A, lmin, D, n, 991u + (unsigned) (n * 7 + kind), prec);

            probes = arbsdp_lambda_min_lower_bound_seeded(d, A, NULL, prec);
            sb = slack_bits(d, lmin, A, prec);
            CHECK(d_le_exact(d, lmin), "ml6 tight: seeded d <= lmin (rigor)");
            CHECK(sb <= 64, "ml6 tight: seeded slack <= 2^-(prec-64)*scale");
            CHECK(probes <= 8, "ml6 probes: seeded path uses <= 8 cho probes");

            if (sb > worst_seeded) worst_seeded = sb;
            if (probes > worst_probes) worst_probes = probes;

            arf_nan(nanseed);
            arbsdp_lambda_min_lower_bound_seeded(d, A, nanseed, prec);
            sb = slack_bits(d, lmin, A, prec);
            CHECK(d_le_exact(d, lmin), "ml6 tight: fallback d <= lmin (rigor)");
            CHECK(sb <= 64, "ml6 tight: fallback slack <= 2^-(prec-64)*scale");
            if (sb > worst_fallback) worst_fallback = sb;

            arf_clear(nanseed);
            arb_clear(d);
            arb_mat_clear(A);
            fmpq_clear(lmin);
            _fmpq_vec_clear(D, n);
        }
    }
    printf("  [ml6] prec=512 worst log2(slack/scale)+prec: seeded=%.0f fallback=%.0f;"
           " max seeded probes=%ld\n", worst_seeded, worst_fallback, (long) worst_probes);
}

/* PROPERTY: d <= lambda_min EXACTLY, for random constructed spectra (singular
 * PSD, tiny, clustered, indefinite, wide), n in {2,3,5,8}, prec in {64,128,512},
 * 5 Householder seeds each = 300 cases. */
static void
test_lambda_min_property(void)
{
    const slong ns[] = { 2, 3, 5, 8 };
    const slong precs[] = { 64, 128, 512 };
    int bad = 0, total = 0;
    for (int ni = 0; ni < 4; ni++)
        for (int pi = 0; pi < 3; pi++)
            for (int kind = 0; kind < 5; kind++)
                for (unsigned s = 0; s < 5; s++) {
                    slong n = ns[ni], prec = precs[pi];
                    fmpq *D = _fmpq_vec_init(n);
                    fmpq_t lmin;
                    arb_mat_t A;
                    arb_t d;
                    fmpq_init(lmin);
                    arb_mat_init(A, n, n);
                    arb_init(d);
                    fill_spectrum(D, n, kind, 13u * s + (unsigned) n);
                    spectral_fixture(A, lmin, D, n, 5u + 101u * s + (unsigned) kind, prec);
                    arbsdp_lambda_min_lower_bound(d, A, prec);
                    total++;
                    if (!d_le_exact(d, lmin))
                        bad++;
                    arb_clear(d);
                    arb_mat_clear(A);
                    fmpq_clear(lmin);
                    _fmpq_vec_clear(D, n);
                }
    CHECK(bad == 0, "ml6 property: d <= lambda_min for every constructed spectrum");
    printf("  [ml6] property: %d/%d constructed spectra bracketed (d <= lmin)\n",
           total - bad, total);
}

/* ADVERSARIAL SEED: the seed is a non-rigorous hint; garbage must never leak.
 * Seeds: NaN, +inf, -inf, lmin+1, lmin + 2^-400 (just above: the classic
 * "return the seed" trap), +2^1000, -2^1000.  d <= lmin must hold for all, d
 * must be finite (the Gershgorin floor is finite for finite A). */
static void
test_lambda_min_adversarial_seed(void)
{
    const slong prec = 512, n = 6;
    fmpq *D = _fmpq_vec_init(n);
    fmpq_t lmin;
    arb_mat_t A;
    arb_t d;
    arf_t seed;
    int bad = 0;
    fmpq_init(lmin);
    arb_mat_init(A, n, n);
    arb_init(d);
    arf_init(seed);
    for (int kind = 0; kind < 4; kind++) {
        fill_spectrum(D, n, kind, 4242u);
        spectral_fixture(A, lmin, D, n, 31337u + (unsigned) kind, prec);
        for (int t = 0; t < 7; t++) {
            arf_t q;
            arf_init(q);
            arf_set_fmpq(q, lmin, prec + 64, ARF_RND_NEAR);
            switch (t) {
            case 0: arf_nan(seed); break;
            case 1: arf_pos_inf(seed); break;
            case 2: arf_neg_inf(seed); break;
            case 3: arf_add_si(seed, q, 1, prec + 64, ARF_RND_NEAR); break;
            case 4: arf_one(seed); arf_mul_2exp_si(seed, seed, -400);
                    arf_add(seed, q, seed, prec + 64, ARF_RND_UP); break;
            case 5: arf_one(seed); arf_mul_2exp_si(seed, seed, 1000); break;
            default: arf_one(seed); arf_mul_2exp_si(seed, seed, 1000); arf_neg(seed, seed); break;
            }
            arbsdp_lambda_min_lower_bound_seeded(d, A, seed, prec);
            if (!d_le_exact(d, lmin) || !arb_is_finite(d))
                bad++;
            arf_clear(q);
        }
    }
    CHECK(bad == 0, "ml6 adversarial seed: d <= lambda_min and finite for every garbage seed");
    printf("  [ml6] adversarial seeds: %d/28 violations\n", bad);
    arf_clear(seed);
    arb_clear(d);
    arb_mat_clear(A);
    fmpq_clear(lmin);
    _fmpq_vec_clear(D, n);
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
    test_lambda_min_growth_exhausted();
    test_lambda_min_tight_and_cheap();
    test_lambda_min_property();
    test_lambda_min_adversarial_seed();
    test_dual_residual();
    test_apriori();

    /* Free FLINT's thread-local caches (fmpz pool, arf scratch, registered
     * cleanup hooks) so valgrind reports no possibly-lost / reachable
     * blocks at exit (bead arb-prec-IPM-7o8 memcheck gate). */
    flint_cleanup_master();

    if (failures != 0) {
        fprintf(stderr, "test_certify: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_certify (verified_psd, ball-mode)\n");
    return EXIT_SUCCESS;
}
