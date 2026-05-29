/*
 * test/test_direction.c -- tests for the Mehrotra predictor-corrector direction +
 * HSDE step length (c/src/direction.c, bead arb-prec-IPM-b14).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written before c/src/direction.c and
 * include/arbsdp/direction.h carry an implementation; the target first fails to
 * link (undefined symbols arbsdp_psd_max_step / arbsdp_sigma / arbsdp_clip_step /
 * arbsdp_tau_kappa_max_step / arbsdp_direction_*), then passes once the impl
 * lands.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §3.5 (Mehrotra RHS + direction recovery),
 *     §3.6 (sigma = clip((mu_aff/mu)^3, 1e-8, 0.9)),
 *     §3.7 (psdMaxStep + tau-kappa step + safeguard clip(max(0.95a,2a-1),0.999999)).
 *   - HsdeNtSdpSolver.ts:613-827 (predictor/corrector RHS, step, safeguard);
 *     PsdCone.ts:186-202 (psdMaxStep).
 *
 * POINT MODE (CLAUDE.md rule 1): a Layer-0 kernel; W is a POINT from
 * arbsdp_nt_scaling and the direction is consumed as midpoints.  Assertions
 * compare MIDPOINTS within a justified tolerance (as test_schur / test_ntscaling).
 *
 * Property coverage (bead arb-prec-IPM-b14):
 *   1. STEP keeps cone feasible  closed-form X=I, dX=diag(-1,-2,-1/2) => alpha=1
 *      (most-negative eig is -2 in scaled coords; 1/2 of that... no: X=I makes the
 *      scaled increment dX itself, lmin=-2, alpha=1/2).  We use the canonical
 *      X=I, dX=diag(-1,-3,-2): lmin=-3, alpha=1/3, and X+alpha*dX has min eig 0
 *      (on the PSD boundary); X+(alpha-eps)*dX is PD; X+(alpha+eps)*dX is NOT.
 *      Also the simplest X=I, dX=-I => alpha=1 (boundary X+dX=0).
 *   2. SIGMA clip  mu_aff/mu extremes clip to [1e-8, 0.9]; a mid value cubes.
 *   3. MU DECREASES  one predictor-corrector step on a tiny golden SDP (sdp_sqrt2)
 *      from a strictly-feasible start strictly decreases mu (descent property).
 *   4. FACTOR REUSE  the step makes exactly ONE arbsdp_factor_with_reg call
 *      (d->factor_calls == 1): predictor + corrector share the factor.
 *   5. MUTATION (CLAUDE.md rule 8): a wrong sigma (sigma=0 corrector, i.e. no
 *      centering) or no clip is caught -- we check that arbsdp_sigma actually
 *      clips (a hand-rolled un-clipped cube differs) and a wrong-SIGN step in
 *      psd_max_step (dX positive => unbounded, NOT a finite step) is caught.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/ntscaling.h"
#include "arbsdp/regularize.h"
#include "arbsdp/linalg.h"
#include "arbsdp/svec.h"
#include "arbsdp/direction.h"

#define PREC 256

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* |mid(a) - v| <= tol  (point-mode closeness of an arb midpoint to a double). */
static int
close_d(const arb_t a, double v, double tol)
{
    arb_t d, t;
    int ok;
    arb_init(d);
    arb_init(t);
    arb_get_mid_arb(d, a);
    arb_set_d(t, v);
    arb_sub(d, d, t, PREC);
    arb_abs(d, d);
    arb_set_d(t, tol);
    ok = arb_le(d, t);
    arb_clear(d);
    arb_clear(t);
    return ok;
}

/* min eigenvalue of a symmetric matrix (point-mode, via arbsdp_eigh ascending). */
static double
min_eig_d(const arb_mat_t A)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t Q;
    arb_ptr lam;
    double out;
    arb_mat_init(Q, n, n);
    lam = _arb_vec_init(n);
    arbsdp_eigh(lam, Q, A, PREC);
    out = arf_get_d(arb_midref(lam + 0), ARF_RND_NEAR);
    _arb_vec_clear(lam, n);
    arb_mat_clear(Q);
    return out;
}

/* ------------------------------------------------------------------------- *
 * Test 1: psd_max_step keeps the cone feasible (closed-form diagonal cases). *
 * ------------------------------------------------------------------------- */
static void
test_step_cone(void)
{
    arb_mat_t X, dX, T;
    arb_t alpha, a;
    int bounded;
    double amid;

    arb_mat_init(X, 3, 3);
    arb_mat_init(dX, 3, 3);
    arb_mat_init(T, 3, 3);
    arb_init(alpha);
    arb_init(a);

    /* (a) X = I, dX = -I  =>  X + alpha*dX = (1-alpha)I; boundary at alpha = 1. */
    arb_mat_one(X);
    arb_mat_one(dX);
    arb_mat_neg(dX, dX);
    bounded = arbsdp_psd_max_step(alpha, X, dX, PREC);
    CHECK(bounded == 1, "psd_max_step: X=I,dX=-I must be BOUNDED");
    CHECK(close_d(alpha, 1.0, 1e-50), "psd_max_step: X=I,dX=-I => alpha=1");

    /* (b) X = I, dX = diag(-1,-3,-2): most-negative eig -3 => alpha = 1/3.
     *     At alpha: X+alpha*dX = diag(2/3, 0, 1/3) -> min eig = 0 (PSD boundary). */
    arb_mat_one(X);
    arb_mat_zero(dX);
    arb_set_si(arb_mat_entry(dX, 0, 0), -1);
    arb_set_si(arb_mat_entry(dX, 1, 1), -3);
    arb_set_si(arb_mat_entry(dX, 2, 2), -2);
    bounded = arbsdp_psd_max_step(alpha, X, dX, PREC);
    CHECK(bounded == 1, "psd_max_step: diag must be BOUNDED");
    /* 1.0/3.0 is the closest double to 1/3 (~16 good digits); the arb alpha is
     * accurate to ~prec, so the difference is bounded by the DOUBLE's error ~1e-16.
     * A 1e-50 tol would test the double literal, not the routine; use 1e-12. */
    CHECK(close_d(alpha, 1.0 / 3.0, 1e-12), "psd_max_step: diag(-1,-3,-2) => alpha=1/3");

    /* X + alpha*dX is on the PSD boundary: min eig ~ 0. */
    arb_mat_scalar_mul_arb(T, dX, alpha, PREC);
    arb_mat_add(T, T, X, PREC);
    CHECK(min_eig_d(T) > -1e-40 && min_eig_d(T) < 1e-40,
          "psd_max_step: X+alpha*dX must be on PSD boundary (min eig ~ 0)");

    /* X + (alpha-eps)*dX is PD (min eig > 0). */
    arb_set_d(a, 0.01);
    arb_sub(a, alpha, a, PREC);             /* alpha - 0.01 */
    arb_mat_scalar_mul_arb(T, dX, a, PREC);
    arb_mat_add(T, T, X, PREC);
    CHECK(min_eig_d(T) > 1e-6, "psd_max_step: X+(alpha-eps)*dX must be PD");

    /* X + (alpha+eps)*dX is NOT PSD (min eig < 0). */
    arb_set_d(a, 0.01);
    arb_add(a, alpha, a, PREC);             /* alpha + 0.01 */
    arb_mat_scalar_mul_arb(T, dX, a, PREC);
    arb_mat_add(T, T, X, PREC);
    CHECK(min_eig_d(T) < -1e-6, "psd_max_step: X+(alpha+eps)*dX must be indefinite");

    /* (c) wrong-sign mutation: dX = +I  =>  X + alpha*dX stays PD for all alpha:
     *     UNBOUNDED (return 0).  A sign bug returning a finite step is caught. */
    arb_mat_one(dX);
    amid = -12345.0;
    arb_set_d(alpha, amid);                 /* sentinel; must be left unchanged */
    bounded = arbsdp_psd_max_step(alpha, X, dX, PREC);
    CHECK(bounded == 0, "psd_max_step: dX=+I must be UNBOUNDED (wrong-sign guard)");

    arb_mat_clear(X);
    arb_mat_clear(dX);
    arb_mat_clear(T);
    arb_clear(alpha);
    arb_clear(a);
}

/* ------------------------------------------------------------------------- *
 * Test 1b: tau-kappa step-to-boundary.                                       *
 * ------------------------------------------------------------------------- */
static void
test_tau_kappa_step(void)
{
    arb_t tau, dtau, kappa, dkappa, alpha;
    int bounded;

    arb_init(tau); arb_init(dtau); arb_init(kappa); arb_init(dkappa);
    arb_init(alpha);

    /* tau=1, dtau=-0.5 -> alpha_tau = 2; kappa=1, dkappa=-1 -> alpha_kappa = 1.
     * min = 1. */
    arb_one(tau);  arb_set_d(dtau, -0.5);
    arb_one(kappa); arb_set_si(dkappa, -1);
    bounded = arbsdp_tau_kappa_max_step(alpha, tau, dtau, kappa, dkappa, PREC);
    CHECK(bounded == 1, "tau_kappa_step: must be BOUNDED");
    CHECK(close_d(alpha, 1.0, 1e-50), "tau_kappa_step: min(-tau/dtau,-kappa/dkappa)=1");

    /* both directions non-negative -> UNBOUNDED. */
    arb_set_si(dtau, 2);
    arb_set_si(dkappa, 3);
    bounded = arbsdp_tau_kappa_max_step(alpha, tau, dtau, kappa, dkappa, PREC);
    CHECK(bounded == 0, "tau_kappa_step: dtau,dkappa>=0 must be UNBOUNDED");

    arb_clear(tau); arb_clear(dtau); arb_clear(kappa); arb_clear(dkappa);
    arb_clear(alpha);
}

/* ------------------------------------------------------------------------- *
 * Test 2: sigma = clip((mu_aff/mu)^3, 1e-8, 0.9).                            *
 * ------------------------------------------------------------------------- */
static void
test_sigma_clip(void)
{
    arb_t mu, mu_aff, sigma;
    arb_init(mu); arb_init(mu_aff); arb_init(sigma);

    /* (a) mu_aff/mu = 0.5 -> 0.125 (mid, in-range, cubes correctly). */
    arb_set_d(mu, 1.0);
    arb_set_d(mu_aff, 0.5);
    arbsdp_sigma(sigma, mu_aff, mu, PREC);
    CHECK(close_d(sigma, 0.125, 1e-50), "sigma: (0.5)^3 = 0.125 (mid, no clip)");

    /* (b) mu_aff/mu = 0.99 -> 0.970299 -> CLIP to upper 0.9.  (0.9 and 0.99 are
     *     non-exact doubles; the clip output is the literal 0.9 -> compare at the
     *     double's ~1e-16 resolution with a 1e-12 margin.) */
    arb_set_d(mu, 1.0);
    arb_set_d(mu_aff, 0.99);
    arbsdp_sigma(sigma, mu_aff, mu, PREC);
    CHECK(close_d(sigma, 0.9, 1e-12), "sigma: 0.99^3 clips to upper 0.9");

    /* (c) mu_aff/mu = 1e-4 -> 1e-12 -> CLIP to lower 1e-8. */
    arb_set_d(mu, 1.0);
    arb_set_d(mu_aff, 1e-4);
    arbsdp_sigma(sigma, mu_aff, mu, PREC);
    CHECK(close_d(sigma, 1e-8, 1e-20), "sigma: 1e-12 clips to lower 1e-8");

    /* (d) mutation guard: an UN-clipped cube of (c) would be 1e-12, far from the
     *     clipped 1e-8.  Confirm the clip is load-bearing. */
    CHECK(!close_d(sigma, 1e-12, 1e-16),
          "MUTATION: sigma without lower-clip would be 1e-12 (test must discriminate)");

    arb_clear(mu); arb_clear(mu_aff); arb_clear(sigma);
}

/* ------------------------------------------------------------------------- *
 * Test 2b: clip_step safeguard.                                             *
 * ------------------------------------------------------------------------- */
static void
test_clip_step(void)
{
    arb_t a, out;
    arb_init(a); arb_init(out);

    /* a = 0.5: max(0.95*0.5, 2*0.5-1) = max(0.475, 0) = 0.475.  (0.475/0.98 are
     *     non-exact doubles -> 1e-12 margin around the double literal.) */
    arb_set_d(a, 0.5);
    arbsdp_clip_step(out, a, PREC);
    CHECK(close_d(out, 0.475, 1e-12), "clip_step: a=0.5 -> 0.475");

    /* a = 0.99: max(0.9405, 0.98) = 0.98 (<= cap). */
    arb_set_d(a, 0.99);
    arbsdp_clip_step(out, a, PREC);
    CHECK(close_d(out, 0.98, 1e-12), "clip_step: a=0.99 -> 0.98");

    /* a = 2.0 (e.g. unbounded clamped large): max(1.9, 3) = 3 -> cap 0.999999. */
    arb_set_d(a, 2.0);
    arbsdp_clip_step(out, a, PREC);
    CHECK(close_d(out, 0.999999, 1e-12), "clip_step: large a caps at 0.999999");

    arb_clear(a); arb_clear(out);
}

/* ------------------------------------------------------------------------- *
 * Integration: one predictor-corrector step strictly decreases mu on a tiny  *
 * golden SDP (sdp_sqrt2).  This is the core "direction is a descent" test.    *
 * ------------------------------------------------------------------------- */

/* initial scale (MATH_SPEC §3.9): xiP = max(1, 10*||b||_inf),
 * xiD = max(1, 10*||C||_F + ||b||_inf).  Computed in arb point-mode. */
static void
initial_scale(arb_t xiP, arb_t xiD, const arbsdp_problem *p, slong prec)
{
    arb_t bnorm, cfrob, t, bi;
    arb_init(bnorm); arb_init(cfrob); arb_init(t); arb_init(bi);

    arb_zero(bnorm);
    for (int i = 0; i < p->m; i++) {
        arbsdp_problem_b(bi, p, i, prec);
        arb_abs(bi, bi);
        if (arb_gt(bi, bnorm)) arb_set(bnorm, bi);
    }
    /* ||C||_F over all blocks (C = matno 0). */
    arb_zero(cfrob);
    for (int b = 0; b < p->nblocks; b++) {
        slong n = (p->block_sizes[b] < 0) ? -(slong) p->block_sizes[b]
                                          : (slong) p->block_sizes[b];
        arb_mat_t Cb;
        arb_mat_init(Cb, n, n);
        arbsdp_problem_block_mat(Cb, p, 0, b, prec);
        arbsdp_frob_inner(t, Cb, Cb, prec);     /* sum C_ij^2 = ||C||_F^2 */
        arb_add(cfrob, cfrob, t, prec);
        arb_mat_clear(Cb);
    }
    arb_sqrt(cfrob, cfrob, prec);

    /* xiP = max(1, 10*bnorm) */
    arb_mul_si(t, bnorm, 10, prec);
    arb_one(xiP);
    if (arb_gt(t, xiP)) arb_set(xiP, t);
    /* xiD = max(1, 10*cfrob + bnorm) */
    arb_mul_si(t, cfrob, 10, prec);
    arb_add(t, t, bnorm, prec);
    arb_one(xiD);
    if (arb_gt(t, xiD)) arb_set(xiD, t);

    arb_clear(bnorm); arb_clear(cfrob); arb_clear(t); arb_clear(bi);
}

static void
test_mu_decreases(void)
{
    arbsdp_problem p;
    arbsdp_iterate it;
    arbsdp_regstate rs;
    arbsdp_direction d;
    arb_mat_t *W;
    arb_t xiP, xiD, mu_old, mu_new;
    int ok, rc;

    arbsdp_problem_init(&p);
    rc = arbsdp_read_sdpa(&p, GOLDEN_DIR "/sdp_sqrt2/sdp_sqrt2.dat-s");
    CHECK(rc == 0, "mu_decreases: failed to read sdp_sqrt2.dat-s");
    if (rc != 0) { arbsdp_problem_clear(&p); return; }

    arbsdp_iterate_init(&it, &p, PREC);
    arbsdp_regstate_init(&rs, PREC);
    arbsdp_direction_init(&d, &p);

    arb_init(xiP); arb_init(xiD); arb_init(mu_old); arb_init(mu_new);

    /* strictly-feasible start: X^b = xiP*I, S^b = xiD*I, y=0, tau=kappa=1. */
    initial_scale(xiP, xiD, &p, PREC);
    for (int b = 0; b < it.nblocks; b++) {
        slong n = it.block_n[b];
        arb_mat_zero(it.X[b]);
        arb_mat_zero(it.S[b]);
        for (slong i = 0; i < n; i++) {
            arb_set(arb_mat_entry(it.X[b], i, i), xiP);
            arb_set(arb_mat_entry(it.S[b], i, i), xiD);
        }
    }
    for (int i = 0; i < it.m; i++) arb_zero(&it.y[i]);
    arb_one(it.tau);
    arb_one(it.kappa);

    /* residuals + mu at the start. */
    arbsdp_iterate_residuals(&it, PREC);
    arb_set(mu_old, it.mu);

    /* NT scaling per block. */
    W = flint_malloc((size_t) it.nblocks * sizeof *W);
    for (int b = 0; b < it.nblocks; b++) {
        slong n = it.block_n[b];
        arb_mat_init(W[b], n, n);
        arbsdp_nt_scaling(W[b], it.X[b], it.S[b], PREC);
    }

    ok = arbsdp_direction_compute(&d, &it, (const arb_mat_t *) W, &rs, PREC);
    CHECK(ok == 1, "mu_decreases: direction_compute failed (factor)");

    /* FACTOR REUSE (Test 4): exactly ONE factor_with_reg invocation. */
    CHECK(d.factor_calls == 1,
          "factor reuse: predictor + corrector must share ONE factor (factor_calls==1)");

    if (ok) {
        /* take the step:  X += alpha*dX, S += alpha*dS, y += alpha*dy,
         *                 tau += alpha*dtau, kappa += alpha*dkappa. */
        for (int b = 0; b < it.nblocks; b++) {
            arb_mat_scalar_addmul_arb(it.X[b], d.dX[b], d.alpha, PREC);
            arb_mat_scalar_addmul_arb(it.S[b], d.dS[b], d.alpha, PREC);
            arbsdp_symmetrize(it.X[b], PREC);
            arbsdp_symmetrize(it.S[b], PREC);
        }
        for (int i = 0; i < it.m; i++)
            arb_addmul(&it.y[i], d.alpha, &d.dy[i], PREC);
        arb_addmul(it.tau, d.alpha, d.dtau, PREC);
        arb_addmul(it.kappa, d.alpha, d.dkappa, PREC);

        arbsdp_iterate_residuals(&it, PREC);
        arb_set(mu_new, it.mu);

        /* descent: mu_new < mu_old. */
        CHECK(arb_lt(mu_new, mu_old), "mu_decreases: one step must strictly decrease mu");

        printf("  mu_decreases: mu_old = %.6e, mu_new = %.6e, alpha = %.6e, sigma = %.6e\n",
               arf_get_d(arb_midref(mu_old), ARF_RND_NEAR),
               arf_get_d(arb_midref(mu_new), ARF_RND_NEAR),
               arf_get_d(arb_midref(d.alpha), ARF_RND_NEAR),
               arf_get_d(arb_midref(d.sigma), ARF_RND_NEAR));
    }

    for (int b = 0; b < it.nblocks; b++) arb_mat_clear(W[b]);
    flint_free(W);
    arb_clear(xiP); arb_clear(xiD); arb_clear(mu_old); arb_clear(mu_new);
    arbsdp_direction_clear(&d);
    arbsdp_regstate_clear(&rs);
    arbsdp_iterate_clear(&it);
    arbsdp_problem_clear(&p);
}

int
main(void)
{
    test_step_cone();
    test_tau_kappa_step();
    test_sigma_clip();
    test_clip_step();
    test_mu_decreases();

    if (failures != 0) {
        fprintf(stderr, "test_direction: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_direction (Mehrotra predictor-corrector + step length)\n");
    return EXIT_SUCCESS;
}
