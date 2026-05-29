/*
 * include/arbsdp/direction.h -- Layer-0 (POINT MODE) Mehrotra predictor-corrector
 * search direction + HSDE step length for the NT-scaled HSDE IPM.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * The search direction (dX, dy, dS, dtau, dkappa) and the step length alpha are
 * computed INSIDE the HSDE-NT IPM loop (Layer 0) from POINT-mode inputs (the NT
 * scaling W, the regularized Schur factor, the current residuals).  Everything
 * here is consumed as MIDPOINTS; ball *radii* are NOT trusted.  Layer 1 derives
 * ALL rigor independently (verified Cholesky of the dual residual, certify.c),
 * never from the direction.  arb_mat_cho / arb_mat_solve_cho_precomp (reached via
 * schur.h / regularize.h) are themselves rigorous, but their returns are used
 * here only as a fast PD check + factor source for the point-mode solve.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §3.5 (Mehrotra predictor sigma=0 + corrector with the
 *     E_x / Rq cross-term; the three RHS sharing ONE Schur factor; the direction
 *     recovery dS/dX and the scalar dtau/dkappa updates), §3.6 (sigma =
 *     clip((mu_aff/mu)^3, 1e-8, 0.9)), §3.7 (psdMaxStep + tau-kappa step +
 *     Mehrotra safeguard clip(max(0.95*a, 2a-1), 0.999999)).
 *   - HsdeNtSdpSolver.ts:
 *       :126-163  buildNtFactor (the G-factor + sv, ported locally in
 *                 direction.c -- needed by the corrector's Rq and S^{-1});
 *       :511-539  W*A*W / W*C*W / W*r_d*W caches (W*C*W and W*r_d*W are the b14
 *                 caches; the W*A*W cache is the schur.h companion);
 *       :574-611  data direction (M*dy1 = b + <A_i, W*C*W>; dS1, dX1; dtau_denom);
 *       :613-660  affine predictor (sigma=0, E_x_aff = -X) + scalar dtau_aff/dkap;
 *       :672-695  affine step, mu_aff, sigma = clip((mu_aff/mu)^3, 1e-8, 0.9);
 *       :697-806  combined corrector (E_x_comb = -X + sigma*mu*S^{-1} - Rq;
 *                 Rq from hdZ = G dS_aff G^T, hdX = diag(-sv) - hdZ, RqSv =
 *                 2*sym(hdX*hdZ)/(sv_i+sv_j), Rq = G^T RqSv G); E_tau_comb;
 *                 scalar dtau/dkappa; full combine dy/dS/dX;
 *       :820-827  combined step length + safeguard clipStep(max(0.95a, 2a-1));
 *       :1168-1175 clipStep (cap at 0.999999) and clip (lo,hi).
 *   - cone/PsdCone.ts:186-202 (psdMaxStep: L L^T = X; M = L^{-1} dX L^{-T};
 *     alpha = lmin>=0 ? Inf : 1/(-lmin)).  Implemented here via the congruent
 *     symmetric form lmin(X^{-1/2} dX X^{-1/2}) (same spectrum), reusing the
 *     point-mode eig kernels (linalg.h).
 *   - docs/FLINT_NOTES.md -- factor-once/solve-many (arb_mat_solve_cho_precomp);
 *     the predictor and corrector SHARE one Schur factor (b14 bead note).
 *
 * REUSE (CLAUDE.md, no duplication): arbsdp_schur_assemble/_solve (schur.h),
 * arbsdp_factor_with_reg (regularize.h), arbsdp_nt_scaling (ntscaling.h),
 * arbsdp_apply_At + the iterate (iterate.h), arbsdp_psd_invsqrt / arbsdp_eigh
 * (linalg.h), arbsdp_frob_inner / arbsdp_symmetrize (svec.h),
 * arbsdp_problem_block_mat / arbsdp_problem_b (problem.h).
 *
 * ==========================================================================
 * STEP-LENGTH CONVENTION (matches the HSDE solver)
 * ==========================================================================
 * The HSDE iterate walks X, S, y, tau, kappa in LOCKSTEP with a SINGLE alpha
 * (HsdeStepLength.ts banner: "there is no alpha_primal vs alpha_dual distinction
 * in HSDE").  alpha = min(alpha_cone_primal, alpha_cone_dual, alpha_tau_kappa),
 * then the Mehrotra safeguard.  The affine (predictor) step is the same min
 * additionally capped at 1.0 (HsdeNtSdpSolver.ts:678) and is used ONLY to form
 * mu_aff -> sigma; it is not taken.
 *
 * Arb memory discipline (CLAUDE.md rule 7): the caller owns init/clear of every
 * output (the arbsdp_direction struct via _init/_clear); these routines allocate
 * only scoped temporaries they clear themselves.  Dimension / PD preconditions
 * are asserted (CLAUDE.md rule 5, fail fast).
 */

#ifndef ARBSDP_DIRECTION_H
#define ARBSDP_DIRECTION_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/regularize.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Scalar Mehrotra helpers (exposed for direct unit testing -- MATH_SPEC §3.6,
 * §3.7; HsdeNtSdpSolver.ts:694-695, :827, :1168-1175).
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_sigma -- the Mehrotra centering parameter
 *     sigma = clip( (mu_aff / max(mu, 1e-300))^3 , 1e-8, 0.9 )
 * (HsdeNtSdpSolver.ts:694-695, MATH_SPEC §3.6).  All operands are POINT-mode
 * scalars; `out`, `mu_aff`, `mu` must be initialized arb_t.  out may alias neither
 * input restriction is required (it is written last).
 */
void arbsdp_sigma(arb_t out, const arb_t mu_aff, const arb_t mu, slong prec);

/*
 * arbsdp_clip_step -- the Mehrotra step safeguard applied to a RAW max-step a:
 *     clip( max(0.95*a, 2*a - 1), 0, 0.999999 )
 * (HsdeNtSdpSolver.ts:827 + clipStep :1168-1172, MATH_SPEC §3.7).  `out` and `a`
 * are initialized arb_t (POINT mode).
 */
void arbsdp_clip_step(arb_t out, const arb_t a, slong prec);

/* -------------------------------------------------------------------------
 * Step-to-boundary (PsdCone.ts:186-202; HsdeStepLength.ts:48-64).
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_psd_max_step -- maximum alpha >= 0 keeping X + alpha*dX positive
 * semidefinite (POINT MODE).  Computes the most-negative generalized eigenvalue
 * lmin of (dX, X) via the congruent symmetric form lmin(X^{-1/2} dX X^{-1/2})
 * (same spectrum as PsdCone.ts's L^{-1} dX L^{-T}); then
 *     alpha = (lmin >= 0) ? UNBOUNDED : 1 / (-lmin).
 *
 * Returns 1 if the step is BOUNDED (some negative eigenvalue), writing the finite
 * boundary alpha into `alpha_out`; returns 0 if UNBOUNDED (X + alpha*dX stays PSD
 * for all alpha >= 0), in which case `alpha_out` is left unchanged.  Caller treats
 * a 0 return as +Infinity in the min().
 *
 * `alpha_out` is an initialized arb_t; X and dX are square n x n (asserted same
 * size).  X must be PD (the IPM keeps the iterate strictly interior); the
 * invsqrt floors a near-singular X (CLAUDE.md rule 5 boundary is the caller's).
 * POINT MODE: alpha_out is a midpoint; radii not trusted.
 */
int arbsdp_psd_max_step(arb_t alpha_out, const arb_mat_t X, const arb_mat_t dX,
                        slong prec);

/*
 * arbsdp_tau_kappa_max_step -- maximum alpha >= 0 keeping tau + alpha*dtau >= 0
 * AND kappa + alpha*dkappa >= 0 (HsdeStepLength.ts:48-64).  alpha_tau = -tau/dtau
 * if dtau < 0 else +Inf; alpha_kappa likewise; result = min.
 *
 * Returns 1 if BOUNDED (dtau<0 or dkappa<0), writing the binding alpha into
 * `alpha_out`; returns 0 if UNBOUNDED (neither binds), `alpha_out` unchanged.
 */
int arbsdp_tau_kappa_max_step(arb_t alpha_out, const arb_t tau, const arb_t dtau,
                              const arb_t kappa, const arb_t dkappa, slong prec);

/* -------------------------------------------------------------------------
 * The full Mehrotra predictor-corrector direction + step.
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_direction -- the combined search direction (dX, dy, dS, dtau, dkappa)
 * and the safeguarded combined step length `alpha`, plus the centering sigma and
 * the affine/combined mu's for diagnostics.  Caller owns init/clear via
 * arbsdp_direction_init / _clear.
 */
typedef struct {
    int        m;          /* number of constraints                       */
    int        nblocks;    /* number of blocks                            */
    slong     *block_n;    /* length nblocks: side of block b             */

    arb_mat_t *dX;         /* length nblocks; combined primal direction   */
    arb_mat_t *dS;         /* length nblocks; combined dual direction     */
    arb_ptr    dy;         /* length m                                    */
    arb_t      dtau;       /* combined tau direction                      */
    arb_t      dkappa;     /* combined kappa direction                    */

    arb_t      alpha;      /* safeguarded combined step length            */
    arb_t      sigma;      /* Mehrotra centering parameter (clipped)       */
    arb_t      mu_aff;     /* affine mu (predictor)                       */

    /* Diagnostics for factor-reuse verification (CLAUDE.md rule 8) and the
     * regularization audit: the number of arbsdp_factor_with_reg ATTEMPTS made by
     * the LAST arbsdp_direction_compute call.  factor_calls is the number of TIMES
     * a factorization routine was INVOKED (must be exactly 1 per step -- the
     * predictor + corrector SHARE the single factor; b14 bead note).             */
    int        factor_attempts;  /* attempts inside the one factor_with_reg call  */
    int        factor_calls;     /* number of factor_with_reg invocations (== 1)  */
} arbsdp_direction;

/*
 * arbsdp_direction_init -- allocate the direction storage for problem `p`.  Pair
 * with arbsdp_direction_clear.  All fields zeroed.
 */
void arbsdp_direction_init(arbsdp_direction *d, const arbsdp_problem *p);

/*
 * arbsdp_direction_clear -- free everything _init allocated and re-zero.
 */
void arbsdp_direction_clear(arbsdp_direction *d);

/*
 * arbsdp_direction_compute -- the one Mehrotra predictor-corrector step.
 *
 * Inputs:
 *   it        the current HSDE iterate with residuals already filled
 *             (arbsdp_iterate_residuals must have been called at `prec`).
 *   W_blocks  the per-block NT scaling (length it->nblocks; W_blocks[b] from
 *             arbsdp_nt_scaling for (X[b], S[b])).  POINT mode.
 *   rs        the carried 3-way regularization state (regularize.h).  The Schur
 *             factor is produced ONCE here via arbsdp_factor_with_reg and reused
 *             for all three RHS (data + affine + combined): factor-once/solve-many
 *             (FLINT_NOTES.md; b14 bead note).
 *   prec      working precision.
 *
 * Output: the combined (dX, dy, dS, dtau, dkappa), the safeguarded `alpha`,
 * `sigma`, `mu_aff`, and the factor_attempts / factor_calls diagnostics.
 *
 * Returns 1 on success; 0 if the Schur factor failed even fully regularized
 * (arbsdp_factor_with_reg returned 0) -- reported honestly (CLAUDE.md rule 5);
 * the direction outputs are then undefined and the caller / precision controller
 * must escalate.
 *
 * d->m / d->nblocks must match it->m / it->nblocks (asserted).
 * POINT MODE: all outputs are midpoints; radii not trusted.
 */
int arbsdp_direction_compute(arbsdp_direction *d, const arbsdp_iterate *it,
                             const arb_mat_t *W_blocks, arbsdp_regstate *rs,
                             slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_DIRECTION_H */
