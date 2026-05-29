/*
 * include/arbsdp/regularize.h -- Layer-0 (POINT MODE) adaptive 3-way Tikhonov
 * regularization of the Schur-complement Cholesky factorization.
 *
 * ==========================================================================
 * POINT MODE -- PERTURBS THE SOLVED SYSTEM, NOT THE CERTIFIED ONE
 * ==========================================================================
 * (CLAUDE.md rule 1, invariant 4 -- NON-NEGOTIABLE BOUNDARY.)
 *
 * Each Layer-0 IPM iteration factors the m x m Schur matrix M (schur.h) to
 * compute the Mehrotra search direction.  On real problems M is routinely ill-
 * conditioned, indefinite, or numerically singular (the condition number grows
 * like 1/mu by design -- CLAUDE.md invariant 3), so a plain arb_mat_cho fails.
 * The fix is a 3-way Tikhonov diagonal lift:
 *
 *     M_reg = M + (delta_p + delta_d + delta_g) * I            (MATH_SPEC §4.1)
 *
 * and arbsdp_factor_with_reg factors M_reg instead of M, bumping the delta tiers
 * on failure until cho succeeds (or all tiers are capped -> honest failure).
 *
 * CRITICAL SEPARATION (invariant 4):  M_reg is what Layer 0 factors and solves;
 * the regularized direction is only a STARTING GUESS.  Layer 1 (certify.c) MUST
 * certify the ORIGINAL, UNPERTURBED problem -- it forms the dual residual
 * Z = C - sum_i y_i A_i from the actual problem matrices, never any regularized
 * version, and absorbs the iterate's error into the rigorous lambda_min bound.
 * The applied delta values are kept in arbsdp_regstate precisely so the
 * perturbation is AUDITABLE: the caller can see exactly what was added to M's
 * diagonal and is responsible for never feeding M_reg into the certifier.  Keep
 * the two books strictly separate (MATH_SPEC §4.4).
 *
 * Although arb_mat_cho is itself a rigorous ball routine (a successful cho is a
 * PROOF that M_reg is PD), here its return is used only as a fast PD check + a
 * factor source for the POINT-mode direction solve.  No rigor about the
 * ORIGINAL problem is claimed from this module (CLAUDE.md rule 1).
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §4 (three-way Tikhonov), §4.2 (diagnose heuristic),
 *     §4.3 (retry loop), §10.4 (DECIDED: carry delta across iterations, match
 *     TS -- no reset-per-iter), §10.6 (DECIDED: maxRefactor = 20).
 *   - solver-ipm/src/solver/Regularization.ts:
 *       :96-115  three tiers + caps (cap_p=1e-2, cap_d=1e+2, cap_g=1e+2) and
 *                bump factors (primal x10, dual x100, gap x100);
 *       :174-192 makeSdpDiagnose -- per-constraint Frobenius row norm
 *                rowNorm_i = sqrt(sum_b ||A_i^b||_F^2); threshold = 1e-10 *
 *                max(globalNorm, 1e-300); diagnose = "primal" if
 *                rowNorm_r < threshold else "gap";
 *       :231-269 factorWith3Way -- retry up to maxRefactor; on failure
 *                diagnose+bump; if diagnosed tier capped, fall through to
 *                gap -> dual -> primal; all capped -> success = false;
 *       :278-305 tryBump -- clamp at cap; use max(jitter, 1e-12) as the
 *                multiplicative base so the first bump off zero is non-zero.
 *   - solver-ipm/src/solver/Defaults.ts:40-46 -- initialJitter = 1e-12 (lives
 *     on the GAP tier), caps, bump factors.
 *   - docs/FLINT_NOTES.md -- arb_mat_cho returns 1 iff PROVEN PD.
 *
 * ==========================================================================
 * PORTING NOTE -- the missing failRow (a faithful, documented deviation)
 * ==========================================================================
 * The TS factorWith3Way drives diagnose() with the ROW INDEX at which the
 * in-place Cholesky hit a non-positive pivot (choleskyInPlace returns it).
 * FLINT's rigorous arb_mat_cho returns only 1/0 -- it does NOT report a failing
 * row.  We therefore port the diagnose SEMANTICS, not its per-row signature:
 *
 *   - We precompute the per-constraint Frobenius row norms once (exactly
 *     makeSdpDiagnose's rowNorms) and the global max.
 *   - On a cho failure we diagnose "primal" iff SOME constraint row has
 *     rowNorm < 1e-10 * max(globalNorm, 1e-300) (a primal-degenerate row is
 *     present), else "gap".  This matches the TS intent: a primal-degenerate
 *     row drives a primal bump; otherwise the generic gap fallback fires.  When
 *     no failing-row index is available, "is there a degenerate row at all" is
 *     the most defensible row-agnostic reading of the TS rule.
 *
 * The dual tier is reachable only via the cap fall-through (gap -> dual ->
 * primal), exactly as in TS (whose v1 diagnose never returns "dual" either --
 * the Schur RHS h is unavailable at factor time; Regularization.ts:39-53).
 *
 * Arb memory discipline (CLAUDE.md rule 7): arbsdp_regstate owns three arb_t
 * deltas; pair every arbsdp_regstate_init with arbsdp_regstate_clear.
 * arbsdp_factor_with_reg allocates only scoped temporaries it clears itself and
 * never mutates M.  Dimension preconditions are asserted (CLAUDE.md rule 5).
 */

#ifndef ARBSDP_REGULARIZE_H
#define ARBSDP_REGULARIZE_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_regstate -- mutable regularization state carried ACROSS iterations
 * (MATH_SPEC §10.4: carry, do not reset per iteration).
 *
 * delta_p, delta_d, delta_g are the three cumulative Tikhonov tiers; the
 * diagonal lift actually applied to M is their SUM (delta_p + delta_d + delta_g).
 * The bumps_* / refactors counters are diagnostic (how many times each tier was
 * bumped, and the total successful-bump count over the state's lifetime),
 * mirroring RegState.bumps* / refactors in Regularization.ts.
 *
 * Caps and bump factors are NOT stored here -- they are fixed module constants
 * matching Defaults.ts (cap_p=1e-2/x10, cap_d=1e+2/x100, cap_g=1e+2/x100).
 */
typedef struct {
    arb_t delta_p;   /* primal tier in [0, 1e-2], bump x10                    */
    arb_t delta_d;   /* dual   tier in [0, 1e+2], bump x100                   */
    arb_t delta_g;   /* gap    tier in [0, 1e+2], bump x100; init 1e-12       */
    ulong bumps_p;   /* number of primal bumps over this state's lifetime     */
    ulong bumps_d;   /* number of dual bumps                                  */
    ulong bumps_g;   /* number of gap bumps                                   */
    ulong refactors; /* total successful bumps (refactor attempts triggered)  */
} arbsdp_regstate;

/*
 * arbsdp_regstate_init -- initialize the state to its per-solve defaults
 * (Defaults.ts:40-46): delta_p = delta_d = 0, delta_g = initialJitter = 1e-12;
 * all counters 0.  Allocates the three arb_t deltas; pair with
 * arbsdp_regstate_clear.  `prec` controls the rounding of the 1e-12 literal.
 */
void arbsdp_regstate_init(arbsdp_regstate *rs, slong prec);

/*
 * arbsdp_regstate_reset -- reset the deltas/counters back to the init defaults
 * WITHOUT reallocating (delta_p=delta_d=0, delta_g=1e-12, counters 0).
 *
 * NOTE (MATH_SPEC §10.4): the IPM main loop must NOT call this between
 * iterations -- the binding decision is to CARRY delta over.  Reset is provided
 * for starting a fresh solve with a reused state and for tests.
 */
void arbsdp_regstate_reset(arbsdp_regstate *rs, slong prec);

/*
 * arbsdp_regstate_clear -- free the three arb_t deltas and re-zero the struct
 * (CLAUDE.md rule 7: matching init/clear, no leaks).
 */
void arbsdp_regstate_clear(arbsdp_regstate *rs);

/*
 * arbsdp_factor_with_reg -- factor M with adaptive 3-way Tikhonov regularization.
 *
 *   L      out: lower-triangular Cholesky factor of M + lift*I on success
 *               (lift = delta_p + delta_d + delta_g, the CARRIED state).
 *               L must be an initialized m x m arb_mat (asserted square, same
 *               side as M).  Untouched on failure.
 *   M      in : the m x m symmetric Schur matrix (POINT mode).  NOT modified.
 *   rs     in/out: the carried regularization state.  On entry the lift starts
 *               from the CARRIED deltas (NOT reset -- MATH_SPEC §10.4); on exit
 *               the deltas reflect any bumps applied this call.
 *   p      in : the problem, used ONLY as the diagnose input (per-constraint
 *               Frobenius row norms across blocks; Regularization.ts:174-192).
 *               p->m must equal arb_mat_nrows(M) (asserted).
 *   prec   in : working precision.
 *   attempts_out  out (optional, may be NULL): the number of factorization
 *               ATTEMPTS made this call (1 if M+carried-lift factored first try;
 *               up to maxRefactor = 20).  Always in [1, 20] on return.
 *
 * Returns 1 iff a clean factor was produced (M + lift*I provably PD), with `L`
 * holding it and `rs` updated.  Returns 0 on HONEST failure (every tier capped
 * and M + max-lift*I still not provably PD) -- the caller / precision controller
 * is expected to escalate prec (b16), per CLAUDE.md rule 5 (fail fast, fail
 * loud; never a silently-fudged factor).
 *
 * Retry loop (Regularization.ts:231-269): up to maxRefactor = 20 attempts; each
 * failure diagnoses primal-vs-gap (see header porting note) and bumps that tier,
 * falling through gap -> dual -> primal when the diagnosed tier is capped.
 */
int arbsdp_factor_with_reg(arb_mat_t L, const arb_mat_t M, arbsdp_regstate *rs,
                           const arbsdp_problem *p, slong prec,
                           int *attempts_out);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_REGULARIZE_H */
