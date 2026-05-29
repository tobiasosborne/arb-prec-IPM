/*
 * include/arbsdp/solve.h -- the Layer-0 HSDE-NT Mehrotra interior-point solve
 * DRIVER: the first end-to-end solver.  Wires the per-iteration machinery
 * (iterate / residuals / convergence / NT scaling / Schur / regularization /
 * Mehrotra direction + step) into a single fixed-precision solve loop.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * The whole loop runs in Arb *point* arithmetic at a FIXED working `prec`,
 * consumed as MIDPOINTS; ball radii are NOT trusted.  The recovered objective
 * value and the returned iterate are a POINT-mode approximate solution -- they
 * certify NOTHING.  Layer 1 (certify.c, b18/b19) derives ALL rigor independently
 * from the verified Cholesky of the dual residual; the precision controller
 * (b16) adds adaptive escalation.  This bead is deliberately the FIXED-precision
 * driver -- precision escalation on a failed/over-regularized factor is b16; here
 * we return an honest ARBSDP_SOLVE_NUMERICAL status (CLAUDE.md rule 5, fail loud).
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §3 (the full iteration), §3.9 (initial point), §3.8
 *     (convergence / exit), §6 (precision -- here FIXED), §1 (sign convention).
 *   - HsdeNtSdpSolver.ts solveHsdeSdpNt() (the port target):
 *       :291-296  initialScale + X^b = xiP*I, S^b = xiD*I, y=0, tau=kappa=1;
 *       :369-491  per-iteration order: residuals/objectives/mu, convergence +
 *                 best-iterate, exit on terminal/limit/stall;
 *       :502-509  per-block NT factor build;
 *       :541-558  Schur assemble + symmetrise;
 *       :820-846  combined step length + safeguard + update + stall detection;
 *       :929-1001 purifyAndReturn: pObjP = sum_b <C^b, X^b/tau>, the sign flip
 *                 on maximize (primalObj = prob.maximize ? -pObjP : pObjP).
 *   - Defaults.ts:34-49 (feasTol=1e-8, optTol=1e-8, iterLimit=500, stallIterCap=10).
 *   - CLAUDE.md invariant 7 (internal min -<C,X>; flip sign on output),
 *     invariant 4 (regstate perturbs the SOLVED system; carry ONE across iters).
 *
 * ==========================================================================
 * OBJECTIVE RECOVERY (CLAUDE.md invariant 7 -- get this wrong and every
 * objective is negated)
 * ==========================================================================
 * The iterate carries the INTERNAL minimization cost C_int^b = -C_file^b
 * (iterate.h), so the loop's internal primal objective is
 *
 *     pObj_internal = sum_b <C_int^b, X^b> = -<C_file, X>.
 *
 * The reported user-facing value (max <C_file, X> convention) is recovered by
 * purifying (divide by tau) and flipping the internal sign:
 *
 *     value_user = - (pObj_internal / tau)
 *                = - (sum_b <C_int^b, X^b/tau>)
 *                = sum_b <C_file^b, X^b/tau>          (since C_int = -C_file).
 *
 * This is exactly HsdeNtSdpSolver.ts:992 `primalObj = prob.maximize ? -pObjP :
 * pObjP` where pObjP = sum_b <C^b (= the TS-internal NEGATED cost), X^b/tau>: the
 * TS solver stores C already negated, so its `-pObjP` for a maximize problem is
 * our `-(sum_b <C_int^b, X^b/tau>)` = `sum_b <C_file^b, X^b/tau>`.  All golden
 * problems are maximize=1.  For a minimize problem the value is +(pObj/tau).
 *
 * Arb memory discipline (CLAUDE.md rule 7): arbsdp_result_init allocates the arb
 * fields and the embedded iterate handle; pair every arbsdp_result_init with
 * arbsdp_result_clear.  arbsdp_solve allocates only scoped per-iteration
 * temporaries it clears every iteration (no growth across the loop).
 */

#ifndef ARBSDP_SOLVE_H
#define ARBSDP_SOLVE_H

#include <flint/arb.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/convergence.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_solve_status -- the terminal status of a fixed-precision solve.
 *
 * The optimal / infeasibility codes REUSE the Layer-0 convergence enum
 * (convergence.h arbsdp_status) values so a caller can compare directly; the
 * resource/numerical exits add codes the per-iteration test does not produce.
 * Note (CLAUDE.md rule 1): these are POINT-mode classifications steering the
 * loop -- they are NOT the rigorous Layer-1 status (b19).
 */
typedef enum {
    ARBSDP_SOLVE_OPTIMAL           = 1,  /* converged: 6-flag optimal (== ARBSDP_STATUS_OPTIMAL) */
    ARBSDP_SOLVE_PRIMAL_INFEASIBLE = 2,  /* (b20) currently never set                            */
    ARBSDP_SOLVE_DUAL_INFEASIBLE   = 3,  /* (b20) currently never set                            */
    ARBSDP_SOLVE_ITER_LIMIT        = 4,  /* hit iter_limit without converging                    */
    ARBSDP_SOLVE_STALL             = 5,  /* stall counter hit stall cap (mu stopped decreasing)  */
    ARBSDP_SOLVE_NUMERICAL         = 6   /* factor failed even fully regularized (escalate: b16) */
} arbsdp_solve_status;

/*
 * arbsdp_solve_params -- the fixed-precision solve controls (Defaults.ts:34-49).
 * Pass ARBSDP_FEAS_TOL / ARBSDP_OPT_TOL / ARBSDP_ITER_LIMIT / ARBSDP_STALL_ITER_CAP
 * (convergence.h) for the TS defaults.  feas_tol/opt_tol are the 6-flag scaling
 * tolerances; iter_limit caps iterations; stall_cap caps consecutive non-decreasing
 * mu iterations before an honest ARBSDP_SOLVE_STALL exit.
 */
typedef struct {
    double feas_tol;     /* primal+dual feasibility tolerance (Defaults: 1e-8)   */
    double opt_tol;      /* optimality-gap tolerance         (Defaults: 1e-8)    */
    int    iter_limit;   /* hard iteration cap               (Defaults: 500)     */
    int    stall_cap;    /* consecutive-stall cap            (Defaults: 10)      */
} arbsdp_solve_params;

/*
 * arbsdp_solve_default_params -- fill `out` with the TS Defaults.ts values
 * (feas_tol=1e-8, opt_tol=1e-8, iter_limit=500, stall_cap=10).
 */
void arbsdp_solve_default_params(arbsdp_solve_params *out);

/*
 * arbsdp_result -- the minimal Layer-0 solve result (the full rigorous
 * asdp_result with [lb,ub] enclosures is b19).
 *
 * `value` is the recovered USER-FACING objective (max <C_file,X> for a maximize
 * problem; see the OBJECTIVE RECOVERY banner).  POINT MODE: it is a midpoint, not
 * a certified enclosure.  `mu` is the final complementarity measure; `iters` the
 * number of iterations taken; `status` the terminal classification.
 *
 * `it` is the FINAL (best) iterate handle so b16/b18/b19 can consume the solved
 * (X, y, S, tau, kappa) for certification / refinement.  It is owned by the
 * result (init'd by arbsdp_result_init, freed by arbsdp_result_clear) and shares
 * NO storage with the problem (which it borrows; the problem must outlive the
 * result).  The iterate's variables are the UNPURIFIED HSDE iterate (still
 * carrying tau); purify (divide by tau) at the consumption site.
 */
typedef struct {
    arbsdp_solve_status status;   /* terminal classification                    */
    arb_t               value;    /* recovered user-facing objective (POINT)    */
    arb_t               mu;       /* final complementarity measure              */
    int                 iters;    /* iterations taken                           */
    arbsdp_iterate      it;       /* final (best) HSDE iterate (owned)          */
} arbsdp_result;

/*
 * arbsdp_result_init -- initialize the result for problem `p` at precision
 * `prec` (allocates value, mu, and the embedded iterate via arbsdp_iterate_init).
 * `p` is borrowed and must outlive the result.  Pair with arbsdp_result_clear.
 * Typically the caller does NOT call this directly -- arbsdp_solve initializes
 * the result it is handed (the caller passes an UNINITIALIZED arbsdp_result).
 */
void arbsdp_result_init(arbsdp_result *res, const arbsdp_problem *p, slong prec);

/*
 * arbsdp_result_clear -- free everything arbsdp_result_init allocated (value, mu,
 * the embedded iterate) and re-zero the struct (CLAUDE.md rule 7).  Does NOT free
 * the borrowed problem.
 */
void arbsdp_result_clear(arbsdp_result *res);

/*
 * arbsdp_solve -- run the fixed-precision HSDE-NT Mehrotra IPM on problem `p` at
 * working precision `prec`, filling `res` (which the function INITIALIZES; the
 * caller passes an uninitialized arbsdp_result and must arbsdp_result_clear it
 * afterwards regardless of the return value).
 *
 * Algorithm (HsdeNtSdpSolver.ts solveHsdeSdpNt, MATH_SPEC §3):
 *   1. Starting iterate: X^b = xiP*I, S^b = xiD*I, y=0, tau=kappa=1 with
 *      xiP = max(1, 10*||b||_inf), xiD = max(1, 10*||C||_F + ||b||_inf)
 *      (MATH_SPEC §3.9, HsdeNtSdpSolver.ts:1134-1143).
 *   2. Loop to iter_limit: residuals+mu; 6-flag convergence (stop on optimal /
 *      limit / stall); per-block NT scaling W = nt_scaling(X,S); Schur assemble;
 *      factor_with_reg (carrying ONE regstate across iters, MATH_SPEC §10.4);
 *      Mehrotra predictor-corrector direction + safeguarded alpha; update
 *      (X+=a*dX, S+=a*dS, y+=a*dy, tau+=a*dtau, kappa+=a*dkappa); symmetrize X,S.
 *   3. Recover res->value (purify + flip sign per the OBJECTIVE RECOVERY banner).
 *
 * Returns the terminal status (also stored in res->status).  On a factorization
 * that fails even fully regularized, returns ARBSDP_SOLVE_NUMERICAL honestly
 * (precision escalation is b16) -- res still holds the best iterate so far and a
 * best-effort recovered value.
 *
 * POINT MODE (CLAUDE.md rule 1): the result is an approximate solution; it
 * certifies nothing.
 */
arbsdp_solve_status arbsdp_solve(arbsdp_result *res, const arbsdp_problem *p,
                                 slong prec, const arbsdp_solve_params *params);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_SOLVE_H */
