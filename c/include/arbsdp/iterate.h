/*
 * include/arbsdp/iterate.h -- the HSDE iterate state and the residual / objective
 * machinery the whole Layer-0 solver (b12-b16) builds on.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * The arbsdp_iterate carries the (X, y, S, tau, kappa) point of the homogeneous
 * self-dual embedding (HSDE) through the interior-point loop.  Everything here is
 * Layer 0: arb arithmetic at a fixed working `prec`, consumed as MIDPOINTS; ball
 * radii are NOT trusted (Layer 1 derives all rigor independently).  Convergence
 * flags below are float64-comparable scaled residuals exactly as in the TS
 * reference -- they steer the loop, they do not certify anything.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §1 (standard form; internal min -<C,X>; sign convention),
 *     §3.1 (HSDE system + mu denominator n+1), §3.2 (the three residuals),
 *     §3.8 (6-flag convergence with ||b||,||c|| scaling), §3.9 (initial point).
 *   - HsdeIterate.ts (the iterate contract + mu denominator), Residuals.ts
 *     (r_p/r_d/r_g signs + objectives), Convergence.ts (the 6-flag tree),
 *     HsdeNtSdpSolver.ts:371-407 (residual assembly), :1043-1132
 *     (checkHsdeTermination -- the scaled rho_p/rho_d/rho_g + prstatus + status),
 *     Defaults.ts:34-49 (feasTol/optTol/iterLimit/stallIterCap).
 *   - CLAUDE.md invariant 7 (internal min -<C,X>), invariant 8 (HSDE tau,kappa).
 *
 * ==========================================================================
 * SIGN CONVENTION (CLAUDE.md invariant 7 -- get this wrong and every objective
 * is negated)
 * ==========================================================================
 * libarbsdp's problem.h stores C VERBATIM from the .dat-s file (the user-facing
 * cost), unlike the TS SdpProblem which negates C at parse time for its internal
 * `min -<C,X>`.  The internal loop ALWAYS MINIMIZES <C_int,X>; to make that
 * minimization express the user's objective, this module forms the internal cost
 * block once at init with the orientation that depends on the sense (MATH_SPEC
 * §1.2, invariant 7, bead arb-prec-IPM-n1x):
 *
 *     C_int^b = -C_file^b   when MAXIMIZE (the SDPA default): min<-C_file,X> = max
 *     C_int^b = +C_file^b   when MINIMIZE (bead n1x):         min<+C_file,X> = min
 *
 * recover_value (solve.c) applies the MATCHING output sign (maximize ? -pObj/tau
 * : +pObj/tau), so the two together return the correct user value in both modes.
 * (PRE-n1x C_int was negated UNCONDITIONALLY, so a minimize problem was solved as
 * a maximize and recover_value returned -(max<C,X>) instead of min<C,X>.)  Then,
 * byte-for-byte with HsdeNtSdpSolver.ts:371-407 (pObj = sum_b <C_int^b, X^b>):
 *
 *     r_p_i  =  sum_b <A_i^b, X^b>  -  b_i * tau               (primal, m-vector)
 *     r_d^b  =  sum_i y_i A_i^b  +  S^b  -  C_int^b * tau      (dual, per block)
 *     r_g    =  -sum_b <C_int^b, X^b>  +  b^T y  -  kappa      (gap, scalar)
 *     pObj   =  sum_b <C_int^b, X^b>                           (internal min obj)
 *     dObj   =  b^T y
 *     mu     =  (sum_b <X^b, S^b>  +  tau*kappa) / (totalConeDim + 1)
 *
 * These formulas are identical in both senses; only the orientation of C_int (set
 * once at init per the table above) differs.  The user-facing optimum is recovered
 * downstream (solve.c recover_value) as -(pObj/tau) when maximize, +(pObj/tau)
 * when minimize -- the matching output sign for C_int's orientation (NOT this
 * module -- here we only carry the internal-form objectives).
 *
 * ==========================================================================
 * BLOCK LAYOUT
 * ==========================================================================
 * X, S, and the per-block dual residual R_d are arrays of `nblocks` arb_mat, one
 * per block, each of side |block_sizes[b]| -- exactly the side returned by
 * arbsdp_problem_block_mat.  A diagonal (LP) block (block_sizes[b] < 0) is stored
 * as a full |block_sizes[b]| x |block_sizes[b]| matrix whose off-diagonals are
 * zero (the materializer writes only i==j entries for such blocks); we keep the
 * dense n x n storage uniformly so the same arb_mat code path serves PSD and LP
 * blocks.  y is a length-m arb vector (arb_ptr).  tau, kappa are global scalars.
 *
 * Arb memory discipline (CLAUDE.md rule 7): arbsdp_iterate_init allocates X, S,
 * R_d, y and all scalars; arbsdp_iterate_clear frees every one.  The iterate
 * borrows a pointer to its arbsdp_problem (NOT owned) -- the caller keeps the
 * problem alive for the iterate's lifetime.
 */

#ifndef ARBSDP_ITERATE_H
#define ARBSDP_ITERATE_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_iterate -- the HSDE iterate (X, y, S, tau, kappa) plus the cost cache,
 * the residual storage, and the scalar diagnostics the convergence test reads.
 */
typedef struct {
    const arbsdp_problem *prob;   /* borrowed; not owned                       */
    int        m;                 /* number of constraints                     */
    int        nblocks;           /* number of blocks                          */
    slong     *block_n;           /* length nblocks: side |block_sizes[b]|     */
    slong      total_cone_dim;    /* sum_b block_n[b] (the mu denom is +1)     */

    /* --- iterate variables (POINT MODE midpoints) --------------------------- */
    arb_mat_t *X;                 /* length nblocks; X[b] is n_b x n_b symmetric*/
    arb_mat_t *S;                 /* length nblocks; S[b] is n_b x n_b symmetric*/
    arb_ptr    y;                 /* length m                                  */
    arb_t      tau;               /* primal homogenization scalar (> 0)        */
    arb_t      kappa;             /* dual homogenization scalar (> 0)          */

    /* --- cached internal cost C_int^b (= -C_file^b if maximize, else +C_file^b;
     *     sign convention above, bead n1x) ----------------------------------- */
    arb_mat_t *C_int;             /* length nblocks                            */

    /* --- residual storage (filled by arbsdp_iterate_residuals) ------------- */
    arb_ptr    r_p;               /* length m: primal residual                 */
    arb_mat_t *R_d;               /* length nblocks: per-block dual residual   */
    arb_t      r_g;               /* gap residual (scalar)                     */

    /* --- scalar diagnostics (filled by arbsdp_iterate_residuals) ----------- */
    arb_t      mu;                /* (sum<X,S> + tau*kappa)/(total_cone_dim+1) */
    arb_t      pObj;             /* sum_b <C_int^b, X^b> (internal min obj)    */
    arb_t      dObj;             /* b^T y                                       */
    arb_t      primal_inf;        /* ||r_p||_inf                               */
    arb_t      dual_inf;          /* max_b ||R_d^b||_inf                        */
    arb_t      gap_inf;           /* |r_g|                                      */
} arbsdp_iterate;

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_iterate_init -- allocate the iterate for problem `p` and cache the
 * internal cost blocks C_int^b at precision `prec` (= -C_file^b when p->maximize,
 * = +C_file^b when minimize; sign convention banner, bead n1x).  All iterate
 * variables (X, S, y, tau, kappa) are zeroed; tau, kappa are set to 1 (the HSDE
 * default, MATH_SPEC §3.9).  `p` is borrowed (not owned) and must outlive the
 * iterate.  Pair with arbsdp_iterate_clear.
 */
void arbsdp_iterate_init(arbsdp_iterate *it, const arbsdp_problem *p, slong prec);

/*
 * arbsdp_iterate_clear -- free everything arbsdp_iterate_init allocated and
 * re-zero the struct (CLAUDE.md rule 7).  Does NOT free the borrowed problem.
 */
void arbsdp_iterate_clear(arbsdp_iterate *it);

/* -------------------------------------------------------------------------
 * Constraint operator (residuals need A and its adjoint A^T)
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_apply_A -- out_i = <A_i, X> = sum_b frobInner(A_i^b, X^b), i = 1..m.
 *
 * `out` must point to at least p->m initialized arb elements.  `X_blocks` is an
 * array of p->nblocks arb_mat, X_blocks[b] of side |block_sizes[b]| (asserted).
 * A_i^b is materialized on demand via arbsdp_problem_block_mat (matno = i+1).
 * Frobenius inner product is via arbsdp_frob_inner (the "2v" off-diagonal
 * doubling lives there; see problem.h / svec.h).
 */
void arbsdp_apply_A(arb_ptr out, const arbsdp_problem *p,
                    const arb_mat_t *X_blocks, slong prec);

/*
 * arbsdp_apply_At -- (A^T y)^b = sum_i y_i A_i^b, for each block b.
 *
 * `out_blocks` is an array of p->nblocks PRE-INITIALIZED arb_mat, out_blocks[b]
 * of side |block_sizes[b]| (asserted); each is zeroed then accumulated.  `y`
 * points to p->m arb elements.  A_i^b is materialized via
 * arbsdp_problem_block_mat (matno = i+1).
 *
 * Adjoint identity (the b11 test): y^T A(X) == sum_b <(A^T y)^b, X^b> for any y
 * and any symmetric X.
 */
void arbsdp_apply_At(arb_mat_t *out_blocks, const arbsdp_problem *p,
                     arb_srcptr y, slong prec);

/* -------------------------------------------------------------------------
 * Residuals + objectives + mu (the 3-row HSDE residuals)
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_iterate_residuals -- from the current (X, y, S, tau, kappa) fill r_p,
 * R_d, r_g, mu, pObj, dObj, primal_inf, dual_inf, gap_inf at precision `prec`.
 *
 * Formulas (HsdeNtSdpSolver.ts:371-407, MATH_SPEC §3.2, signs per the banner):
 *     r_p_i = sum_b <A_i^b, X^b> - b_i*tau
 *     R_d^b = sum_i y_i A_i^b + S^b - C_int^b*tau
 *     r_g   = -pObj + dObj - kappa     (pObj = sum_b<C_int^b,X^b>, dObj = b^T y)
 *     mu    = (sum_b <X^b,S^b> + tau*kappa) / (total_cone_dim + 1)
 *
 * POINT MODE: midpoints; radii not trusted.
 */
void arbsdp_iterate_residuals(arbsdp_iterate *it, slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_ITERATE_H */
