/*
 * include/arbsdp/schur.h -- Layer-0 (POINT MODE) Nesterov-Todd-scaled Schur
 * complement assembly and its factor-once/solve-many interface for the Mehrotra
 * predictor-corrector step.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * The Schur matrix M is built inside the HSDE-NT IPM loop (Layer 0) from the
 * POINT-mode NT scaling W (arbsdp_nt_scaling, ntscaling.h).  M and its factor are
 * consumed as MIDPOINTS to compute search directions; ball *radii* are NOT
 * trusted.  Layer 1 derives ALL rigor independently (verified Cholesky of the
 * dual residual, certify.c), never from M.  arb_mat_cho / arb_mat_solve_cho_precomp
 * are themselves rigorous ball routines, but here cho's return is used only as a
 * fast PD check + factor source for the (point-mode) direction solve.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §3.4: the NT-scaled Schur complement
 *         M_ik = sum_b <A_i^b, W^b A_k^b W^b>,
 *     symmetric and PSD in exact arithmetic (a Gram matrix; §2.4).
 *   - HsdeNtSdpSolver.ts:511-558: cache WAW[b][k] = W^b A_k^b W^b per block, then
 *     M_ik += frobInner(A_i^b, WAW[b][k]); finally symmetrise by averaging.
 *   - docs/FLINT_NOTES.md: arb_mat_cho returns 1 iff A proven PD;
 *     arb_mat_solve_cho_precomp reuses ONE factor for many RHS (factor-once /
 *     solve-many -- the predictor and corrector share a single Schur factor).
 *
 * ==========================================================================
 * AUDITION (CLAUDE.md rule 3) -- symmetric W vs the HSDE G-factor
 * ==========================================================================
 * Two NT representations are available for the Schur input (b12 bead note):
 *   (1) the SYMMETRIC W from arbsdp_nt_scaling (b06), W = X^{1/2}(X^{1/2} S
 *       X^{1/2})^{-1/2} X^{1/2}, satisfying W S W = X; M_ik = <A_i, W A_k W>;
 *   (2) the TS HSDE G-factor (buildNtFactor: W = G^T G), with the Gram-form route
 *       M_ik = <G A_i G^T, G A_k G^T>.
 * KEY FACT: W = G^T G is the UNIQUE symmetric PD square root solving W S W = X, so
 * BOTH representations yield the SAME W and hence the SAME M.  The G-factor's
 * distinct role in the TS solver is the DIRECTION RECOVERY (hdZ = G dS G^T;
 * HsdeNtSdpSolver.ts:703-758), NOT a differently-conditioned Schur assembly.
 *
 * Measured (docs/probes/schur_nt_audition.c, prec=256, n=m=3 fixture):
 *     max|W - G^T G|           = 1.21e-76   (same symmetric W)
 *     max|M_routeA - M_routeB| = 1.10e-75   (same Schur matrix)
 *     kappa(M) route A         = 8.246e+00
 *     kappa(M) route B         = 8.246e+00  (identical conditioning)
 *
 * DECISION: implement the SYMMETRIC-W path (route 1).  The G-factor gives ZERO
 * conditioning benefit for the Schur assembly (it produces the same M to ~prec
 * accuracy), and we already have the symmetric W from b06 -- threading a second
 * NT representation through Layer 0 only for Schur would be duplication with no
 * payoff.  The G-factor remains the right tool for direction recovery in a later
 * bead, where it is genuinely needed (the diag(+-sv) identities).  Decision and
 * numbers recorded; not a silent default (CLAUDE.md rule 3).
 *
 * Arb memory discipline (CLAUDE.md rule 7): callers own init/clear of every
 * argument (M, L, W_blocks, rhs, x).  These functions allocate only scoped
 * temporaries they clear themselves.  Dimension preconditions are asserted
 * (CLAUDE.md rule 5, fail fast).
 */

#ifndef ARBSDP_SCHUR_H
#define ARBSDP_SCHUR_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_schur_assemble -- build the m x m symmetric Schur matrix
 *     M_ik = sum_b <A_i^b, W^b A_k^b W^b>          (MATH_SPEC §3.4)
 * at precision `prec` into the pre-initialized arb_mat `M`.
 *
 * `M` must be an initialized m x m arb_mat (m = p->m; asserted square m x m).
 * `W_blocks` is an array of p->nblocks arb_mat: W_blocks[b] is the per-block NT
 * scaling (from arbsdp_nt_scaling), of side |block_sizes[b]| (asserted), POINT
 * mode (its radii are not trusted).  A_i^b is materialized on demand via
 * arbsdp_problem_block_mat (matno = i+1); the Frobenius inner product is
 * arbsdp_frob_inner (the off-diagonal "2v" doubling lives there; see svec.h).
 *
 * Per-block inner-loop structure (symmetry exploited): for each block b, the
 * cache WAW_k = sym(W^b A_k^b W^b) is formed once per k (two mat-muls), then only
 * the UPPER triangle i <= k of M is accumulated via <A_i^b, WAW_k> and the result
 * mirrored to the lower triangle at the end (M is symmetric in exact arithmetic;
 * mirroring also removes the point-mode asymmetry, matching the TS averaging).
 * This halves the frobInner count versus the TS full-square loop.
 *
 * POINT MODE: M is a midpoint; radii are not trusted (see header banner).
 */
void arbsdp_schur_assemble(arb_mat_t M, const arbsdp_problem *p,
                           const arb_mat_t *W_blocks, slong prec);

/*
 * arbsdp_schur_factor -- L <- Cholesky factor of M, via arb_mat_cho.
 *
 * Returns 1 iff M is PROVEN positive-definite over the input balls (and fills the
 * lower-triangular factor L with L L^T enclosing M); returns 0 otherwise.  No
 * regularization is applied here -- a 0 return is reported HONESTLY (CLAUDE.md
 * rule 5, fail loud).  3-way Tikhonov regularization-on-failure is a separate bead
 * (b13); this routine never silently fudges a non-PD M.
 *
 * `L` and `M` must be initialized m x m arb_mat (asserted square, same side).
 * The returned L is the factor-once half of the factor-once/solve-many pattern:
 * pass it to arbsdp_schur_solve for each RHS (predictor + corrector).
 */
int arbsdp_schur_factor(arb_mat_t L, const arb_mat_t M, slong prec);

/*
 * arbsdp_schur_solve -- x <- M^{-1} rhs using the precomputed factor L, via
 * arb_mat_solve_cho_precomp (FLINT_NOTES.md: reuse ONE cho factor for many RHS).
 *
 * `L` is the factor produced by a SUCCESSFUL arbsdp_schur_factor (return 1).
 * `rhs` and `x` are m x ncols arb_mat (typically ncols = 1; asserted compatible:
 * x and rhs have the same shape, L is m x m, rhs has m rows).  Reusing L across
 * the Mehrotra predictor and corrector solves is the whole point -- factor once
 * (arbsdp_schur_factor), solve many (this routine), per IPM iteration.
 *
 * POINT MODE: x is a midpoint; radii are not trusted.
 */
void arbsdp_schur_solve(arb_mat_t x, const arb_mat_t L, const arb_mat_t rhs,
                        slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_SCHUR_H */
