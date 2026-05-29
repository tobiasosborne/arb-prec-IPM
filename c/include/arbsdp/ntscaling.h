/*
 * include/arbsdp/ntscaling.h -- Layer-0 (POINT MODE) Nesterov-Todd scaling for
 * the PSD cone.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * arbsdp_nt_scaling is used INSIDE the HSDE-NT IPM loop (Layer 0), which runs in
 * Arb *point* arithmetic at adaptive precision.  It is built on the point-mode
 * spectral kernels in linalg.h (cyclic-Jacobi eig / psd_sqrt / psd_invsqrt), so
 * its output is consumed as a POINT (midpoint); the ball *radii* are NOT trusted.
 * Layer 1 derives ALL rigor independently (verified Cholesky), never from W.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §3.3: the NT scaling W^b per block satisfies
 *         W^b S^b W^b = X^b
 *     (Todd-Toh-Tutuncu 1998 "On the Nesterov-Todd direction in SDP").
 *   - PsdCone.ts ntScaling (lines 169-183) -- the port target:
 *         W = X^{1/2} (X^{1/2} S X^{1/2})^{-1/2} X^{1/2},
 *     with the intermediate `inner = X^{1/2} S X^{1/2}` symmetrized before the
 *     inverse square root and the final W symmetrized at the end.
 *
 * This symmetric-factor form is one of several equivalent NT constructions; it is
 * the one the TS golden master uses, so libarbsdp ports it exactly to keep the
 * golden-master comparison meaningful (CLAUDE.md rule 4).  (MATH_SPEC §3.3 also
 * records an alternative Cholesky/eig "buildNtFactor" form used by the HSDE
 * solver for the G-factor; that is a separate routine, out of scope here.)
 *
 * VERIFICATION of W S W = X for the symmetric form:
 *     Let H = X^{1/2}, M = (H S H)^{-1/2}.  Then W = H M H, and (with H, M
 *     symmetric)
 *         W S W = H M H S H M H = H M (H S H) M H = H M M^{-2} M H = H H = X,
 *     using M (H S H) M = M M^{-2} M = I.
 *
 * Arb memory discipline (CLAUDE.md rule 7): the caller owns init/clear of W, X,
 * S; this routine allocates only scoped temporaries that it clears itself.
 * Square / matching-size / no-alias preconditions are asserted (CLAUDE.md rule 5,
 * fail fast).  PD of X and S is assumed by the caller (the IPM keeps the iterate
 * strictly interior); a non-PD input is clamped by the underlying psd_sqrt /
 * psd_invsqrt floors rather than detected here.
 */

#ifndef ARBSDP_NTSCALING_H
#define ARBSDP_NTSCALING_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_nt_scaling -- W <- X^{1/2} (X^{1/2} S X^{1/2})^{-1/2} X^{1/2} (POINT MODE).
 *
 * Computes the symmetric NT scaling satisfying  W S W = X  for symmetric PD X, S.
 * The intermediate inner = X^{1/2} S X^{1/2} and the result W are symmetrized.
 *
 * `W` must be an initialized n x n arb_mat; X and S must be square n x n
 * (asserted square, same size).  W must NOT alias X or S.
 *
 * POINT MODE: the result is a midpoint; radii are not trusted (see banner).
 */
void arbsdp_nt_scaling(arb_mat_t W, const arb_mat_t X, const arb_mat_t S,
                       slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_NTSCALING_H */
