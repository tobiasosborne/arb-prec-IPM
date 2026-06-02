/*
 * include/arbsdp/certify.h -- Layer-1 (RIGOROUS, BALL-arithmetic) certification
 * primitives: verified positive-definiteness, a Gershgorin lambda_min floor, and
 * the verified-Cholesky-shift lambda_min lower bound.
 *
 * ==========================================================================
 * BALL MODE -- THE ONLY SOURCE OF RIGOR (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * Unlike linalg.c (POINT mode, Layer 0), every routine here operates in Arb
 * *ball* arithmetic and the result is a THEOREM about the true matrix enclosed
 * by the input balls.  These are the first Layer-1 kernels.  They never call the
 * point-mode spectral routines (arbsdp_eigh / psd_sqrt) -- rigorous symmetric
 * eigendecomposition is Arb's weak spot (linalg.h AUDITION banner), so all rigor
 * derives from verified Cholesky (arb_mat_cho), Arb's strength.
 *
 * AUDITION (CLAUDE.md rule 3): the lambda_min lower-bound method was auditioned
 * in docs/MATH_SPEC.md §5.3.1 (bead arb-prec-IPM-b27).  Verified-Cholesky shift
 * with a Gershgorin seed and bisection on the monotone predicate
 *     g(s) = "arb_mat_cho(A - s*I) succeeds"
 * WON over rigorous symmetric eig (arb's weak axis; vacuous near the boundary),
 * Weyl/Rump perturbation (needs an approximate eig + interval matmul; ~10-30x a
 * Cholesky), single-shot Gershgorin (kept as the bisection SEED only), and
 * Newton/secant on the shift (degenerate: d(lambda_min)/ds = -1).  See §5.3.1 for
 * the comparison table and the loser-by-loser rationale.
 *
 * Memory discipline (CLAUDE.md rule 7): callers own init/clear of every argument
 * (the arb_mat inputs and the arb / arb_mat outputs); these functions allocate
 * only scoped temporaries that they clear themselves.  Square-matrix
 * preconditions are asserted (CLAUDE.md rule 5).
 *
 * References (cited again at the implementation sites in certify.c):
 *   - Rump, BIT 2006, 46:433-452, Thm 2.3 / Cor 2.4: a completed (ball/interval)
 *     Cholesky of A certifies A positive definite.
 *   - Gershgorin 1931 (disk theorem): lambda_min(A) >= min_i (A_ii - sum_{j!=i}|A_ij|).
 *   - Higham-Strabic-Sego, SIAM Review 2016, 58(2):245-263: bisection definiteness
 *     oracle (pure bisection on the shift, not Newton).
 *   - Jansson-Chaykin-Keil 2007, SIAM J. Numer. Anal. 46(1):180-200: consumer of
 *     d_b <= lambda_min(Z^b) in the Layer-1 lower bound (b18/b19).
 *   - FLINT 3.0.1 arb_mat.h:378: arb_mat_cho returns 1 = CERTAINLY PD over the
 *     input ball enclosures.
 */

#ifndef ARBSDP_CERTIFY_H
#define ARBSDP_CERTIFY_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_verified_psd -- return 1 IFF A is CERTIFIED positive definite (BALL).
 *
 * Mechanism: symmetrize a scratch copy of A (arb_mat_cho reads only the lower
 * triangle), then attempt arb_mat_cho into a scratch L.  By the FLINT 3.0.1
 * contract (arb_mat.h:378) and Rump BIT 2006 Cor 2.4, a return of 1 is a PROOF
 * that A -- over its input ball enclosures -- is positive definite.  A return of
 * 0 means NOT-provably-PD: A is indefinite/semidefinite (e.g. singular, with
 * lambda_min = 0) OR the precision was insufficient to complete the factorization.
 * A is square (asserted).  prec is the Cholesky working precision in bits.
 */
int arbsdp_verified_psd(const arb_mat_t A, slong prec);

/*
 * arbsdp_gershgorin_lower_bound -- ALWAYS-VALID rigorous lower bound on
 * lambda_min(A) via the Gershgorin disk theorem (1931):
 *
 *     out <= lambda_min(A),   out = min_i ( lbound(A_ii) - sum_{j!=i} ubound(|A_ij|) ).
 *
 * Diagonal entries are taken at their LOWER endpoint and off-diagonal magnitudes
 * at their UPPER endpoint (arb_get_lbound_arf / arb_get_ubound_arf), and the
 * arithmetic rounds conservatively (down), so `out` is a guaranteed lower bound
 * for the true matrix enclosed by the balls.  Valid for ANY symmetric A (no PD
 * assumption); used as the SEED + safe return floor for the bisection below.
 * A is square (asserted); `out` must be initialized.
 */
void arbsdp_gershgorin_lower_bound(arb_t out, const arb_mat_t A, slong prec);

/*
 * arbsdp_lambda_min_lower_bound -- rigorous lower bound d on lambda_min(A) via
 * the verified-Cholesky shift + bisection (MATH_SPEC §5.3 / §5.3.1).
 *
 * GUARANTEE (THEOREM, CLAUDE.md rule 2): d <= lambda_min(A).  The guarantee is on
 * d's VALUE -- d is returned as the last shift s with a SUCCESSFUL ball Cholesky
 * of (A - s*I), and the shift is subtracted at its UPPER endpoint when forming
 * (A - s*I), so a success proves A - upper(s)*I >= 0, i.e. lambda_min(A) >= s.
 * A returned d may be <= 0 for indefinite / rank-deficient A (the SDP-boundary
 * case): that is a FIRST-CLASS valid output (it feeds the min(0,d) penalty in the
 * Jansson lower bound), NOT an error.  d NEVER exceeds lambda_min(A).
 *
 * A is square (asserted); `d` must be initialized.  prec is the working precision.
 */
void arbsdp_lambda_min_lower_bound(arb_t d, const arb_mat_t A, slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_CERTIFY_H */
