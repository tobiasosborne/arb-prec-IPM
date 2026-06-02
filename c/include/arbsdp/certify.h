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

#include "arbsdp/problem.h"

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

/* ==========================================================================
 * Layer-1 dual residual + a-priori bounds (bead arb-prec-IPM-b18)
 * ==========================================================================
 * These feed the Jansson-Chaykin-Keil 2007 lower/upper bound assembled in b19.
 * The lower bound needs (a) the dual residual Z^b = C^b - sum_i y_i A_i^b, whose
 * lambda_min lower bound (arbsdp_lambda_min_lower_bound above) supplies the
 * d_b <= lambda_min(Z^b) penalty term, and (b) the a-priori trace bounds
 * xbar_b >= tr(X*_b) (CLAUDE.md invariant 5: REQUIRED for a finite lb).
 */

/*
 * arbsdp_dual_residual -- per-block dual residual in BALL arithmetic (RIGOR):
 *
 *     Z[b] = C[b] - sum_{i=1}^{m} y[i-1] * A_i[b]      (b = 0 .. p->nblocks-1)
 *
 * where C[b] = the problem's objective matrix (matno 0, materialized verbatim via
 * arbsdp_problem_block_mat -- NOT the solver's sign-flipped C_int = -C_file; see
 * the SIGN note below) and A_i[b] = constraint matrix matno i for the same block.
 *
 * RIGOR (CLAUDE.md rule 2): the returned Z[b] is a true ball ENCLOSURE -- every
 * entry's ball provably contains the exact C[b] - sum_i y_i A_i[b].  The linear
 * combination is accumulated per entry with arb_dot (ONE rounding for the whole
 * length-m dot product), so the radius rigorously absorbs all rounding error.
 * Each Z[b] is symmetrized (arbsdp_symmetrize) because the downstream verified
 * Cholesky / lambda_min routines read only the lower triangle.
 *
 * SIGN (CLAUDE.md skepticism, rule 8; REPORTED to b19, which owns the sign
 * decision): this routine is SIGN-AGNOSTIC.  It forms Z literally from C (matno 0,
 * file sign) and the A_i (matno 1..m, file sign, exactly as arbsdp_apply_At does).
 * It does NOT choose or purify the sign of y -- the CALLER supplies y already in
 * the convention that makes Z = C - sum y_i A_i the intended dual slack.  (The
 * Layer-0 solver works in min -<C,X> with C_int = -C_file and forms sum_i y_i A_i
 * via apply_At; mapping its y to the y this routine expects is the caller's job.)
 *
 * `Z` is an array of p->nblocks PRE-INITIALIZED arb_mat (caller owns init/clear),
 * each of side |p->block_sizes[b]|.  `y` is an arb vector of length p->m (the
 * approximate dual lifted to balls).  prec is the certification precision (the
 * caller passes prec_c = prec + 128; MATH_SPEC §10, decision 2).  All arguments
 * non-NULL and dimensions asserted (CLAUDE.md rule 5).
 */
void arbsdp_dual_residual(arb_mat_t *Z, const arbsdp_problem *p, arb_srcptr y,
                          slong prec);

/*
 * arbsdp_apriori -- a-priori bounds the certifier needs but cannot derive from
 * the problem data alone (CLAUDE.md invariant 5: NO silent boundedness).
 *
 *   xbar[b]  -- an a-priori TRACE bound  xbar[b] >= tr(X*_b)  on the b-th block of
 *               an optimal primal X*.  Convention: a TRACE bound (sum of the
 *               block's diagonal), with NO block-dimension factor (MATH_SPEC
 *               §5.3.1, bead arb-prec-IPM-om9).  REQUIRED for a finite Jansson
 *               lower bound: an UNSET xbar[b] means tr(X_b) is NOT a-priori
 *               bounded, so b19 reports lb = -inf for that block honestly --
 *               it NEVER silently assumes boundedness.
 *   ybar     -- an a-priori bound on the dual (norm).  RE-SCOPED (b18 derivation,
 *               MATH_SPEC §5.5): the FINITE upper bound (UB-A) is the dual-side
 *               MIRROR of the lower bound and uses xbar, NOT ybar.  ybar is the
 *               input for the SECONDARY residual upper bound (UB-B, Jansson 2009
 *               Thm 4.2), which needs a primal-PSD projection arbsdp does not yet
 *               have -- so ybar is currently unused by b19.  Tracked for UB-B.
 *
 * The set/unset distinction is carried by the *_set FLAGS, not by a sentinel
 * value (CLAUDE.md invariant 5 + rule 8): the flag is the single source of truth
 * for "is this bound supplied", so an explicitly-set huge xbar[b] is still a
 * FINITE bound, distinct from "unset" (= +inf).  This struct only STORES the
 * bounds + flags; b19 consumes them.
 *
 * Memory discipline (CLAUDE.md rule 7): arbsdp_apriori_init allocates the arrays
 * and the arb_t ybar; arbsdp_apriori_clear frees them.  The setters copy `value`
 * (arb_set), so the caller retains ownership of its own arb_t.
 */
typedef struct {
    int      nblocks;
    arb_ptr  xbar;       /* length nblocks: a-priori bound xbar[b] >= tr(X*_b)    */
    int     *xbar_set;   /* length nblocks: 1 iff a finite xbar[b] was supplied   */
    arb_t    ybar;       /* dual-norm bound; UB-B fallback only (NOT UB-A); see hdr */
    int      ybar_set;   /* 1 iff ybar was supplied                              */
} arbsdp_apriori;

/* arbsdp_apriori_init -- allocate for nblocks blocks; all bounds start UNSET
 * (xbar_set[b] = 0 for all b, ybar_set = 0).  nblocks >= 0 (asserted). */
void arbsdp_apriori_init(arbsdp_apriori *ab, int nblocks);

/* arbsdp_apriori_clear -- free all heap owned by the struct and re-zero it.
 * Idempotent on an init'd struct (CLAUDE.md rule 7). */
void arbsdp_apriori_clear(arbsdp_apriori *ab);

/* arbsdp_apriori_set_xbar -- set xbar[b] = value and xbar_set[b] = 1 (copies
 * value).  b in 0..nblocks-1 (asserted). */
void arbsdp_apriori_set_xbar(arbsdp_apriori *ab, int b, const arb_t value);

/* arbsdp_apriori_set_ybar -- set ybar = value and ybar_set = 1 (copies value). */
void arbsdp_apriori_set_ybar(arbsdp_apriori *ab, const arb_t value);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_CERTIFY_H */
