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
#include "arbsdp/iterate.h"

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

/* ==========================================================================
 * The rigorous two-sided bracket [lb, ub] (bead arb-prec-IPM-b19)
 * ==========================================================================
 * THIS IS THE PROJECT'S PRODUCT (CLAUDE.md "What this is"): the first rigorous
 * interval certificate that PROVABLY CONTAINS the SDP optimum
 *     p*_ext = max <C_file, X>  s.t.  <A_i,X> = b_i,  X >= 0.
 * Everything below runs in Arb *ball* arithmetic at the certification precision
 * prec_c and produces THEOREMS (CLAUDE.md rule 1, invariant 2; the only rigor
 * source is the verified-Cholesky-shift lambda bound, never eigendecomposition).
 *
 * MATH_SPEC §5.4 (lower bound, Jansson-Chaykin-Keil 2007 Thm 3.2) and §5.5
 * (upper bound UB-A, its dual-side trace-mirror).  Both consume the SAME external
 * dual y_ext = -y_int (y_int = it->y / it->tau) and the SAME file-sign data
 * (C_file, A_i) via arbsdp_dual_residual; passing y_int (the WRONG sign) forms
 * Z_int = -Z_ext, swaps lambda_min<->lambda_max, and yields a bracket that
 * EXCLUDES the optimum (P0; §5.5 "Sign contract").
 */

/*
 * arbsdp_lambda_max_upper_bound -- rigorous UPPER bound dhi on lambda_max(A) via
 * lambda_max(A) = -lambda_min(-A) (MATH_SPEC §5.5):
 *
 *     dhi = -arbsdp_lambda_min_lower_bound(-A) >= lambda_max(A).
 *
 * GUARANTEE (THEOREM, CLAUDE.md rule 2): dhi >= lambda_max(A).  A successful ball
 * Cholesky of (-A - s*I) proves lambda_min(-A) >= s, i.e. lambda_max(A) <= -s;
 * the b17 kernel returns the largest such s as d_lo(-A), so dhi = -d_lo(-A) is a
 * rigorous upper bound (Rump 2006 Cor 2.4; FLINT arb_mat_cho contract).  A is
 * square (asserted); `out` must be initialized.  prec is the working precision.
 */
void arbsdp_lambda_max_upper_bound(arb_t out, const arb_mat_t A, slong prec);

/*
 * arbsdp_lower_bound -- the Jansson lower bound (MATH_SPEC §5.4) for the EXTERNAL
 * max problem, assembled in ball arithmetic:
 *
 *     lb = b^T y_ext + sum_b min(0, dlo_b) * xbar_b,
 *
 * with dlo_b = arbsdp_lambda_min_lower_bound(Z_ext^b) <= lambda_min(Z_ext^b),
 * Z_ext^b = arbsdp_dual_residual(., p, y_ext, prec) = C_file^b - sum_i (y_ext)_i A_i^b.
 *
 * HONEST -inf (CLAUDE.md invariant 5, rule 2): if for some block dlo_b < 0 AND
 * xbar_set[b] == 0 (no a-priori trace bound supplied), lb = -inf (arb set to
 * negative infinity) -- NEVER a fabricated finite bound.
 *
 * `lb` is the assembled ball (the caller takes its LOWER endpoint for the
 * rigorous scalar; arbsdp_certify_bracket does this).  `y_ext` is the external
 * dual (length p->m), ALREADY tau-purified and negated by the caller -- this
 * routine is the §5.4 ASSEMBLY and does not re-derive the sign.  `ab` supplies
 * the xbar trace bounds + set flags.  All arguments non-NULL; ab->nblocks must
 * equal p->nblocks (asserted).  prec is the certification precision prec_c.
 */
void arbsdp_lower_bound(arb_t lb, const arbsdp_problem *p, arb_srcptr y_ext,
                        const arbsdp_apriori *ab, slong prec);

/*
 * arbsdp_upper_bound -- the dual-side UPPER bound (MATH_SPEC §5.5, UB-A) for the
 * EXTERNAL max problem, assembled in ball arithmetic:
 *
 *     ub = b^T y_ext + sum_b max(0, dhi_b) * xbar_b,
 *
 * with dhi_b = arbsdp_lambda_max_upper_bound(Z_ext^b) >= lambda_max(Z_ext^b),
 * Z_ext^b as in arbsdp_lower_bound (SAME y_ext, SAME file-sign data).
 *
 * HONEST +inf (CLAUDE.md invariant 5, rule 2): if for some block dhi_b > 0 AND
 * xbar_set[b] == 0, ub = +inf -- NEVER a fabricated finite bound.  The clamp
 * max(0, dhi_b) (mirroring min(0, dlo_b) for the lower bound) is the safe
 * over-estimate of <Z_b, X_b> when Z_b is negative definite.
 *
 * `ub` is the assembled ball (the caller takes its UPPER endpoint).  Arguments,
 * sign contract, and asserts as arbsdp_lower_bound.
 */
void arbsdp_upper_bound(arb_t ub, const arbsdp_problem *p, arb_srcptr y_ext,
                        const arbsdp_apriori *ab, slong prec);

/*
 * arbsdp_primal_lower_bound -- a rigorous PRIMAL lower bound on the EXTERNAL max
 * optimum via the Rayleigh quotient of a FEASIBLE block-diagonal rank-1 primal
 * point, for the FULLY-SEPARABLE per-block-trace family (bead arb-prec-IPM-oc7;
 * generalizes the single-block vry case to nblocks >= 1).  This is the PRIMAL
 * MIRROR of arbsdp_lower_bound (the DUAL-residual Jansson bound): a feasible
 * PRIMAL point gives a lower bound; a feasible DUAL point gives an upper bound --
 * here we exhibit a feasible primal.
 *
 * METHOD.  For the fully-separable per-block-trace external problem -- m blocks,
 * each constraint A_i exactly the identity on exactly ONE block b_i and zero
 * elsewhere (so block b has the single constraint tr(X_b) = beta_b), with beta_b
 * >= 0 -- the optimum is sum_b beta_b * lambda_max(C_b).  For ANY block vectors
 * v_b != 0, the block direct-sum
 *     X_hat = (+)_b  beta_b * v_b v_b^T / (v_b^T v_b)
 * is EXACTLY feasible: each block is PSD (a rank-1 outer product scaled by
 * beta_b >= 0 -- PSD by construction, no Cholesky), tr(X_hat_b) = beta_b, and no
 * constraint couples two blocks.  Its objective
 *     <C, X_hat> = sum_b beta_b * (v_b^T C_b v_b)/(v_b^T v_b) = sum_b beta_b R(v_b)
 * with R(v_b) <= lambda_max(C_b) for EVERY v_b (Courant-Fischer).  Hence
 *     lb := sum_b beta_b * (v_b^T C_b v_b)/(v_b^T v_b)
 * (at its lower endpoint) is a rigorous LOWER bound on the optimum, for ANY v_b.
 * It is TIGHT when each v_b approximates the top eigenvector of C_b.  The
 * single-block single-trace vry case is exactly nblocks == 1 (beta_0 = b_1).
 *
 * RIGOR (CLAUDE.md rule 2, invariant 2): each per-block quotient is evaluated in
 * Arb *ball* arithmetic and the blocks are summed in ball arithmetic, so lb is a
 * true enclosure and lb's lower endpoint <= opt REGARDLESS of solve quality -- the
 * bound holds for ANY fixed v_b.  CRUCIALLY this needs NO eigenvalue rigor and NO
 * PSD certification of X: the rank-1 blocks are PSD by construction, which is
 * exactly why it works at the PSD boundary where a verified Cholesky of X FAILS
 * (project invariant 1).  Each v_b is chosen POINT-MODE / approximately
 * (arbsdp_eigh's top column of the solver's X_b, frozen to an EXACT midpoint
 * vector); it is used ONLY as the primal direction and is NOT a rigor dependency.
 * A bad v_b merely loosens lb; it can never make it unsound.
 *
 * APPLICABILITY (CONSERVATIVE -- a wrong ACCEPT is a P0, so it prefers rejecting):
 * returns 1 and sets lb_out only when the block-diagonal rank-1 trace construction
 * is provably valid -- m == nblocks (== it->nblocks == it->m); each constraint A_i
 * (matno i) materializes EXACTLY to the identity on EXACTLY one block (every
 * diagonal entry arb_is_one, every off-diagonal arb_is_zero) and EXACTLY zero on
 * every other block; the map i -> b_i is a bijection onto {0..nblocks-1} (each
 * block pinned by exactly one constraint); and each beta_b = b_{i_b-1} has lower
 * endpoint >= 0.  Otherwise returns 0 with lb_out untouched, so the caller falls
 * back to the rigorous dual bound (safe).  Coupled / multi-constraint-per-block /
 * higher-rank feasible points are oc7 (b)/(c).
 *
 * Citation: R(v) <= lambda_max(C) is standard (Parlett, "The Symmetric
 * Eigenvalue Problem", 1998, the Rayleigh-quotient / Courant-Fischer min-max);
 * this is the primal mirror of the dual-residual Jansson lower bound.
 *
 * Memory (CLAUDE.md rule 7): lb_out is a caller-init'd output; all temporaries are
 * scoped and cleared.  p, it non-NULL.  prec is the working precision.
 *
 * returns 1 and sets lb_out (a rigorous lower-bound ball) if applicable; returns
 * 0 (lb_out untouched) if not.
 */
int arbsdp_primal_lower_bound(arb_t lb_out, const arbsdp_problem *p,
                              const arbsdp_iterate *it, slong prec);

/*
 * arbsdp_certify_status -- the Layer-1 status of a bracket (MATH_SPEC §5.6, the
 * finiteness subset; the gap <= tol "optimal" refinement is Layer-2).
 */
typedef enum {
    ARBSDP_CERTIFY_OPTIMAL_BRACKET = 1, /* both lb, ub finite: a rigorous bracket  */
    ARBSDP_CERTIFY_INCONCLUSIVE    = 2  /* lb=-inf or ub=+inf, or tau unhealthy     */
} arbsdp_certify_status;

/*
 * arbsdp_certify_bracket -- THE PRODUCT (bead arb-prec-IPM-b19): the rigorous
 * two-sided certificate [lb, ub] for the external max problem (MATH_SPEC §5.4 +
 * §5.5).  Returns a status; fills lb, ub.
 *
 * GUARANTEE (THEOREM, CLAUDE.md rule 2): lb <= p*_ext <= ub.  lb is taken at the
 * Arb LOWER endpoint and ub at the Arb UPPER endpoint of the assembled balls, so
 * the scalar interval [lb, ub] rigorously contains the optimum even after all
 * ball rounding.  A bracket that EVER excludes a known optimum is a P0 bug.
 *
 * THE P0 SIGN LOCUS (computed in EXACTLY ONE place here): the external dual is
 *     y_ext[i] = -( it->y[i] / it->tau )                       (tau-purify, THEN negate)
 * formed once into a scratch vector and passed to BOTH arbsdp_lower_bound and
 * arbsdp_upper_bound (MATH_SPEC §5.5 sign contract).
 *
 * TAU GUARD (CLAUDE.md rule 5): if it->tau < TAU_HEALTHY (1e-6, the solve.c
 * convergence threshold) the iterate is on the HSDE infeasibility path (bead
 * b20), not a usable approximate optimum -- this routine does NOT purify/certify;
 * it returns ARBSDP_CERTIFY_INCONCLUSIVE with lb=-inf, ub=+inf.
 *
 * STATUS: ARBSDP_CERTIFY_OPTIMAL_BRACKET iff both lb, ub are finite; otherwise
 * ARBSDP_CERTIFY_INCONCLUSIVE (an infinite side or an unhealthy tau).
 *
 * Memory (CLAUDE.md rule 7): `lb`, `ub` are caller-init'd outputs; the scratch
 * y_ext vector and the assembled balls are scoped here and cleared.  p, it, ab
 * non-NULL; ab->nblocks == p->nblocks == it->nblocks (asserted).  prec_c is the
 * certification precision (the caller passes final_prec + 128; MATH_SPEC §10).
 */
arbsdp_certify_status arbsdp_certify_bracket(arb_t lb, arb_t ub,
                                             const arbsdp_problem *p,
                                             const arbsdp_iterate *it,
                                             const arbsdp_apriori *ab,
                                             slong prec_c);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_CERTIFY_H */
