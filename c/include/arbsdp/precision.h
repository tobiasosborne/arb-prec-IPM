/*
 * include/arbsdp/precision.h -- adaptive precision-escalation around the
 * fixed-precision Layer-0 driver (arbsdp_solve, b15).
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * This is still POINT MODE.  Escalating precision improves the APPROXIMATE
 * solve so hard cases (irrational / rank-deficient boundary optima) converge;
 * it certifies NOTHING.  All rigor is Layer 1 (certify.c, b18/b19), which
 * derives its [lb,ub] independently in ball arithmetic.  This module only
 * decides WHICH working precision the point-mode solve runs at, and reports an
 * honest limit when no precision within the cap meets the accuracy target
 * (CLAUDE.md rule 5 -- never a false OPTIMAL).
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §6 (precision controller: initial prec, escalation
 *     triggers, doubling vs model-based, caps -> "limit"/"inconclusive").
 *   - docs/MATH_SPEC.md §10.5 (BINDING: stall = muNew > 0.99*mu, K = 10).
 *   - PRD.md §4.3 (adaptive precision: start prec0, escalate on Schur
 *     ill-conditioning / mu stall, model prec ~ c*(-log2 mu)+margin, prec_max/
 *     iter_max caps -> honest limit).
 *   - c/include/arbsdp/solve.h -- the fixed-prec driver this wraps (REUSED, not
 *     reimplemented: arbsdp_solve / arbsdp_result / arbsdp_solve_default_params).
 *
 * ==========================================================================
 * AUDITION (CLAUDE.md rule 3 -- do not default to the first option).
 * Escalation strategies driven against the real arbsdp_solve on the motivating
 * case golden/sdp_sqrt2 (irrational rank-deficient boundary optimum), measuring
 * the total work to reach a 20-digit OPTIMAL from prec0=128:
 * ==========================================================================
 *
 *   strategy                         | solves | sum_prec | sum_iters | outcome
 *   ---------------------------------+--------+----------+-----------+---------
 *   (a) geometric doubling           |    6   |   8064   |    45     | REACHED
 *   (b) model-based prec += c*log2   |   79   | 159030   |   815     | REACHED
 *       (mu_ach/tol), c=2            |        |          |           | (20x!)
 *   (c) hybrid (model jump, then x2) |    6   |   7878   |    44     | REACHED
 *
 * WINNER: (a) geometric doubling  prec <- 2*prec  (MATH_SPEC §6 default).
 *   - Model-based (b) is REJECTED: 20x the work.  A single failed solve's
 *     ACHIEVED mu does not predict the precision needed to push PAST the cone-
 *     collapse floor on a rank-deficient boundary -- the achievable mu floor is
 *     itself set by precision in a way one failed solve cannot see, so the model
 *     takes dozens of tiny under-predicting increments.  The MATH_SPEC §6 model
 *     prec ~ c*(-log2 mu) describes the precision to MAINTAIN accuracy at a given
 *     mu, not to REACH a deeper mu; using it as an escalation step misapplies it.
 *   - Hybrid (c) ties (a) (a marginally smaller first jump) but pays for it with
 *     a fragile model term that (b) shows can misbehave; no meaningful saving.
 *   - Doubling (a) is the simplest and most robust, matches the spec default,
 *     and ties the best alternative on the motivating case.  Adopted.
 *
 * MEASURED HONEST CEILING (CLAUDE.md rule 5): on sdp_sqrt2 the point-mode driver
 * caps OPTIMAL accuracy near ~28-29 digits -- the iterate is driven out of the
 * cone (exit NUMERICAL) around mu ~ 1e-28 before a tighter gap test fires, and
 * the floor barely improves past ~6000 bits (prec=6144 -> 28.6 digits;
 * prec=12288 -> 29.9 digits).  Requesting more than the achievable accuracy is
 * reported as an honest ARBSDP_ADAPTIVE_LIMIT, never a false OPTIMAL.
 *
 * ==========================================================================
 * TARGET / TOLERANCE COUPLING
 * ==========================================================================
 * The fixed driver's 6-flag convergence (convergence.c) fires OPTIMAL once the
 * scaled primal/dual/gap residuals fall below feas_tol/opt_tol.  With the default
 * 1e-8 tolerances sdp_sqrt2 reaches OPTIMAL but PLATEAUS at ~10 digits regardless
 * of precision (the loop stops the instant the 1e-8 gap is met).  To realise the
 * accuracy the higher precision buys, the adaptive driver TIGHTENS the inner
 * tolerance to match the request:  opt_tol = feas_tol = 10^(-target_digits).
 * (Target/tolerance coupling is UNCHANGED from the prior gate.)
 *
 * CROSS-PRECISION STABILITY GATE (bead arb-prec-IPM-p68).  An OPTIMAL solve at
 * precision P is only a CANDIDATE.  The next-higher-precision (2P) solve is a
 * CONFIRMING solve.  The candidate is CONFIRMED iff the confirming solve is ALSO
 * OPTIMAL and its recovered objective value agrees with the candidate's to at
 * least target_digits relative decimal digits (arbsdp_adaptive_confirmed).  On
 * confirmation the controller RETURNS THE CANDIDATE: final_prec stays at P and
 * value/iters/iterate are the candidate's -- the confirming solve is verification
 * overhead, discarded.  Bits/digit does NOT regress relative to the old gate.
 * If the confirming solve is itself OPTIMAL but does not agree, it becomes the
 * new candidate and escalation continues.  If prec_max is reached without a
 * confirmation, ARBSDP_ADAPTIVE_LIMIT is returned (honest; never a false OPTIMAL).
 * Edge: prec0 == prec_max means no second precision exists; the gate can only
 * return ARBSDP_ADAPTIVE_LIMIT.
 *
 * HONEST FRAMING (CLAUDE.md rule 1): ARBSDP_ADAPTIVE_OPTIMAL is a POINT-MODE
 * HEURISTIC stop.  It catches precision-limited inaccuracy and cross-precision
 * instability (the p68 class: a precision-starved OPTIMAL whose value moves when
 * precision rises is rejected).  It is NOT a rigorous guarantee of objective
 * accuracy: two solves that both converge to the same tolerance-limited point and
 * agree with each other but are both short of the true optimum are not caught
 * here.  The RIGOROUS objective-accuracy guarantee is Layer 1's certified bracket
 * width (certify.c, b18/b19), which derives [lb,ub] independently in ball
 * arithmetic.  A follow-up will gate on the Layer-1 bracket once its lower bound
 * is tight (bead arb-prec-IPM-oc7).
 *
 * Arb memory discipline (CLAUDE.md rule 7): arbsdp_adaptive_result owns one
 * embedded arbsdp_result (the final solve) and a heap prec_history array; each
 * intermediate re-solve init's and clears its own arbsdp_result cleanly (no
 * growth across re-solves).  Pair arbsdp_solve_adaptive (which initialises the
 * result it is handed) with arbsdp_adaptive_result_clear.
 */

#ifndef ARBSDP_PRECISION_H
#define ARBSDP_PRECISION_H

#include <flint/arb.h>

#include "arbsdp/problem.h"
#include "arbsdp/solve.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_adaptive_status -- terminal classification of an adaptive solve.
 * POINT MODE (CLAUDE.md rule 1): NOT the rigorous Layer-1 status (b19).
 */
typedef enum {
    ARBSDP_ADAPTIVE_OPTIMAL = 1,  /* a candidate OPTIMAL solve was CONFIRMED by the next-higher precision (cross-precision stability, p68) */
    ARBSDP_ADAPTIVE_LIMIT   = 2,  /* prec_max reached without meeting target (honest) */
    ARBSDP_ADAPTIVE_INFEASIBLE = 3 /* a terminal infeasibility classification (POINT MODE, epic.2):
                                    * the fixed solve classified PRIMAL/DUAL infeasible
                                    * via the (tau,kappa) collapse; this is precision-
                                    * INDEPENDENT (more precision will not change it) so
                                    * the controller does NOT escalate.  See
                                    * res->res.status for primal vs dual.  NOT the
                                    * rigorous ball-verified Farkas cert (bead b20). */
} arbsdp_adaptive_status;

/*
 * arbsdp_precision_params -- the adaptive-escalation controls (MATH_SPEC §6).
 *
 *   prec0          starting working precision in bits (e.g. 128).  MATH_SPEC §6:
 *                  prec_0 = max(128, ceil(digits*log2(10)) + margin).
 *   prec_max       hard cap on working precision in bits (e.g. 8192).  On hitting
 *                  it without meeting the target -> ARBSDP_ADAPTIVE_LIMIT.
 *   escalation     geometric factor applied on each escalation (AUDITION winner:
 *                  2 = doubling).  Must be >= 2.
 *   target_digits  requested decimal digits of accuracy.  Drives the inner
 *                  feas/opt tolerance (10^-target_digits) and the accuracy gate.
 *   iter_limit     hard per-solve iteration cap (forwarded to arbsdp_solve;
 *                  Defaults.ts: 500).
 *   stall_cap      consecutive-stall cap (MATH_SPEC §10.5: K=10, muNew>0.99*mu;
 *                  forwarded to arbsdp_solve).
 */
typedef struct {
    slong  prec0;
    slong  prec_max;
    int    escalation;
    int    target_digits;
    int    iter_limit;
    int    stall_cap;
} arbsdp_precision_params;

/*
 * arbsdp_precision_default_params -- fill `out` with sensible defaults:
 * prec0=128, prec_max=8192, escalation=2 (doubling), target_digits=30,
 * iter_limit=ARBSDP_ITER_LIMIT (500), stall_cap=ARBSDP_STALL_ITER_CAP (10).
 */
void arbsdp_precision_default_params(arbsdp_precision_params *out);

/*
 * arbsdp_prec_step -- one entry of the escalation trace (MATH_SPEC §7.1
 * prec_history / gap_history).  Records what each precision tried produced.
 *   prec    working precision used for this solve
 *   status  the fixed-driver terminal status at that precision
 *   mu       the final complementarity measure (midpoint; diagnostic)
 *   digits   achieved decimal-digit accuracy implied by mu (-log10(mu)),
 *            a POINT-mode diagnostic used by the accuracy gate
 */
typedef struct {
    slong               prec;
    arbsdp_solve_status status;
    double              mu;
    double              digits;
} arbsdp_prec_step;

/*
 * arbsdp_adaptive_result -- the adaptive-solve result.
 *
 *   status            ARBSDP_ADAPTIVE_OPTIMAL / ARBSDP_ADAPTIVE_LIMIT.
 *   res               the embedded fixed-precision result from the FINAL (best)
 *                     solve -- owns its arb fields + iterate (solve.h).  Consume
 *                     res.value / res.it for the recovered point-mode solution.
 *   final_prec        the working precision of the final solve.
 *   prec_history      heap array (length prec_history_len) of every precision
 *                     tried, in order, with its outcome (MATH_SPEC §7.1).
 *   prec_history_len  number of precisions tried (>= 1).
 *
 * Owned by the result: res (embedded arbsdp_result) and prec_history (heap).
 * arbsdp_solve_adaptive INITIALISES this (caller passes uninitialised); pair
 * with arbsdp_adaptive_result_clear (CLAUDE.md rule 7).
 */
typedef struct {
    arbsdp_adaptive_status status;
    arbsdp_result          res;
    slong                  final_prec;
    arbsdp_prec_step      *prec_history;
    int                    prec_history_len;
} arbsdp_adaptive_result;

/*
 * arbsdp_solve_adaptive -- run arbsdp_solve at prec0 and escalate (per the
 * AUDITION winner: prec <- escalation*prec) until the accuracy target is met or
 * prec_max is reached.  `res` is INITIALISED by this function (the caller passes
 * an uninitialised arbsdp_adaptive_result and must arbsdp_adaptive_result_clear
 * it afterwards regardless of the return value).
 *
 * Policy (MATH_SPEC §6, AUDITION, bead arb-prec-IPM-p68):
 *   1. tol <- 10^(-target_digits); set inner feas_tol = opt_tol = tol.
 *      (Target/tolerance coupling is unchanged.)
 *   2. prec <- prec0.  Loop:
 *        - re-solve cleanly at prec (a fresh arbsdp_result init/solve/clear --
 *          no warm-start in v1; see LIMITATION); record a prec_history step.
 *        - if status != OPTIMAL, advance prec or return ARBSDP_ADAPTIVE_LIMIT.
 *        - if status == OPTIMAL and no candidate yet, PROMOTE to candidate at P.
 *          Escalate prec to 2P (the confirming precision).
 *        - confirming solve at 2P:
 *            if OPTIMAL and agrees with candidate to >= target_digits digits
 *              (arbsdp_adaptive_confirmed): RETURN THE CANDIDATE result
 *              (final_prec = P, value/iters from the candidate solve) ->
 *              ARBSDP_ADAPTIVE_OPTIMAL, done.  The confirming solve is discarded.
 *            if OPTIMAL but does not agree: the confirming solve becomes the new
 *              candidate; escalate and continue.
 *            if not OPTIMAL: keep the existing candidate; continue.
 *        - if prec >= prec_max: return the best-so-far result ->
 *          ARBSDP_ADAPTIVE_LIMIT (honest; never a false OPTIMAL), done.
 *        - else prec <- min(escalation*prec, prec_max) and continue.
 *   Edge: prec0 == prec_max means no confirming precision exists; the result can
 *   only be ARBSDP_ADAPTIVE_LIMIT.  The "best-so-far" kept on a LIMIT is the
 *   best candidate or last iterate seen, so Layer 1 can still certify whatever
 *   bound it can (MATH_SPEC §6 graceful degradation).
 *
 * Returns the adaptive status (also stored in res->status).
 *
 * LIMITATION (v1, documented per the bead): each escalation is a CLEAN re-solve
 * from the initial point at the higher precision -- there is NO warm-start from
 * the lower-precision iterate.  A warm-start (lift the prior iterate to the new
 * precision) is an optional follow-up that would cut total iterations; it is not
 * implemented here.  POINT MODE (CLAUDE.md rule 1): the result certifies nothing.
 */
arbsdp_adaptive_status arbsdp_solve_adaptive(arbsdp_adaptive_result *res,
                                             const arbsdp_problem *p,
                                             const arbsdp_precision_params *pp);

/*
 * arbsdp_adaptive_confirmed -- the CROSS-PRECISION STABILITY gate (bead
 * arb-prec-IPM-p68).  A candidate OPTIMAL solve at precision P is CONFIRMED by
 * the next (precision-2P) CONFIRMING solve iff that confirming solve is ALSO
 * OPTIMAL and its recovered value agrees with the candidate's value to at least
 * target_digits decimal digits (relative).
 *
 * Returns 1 IFF (conf_status == ARBSDP_SOLVE_OPTIMAL) AND the relative agreement
 * between v_cand and v_conf is >= target_digits digits, where
 *     agreement = -log10( |v_cand - v_conf| / max(1, |v_cand|) ).
 * Exact agreement (zero / underflowing-small difference) => +inf => confirmed.
 *
 * POINT MODE (CLAUDE.md rule 1): this is a HEURISTIC stop on the approximate
 * point-mode solve -- a precision-starved OPTIMAL whose recovered value MOVES
 * when precision rises is rejected (the p68 class).  It is NOT a rigor claim;
 * all rigor is Layer 1's bracket (certify.c, b18/b19).  `prec` is the working
 * precision for the internal ball-arithmetic agreement computation.
 */
int arbsdp_adaptive_confirmed(arbsdp_solve_status conf_status,
                              const arb_t v_cand, const arb_t v_conf,
                              int target_digits, slong prec);

/*
 * arbsdp_adaptive_result_clear -- free everything arbsdp_solve_adaptive
 * allocated (the embedded arbsdp_result via arbsdp_result_clear, and the
 * prec_history heap array) and re-zero the struct (CLAUDE.md rule 7).  Does NOT
 * free the borrowed problem.
 */
void arbsdp_adaptive_result_clear(arbsdp_adaptive_result *res);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_PRECISION_H */
