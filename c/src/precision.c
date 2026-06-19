/*
 * src/precision.c -- adaptive precision-escalation around the fixed-precision
 * Layer-0 driver (arbsdp_solve, b15).  Bead arb-prec-IPM-b16.
 *
 * POINT MODE (CLAUDE.md rule 1, invariant 2): escalating precision improves the
 * APPROXIMATE solve so hard cases converge; it certifies NOTHING.  All rigor is
 * Layer 1 (b18/b19).  See the full GROUND TRUTH + AUDITION banner in
 * include/arbsdp/precision.h -- the chosen escalation strategy is geometric
 * DOUBLING (MATH_SPEC §6 default), which beat model-based escalation 6 solves to
 * 79 on the motivating case golden/sdp_sqrt2 (20x less work) and tied the hybrid.
 *
 * REUSE (CLAUDE.md, bead instruction): this wraps arbsdp_solve / arbsdp_result;
 * it does NOT reimplement the IPM loop.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §6 (initial prec, escalation, caps -> honest "limit").
 *   - docs/MATH_SPEC.md §10.5 (stall = muNew>0.99*mu, K=10; forwarded to solve).
 *   - PRD.md §4.3 (prec0, prec<-2*prec, prec_max/iter_max caps -> inconclusive).
 *   - c/include/arbsdp/precision.h (the contract).
 *
 * Arb memory discipline (CLAUDE.md rule 7): each escalation is a CLEAN re-solve
 * (a fresh arbsdp_result init/solve, copied into the kept result, then cleared)
 * -- no growth across re-solves.  arbsdp_solve_adaptive owns the embedded result
 * and the prec_history heap; arbsdp_adaptive_result_clear frees both.
 */

#include <math.h>
#include <stdlib.h>

#include <flint/arb.h>

#include "arbsdp/precision.h"
#include "arbsdp/solve.h"
#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/convergence.h"

void
arbsdp_precision_default_params(arbsdp_precision_params *out)
{
    out->prec0         = 128;                    /* MATH_SPEC §6 floor          */
    out->prec_max      = 8192;                   /* MATH_SPEC §6 cap (e.g.)     */
    out->escalation    = 2;                      /* AUDITION winner: doubling   */
    out->target_digits = 30;                     /* requested accuracy          */
    out->iter_limit    = ARBSDP_ITER_LIMIT;      /* 500 (Defaults.ts)           */
    out->stall_cap     = ARBSDP_STALL_ITER_CAP;  /* 10  (MATH_SPEC §10.5)       */
}

void
arbsdp_adaptive_result_clear(arbsdp_adaptive_result *res)
{
    arbsdp_result_clear(&res->res);
    if (res->prec_history != NULL) {
        free(res->prec_history);
        res->prec_history = NULL;
    }
    res->prec_history_len = 0;
    res->final_prec       = 0;
    res->status           = ARBSDP_ADAPTIVE_LIMIT;
}

/*
 * achieved_digits_from_mu -- POINT-mode accuracy diagnostic.  The complementarity
 * measure mu is an upper bound on the duality gap; once mu < 10^-d the objective
 * is accurate to ~d digits.  Reported in prec_history (MATH_SPEC §7.1
 * gap_history).  Returns -log10(mu), or a large sentinel when mu underflows.
 * NOTE: this is a DIAGNOSTIC only.  The accuracy gate is now
 * arbsdp_adaptive_confirmed (cross-precision stability, bead arb-prec-IPM-p68):
 * an OPTIMAL solve is accepted only when the next-higher precision confirms it.
 * This mu-based digit count is NOT the gate (CLAUDE.md rule 1).
 */
static double
achieved_digits_from_mu(const arb_t mu)
{
    /* Operate on the arf MIDPOINT directly (the codebase idiom -- cf. solve.c
     * mid_is_pos_finite); arb_is_finite/arb_is_zero on a const arb_t trip a
     * GCC -Wstringop-overread false positive when inlined. */
    const arf_struct *m = arb_midref(mu);
    double v;
    if (!arf_is_finite(m) || arf_is_zero(m)) return 1e9;
    v = fabs(arf_get_d(m, ARF_RND_NEAR));
    if (v <= 0.0) return 1e9;
    return -log10(v);
}

/*
 * arbsdp_adaptive_confirmed -- the CROSS-PRECISION STABILITY gate (bead
 * arb-prec-IPM-p68).  See the contract + doc comment in precision.h.  A
 * candidate OPTIMAL solve is CONFIRMED iff the next (higher-precision) confirming
 * solve is ALSO OPTIMAL and its recovered value agrees with the candidate to
 * >= target_digits relative decimal digits.  The relative-agreement formula
 * MIRRORS test_precision.c digits_matched:
 *     err = |v_cand - v_conf|;  rel = max(1, |v_cand|);  d = -log10(mid(err/rel));
 * a zero / non-finite-small err is exact agreement => +inf => confirmed.
 *
 * POINT MODE (CLAUDE.md rule 1): this is a heuristic stop on the approximate
 * point-mode solve, NOT a rigor claim; rigor is Layer 1's bracket (b18/b19).
 * A precision-starved OPTIMAL whose value MOVES when precision rises is the p68
 * class -- here it is rejected (d < target_digits).
 */
int
arbsdp_adaptive_confirmed(arbsdp_solve_status conf_status,
                          const arb_t v_cand, const arb_t v_conf,
                          int target_digits, slong prec)
{
    arb_t err, rel, one;
    double d;

    /* The confirming solve must itself be a clean OPTIMAL (precision.h). */
    if (conf_status != ARBSDP_SOLVE_OPTIMAL) return 0;

    arb_init(err); arb_init(rel); arb_init(one);

    arb_sub(err, v_cand, v_conf, prec);
    arb_abs(err, err);

    arb_abs(rel, v_cand);
    arb_one(one);
    if (arb_lt(rel, one)) arb_set(rel, one);
    arb_div(err, err, rel, prec);

    if (arb_is_zero(err) || !arb_is_finite(err)) {
        d = 1e9;                                /* exact agreement => +inf       */
    } else {
        double e = arf_get_d(arb_midref(err), ARF_RND_NEAR);
        d = (e <= 0.0) ? 1e9 : -log10(e);
    }

    arb_clear(err); arb_clear(rel); arb_clear(one);
    return d >= (double) target_digits;
}

/*
 * arbsdp_result_swap -- shallow struct swap of two arbsdp_result.  SAFE because
 * arbsdp_result holds an inline arb_struct value/mu plus an arbsdp_iterate `it`
 * whose handles point to HEAP, with NO pointer into the struct itself: a shallow
 * move transfers ownership of every heap allocation intact (no fix-up needed).
 * Used by arbsdp_solve_adaptive to MOVE a confirming solve into the kept slot
 * without a deep copy (CLAUDE.md rule 7: ownership moves, never duplicates).
 */
static void
arbsdp_result_swap(arbsdp_result *a, arbsdp_result *b)
{
    arbsdp_result t = *a;
    *a = *b;
    *b = t;
}

arbsdp_adaptive_status
arbsdp_solve_adaptive(arbsdp_adaptive_result *res, const arbsdp_problem *p,
                      const arbsdp_precision_params *pp)
{
    arbsdp_result scratch;            /* confirming solves land here (local)   */
    arbsdp_solve_params sp;
    double tol;
    slong  prec, prec_max, prec0, p_cand;
    int    escalation, cap, count, met, have_cand, res_solved;

    /* --- params / tolerance coupling (precision.h banner) ----------------- */
    prec0      = pp->prec0      > 0 ? pp->prec0      : 128;
    prec_max   = pp->prec_max   > 0 ? pp->prec_max   : 8192;
    escalation = pp->escalation >= 2 ? pp->escalation : 2;
    if (prec_max < prec0) prec_max = prec0;       /* sane: cap >= start        */

    /* Inner feas/opt tolerance = 10^-target_digits so the gap test fires at the
     * requested accuracy (precision.h TARGET/TOLERANCE COUPLING).  Clamp the
     * exponent so pow() stays finite for absurd targets. */
    {
        int td = pp->target_digits;
        if (td < 1)   td = 1;
        if (td > 300) td = 300;
        tol = pow(10.0, -(double) td);
    }
    sp.feas_tol   = tol;
    sp.opt_tol    = tol;
    sp.iter_limit = pp->iter_limit > 0 ? pp->iter_limit : ARBSDP_ITER_LIMIT;
    sp.stall_cap  = pp->stall_cap  > 0 ? pp->stall_cap  : ARBSDP_STALL_ITER_CAP;

    /* --- bound the number of escalation steps for the history allocation ---
     * Geometric: prec0, prec0*f, prec0*f^2, ... <= prec_max, plus the capped
     * final step.  count = floor(log_f(prec_max/prec0)) + 2 is a safe upper
     * bound; we grow if we ever exceed it (we will not). */
    {
        double span = log((double) prec_max / (double) prec0) /
                      log((double) escalation);
        cap = (int) span + 3;
        if (cap < 2) cap = 2;
    }
    res->prec_history     = (arbsdp_prec_step *) malloc((size_t) cap *
                                                        sizeof(arbsdp_prec_step));
    res->prec_history_len = 0;

    /* CROSS-PRECISION STABILITY controller (bead arb-prec-IPM-p68).  The old gate
     * (an OPTIMAL solve at the tightened tolerance) was a POINT-mode heuristic that
     * accepted a precision-STARVED OPTIMAL whose recovered objective was short of
     * target_digits (p68: trivial_2x2 stamped OPTIMAL@128 with 6.7/12 digits).  The
     * fix: an OPTIMAL solve at precision P is only a CANDIDATE; the NEXT (precision-
     * 2P) solve CONFIRMS it iff that solve is ALSO OPTIMAL and its value agrees with
     * the candidate to >= target_digits (arbsdp_adaptive_confirmed).  On confirmation
     * we RETURN THE CANDIDATE (final_prec stays at P_candidate -- the confirming solve
     * is verification overhead, discarded -- so bits/digit does NOT regress).  This
     * still certifies NOTHING (CLAUDE.md rule 1); rigor is Layer 1's bracket (b18/b19).
     *
     * MEMORY (CLAUDE.md rule 7): res->res is the live CANDIDATE (the single owned
     * slot at every point); `scratch` holds each confirming solve.  arbsdp_solve
     * INITIALISES the slot it is handed, so the FIRST solve into res->res / scratch
     * needs no pre-init.  On a confirming solve we either (a) confirm -> clear
     * scratch, return the candidate; (b) promote scratch to the new candidate ->
     * clear the OLD res->res FIRST, then MOVE scratch into res->res (shallow swap;
     * the swapped-out old slot is already cleared so the swap leaves res->res owning
     * the new solve and scratch holding the cleared husk we do NOT touch); or (c)
     * reject scratch -> clear scratch, keep the candidate.  Exactly ONE owned result
     * lives in res->res at the end; scratch is always cleared or moved, never both. */
    prec       = prec0;
    count      = 0;
    met        = 0;
    have_cand  = 0;
    p_cand     = prec0;
    res_solved = 0;

    for (;;) {
        arbsdp_solve_status st;
        arbsdp_result      *slot;       /* the result this iteration solves into  */

        /* With no candidate, every solve lands in res->res (it is both the slot we
         * try AND the best-effort kept result).  arbsdp_solve re-INITIALISES the
         * slot it is handed WITHOUT freeing prior contents, so a non-first solve
         * into an already-solved res->res must clear it first or it leaks (CLAUDE.md
         * rule 7).  Once a candidate exists, confirming solves land in `scratch`
         * and res->res is left untouched (the kept candidate). */
        if (!have_cand) {
            if (res_solved) arbsdp_result_clear(&res->res);
            slot = &res->res;
        } else {
            slot = &scratch;            /* fresh husk: first use raw; reuse is */
        }                               /* a cleared/freed husk from prior pass */
        st = arbsdp_solve(slot, p, prec, &sp);
        if (slot == &res->res) res_solved = 1;

        /* Record the step (MATH_SPEC §7.1 prec_history) from whichever slot we
         * just filled -- the diagnostics (mu, achieved_digits_from_mu) are kept
         * intact; they are NO LONGER the gate (bead p68). */
        if (res->prec_history_len < cap) {
            arbsdp_prec_step *step = &res->prec_history[res->prec_history_len];
            step->prec   = prec;
            step->status = st;
            step->mu     = arf_get_d(arb_midref(slot->mu), ARF_RND_NEAR);
            step->digits = achieved_digits_from_mu(slot->mu);
            res->prec_history_len++;
        }
        count++;

        /* POINT-MODE infeasibility is TERMINAL and precision-INDEPENDENT (bead
         * arb-prec-IPM-epic.2): a PRIMAL/DUAL infeasible classification will not
         * change with more precision (it is a (tau,kappa)-collapse status class,
         * not a precision-starved gap), so do NOT escalate.  Return immediately
         * with res->res holding the infeasible solve and res->status set; the
         * caller reads res->res.status for primal vs dual.  NOT the rigorous
         * ball-verified Farkas cert (bead b20).
         *
         * MEMORY (CLAUDE.md rule 7): the prec_history step is already recorded
         * above from `slot`.  For a genuinely infeasible problem the FIRST solve
         * classifies infeasible, so have_cand is false and slot == &res->res (the
         * live owned slot) -- nothing to move.  Defensively, if a candidate
         * existed (slot == &scratch), MOVE scratch into res->res exactly as the
         * OPTIMAL-promotion path does (clear the old res->res first, then shallow
         * swap) so res->res owns the infeasible solve and the cleared husk is left
         * in scratch (not double-cleared).  Exactly one owned result lives in
         * res->res; scratch is moved, never leaked. */
        if (st == ARBSDP_SOLVE_PRIMAL_INFEASIBLE ||
            st == ARBSDP_SOLVE_DUAL_INFEASIBLE) {
            if (slot != &res->res) {
                arbsdp_result_clear(&res->res);
                arbsdp_result_swap(&res->res, &scratch);
            }
            res->final_prec = prec;
            res->status     = ARBSDP_ADAPTIVE_INFEASIBLE;
            return res->status;
        }

        if (!have_cand) {
            /* No candidate yet (the very first solve, or every prior solve was
             * non-OPTIMAL).  An OPTIMAL solve becomes the candidate; otherwise the
             * solve stays in res->res as the best-effort kept result.  Either way
             * final_prec advances so an all-NUMERICAL run reports the last prec. */
            res->final_prec = prec;
            if (st == ARBSDP_SOLVE_OPTIMAL) {
                have_cand = 1;
                p_cand    = prec;
            }
        } else {
            /* We have a candidate (in res->res @ p_cand); `scratch` is the
             * confirming solve @ prec (= 2*p_cand). */
            if (arbsdp_adaptive_confirmed(st, res->res.value, scratch.value,
                                          pp->target_digits, prec)) {
                /* CONFIRMED: return the candidate (final_prec = p_cand, value/
                 * iters/iterate all from the candidate solve).  Discard scratch. */
                arbsdp_result_clear(&scratch);
                res->final_prec = p_cand;
                met = 1;
                break;
            }
            if (st == ARBSDP_SOLVE_OPTIMAL) {
                /* Not confirmed but the confirming solve is itself OPTIMAL: it
                 * becomes the NEW candidate.  Clear the OLD candidate FIRST, then
                 * MOVE scratch into res->res via a shallow swap (the swapped-out
                 * old slot is the just-cleared husk; we leave it in scratch and do
                 * NOT clear it again -- no double-clear, no leak). */
                arbsdp_result_clear(&res->res);
                arbsdp_result_swap(&res->res, &scratch);
                p_cand          = prec;
                res->final_prec = prec;
            } else {
                /* Confirming solve failed to be OPTIMAL: reject it, keep the
                 * candidate.  final_prec stays at the candidate (the kept result).
                 * Clear scratch (its arbsdp_solve init is balanced here). */
                arbsdp_result_clear(&scratch);
            }
        }

        /* Hit the cap without a confirmation -> honest LIMIT (CLAUDE.md rule 5).
         * res->res still holds the best candidate (or last solve if never OPTIMAL)
         * so Layer 1 can certify whatever bound it can. */
        if (prec >= prec_max) {
            break;
        }

        /* Escalate (AUDITION winner: geometric doubling), clamped to prec_max so
         * the cap is always tried exactly once. */
        {
            slong next = prec * (slong) escalation;
            if (next > prec_max) next = prec_max;
            if (next <= prec)    next = prec_max;   /* defensive: always advance */
            prec = next;
        }
    }

    res->status = met ? ARBSDP_ADAPTIVE_OPTIMAL : ARBSDP_ADAPTIVE_LIMIT;
    return res->status;
}
