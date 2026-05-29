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
 * NOTE: the OPTIMAL accuracy GATE is the tightened-tolerance convergence flag
 * (see solve_meets_target), NOT this midpoint diagnostic (CLAUDE.md rule 1).
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
 * The accuracy GATE.  The fixed driver's 6-flag convergence (convergence.c)
 * fires OPTIMAL only when the scaled primal/dual/gap residuals are all below the
 * tolerance we set (10^-target_digits).  An OPTIMAL exit at that tolerance
 * therefore certifies (in point mode) the objective to ~target_digits digits.
 * The gate is thus exactly: the fixed status is OPTIMAL.  We do NOT consult the
 * untrusted midpoint mu for the gate (CLAUDE.md rule 1) -- only for diagnostics.
 */
static int
solve_meets_target(arbsdp_solve_status status)
{
    return status == ARBSDP_SOLVE_OPTIMAL;
}

arbsdp_adaptive_status
arbsdp_solve_adaptive(arbsdp_adaptive_result *res, const arbsdp_problem *p,
                      const arbsdp_precision_params *pp)
{
    arbsdp_solve_params sp;
    double tol;
    slong  prec, prec_max, prec0;
    int    escalation, cap, count, met;

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

    /* The kept result.  arbsdp_solve INITIALISES the result it is handed (the
     * caller passes an UNINITIALISED arbsdp_result -- solve.h), so we do NOT
     * pre-init here.  On each escalation we arbsdp_result_clear the previous
     * solve before the next arbsdp_solve re-initialises it at the higher prec
     * (clean re-solve, no warm-start -- precision.h LIMITATION).  This keeps
     * exactly ONE live arbsdp_result, so there is no growth across re-solves
     * (CLAUDE.md rule 7).  arbsdp_adaptive_result_clear's final
     * arbsdp_result_clear is then balanced by the LAST arbsdp_solve's init. */
    prec  = prec0;
    count = 0;
    met   = 0;

    for (;;) {
        arbsdp_solve_status st;

        /* Clear the previous solve before re-solving at the escalated prec; the
         * first pass has nothing to clear (arbsdp_solve will init). */
        if (count > 0) {
            arbsdp_result_clear(&res->res);
        }

        st = arbsdp_solve(&res->res, p, prec, &sp);

        /* Record the step (MATH_SPEC §7.1 prec_history). */
        if (res->prec_history_len < cap) {
            arbsdp_prec_step *step = &res->prec_history[res->prec_history_len];
            step->prec   = prec;
            step->status = st;
            step->mu     = arf_get_d(arb_midref(res->res.mu), ARF_RND_NEAR);
            step->digits = achieved_digits_from_mu(res->res.mu);
            res->prec_history_len++;
        }
        res->final_prec = prec;
        count++;

        /* Accuracy gate: OPTIMAL at the tightened tolerance (point-mode). */
        if (solve_meets_target(st)) {
            met = 1;
            break;
        }

        /* Hit the cap without meeting the target -> honest LIMIT (rule 5). */
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
