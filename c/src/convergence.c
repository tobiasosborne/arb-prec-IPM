/*
 * src/convergence.c -- the 6-flag HSDE convergence test (POINT MODE).
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §3.8 (rho_p, rho_d, rho_g; ||b||/||C|| scaling; prstatus;
 *     anti-collapse floor; optimal/running tree; default tolerances; the arb
 *     midpoint evaluation rule of bead arb-prec-IPM-qx4).
 *   - HsdeNtSdpSolver.ts:1043-1132 (checkHsdeTermination -- the exact formulas).
 *   - HsdeNtSdpSolver.ts:431-436 (the best-iterate `achieved` ranking metric).
 *   - Convergence.ts (the REL/ABS 6-flag shape).
 *   - Defaults.ts:34-49 (feasTol/optTol).
 *   - Andersen-Roos-Terlaky 2003, HSDE section: tau -> 0 with kappa > 0 carries
 *     the infeasibility certificate (the §3.8.1 classification).
 *
 * ARITHMETIC (bead arb-prec-IPM-qx4).  The TS reference evaluates this test in
 * float64.  Ported literally (as before qx4) the purified gap
 * |pObj/tau - dObj/tau| is a difference of two doubles, quantized at 1 ulp of
 * the objective (~2.2e-16 relative): for opt_tol below ~1e-16 the flag fired
 * only when the two doubles collided and missed gaps that straddle a rounding
 * boundary -- the stop decision was rounding noise above ~16 digits.  Here every
 * quantity is formed in arf (the arb MIDPOINT type) at the working precision
 * `prec`, round-to-nearest, and every decision is an arf_cmp.  This is POINT
 * mode (CLAUDE.md rule 1): radii are never read, so no comparison is ever
 * "undecided" -- there is no ball reasoning here and none is claimed.  The
 * float64 fields of arbsdp_convergence are diagnostic copies only.
 *
 * REUSE (no duplication): the cost norm cFrob is taken over the iterate's cached
 * C_int blocks (||C_int||_F = ||C_file||_F, negation preserves the norm); ||b||
 * from arbsdp_problem_b (problem.h).
 *
 * Arb memory discipline (CLAUDE.md rule 7): every arf_init/arb_init below has
 * its matching clear in the same function.
 */

#include <assert.h>
#include <math.h>

#include <flint/flint.h>
#include <flint/arf.h>
#include <flint/arb.h>

#include "arbsdp/convergence.h"

#define RND ARF_RND_NEAR

/* Midpoint comparisons.  A NaN operand makes every predicate false, so a
 * corrupted quantity can never satisfy a convergence/termination test. */
static int
af_le(const arf_t a, const arf_t b)
{
    if (arf_is_nan(a) || arf_is_nan(b)) return 0;
    return arf_cmp(a, b) <= 0;
}

static int
af_lt(const arf_t a, const arf_t b)
{
    if (arf_is_nan(a) || arf_is_nan(b)) return 0;
    return arf_cmp(a, b) < 0;
}

static int
af_gt(const arf_t a, const arf_t b)
{
    if (arf_is_nan(a) || arf_is_nan(b)) return 0;
    return arf_cmp(a, b) > 0;
}

/* r <- max(r, x) on midpoints (NaN x is ignored; NaN r is replaced). */
static void
af_max_inplace(arf_t r, const arf_t x)
{
    if (arf_is_nan(x)) return;
    if (arf_is_nan(r) || arf_cmp(x, r) > 0) arf_set(r, x);
}

/* r <- 10^-k as the arf midpoint, at precision prec (k >= 0). */
static void
af_ten_pow_neg(arf_t r, ulong k, slong prec)
{
    arb_t t;
    arb_init(t);
    arb_ui_pow_ui(t, 10, k, prec);
    arb_inv(t, t, prec);
    arf_set(r, arb_midref(t));
    arb_clear(t);
}

/* ||b||_inf over the problem RHS (max |mid b_i|). */
static void
b_inf_norm(arf_t out, const arbsdp_problem *p, slong prec)
{
    arb_t bi;
    arf_t a;
    arb_init(bi);
    arf_init(a);
    arf_zero(out);
    for (int i = 0; i < p->m; i++) {
        arbsdp_problem_b(bi, p, i, prec);
        arf_abs(a, arb_midref(bi));
        af_max_inplace(out, a);
    }
    arb_clear(bi);
    arf_clear(a);
}

/* ||C||_F = sqrt(sum_b sum_{ij} (mid C_int^b_{ij})^2).  ||C_int||_F =
 * ||C_file||_F (negation preserves the Frobenius norm).  Uses the iterate's
 * cached C_int blocks (no re-materialization). */
static void
c_frob_norm(arf_t out, const arbsdp_iterate *it, slong prec)
{
    arf_zero(out);
    for (int b = 0; b < it->nblocks; b++) {
        slong n = arb_mat_nrows(it->C_int[b]);
        for (slong i = 0; i < n; i++)
            for (slong j = 0; j < n; j++) {
                const arf_struct *c = arb_midref(arb_mat_entry(it->C_int[b], i, j));
                arf_addmul(out, c, c, prec, RND);
            }
    }
    arf_sqrt(out, out, prec, RND);
}

void
arbsdp_check_convergence_arb(arbsdp_convergence *out, arb_t achieved,
                             const arbsdp_iterate *it,
                             const arb_t feas_tol, const arb_t opt_tol,
                             slong prec)
{
    assert(out != NULL);
    assert(it != NULL);

    const arbsdp_problem *p = it->prob;

    /* Iterate scalars, read as MIDPOINTS (point mode; radii untrusted). */
    const arf_struct *tau       = arb_midref(it->tau);
    const arf_struct *kappa     = arb_midref(it->kappa);
    const arf_struct *primalInf = arb_midref(it->primal_inf);
    const arf_struct *dualInf   = arb_midref(it->dual_inf);
    const arf_struct *pObj      = arb_midref(it->pObj);
    const arf_struct *dObj      = arb_midref(it->dObj);
    const arf_struct *eps_pd    = arb_midref(feas_tol);   /* eps_p = eps_d */
    const arf_struct *eps_g     = arb_midref(opt_tol);

    arf_t one, bInfNorm, cFrob, den, rhoP, rhoD, rhoG, pPure, dPure, gapAbs,
          absP, absD, gapScale, tk, prstatus, t, ach;
    arf_init(one); arf_init(bInfNorm); arf_init(cFrob); arf_init(den);
    arf_init(rhoP); arf_init(rhoD); arf_init(rhoG); arf_init(pPure);
    arf_init(dPure); arf_init(gapAbs); arf_init(absP); arf_init(absD);
    arf_init(gapScale); arf_init(tk); arf_init(prstatus); arf_init(t);
    arf_init(ach);

    arf_one(one);
    const int tau_pos = (arf_sgn(tau) > 0) && arf_is_finite(tau);

    /* Scaling norms (HsdeNtSdpSolver.ts:1056-1059):
     *   bInfNorm = max(1, ||b||_inf);  cFrob = max(1, ||C||_F). */
    b_inf_norm(bInfNorm, p, prec);
    af_max_inplace(bInfNorm, one);
    c_frob_norm(cFrob, it, prec);
    af_max_inplace(cFrob, one);

    /* rho_p = primalInf/(tau*eps_p*(1+bInfNorm)); rho_d analogous with cFrob
     * (HsdeNtSdpSolver.ts:1069-1070).  tau <= 0 => +inf (no purification). */
    if (tau_pos) {
        arf_add(den, one, bInfNorm, prec, RND);
        arf_mul(den, den, eps_pd, prec, RND);
        arf_mul(den, den, tau, prec, RND);
        arf_div(rhoP, primalInf, den, prec, RND);

        arf_add(den, one, cFrob, prec, RND);
        arf_mul(den, den, eps_pd, prec, RND);
        arf_mul(den, den, tau, prec, RND);
        arf_div(rhoD, dualInf, den, prec, RND);

        /* Purified objectives and gap (HsdeNtSdpSolver.ts:1072-1076), formed
         * at `prec` -- the qx4 fix: no float64 difference of the objectives. */
        arf_div(pPure, pObj, tau, prec, RND);
        arf_div(dPure, dObj, tau, prec, RND);
        arf_sub(gapAbs, pPure, dPure, prec, RND);
        arf_abs(gapAbs, gapAbs);
        arf_abs(absP, pPure);
        arf_abs(absD, dPure);
        /* gapScale = max(1, min(|pObjPure|, |dObjPure|)) */
        arf_set(gapScale, af_lt(absP, absD) ? absP : absD);
        af_max_inplace(gapScale, one);
        arf_mul(den, eps_g, gapScale, prec, RND);
        arf_div(rhoG, gapAbs, den, prec, RND);
    } else {
        arf_pos_inf(rhoP);
        arf_pos_inf(rhoD);
        arf_pos_inf(rhoG);
        arf_pos_inf(gapAbs);
        arf_pos_inf(absP);
    }

    /* prstatus = (tau-kappa)/max(tau+kappa, tiny) (HsdeNtSdpSolver.ts:1078).
     * The TS 1e-300 floor only guards a zero denominator; the decisions that
     * read prstatus all run after the anti-collapse floor (tau+kappa >= 1e-8)
     * or the solve loop's tau,kappa > 0 validity gate, so a non-positive
     * tau+kappa simply yields prstatus = 0 (neither branch). */
    arf_add(tk, tau, kappa, prec, RND);
    if (arf_sgn(tk) > 0) {
        arf_sub(t, tau, kappa, prec, RND);
        arf_div(prstatus, t, tk, prec, RND);
    } else {
        arf_zero(prstatus);
    }

    /* Diagnostic float64 copies (never re-used for a decision). */
    out->rho_p    = arf_get_d(rhoP, RND);
    out->rho_d    = arf_get_d(rhoD, RND);
    out->rho_g    = arf_get_d(rhoG, RND);
    out->prstatus = arf_get_d(prstatus, RND);

    /* The 6 flags (Convergence.ts shape; REL uses the purified rho-form, ABS
     * compares the un-purified inf-norms / gap directly to the tolerance). */
    out->flag_rel_primal = af_le(rhoP, one);
    out->flag_rel_dual   = af_le(rhoD, one);
    out->flag_rel_gap    = af_le(rhoG, one);
    out->flag_abs_primal = af_le(primalInf, eps_pd);
    out->flag_abs_dual   = af_le(dualInf, eps_pd);
    out->flag_abs_gap    = arf_is_finite(gapAbs) && af_le(gapAbs, eps_g);

    /* prstatus > 1/2 (exact dyadic constant). */
    arf_one(t);
    arf_mul_2exp_si(t, t, -1);
    out->prstatus_tau_dominant = af_gt(prstatus, t);

    /* achieved = max(rho_p, rho_d, rho_g, gapAbs/objScale), objScale =
     * max(1, |pObjPure|): the TS best-iterate ranking metric
     * (HsdeNtSdpSolver.ts:431-436).  +inf when tau <= 0. */
    if (achieved != NULL) {
        if (tau_pos) {
            arf_set(ach, rhoP);
            af_max_inplace(ach, rhoD);
            af_max_inplace(ach, rhoG);
            arf_set(den, absP);
            af_max_inplace(den, one);
            arf_div(t, gapAbs, den, prec, RND);
            af_max_inplace(ach, t);
        } else {
            arf_pos_inf(ach);
        }
        arf_set(arb_midref(achieved), ach);
        mag_zero(arb_radref(achieved));
    }

    /* Anti-collapse floor (HsdeNtSdpSolver.ts:1084-1087): tau+kappa < 1e-8 =>
     * degenerate iterate, no classification -> running. */
    af_ten_pow_neg(t, 8, prec);                    /* TAU_KAPPA_FLOOR = 1e-8 */
    if (af_lt(tk, t)) {
        out->status = ARBSDP_STATUS_RUNNING;
        goto done;
    }

    /* Optimal (HsdeNtSdpSolver.ts:1090): rho_p<=1 && rho_d<=1 && rho_g<=1 &&
     * prstatus>0.5. */
    if (out->flag_rel_primal && out->flag_rel_dual && out->flag_rel_gap &&
        out->prstatus_tau_dominant) {
        out->status = ARBSDP_STATUS_OPTIMAL;
        goto done;
    }

    /* POINT-MODE (tau,kappa) infeasibility status-class detection (bead
     * arb-prec-IPM-epic.2, MATH_SPEC §3.8.1).  This is the Layer-0 status CLASS
     * that STEERS the loop -- exactly analogous to ARBSDP_STATUS_OPTIMAL above.
     * It reads iterate midpoints and certifies NOTHING.  CLAUDE.md invariant 8's
     * "verified Farkas certificate in ball arithmetic" governs the RIGOROUS
     * certify-side status (bead b20, separate); it does NOT forbid this
     * point-mode steering status (solve.h documents solve_status as "POINT-mode
     * classifications steering the loop -- NOT the rigorous Layer-1 status").
     *
     * GROUND TRUTH (HsdeNtSdpSolver.ts checkHsdeTermination; Andersen-Roos-
     * Terlaky 2003 HSDE: tau->0, kappa>0 carries the Farkas certificate).  The
     * prstatus<-0.5 (kappa-dominant) guard makes the optimal (tau-dominant,
     * prstatus>0.5) and infeasible regimes NON-overlapping, so a feasible problem
     * (prstatus=+1.0) never fires here.  PRIMAL-INFEASIBLE is tested FIRST: it
     * disambiguates problems that satisfy both predicates.
     *
     * The thresholds are CLASSIFICATION constants, not accuracy tolerances: they
     * stay fixed as target_digits grows (qx4).  eps_inf = max(feas_tol, 1e-8)
     * floors the witness-residual test at 1e-8 for any tighter feas_tol (a
     * Farkas ray is a direction: its residual need not reach the optimality
     * tolerance); WITNESS_FLOOR = 1e-6; prstatus < -1/2; TAU_INFEAS = 1.  They
     * are set into arf at `prec` (1e-8, 1e-6 as 10^-k; -1/2 and 1 exactly). */
    {
        arf_t eps_inf, witness, neg_half, absO;
        arf_init(eps_inf); arf_init(witness); arf_init(neg_half); arf_init(absO);

        af_ten_pow_neg(eps_inf, 8, prec);
        af_max_inplace(eps_inf, eps_pd);           /* max(feas_tol, 1e-8) */
        af_ten_pow_neg(witness, 6, prec);          /* WITNESS_FLOOR = 1e-6 */
        arf_one(neg_half);
        arf_mul_2exp_si(neg_half, neg_half, -1);
        arf_neg(neg_half, neg_half);               /* -1/2, exact */

        /* TAU-COLLAPSE guard (TAU_INFEAS = 1).  The HSDE infeasibility
         * certificate is the tau->0 limit (Andersen-Roos-Terlaky 2003).  tau
         * initialises to 1 (MATH_SPEC §3.9), so tau < 1 means tau is genuinely
         * collapsing.  The prstatus<-0.5 guard ALONE is NOT sufficient: on the
         * extreme disparate-scale FEASIBLE golden disparate_scale_sqrt2 (cond(M)
         * ~ 4e32) the first step transiently blows kappa to ~3.6e16 while tau
         * stays ~10, giving prstatus=-1.0 with a witness test that passes ONLY
         * because dObj (= kappa here) is huge -- a precision-starved transient,
         * NOT a Farkas certificate.  tau < 1 rejects it while admitting every
         * genuine collapse (the de-risk corpus fires at tau in [7.2e-11,
         * 4.1e-4]).  POINT-MODE steering guard, not a rigor claim (rule 1). */
        const int tau_collapsing = af_lt(tau, one);
        const int kappa_dominant = af_lt(prstatus, neg_half);

        /* PRIMAL-INFEASIBLE (Farkas y: Sum y_i A_i <= 0, b^T y > 0): the dual
         * objective grows away from zero (dObj > floor) while the dual residual
         * stays at zero:  dualInf <= eps_inf*(1+|dObj|). */
        arf_abs(absO, dObj);
        arf_add(absO, absO, one, prec, RND);
        arf_mul(absO, absO, eps_inf, prec, RND);
        if (tau_collapsing && kappa_dominant && af_gt(dObj, witness) &&
            af_le(dualInf, absO)) {
            out->status = ARBSDP_STATUS_PRIMAL_INFEASIBLE;
        } else {
            /* DUAL-INFEASIBLE / primal unbounded (recession ray X: A(X)->0,
             * <C,X> < 0 in the internal min form): pObj < -floor while
             * primalInf <= eps_inf*(1+|pObj|). */
            arf_abs(absO, pObj);
            arf_add(absO, absO, one, prec, RND);
            arf_mul(absO, absO, eps_inf, prec, RND);
            arf_neg(witness, witness);
            if (tau_collapsing && kappa_dominant && af_lt(pObj, witness) &&
                af_le(primalInf, absO)) {
                out->status = ARBSDP_STATUS_DUAL_INFEASIBLE;
            } else {
                /* Otherwise still running. */
                out->status = ARBSDP_STATUS_RUNNING;
            }
        }

        arf_clear(eps_inf); arf_clear(witness); arf_clear(neg_half);
        arf_clear(absO);
    }

done:
    arf_clear(one); arf_clear(bInfNorm); arf_clear(cFrob); arf_clear(den);
    arf_clear(rhoP); arf_clear(rhoD); arf_clear(rhoG); arf_clear(pPure);
    arf_clear(dPure); arf_clear(gapAbs); arf_clear(absP); arf_clear(absD);
    arf_clear(gapScale); arf_clear(tk); arf_clear(prstatus); arf_clear(t);
    arf_clear(ach);
}

void
arbsdp_check_convergence(arbsdp_convergence *out, const arbsdp_iterate *it,
                         double feas_tol, double opt_tol, slong prec)
{
    /* A double tolerance is a dyadic rational: arb_set_d is EXACT, so this is
     * not a round trip -- the test itself runs in arb (bead qx4). */
    arb_t ft, ot;
    arb_init(ft);
    arb_init(ot);
    arb_set_d(ft, feas_tol);
    arb_set_d(ot, opt_tol);
    arbsdp_check_convergence_arb(out, NULL, it, ft, ot, prec);
    arb_clear(ft);
    arb_clear(ot);
}
