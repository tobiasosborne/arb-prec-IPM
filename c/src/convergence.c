/*
 * src/convergence.c -- the 6-flag HSDE convergence test (POINT MODE).
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §3.8 (rho_p, rho_d, rho_g; ||b||/||C|| scaling; prstatus;
 *     anti-collapse floor; optimal/running tree; default tolerances).
 *   - HsdeNtSdpSolver.ts:1043-1132 (checkHsdeTermination -- the exact formulas).
 *   - Convergence.ts (the REL/ABS 6-flag shape).
 *   - Defaults.ts:34-49 (feasTol/optTol).
 *
 * POINT MODE (CLAUDE.md rule 1): the scaled ratios are float doubles; their
 * magnitude steers the loop, they certify nothing.  Norms are computed from the
 * problem data in arb at `prec`, then converted to double.
 *
 * REUSE (no duplication): the cost norm cFrob is taken over the iterate's cached
 * C_int blocks (||C_int||_F = ||C_file||_F, negation preserves the norm); ||b||
 * from arbsdp_problem_b (problem.h).
 */

#include <assert.h>
#include <math.h>

#include <flint/flint.h>

#include "arbsdp/convergence.h"

/* convert an arb to double via its midpoint (point mode; radii not trusted). */
static double
arb_to_d(const arb_t a)
{
    arb_t mid;
    double v;
    arb_init(mid);
    arb_get_mid_arb(mid, a);
    v = arf_get_d(arb_midref(mid), ARF_RND_NEAR);
    arb_clear(mid);
    return v;
}

/* ||b||_inf over the problem RHS (max |b_i|), as a double. */
static double
b_inf_norm(const arbsdp_problem *p, slong prec)
{
    arb_t bi, a;
    double best = 0.0;
    arb_init(bi);
    arb_init(a);
    for (int i = 0; i < p->m; i++) {
        arbsdp_problem_b(bi, p, i, prec);
        arb_abs(a, bi);
        double v = arb_to_d(a);
        if (v > best)
            best = v;
    }
    arb_clear(bi);
    arb_clear(a);
    return best;
}

/* ||C||_F = sqrt(sum_b sum_{ij} (C_int^b_{ij})^2), as a double.  ||C_int||_F =
 * ||C_file||_F (negation preserves the Frobenius norm).  Uses the iterate's
 * cached C_int blocks (no re-materialization). */
static double
c_frob_norm(const arbsdp_iterate *it, slong prec)
{
    arb_t sq, acc;
    arb_init(sq);
    arb_init(acc);
    arb_zero(acc);
    for (int b = 0; b < it->nblocks; b++) {
        slong n = arb_mat_nrows(it->C_int[b]);
        for (slong i = 0; i < n; i++)
            for (slong j = 0; j < n; j++) {
                arb_sqr(sq, arb_mat_entry(it->C_int[b], i, j), prec);
                arb_add(acc, acc, sq, prec);
            }
    }
    arb_sqrt(acc, acc, prec);
    double v = arb_to_d(acc);
    arb_clear(sq);
    arb_clear(acc);
    return v;
}

void
arbsdp_check_convergence(arbsdp_convergence *out, const arbsdp_iterate *it,
                         double feas_tol, double opt_tol, slong prec)
{
    assert(out != NULL);
    assert(it != NULL);

    const arbsdp_problem *p = it->prob;

    /* Scaling norms (HsdeNtSdpSolver.ts:1056-1059):
     *   bInfNorm = max(1, ||b||_inf);  cFrob = max(1, ||C||_F). */
    double bInfNorm = b_inf_norm(p, prec);
    if (bInfNorm < 1.0)
        bInfNorm = 1.0;
    double cFrob = c_frob_norm(it, prec);
    if (cFrob < 1.0)
        cFrob = 1.0;

    const double eps_p = feas_tol;
    const double eps_d = feas_tol;
    const double eps_g = opt_tol;

    double tau = arb_to_d(it->tau);
    double kappa = arb_to_d(it->kappa);
    double primalInf = arb_to_d(it->primal_inf);
    double dualInf = arb_to_d(it->dual_inf);
    double pObj = arb_to_d(it->pObj);
    double dObj = arb_to_d(it->dObj);

    /* rho_p = primalInf/(tau*eps_p*(1+bInfNorm)); rho_d analogous with cFrob
     * (HsdeNtSdpSolver.ts:1069-1070). */
    double rhoP = (tau > 0.0)
                      ? primalInf / (tau * eps_p * (1.0 + bInfNorm))
                      : INFINITY;
    double rhoD = (tau > 0.0)
                      ? dualInf / (tau * eps_d * (1.0 + cFrob))
                      : INFINITY;

    /* Purified objectives and gap (HsdeNtSdpSolver.ts:1072-1076). */
    double pObjPure = (tau > 0.0) ? pObj / tau : NAN;
    double dObjPure = (tau > 0.0) ? dObj / tau : NAN;
    double gapAbs = fabs(pObjPure - dObjPure);
    double absP = fabs(pObjPure), absD = fabs(dObjPure);
    double gapScale = (absP < absD) ? absP : absD;
    if (gapScale < 1.0)
        gapScale = 1.0;
    double rhoG = isfinite(gapAbs) ? gapAbs / (eps_g * gapScale) : INFINITY;

    /* prstatus = (tau-kappa)/max(tau+kappa, 1e-300) (HsdeNtSdpSolver.ts:1078). */
    double tk = tau + kappa;
    double prstatus = (tau - kappa) / (tk > 1e-300 ? tk : 1e-300);

    out->rho_p = rhoP;
    out->rho_d = rhoD;
    out->rho_g = rhoG;
    out->prstatus = prstatus;

    /* The 6 flags (Convergence.ts shape; REL uses the purified rho-form, ABS
     * compares the un-purified inf-norms / gap directly to the tolerance). */
    out->flag_rel_primal = (rhoP <= 1.0);
    out->flag_rel_dual = (rhoD <= 1.0);
    out->flag_rel_gap = (rhoG <= 1.0);
    out->flag_abs_primal = (primalInf <= eps_p);
    out->flag_abs_dual = (dualInf <= eps_d);
    out->flag_abs_gap = (isfinite(gapAbs) && gapAbs <= eps_g);

    /* Anti-collapse floor (HsdeNtSdpSolver.ts:1084-1087): tau+kappa < 1e-8 =>
     * degenerate iterate, no classification -> running. */
    const double TAU_KAPPA_FLOOR = 1e-8;
    if (tk < TAU_KAPPA_FLOOR) {
        out->status = ARBSDP_STATUS_RUNNING;
        return;
    }

    /* Optimal (HsdeNtSdpSolver.ts:1090): rho_p<=1 && rho_d<=1 && rho_g<=1 &&
     * prstatus>0.5. */
    if (rhoP <= 1.0 && rhoD <= 1.0 && rhoG <= 1.0 && prstatus > 0.5) {
        out->status = ARBSDP_STATUS_OPTIMAL;
        return;
    }

    /* POINT-MODE (tau,kappa) infeasibility status-class detection wired (bead
     * arb-prec-IPM-epic.2).  This is the Layer-0 status CLASS that STEERS the
     * loop -- exactly analogous to ARBSDP_STATUS_OPTIMAL above (b19's bracket
     * certifies that independently).  It reads float64 from the iterate
     * midpoints and certifies NOTHING.  CLAUDE.md invariant 8's "verified Farkas
     * certificate in ball arithmetic" governs the RIGOROUS user-facing
     * certify-side status (bead b20, separate); it does NOT forbid this
     * point-mode steering status (solve.h documents solve_status as "POINT-mode
     * classifications steering the loop -- NOT the rigorous Layer-1 status").
     * The rigorous ball-verified Farkas cert remains bead b20.
     *
     * GROUND TRUTH (HsdeNtSdpSolver.ts checkHsdeTermination; Andersen-Roos-
     * Terlaky 2003 HSDE: tau->0, kappa>0 carries the Farkas certificate).  The
     * prstatus<-0.5 (kappa-dominant) guard makes the optimal (tau-dominant,
     * prstatus>0.5) and infeasible regimes NON-overlapping, so a feasible problem
     * (prstatus=+1.0) never fires here.  The anti-collapse floor (tk<1e-8) and
     * the optimal check above run FIRST.  PRIMAL-INFEASIBLE is tested FIRST: it
     * disambiguates problems that satisfy both predicates. */
    {
        const double eps_inf = (feas_tol > 1e-8) ? feas_tol : 1e-8;
        const double WITNESS_FLOOR = 1e-6;

        /* TAU-COLLAPSE guard.  The HSDE infeasibility certificate is the tau->0
         * limit (Andersen-Roos-Terlaky 2003): the recession ray / Farkas witness
         * is read off the iterate only as tau collapses.  tau initialises to 1
         * (MATH_SPEC §3.9), so a healthy or merely transient iterate keeps tau of
         * order 1; tau < TAU_INFEAS (=1) means tau is genuinely collapsing.
         *
         * The prstatus<-0.5 (kappa-dominant) guard ALONE is NOT sufficient: on an
         * extreme disparate-scale FEASIBLE problem (golden disparate_scale_sqrt2,
         * cond(M) ~ 4e32) the first point-mode step transiently blows kappa to
         * ~3.6e16 while tau stays ~10, giving prstatus=-1.0 with a witness test
         * that passes ONLY because dObj (= kappa here) is huge -- a precision-
         * starved transient, NOT a Farkas certificate.  Requiring tau < 1 rejects
         * it (tau ~10, GROWN above init) while admitting every genuine collapse
         * (the de-risk corpus fires at tau in [7.2e-11, 4.1e-4], all << 1).  This
         * is a POINT-MODE steering guard, not a rigor claim (CLAUDE.md rule 1);
         * the rigorous ball-verified Farkas cert is bead b20. */
        const double TAU_INFEAS = 1.0;
        const int tau_collapsing = (tau < TAU_INFEAS);

        /* PRIMAL-INFEASIBLE (Farkas y: Sum y_i A_i <= 0, b^T y > 0): the dual
         * objective grows away from zero (dObj > floor) while the dual residual
         * stays at zero (the y is a genuine improving ray of the dual). */
        if (tau_collapsing && prstatus < -0.5 && dObj > WITNESS_FLOOR &&
            dualInf <= eps_inf * (1.0 + fabs(dObj))) {
            out->status = ARBSDP_STATUS_PRIMAL_INFEASIBLE;
            return;
        }

        /* DUAL-INFEASIBLE / primal unbounded (recession ray X: A(X)->0,
         * <C,X> < 0 in the internal min form): the primal objective drops away
         * from zero while the primal residual stays at zero. */
        if (tau_collapsing && prstatus < -0.5 && pObj < -WITNESS_FLOOR &&
            primalInf <= eps_inf * (1.0 + fabs(pObj))) {
            out->status = ARBSDP_STATUS_DUAL_INFEASIBLE;
            return;
        }
    }

    /* Otherwise still running. */
    out->status = ARBSDP_STATUS_RUNNING;
}
