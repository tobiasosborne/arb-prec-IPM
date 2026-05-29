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

    /* Infeasibility-certificate branches are STUBBED here (bead b20 wires the
     * verified Farkas test in ball arithmetic; a float64 heuristic here would
     * violate CLAUDE.md invariant 8).  Everything else is running. */
    out->status = ARBSDP_STATUS_RUNNING;
}
