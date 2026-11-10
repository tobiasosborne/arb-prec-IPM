/*
 * include/arbsdp/convergence.h -- the 6-flag HSDE convergence test (POINT MODE).
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1)
 * ==========================================================================
 * This is the Layer-0 termination heuristic that steers the IPM loop.  It forms
 * the TS reference's scaled ratios (rho_p, rho_d, rho_g) and tau-kappa indicator
 * (prstatus) and reports a status.  It certifies NOTHING -- the rigorous status
 * comes from Layer 1 (certify.c).
 *
 * ARB MIDPOINT ARITHMETIC (bead arb-prec-IPM-qx4, MATH_SPEC §3.8): every ratio,
 * flag and status DECISION is computed from the iterate's arb MIDPOINTS at the
 * working precision and decided by arf midpoint comparisons -- never through
 * float64.  The TS reference does this in float64, where the purified gap
 * |pObj/tau - dObj/tau| is quantized at 1 ulp of the objective (~2.2e-16 rel), so
 * any opt_tol below ~1e-16 made the stop decision rounding noise.  Ball radii
 * are not consulted (point mode, CLAUDE.md rule 1): no comparison can be
 * "undecided".  The double fields of arbsdp_convergence are DIAGNOSTICS only.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - docs/MATH_SPEC.md §3.8 (the 6-flag test, the ||b||/||c|| scaling, the
 *     status tree) and §3.8 Default tolerances.
 *   - HsdeNtSdpSolver.ts:1043-1132 (checkHsdeTermination): bInfNorm = max(1,
 *     ||b||_inf); cFrob = max(1, ||C||_F); rho_p = primalInf/(tau*eps_p*(1+
 *     bInfNorm)); rho_d = dualInf/(tau*eps_d*(1+cFrob)); rho_g = gapAbs/(eps_g*
 *     gapScale) with gapScale = max(1, min(|pObjPure|,|dObjPure|)); prstatus =
 *     (tau-kappa)/max(tau+kappa,1e-300); optimal iff rho_p<=1 && rho_d<=1 &&
 *     rho_g<=1 && prstatus>0.5; anti-collapse floor tau+kappa<1e-8 => running.
 *   - Convergence.ts (the 6-flag REL/ABS pair shape) -- the SDP path uses the
 *     purified rho-form for the relative flags; the absolute flags compare the
 *     un-purified inf-norms directly to the tolerance (REL is the load-bearing
 *     one for the status; ABS is reported for diagnostics).
 *   - Defaults.ts:34-49 (feasTol=1e-8, optTol=1e-8, iterLimit=500,
 *     stallIterCap=10).
 *
 * The cost-norm cFrob is over the INTERNAL cost C_int = -C_file, whose Frobenius
 * norm equals ||C_file||_F (negation preserves the norm), so it is computed from
 * the iterate's cached C_int blocks.
 */

#ifndef ARBSDP_CONVERGENCE_H
#define ARBSDP_CONVERGENCE_H

#include <flint/arb.h>

#include "arbsdp/iterate.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default Layer-0 tolerances (Defaults.ts:34-49). */
#define ARBSDP_FEAS_TOL       1e-8
#define ARBSDP_OPT_TOL        1e-8
#define ARBSDP_ITER_LIMIT     500
#define ARBSDP_STALL_ITER_CAP 10

/* Status codes for the Layer-0 termination test (HsdeNtSdpSolver.ts status
 * vocabulary).  The PRIMAL/DUAL_INFEASIBLE values are set by the point-mode
 * (tau,kappa) classification wired in epic.2 (MATH_SPEC §3.8.1).  This is a
 * Layer-0 steering status only -- it certifies nothing.  The rigorous
 * ball-verified Farkas certificate is bead b20. */
typedef enum {
    ARBSDP_STATUS_RUNNING           = 0,  /* no terminal classification yet     */
    ARBSDP_STATUS_OPTIMAL           = 1,  /* rho<=1 all three + prstatus>0.5    */
    ARBSDP_STATUS_PRIMAL_INFEASIBLE = 2,  /* point-mode (tau,kappa) class (epic.2);
                                           * rigorous Farkas cert: b20          */
    ARBSDP_STATUS_DUAL_INFEASIBLE   = 3   /* point-mode (tau,kappa) class (epic.2);
                                           * rigorous Farkas cert: b20          */
} arbsdp_status;

/*
 * arbsdp_convergence -- the result of the 6-flag test.  The six flags, `status`
 * and `prstatus_tau_dominant` are the DECISIONS, computed in arb midpoint
 * arithmetic at the working precision (bead qx4).  rho_* and prstatus are
 * float64 copies of the arb ratios for DIAGNOSTICS/logging only: they may
 * underflow/overflow (e.g. rho_g < 1e-308 reads 0.0) and must never be used to
 * re-decide convergence.
 */
typedef struct {
    arbsdp_status status;
    /* scaled residual ratios (<= 1 means "feasible/closed to tol") */
    double rho_p;     /* primalInf / (tau*feasTol*(1+max(1,||b||_inf)))   */
    double rho_d;     /* dualInf   / (tau*feasTol*(1+max(1,||C||_F)))     */
    double rho_g;     /* gapAbs    / (optTol*max(1,min(|pObjPure|,|dObjPure|))) */
    double prstatus;  /* (tau-kappa)/max(tau+kappa,1e-300)               */
    /* the 6 flags */
    int flag_rel_primal;   /* rho_p <= 1                                  */
    int flag_rel_dual;     /* rho_d <= 1                                  */
    int flag_rel_gap;      /* rho_g <= 1                                  */
    int flag_abs_primal;   /* primalInf <= feasTol                        */
    int flag_abs_dual;     /* dualInf   <= feasTol                        */
    int flag_abs_gap;      /* gapAbs    <= optTol  (gapAbs = |pObjPure-dObjPure|) */
    /* prstatus > 1/2 (tau-dominant branch), decided in arb; read by the solve
     * driver's soft-optimal test (HsdeNtSdpSolver.ts:474-483). */
    int prstatus_tau_dominant;
} arbsdp_convergence;

/*
 * arbsdp_check_convergence -- run the 6-flag HSDE termination test on `it`
 * (which must have had arbsdp_iterate_residuals called first), with feasibility
 * tolerance `feas_tol` and optimality tolerance `opt_tol`.  Pass
 * ARBSDP_FEAS_TOL / ARBSDP_OPT_TOL for the defaults.  Fills `out`.
 *
 * The scaling norms ||b||_inf and ||C||_F are computed from the problem data at
 * precision `prec` (point mode; kept in arb for the ratios).  status is
 * ARBSDP_STATUS_OPTIMAL iff rho_p<=1 && rho_d<=1 && rho_g<=1 && prstatus>0.5 and
 * the anti-collapse floor (tau+kappa >= 1e-8) holds; ARBSDP_STATUS_PRIMAL/DUAL_
 * INFEASIBLE iff the (tau,kappa) point-mode test fires (MATH_SPEC §3.8.1);
 * otherwise ARBSDP_STATUS_RUNNING.  Rigorous Farkas certs: b20.
 */
void arbsdp_check_convergence(arbsdp_convergence *out, const arbsdp_iterate *it,
                              double feas_tol, double opt_tol, slong prec);

/*
 * arbsdp_check_convergence_arb -- the same test with ARB tolerances (bead qx4).
 * This is the primary entry point; arbsdp_check_convergence converts its double
 * tolerances EXACTLY (arb_set_d: a double is a dyadic rational) and calls it.
 * Use it for tolerances a double cannot carry, e.g. 10^-320 (subnormal) or
 * 10^-400 (underflows to 0): the precision controller passes 10^-target_digits
 * built in arb.  Only the MIDPOINTS of feas_tol / opt_tol are read.
 *
 * If `achieved` is non-NULL it receives (as an exact midpoint, radius 0) the TS
 * best-iterate ranking metric (HsdeNtSdpSolver.ts:431-436)
 *     achieved = max(rho_p, rho_d, rho_g, gapAbs / max(1, |pObjPure|)),
 * +inf when tau <= 0.  Lower is better; the solve driver snapshots on strict
 * improvement (arf midpoint compare).
 */
void arbsdp_check_convergence_arb(arbsdp_convergence *out, arb_t achieved,
                                  const arbsdp_iterate *it,
                                  const arb_t feas_tol, const arb_t opt_tol,
                                  slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_CONVERGENCE_H */
