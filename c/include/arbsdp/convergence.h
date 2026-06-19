/*
 * include/arbsdp/convergence.h -- the 6-flag HSDE convergence test (POINT MODE).
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1)
 * ==========================================================================
 * This is the Layer-0 termination heuristic that steers the IPM loop.  It reduces
 * the iterate's residuals/objectives to float64 scaled ratios (rho_p, rho_d,
 * rho_g) and a tau-kappa indicator (prstatus), exactly as the TS reference does,
 * and reports a status.  It certifies NOTHING -- the rigorous status comes from
 * Layer 1 (certify.c, later beads).
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
 * arbsdp_convergence -- the result of the 6-flag test.  rho_* and prstatus are
 * float doubles (point-mode scaled ratios; their magnitude, not their last bit,
 * is what matters).  The six flags are the REL/ABS primal/dual/gap conjuncts.
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
} arbsdp_convergence;

/*
 * arbsdp_check_convergence -- run the 6-flag HSDE termination test on `it`
 * (which must have had arbsdp_iterate_residuals called first), with feasibility
 * tolerance `feas_tol` and optimality tolerance `opt_tol`.  Pass
 * ARBSDP_FEAS_TOL / ARBSDP_OPT_TOL for the defaults.  Fills `out`.
 *
 * The scaling norms ||b||_inf and ||C||_F are computed from the problem data at
 * precision `prec` (point mode; converted to double for the ratios).  status is
 * ARBSDP_STATUS_OPTIMAL iff rho_p<=1 && rho_d<=1 && rho_g<=1 && prstatus>0.5 and
 * the anti-collapse floor (tau+kappa >= 1e-8) holds; ARBSDP_STATUS_PRIMAL/DUAL_
 * INFEASIBLE iff the (tau,kappa) point-mode test fires (MATH_SPEC §3.8.1);
 * otherwise ARBSDP_STATUS_RUNNING.  Rigorous Farkas certs: b20.
 */
void arbsdp_check_convergence(arbsdp_convergence *out, const arbsdp_iterate *it,
                              double feas_tol, double opt_tol, slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_CONVERGENCE_H */
