/*
 * src/solve.c -- the Layer-0 HSDE-NT Mehrotra interior-point solve DRIVER.
 *
 * Implements include/arbsdp/solve.h (the contract).  Wires the per-iteration
 * machinery (iterate / residuals / convergence / NT scaling / Schur /
 * regularization / Mehrotra direction + step) into a single fixed-precision
 * solve loop.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * The whole loop runs in Arb *point* arithmetic at a FIXED working `prec`,
 * consumed as MIDPOINTS; ball radii are NOT trusted.  The recovered objective
 * value certifies NOTHING (Layer 1 derives all rigor independently).
 *
 * GROUND TRUTH (CLAUDE.md rule 3): solve.h banner (the algorithm steps, the
 * OBJECTIVE RECOVERY sign convention, the starting point); MATH_SPEC §3;
 * HsdeNtSdpSolver.ts solveHsdeSdpNt() (init :291-296/:1134-1143, loop :369-491,
 * NT :502-509, Schur :541-558, step/update :820-846, purify :929-1001);
 * Defaults.ts:34-49.
 *
 * REUSE (CLAUDE.md, no duplication): iterate.h (iterate/residuals/apply_A),
 * convergence.h (6-flag test), ntscaling.h, schur.h, regularize.h (ONE regstate
 * carried across iters -- invariant 4 / MATH_SPEC §10.4), direction.h
 * (predictor-corrector + safeguarded step), svec.h (symmetrize / frob_inner),
 * problem.h.
 */

#include <assert.h>
#include <math.h>

#include <flint/flint.h>
#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/solve.h"
#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/convergence.h"
#include "arbsdp/ntscaling.h"
#include "arbsdp/regularize.h"
#include "arbsdp/direction.h"
#include "arbsdp/svec.h"

/* -------------------------------------------------------------------------
 * Parameters / result lifecycle
 * ---------------------------------------------------------------------- */

void
arbsdp_solve_default_params(arbsdp_solve_params *out)
{
    assert(out != NULL);
    out->feas_tol   = ARBSDP_FEAS_TOL;       /* 1e-8 (Defaults.ts:34-49) */
    out->opt_tol    = ARBSDP_OPT_TOL;        /* 1e-8                     */
    out->iter_limit = ARBSDP_ITER_LIMIT;     /* 500                     */
    out->stall_cap  = ARBSDP_STALL_ITER_CAP; /* 10                      */
}

void
arbsdp_result_init(arbsdp_result *res, const arbsdp_problem *p, slong prec)
{
    assert(res != NULL);
    assert(p != NULL);
    res->status = ARBSDP_SOLVE_NUMERICAL;  /* honest default until set */
    res->iters  = 0;
    arb_init(res->value);
    arb_init(res->mu);
    arbsdp_iterate_init(&res->it, p, prec);
}

void
arbsdp_result_clear(arbsdp_result *res)
{
    if (res == NULL) return;
    arb_clear(res->value);
    arb_clear(res->mu);
    arbsdp_iterate_clear(&res->it);
    res->status = ARBSDP_SOLVE_NUMERICAL;
    res->iters  = 0;
}

/* -------------------------------------------------------------------------
 * Starting iterate (MATH_SPEC §3.9, HsdeNtSdpSolver.ts:1134-1143):
 *   xiP = max(1, 10*||b||_inf),  xiD = max(1, 10*||C||_F + ||b||_inf)
 *   X^b = xiP*I, S^b = xiD*I, y = 0, tau = kappa = 1
 *
 * Mirrors the strictly-feasible start used in test_direction.c (initial_scale).
 * ---------------------------------------------------------------------- */
static void
initial_scale(arb_t xiP, arb_t xiD, const arbsdp_problem *p, slong prec)
{
    arb_t bnorm, cfrob, t, bi;
    arb_init(bnorm);
    arb_init(cfrob);
    arb_init(t);
    arb_init(bi);

    /* ||b||_inf */
    arb_zero(bnorm);
    for (int i = 0; i < p->m; i++) {
        arbsdp_problem_b(bi, p, i, prec);
        arb_abs(bi, bi);
        if (arb_gt(bi, bnorm)) arb_set(bnorm, bi);
    }

    /* ||C||_F = sqrt(sum_b <C^b, C^b>)  (C = matno 0; norm is sign-invariant). */
    arb_zero(cfrob);
    for (int b = 0; b < p->nblocks; b++) {
        slong n = (p->block_sizes[b] < 0) ? -(slong) p->block_sizes[b]
                                          :  (slong) p->block_sizes[b];
        arb_mat_t Cb;
        arb_mat_init(Cb, n, n);
        arbsdp_problem_block_mat(Cb, p, 0, b, prec);
        arbsdp_frob_inner(t, Cb, Cb, prec);
        arb_add(cfrob, cfrob, t, prec);
        arb_mat_clear(Cb);
    }
    arb_sqrt(cfrob, cfrob, prec);

    /* xiP = max(1, 10*||b||_inf) */
    arb_mul_si(t, bnorm, 10, prec);
    arb_one(xiP);
    if (arb_gt(t, xiP)) arb_set(xiP, t);

    /* xiD = max(1, 10*||C||_F + ||b||_inf) */
    arb_mul_si(t, cfrob, 10, prec);
    arb_add(t, t, bnorm, prec);
    arb_one(xiD);
    if (arb_gt(t, xiD)) arb_set(xiD, t);

    arb_clear(bnorm);
    arb_clear(cfrob);
    arb_clear(t);
    arb_clear(bi);
}

static void
set_starting_iterate(arbsdp_iterate *it, const arbsdp_problem *p, slong prec)
{
    arb_t xiP, xiD;
    arb_init(xiP);
    arb_init(xiD);

    initial_scale(xiP, xiD, p, prec);

    for (int b = 0; b < it->nblocks; b++) {
        slong n = it->block_n[b];
        arb_mat_zero(it->X[b]);
        arb_mat_zero(it->S[b]);
        for (slong i = 0; i < n; i++) {
            arb_set(arb_mat_entry(it->X[b], i, i), xiP);
            arb_set(arb_mat_entry(it->S[b], i, i), xiD);
        }
    }
    for (int i = 0; i < it->m; i++) arb_zero(&it->y[i]);
    arb_one(it->tau);
    arb_one(it->kappa);

    arb_clear(xiP);
    arb_clear(xiD);
}

/* -------------------------------------------------------------------------
 * Objective recovery (solve.h OBJECTIVE RECOVERY banner, invariant 7):
 *   value_user = sum_b <C_file^b, X^b/tau> = -(pObj_internal / tau)   [maximize]
 *   value_user = +(pObj_internal / tau)                              [minimize]
 * pObj_internal = sum_b <C_int^b, X^b> is it->pObj (already filled by
 * arbsdp_iterate_residuals).  HsdeNtSdpSolver.ts:992 (primalObj = maximize ?
 * -pObjP : pObjP, pObjP = sum_b <C_int^b, X^b/tau>).
 * ---------------------------------------------------------------------- */
static void
recover_value(arb_t value, const arbsdp_iterate *it, int maximize, slong prec)
{
    /* it->pObj = sum_b <C_int^b, X^b> (internal min objective, UNPURIFIED). */
    arb_div(value, it->pObj, it->tau, prec);   /* pObj_internal / tau */
    if (maximize) arb_neg(value, value);        /* flip sign for maximize */
}

/* -------------------------------------------------------------------------
 * Best-iterate snapshot + soft-optimal classification
 * (HsdeNtSdpSolver.ts:369-491, faithfully ported).
 *
 * The TS reference ranks iterates by the SCALED achieved precision
 *     achieved = max(rho_p, rho_d, rho_g, gapAbs/objScale)
 * (NOT by mu) and snapshots whenever `achieved` improves (:436-445).  It also
 * classifies the snapshot as soft-optimal ("dual-feasible", lifted to optimal on
 * the wire) when the HOMOGENEOUS convergence indicators are at their limits even
 * though the strict 6-flag ratio test floors above 1 (:453-483):
 *     mu <= feas_tol  &&  prstatus > 0.5  &&  tau >= TAU_HEALTHY (1e-6).
 * On a non-strict exit the status is `bestStatus ?? fallback`
 * (finalizeBestOr :913-926): a soft-optimal snapshot yields OPTIMAL, otherwise
 * the honest fallback (iter-limit / stall / numerical).
 *
 * This is exactly why a point-mode HSDE iterate that is driven numerically out
 * of the cone near a rank-deficient boundary still returns its best snapshot and
 * a best-effort value (solve.h: result holds the best iterate + best-effort
 * value), rather than the broken final iterate.
 *
 * Only the primal/dual variables (X, S, y, tau, kappa) are copied; residuals /
 * objectives are recomputed on the snapshot before recovery.
 * ---------------------------------------------------------------------- */
static void
copy_iterate_vars(arbsdp_iterate *dst, const arbsdp_iterate *src, slong prec)
{
    for (int b = 0; b < src->nblocks; b++) {
        arb_mat_set(dst->X[b], src->X[b]);
        arb_mat_set(dst->S[b], src->S[b]);
    }
    for (int i = 0; i < src->m; i++) arb_set(&dst->y[i], &src->y[i]);
    arb_set(dst->tau,   src->tau);
    arb_set(dst->kappa, src->kappa);
    (void) prec;
}

/* A strictly-feasible iterate has tau, kappa, mu all strictly positive and
 * finite.  POINT MODE (CLAUDE.md rule 1): test the MIDPOINTS only -- the ball
 * radii are not trusted, and near a rank-deficient optimum a true-positive but
 * tiny scalar (e.g. kappa ~ 1e-8) has a ball that straddles zero, so
 * arb_is_positive (whole-ball) would spuriously reject a valid iterate. */
static int
mid_is_pos_finite(const arb_t x)
{
    const arf_struct *m = arb_midref(x);
    return arf_is_finite(m) && arf_sgn(m) > 0;
}

static int
iterate_is_valid(const arbsdp_iterate *it)
{
    return mid_is_pos_finite(it->mu)
        && mid_is_pos_finite(it->tau)
        && mid_is_pos_finite(it->kappa);
}

/* epic.6 / CLAUDE rule 1: Layer 0 is point-mode -- ball radii are untrusted.
 * Zero the iterate's radii each IPM step so accumulated (untrusted) radius
 * cannot grow ~1/mu near the boundary and NaN-corrupt a midpoint via
 * arb_rsqrt/arb_inv in NT scaling.  Pure midpoint arithmetic is what rule 1
 * ("solve in points, radii not trusted") requires; rigor is Layer-1's job.
 *
 * In-place radius zeroing ONLY -- mag_zero(arb_radref(.)) touches no allocation,
 * so this adds NO arb_init/arb_clear (CLAUDE.md rule 7).  Covers every iterate
 * variable: X[b], S[b] (all entries), y[i], tau, kappa.  (Residuals/objectives
 * are recomputed downstream from these clean midpoints; mu is derived too.) */
static void
iterate_clear_radii(arbsdp_iterate *it)
{
    for (int b = 0; b < it->nblocks; b++) {
        slong n = it->block_n[b];
        for (slong i = 0; i < n; i++)
            for (slong j = 0; j < n; j++) {
                mag_zero(arb_radref(arb_mat_entry(it->X[b], i, j)));
                mag_zero(arb_radref(arb_mat_entry(it->S[b], i, j)));
            }
    }
    for (int i = 0; i < it->m; i++)
        mag_zero(arb_radref(&it->y[i]));
    mag_zero(arb_radref(it->tau));
    mag_zero(arb_radref(it->kappa));
}

/* achieved = max(rho_p, rho_d, rho_g, gapAbs/objScale), the TS best-iterate
 * ranking metric (HsdeNtSdpSolver.ts:431-435).  gapAbs/objScale is gap_inf
 * (the |pObjPure - dObjPure| absolute flag input) over max(1, |pObjPure|).
 * Computed in double (the ratios are float64 in TS; only their magnitude
 * matters -- POINT mode, rule 1). */
static double
achieved_metric(const arbsdp_convergence *conv, const arbsdp_iterate *it)
{
    double a = conv->rho_p;
    double pobj_pure;
    if (conv->rho_d > a) a = conv->rho_d;
    if (conv->rho_g > a) a = conv->rho_g;

    pobj_pure = arf_get_d(arb_midref(it->pObj), ARF_RND_NEAR)
              / arf_get_d(arb_midref(it->tau), ARF_RND_NEAR);
    {
        /* gapAbs = |pObjPure - dObjPure|; the convergence test already reduced
         * the absolute gap to a float in flag_abs_gap's comparison, but we
         * recompute the ratio here exactly as TS:436 to rank iterates. */
        double dobj_pure = arf_get_d(arb_midref(it->dObj), ARF_RND_NEAR)
                         / arf_get_d(arb_midref(it->tau), ARF_RND_NEAR);
        double gap_abs = pobj_pure - dobj_pure;
        double obj_scale, ratio;
        if (gap_abs < 0) gap_abs = -gap_abs;
        obj_scale = pobj_pure < 0 ? -pobj_pure : pobj_pure;
        if (obj_scale < 1.0) obj_scale = 1.0;
        ratio = gap_abs / obj_scale;
        if (ratio > a) a = ratio;
    }
    return a;
}

/* Soft-optimal: the HSDE homogeneous convergence indicators are at their limits
 * (mu <= feas_tol, tau-dominant, tau not collapsed) even though the strict
 * 6-flag ratio test floors above 1 (HsdeNtSdpSolver.ts:474-483).  TAU_HEALTHY =
 * 1e-6 keeps the 1/tau purification amplification honest. */
#define ARBSDP_TAU_HEALTHY 1e-6
static int
is_soft_optimal(const arbsdp_iterate *it, const arbsdp_convergence *conv,
                double feas_tol)
{
    double mu  = arf_get_d(arb_midref(it->mu),  ARF_RND_NEAR);
    double tau = arf_get_d(arb_midref(it->tau), ARF_RND_NEAR);
    return mu <= feas_tol && conv->prstatus > 0.5 && tau >= ARBSDP_TAU_HEALTHY;
}

/* -------------------------------------------------------------------------
 * The solve loop.
 * ---------------------------------------------------------------------- */

arbsdp_solve_status
arbsdp_solve(arbsdp_result *res, const arbsdp_problem *p, slong prec,
             const arbsdp_solve_params *params)
{
    assert(res != NULL);
    assert(p != NULL);
    assert(params != NULL);

    /* The result owns the iterate; init it (caller passes UNINITIALIZED res). */
    arbsdp_result_init(res, p, prec);

    arbsdp_iterate    *it = &res->it;      /* the WORKING iterate (best at exit) */
    arbsdp_iterate     best;               /* best-iterate snapshot             */
    arbsdp_regstate    rs;                 /* ONE state carried across iters    */
    arbsdp_direction   d;
    arb_mat_t         *W;                  /* per-block NT scaling              */
    double             best_achieved = INFINITY;  /* lowest achieved seen        */
    int                best_soft_optimal = 0;    /* best snapshot is soft-opt   */
    int                have_best = 0;             /* a valid snapshot exists     */
    int                best_iter = 0;             /* iteration of the best snap  */
    int                stall_count = 0;
    /* `fallback` is the honest non-optimal exit reason; `status` becomes
     * OPTIMAL on a strict 6-flag pass, else `best_soft_optimal ? OPTIMAL :
     * fallback` (HsdeNtSdpSolver.ts finalizeBestOr). */
    arbsdp_solve_status fallback = ARBSDP_SOLVE_ITER_LIMIT;
    arbsdp_solve_status status;
    int                strict_optimal = 0;
    /* POINT-MODE infeasibility classification (bead arb-prec-IPM-epic.2): set
     * non-zero when the 6-flag test classifies the FIRING iterate as PRIMAL/
     * DUAL infeasible (tau-collapse).  0 = not infeasible.  This is a Layer-0
     * status CLASS, not a rigorous Farkas cert (b20). */
    arbsdp_solve_status infeasible_status = 0;
    int                iter = 0;

    arb_t mu_old, thresh;                  /* stall detection scratch (cached)  */
    arbsdp_iterate_init(&best, p, prec);
    arbsdp_regstate_init(&rs, prec);
    arbsdp_direction_init(&d, p);
    arb_init(mu_old);
    arb_init(thresh);

    /* per-block NT scaling storage, allocated ONCE (no per-iteration growth) */
    W = flint_malloc((size_t) it->nblocks * sizeof *W);
    for (int b = 0; b < it->nblocks; b++)
        arb_mat_init(W[b], it->block_n[b], it->block_n[b]);

    /* 1. Starting iterate (MATH_SPEC §3.9). */
    set_starting_iterate(it, p, prec);

    /* 2. Loop to iter_limit. */
    for (iter = 0; iter < params->iter_limit; iter++) {
        arbsdp_convergence conv;

        /* residuals + mu (HsdeNtSdpSolver.ts:371-407). */
        arbsdp_iterate_residuals(it, prec);

        /* If the iterate has left the cone (point-mode boundary collapse: tau,
         * kappa or mu non-positive / non-finite), the direction is meaningless;
         * stop honestly and fall back to the best snapshot (CLAUDE.md rule 5;
         * b16 escalates prec past this floor). */
        if (!iterate_is_valid(it)) {
            fallback = ARBSDP_SOLVE_NUMERICAL;
            break;
        }

        /* 6-flag convergence (HsdeNtSdpSolver.ts:1043-1132). */
        arbsdp_check_convergence(&conv, it, params->feas_tol, params->opt_tol,
                                 prec);

        /* Best-iterate tracking by the SCALED achieved metric (TS:431-445):
         * snapshot whenever `achieved` improves, recording whether the snapshot
         * satisfies the soft-optimal homogeneous criteria (TS:474-483). */
        {
            double achieved = achieved_metric(&conv, it);
            if (achieved < best_achieved) {
                copy_iterate_vars(&best, it, prec);
                best_achieved = achieved;
                best_iter = iter;
                best_soft_optimal = is_soft_optimal(it, &conv, params->feas_tol);
                have_best = 1;
            }
        }

        /* Strict optimal exit (TS:485-487 term.status !== "running"). */
        if (conv.status == ARBSDP_STATUS_OPTIMAL) {
            strict_optimal = 1;
            break;
        }

        /* POINT-MODE infeasibility exit (bead arb-prec-IPM-epic.2): the 6-flag
         * test classified the CURRENT iterate `it` (which holds the just-checked
         * residuals -- the FIRING iterate that collapsed tau) as infeasible.
         * Break here so res->it carries that firing iterate (its tau/kappa/pObj/
         * dObj are the certificate data for the certify side, b20).  This is a
         * Layer-0 status CLASS that steers the loop, NOT a rigorous Farkas cert. */
        if (conv.status == ARBSDP_STATUS_PRIMAL_INFEASIBLE) {
            infeasible_status = ARBSDP_SOLVE_PRIMAL_INFEASIBLE;
            break;
        }
        if (conv.status == ARBSDP_STATUS_DUAL_INFEASIBLE) {
            infeasible_status = ARBSDP_SOLVE_DUAL_INFEASIBLE;
            break;
        }

        /* Resource exits (TS:489-491). */
        if (stall_count >= params->stall_cap) {
            fallback = ARBSDP_SOLVE_STALL;
            break;
        }

        /* per-block NT scaling W^b = nt_scaling(X^b, S^b) (MATH_SPEC §3.3).  A
         * nonzero return is the honest cone-exit / accuracy-exhausted signal
         * (bead b30): X^b or its inner factor is no longer strictly PD at this
         * precision -> NUMERICAL (CLAUDE.md rule 5; b16 escalates prec). */
        {
            int nt_ok = 1;
            for (int b = 0; b < it->nblocks; b++)
                if (arbsdp_nt_scaling(W[b], it->X[b], it->S[b], prec))
                    nt_ok = 0;
            if (!nt_ok) {
                fallback = ARBSDP_SOLVE_NUMERICAL;
                break;
            }
        }

        /* Mehrotra predictor-corrector direction + safeguarded alpha.  This
         * assembles + factors (with carried regularization) ONCE and reuses the
         * factor for predictor + corrector (direction.h / schur.h / regularize.h).
         * On a fully-regularized factor failure it returns 0 -> honest NUMERICAL
         * fallback (TS:572 finalizeBestOr("numerical-error")). */
        if (!arbsdp_direction_compute(&d, it, (const arb_mat_t *) W, &rs, prec)) {
            fallback = ARBSDP_SOLVE_NUMERICAL;
            break;
        }

        /* Cache the pre-step mu for the stall test (TS:848 uses muNew vs mu). */
        arb_set(mu_old, it->mu);

        /* Update: X += a*dX, S += a*dS, y += a*dy, tau += a*dtau,
         * kappa += a*dkappa (HsdeNtSdpSolver.ts:828-846), then symmetrize. */
        for (int b = 0; b < it->nblocks; b++) {
            arb_mat_scalar_addmul_arb(it->X[b], d.dX[b], d.alpha, prec);
            arb_mat_scalar_addmul_arb(it->S[b], d.dS[b], d.alpha, prec);
            arbsdp_symmetrize(it->X[b], prec);
            arbsdp_symmetrize(it->S[b], prec);
        }
        for (int i = 0; i < it->m; i++)
            arb_addmul(&it->y[i], d.alpha, &d.dy[i], prec);
        arb_addmul(it->tau,   d.alpha, d.dtau,   prec);
        arb_addmul(it->kappa, d.alpha, d.dkappa, prec);

        /* epic.6 / CLAUDE rule 1: zero the just-updated iterate's untrusted radii
         * BEFORE the stall recompute, so the next iteration's residual /
         * convergence / NT-scaling / direction all read pure midpoints and the
         * accumulated radius cannot grow ~1/mu and NaN-corrupt rsqrt/inv near the
         * PSD boundary.  (Iter-0 starting iterate is already clean.) */
        iterate_clear_radii(it);

        /* Stall detection on homogenized mu (TS:848-852): recompute muNew from
         * the freshly-stepped iterate; if muNew > 0.99*mu_old the step made no
         * real progress -> stall++ else reset.  (Residuals are recomputed at the
         * TOP of the next iteration too; this extra refresh is the stall probe.) */
        arbsdp_iterate_residuals(it, prec);       /* muNew on the stepped iterate */
        arb_set_d(thresh, 0.99);                   /* 0.99 * mu_old               */
        arb_mul(thresh, thresh, mu_old, prec);
        if (!iterate_is_valid(it) || arb_gt(it->mu, thresh))
            stall_count++;
        else
            stall_count = 0;
    }

    /* finalizeBestOr (TS:913-926): a strict pass is OPTIMAL; a POINT-MODE
     * infeasibility classification (epic.2) is terminal next; otherwise a
     * soft-optimal best snapshot lifts to OPTIMAL, else the honest fallback. */
    if (strict_optimal) {
        status = ARBSDP_SOLVE_OPTIMAL;
    } else if (infeasible_status) {
        status = infeasible_status;
    } else if (have_best && best_soft_optimal) {
        status = ARBSDP_SOLVE_OPTIMAL;
    } else {
        status = fallback;
    }

    /* On any NON-strict, NON-infeasible exit, restore the best snapshot so
     * res->it carries the best iterate and res->value the best-effort value
     * (solve.h).  For an INFEASIBLE exit the restore is SKIPPED: res->it must
     * stay the FIRING iterate (its collapsed tau/kappa and pObj/dObj are the
     * certificate data, epic.2) -- the best-by-gap snapshot PREDATES the
     * tau-collapse and would lose it.  On a strict optimal exit the working `it`
     * IS the converged iterate; keep it. */
    if (!strict_optimal && !infeasible_status && have_best) {
        copy_iterate_vars(it, &best, prec);
        iter = best_iter;
    }

    /* Recompute residuals/objectives on the FINAL (best or firing) iterate so
     * res->mu and the recovered value are consistent with what res->it holds. */
    arbsdp_iterate_residuals(it, prec);
    arb_set(res->mu, it->mu);

    /* 3. Recover the user-facing objective (purify + sign flip; banner).  NOTE:
     * for an INFEASIBLE status the recovered value is NOT meaningful (tau -> 0
     * on the firing iterate makes pObj/tau large/meaningless); the STATUS is the
     * result (epic.2).  recover_value still runs so res->value is well-formed. */
    recover_value(res->value, it, p->maximize, prec);

    res->status = status;
    res->iters  = iter;

    /* teardown (CLAUDE.md rule 7: matching clears). */
    for (int b = 0; b < it->nblocks; b++) arb_mat_clear(W[b]);
    flint_free(W);
    arb_clear(mu_old);
    arb_clear(thresh);
    arbsdp_direction_clear(&d);
    arbsdp_regstate_clear(&rs);
    arbsdp_iterate_clear(&best);

    return status;
}
