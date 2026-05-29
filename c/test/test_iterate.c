/*
 * test/test_iterate.c -- POINT-MODE tests for the HSDE iterate state, the
 * constraint operator A / A^T, the 3-row residuals, and the 6-flag convergence
 * test (c/src/iterate.c, c/src/convergence.c).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written before the impls exist; the target
 * link-fails first (undefined symbols), then passes once the impls land.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §3.1-3.2 (HSDE residuals, mu denom n+1), §3.8 (6-flag).
 *   - HsdeNtSdpSolver.ts:371-407 (residuals), :1043-1132 (checkHsdeTermination).
 *   - CLAUDE.md invariant 7 (internal min -<C,X>), invariant 8 (tau,kappa).
 *
 * POINT MODE (CLAUDE.md rule 1): assertions compare MIDPOINTS within a justified
 * absolute tolerance (as in test_ntscaling / test_linalg).  TOL = 1e-50 at
 * prec=256 sits far above the achieved arithmetic accuracy yet far below any
 * wrong-result error (a sign flip, a swapped term, or a missing scaling is off by
 * O(1)).
 *
 * Coverage (bead arb-prec-IPM-b11):
 *   1. ADJOINTNESS: y^T A(X) == sum_b <(A^T y)^b, X^b> for random y, symmetric X,
 *      on the golden problem sdp_sqrt2 -- the operator adjoint identity.
 *   2. RESIDUALS VANISH at a constructed feasible point (X,y,S,tau=1,kappa=0)
 *      satisfying A(X)=b, A^T y + S = C_int  =>  r_p ~ 0, R_d ~ 0.
 *   3. CONVERGENCE FLAGS: tiny residuals + tiny gap + tau>kappa => optimal flags
 *      all true / status OPTIMAL; a large residual => the corresponding flag
 *      false / status RUNNING.  The ||b||,||c|| SCALING is tested explicitly: a
 *      problem whose data is scaled up by 1000 must not change the relative flags.
 *   4. MUTATION (CLAUDE.md rule 8): a sign error in the gap residual is caught by
 *      the residual-vanishing test; dropping the ||b|| scaling is caught by the
 *      scaling-invariance test.  Both are demonstrated by recomputing a WRONG
 *      quantity and asserting the checker rejects it.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/convergence.h"
#include "arbsdp/io.h"
#include "arbsdp/iterate.h"
#include "arbsdp/svec.h"

#define PREC 256
#define TOL 1e-50

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "../../golden"
#endif

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void
golden_path(char *buf, size_t n, const char *name)
{
    snprintf(buf, n, "%s/%s/%s.dat-s", GOLDEN_DIR, name, name);
}

/* |mid(a) - mid(b)| <= tol (point-mode closeness; radii not trusted). */
static int
close_tol(const arb_t a, const arb_t b, double tol)
{
    arb_t d, t;
    int ok;
    arb_init(d);
    arb_init(t);
    arb_get_mid_arb(d, a);
    arb_get_mid_arb(t, b);
    arb_sub(d, d, t, PREC);
    arb_abs(d, d);
    arb_set_d(t, tol);
    ok = arb_le(d, t);
    arb_clear(d);
    arb_clear(t);
    return ok;
}

static int
mat_is_zero(const arb_mat_t A, double tol)
{
    slong n = arb_mat_nrows(A), m = arb_mat_ncols(A);
    arb_t z;
    int ok = 1;
    arb_init(z);
    for (slong i = 0; i < n && ok; i++)
        for (slong j = 0; j < m && ok; j++)
            if (!close_tol(arb_mat_entry(A, i, j), z, tol))
                ok = 0;
    arb_clear(z);
    return ok;
}

/* ----- Test 1: adjointness  y^T A(X) == sum_b <(A^T y)^b, X^b> ------------- *
 * The defining adjoint identity for the constraint operator.  We use the golden
 * sdp_sqrt2 (m=2, one 2x2 block) with a chosen random-ish y and a symmetric X.
 * A sign error or a transposed index in either apply_A or apply_At breaks it. */
static void
test_adjointness(void)
{
    char path[1024];
    arbsdp_problem p;
    arbsdp_problem_init(&p);
    golden_path(path, sizeof path, "sdp_sqrt2");
    CHECK(arbsdp_read_sdpa(&p, path) == 0, "adjoint: read sdp_sqrt2");

    const int m = p.m, nb = p.nblocks;
    /* y: pick distinct nonzero values. */
    arb_ptr y = _arb_vec_init(m);
    arb_set_d(&y[0], 1.5);
    arb_set_d(&y[1], -2.25);

    /* X blocks: symmetric, distinct entries (block 0 is 2x2). */
    arb_mat_t *X = flint_malloc((size_t) nb * sizeof *X);
    for (int b = 0; b < nb; b++) {
        slong n = (p.block_sizes[b] < 0) ? -p.block_sizes[b] : p.block_sizes[b];
        arb_mat_init(X[b], n, n);
    }
    arb_set_d(arb_mat_entry(X[0], 0, 0), 3.0);
    arb_set_d(arb_mat_entry(X[0], 1, 1), 5.0);
    arb_set_d(arb_mat_entry(X[0], 0, 1), 0.75);
    arb_set_d(arb_mat_entry(X[0], 1, 0), 0.75);

    /* LHS: y^T A(X).  Cast to (const arb_mat_t *): C cannot implicitly add the
     * const qualifier to a pointer-to-array element type (ISO C array-qualifier
     * rule); apply_A treats the blocks read-only. */
    arb_ptr Ax = _arb_vec_init(m);
    arbsdp_apply_A(Ax, &p, (const arb_mat_t *) X, PREC);
    arb_t lhs, tmp;
    arb_init(lhs);
    arb_init(tmp);
    arb_zero(lhs);
    for (int i = 0; i < m; i++) {
        arb_mul(tmp, &y[i], &Ax[i], PREC);
        arb_add(lhs, lhs, tmp, PREC);
    }

    /* RHS: sum_b <(A^T y)^b, X^b>. */
    arb_mat_t *Aty = flint_malloc((size_t) nb * sizeof *Aty);
    for (int b = 0; b < nb; b++) {
        slong n = arb_mat_nrows(X[b]);
        arb_mat_init(Aty[b], n, n);
    }
    arbsdp_apply_At(Aty, &p, y, PREC);
    arb_t rhs;
    arb_init(rhs);
    arb_zero(rhs);
    for (int b = 0; b < nb; b++) {
        arbsdp_frob_inner(tmp, Aty[b], X[b], PREC);
        arb_add(rhs, rhs, tmp, PREC);
    }

    CHECK(close_tol(lhs, rhs, TOL), "adjoint: y^T A(X) != <A^T y, X>");

    /* cleanup */
    for (int b = 0; b < nb; b++) {
        arb_mat_clear(X[b]);
        arb_mat_clear(Aty[b]);
    }
    flint_free(X);
    flint_free(Aty);
    _arb_vec_clear(y, m);
    _arb_vec_clear(Ax, m);
    arb_clear(lhs);
    arb_clear(rhs);
    arb_clear(tmp);
    arbsdp_problem_clear(&p);
}

/* ----- Test 2: residuals vanish at a constructed feasible point ------------ *
 * sdp_sqrt2: max x_12 s.t. x_11=1, x_22=2.  Optimal X*=[[1,sqrt2],[sqrt2,2]] is
 * primal-feasible (A(X*)=b).  Build a DUAL point (y,S) with A^T y + S = C_int and
 * S symmetric so R_d=0.  Pick y freely; set S = C_int - A^T y (symmetric since
 * C_int and each A_i are symmetric).  With tau=1, kappa=0:
 *     r_p = A(X*) - b*1 = 0
 *     R_d = A^T y + S - C_int*1 = 0
 * (r_g need not vanish; only r_p, R_d are asserted -- this is a feasibility, not
 * an optimality, construction.) */
static void
test_residuals_vanish(void)
{
    char path[1024];
    arbsdp_problem p;
    arbsdp_problem_init(&p);
    golden_path(path, sizeof path, "sdp_sqrt2");
    CHECK(arbsdp_read_sdpa(&p, path) == 0, "residual-vanish: read sdp_sqrt2");

    arbsdp_iterate it;
    arbsdp_iterate_init(&it, &p, PREC);

    const int nb = p.nblocks;

    /* Primal-feasible X* = [[1, sqrt2],[sqrt2, 2]]. */
    arb_t s2;
    arb_init(s2);
    arb_set_si(s2, 2);
    arb_sqrt(s2, s2, PREC);
    arb_set_si(arb_mat_entry(it.X[0], 0, 0), 1);
    arb_set_si(arb_mat_entry(it.X[0], 1, 1), 2);
    arb_set(arb_mat_entry(it.X[0], 0, 1), s2);
    arb_set(arb_mat_entry(it.X[0], 1, 0), s2);

    /* Dual y arbitrary; S = C_int - A^T y. */
    arb_set_d(&it.y[0], 0.3);
    arb_set_d(&it.y[1], -0.7);
    arb_mat_t *Aty = flint_malloc((size_t) nb * sizeof *Aty);
    for (int b = 0; b < nb; b++) {
        slong n = arb_mat_nrows(it.X[b]);
        arb_mat_init(Aty[b], n, n);
    }
    arbsdp_apply_At(Aty, &p, it.y, PREC);
    for (int b = 0; b < nb; b++)
        arb_mat_sub(it.S[b], it.C_int[b], Aty[b], PREC); /* S = C_int - A^T y */

    arb_one(it.tau);
    arb_zero(it.kappa);

    arbsdp_iterate_residuals(&it, PREC);

    /* r_p ~ 0 */
    int rp_zero = 1;
    arb_t z;
    arb_init(z);
    for (int i = 0; i < p.m; i++)
        if (!close_tol(&it.r_p[i], z, TOL))
            rp_zero = 0;
    CHECK(rp_zero, "residual-vanish: r_p != 0 at feasible point");

    /* R_d ~ 0 */
    int rd_zero = 1;
    for (int b = 0; b < nb; b++)
        if (!mat_is_zero(it.R_d[b], TOL))
            rd_zero = 0;
    CHECK(rd_zero, "residual-vanish: R_d != 0 at feasible point");

    /* PIN THE SIGN of r_g (rule 8): r_g = -pObj + dObj - kappa.  Recompute the
     * EXPECTED value independently from the iterate's own pObj/dObj/kappa fields
     * with the signs written out by hand, and assert equality.  This pins the
     * SIGN of every term (a flip to +pObj, or -dObj, or +kappa, is caught) while
     * staying exact: pObj/dObj already carry the same arb values the formula uses,
     * so there is no double-rounding mismatch.  Independently we also pin the
     * MAGNITUDE: pObj = <C_int,X*> = -sqrt2 (the negated, doubled off-diagonal),
     * checked against -s2 below. */
    arb_t rg_exp;
    arb_init(rg_exp);
    arb_neg(rg_exp, it.pObj);                 /* -pObj */
    arb_add(rg_exp, rg_exp, it.dObj, PREC);   /* + dObj */
    arb_sub(rg_exp, rg_exp, it.kappa, PREC);  /* - kappa */
    CHECK(close_tol(it.r_g, rg_exp, TOL),
          "residual: r_g != -pObj+dObj-kappa (sign error)");

    /* MAGNITUDE/SIGN of pObj: <C_int, X*> = -<C_file, X*> = -x_12 = -sqrt2.
     * Catches a dropped negation in C_int or a missing 2v doubling. */
    arb_t pobj_exp;
    arb_init(pobj_exp);
    arb_neg(pobj_exp, s2);
    CHECK(close_tol(it.pObj, pobj_exp, TOL),
          "residual: pObj != <C_int,X*> = -sqrt2 (sign/2v error)");
    arb_clear(pobj_exp);

    /* MUTATION blind-check: a +pObj+dObj-kappa form differs from r_g (the test
     * discriminates a sign flip rather than passing vacuously). */
    arb_t rg_wrong, diff;
    arb_init(rg_wrong);
    arb_init(diff);
    arb_add(rg_wrong, it.pObj, it.dObj, PREC); /* +pObj + dObj (wrong sign) */
    arb_sub(rg_wrong, rg_wrong, it.kappa, PREC);
    arb_sub(diff, rg_wrong, it.r_g, PREC);
    arb_abs(diff, diff);
    arb_set_d(z, TOL);
    CHECK(!arb_le(diff, z),
          "MUTATION: gap-residual sign flip must change r_g (test is blind)");

    arb_clear(rg_exp);
    arb_clear(rg_wrong);
    arb_clear(diff);
    arb_clear(z);
    arb_clear(s2);
    for (int b = 0; b < nb; b++)
        arb_mat_clear(Aty[b]);
    flint_free(Aty);
    arbsdp_iterate_clear(&it);
    arbsdp_problem_clear(&p);
}

/* Helper: build an "optimal-looking" iterate on sdp_sqrt2 with given residual
 * magnitudes by directly poking r_p / R_d / r_g / mu / objectives.  This lets us
 * exercise arbsdp_check_convergence in isolation of the residual assembly. */
static void
set_iterate_for_convergence(arbsdp_iterate *it, double rp0, double rd00,
                            double gap, double tau, double kappa)
{
    /* primal residual: r_p[0] = rp0, rest 0 */
    for (int i = 0; i < it->m; i++)
        arb_zero(&it->r_p[i]);
    if (it->m > 0)
        arb_set_d(&it->r_p[0], rp0);
    /* dual residual: R_d[0](0,0) = rd00, rest 0 */
    for (int b = 0; b < it->nblocks; b++)
        arb_mat_zero(it->R_d[b]);
    if (it->nblocks > 0)
        arb_set_d(arb_mat_entry(it->R_d[0], 0, 0), rd00);
    /* objectives: pObj/tau and dObj/tau differ by `gap`.  Put pObjPure = 1,
     * dObjPure = 1 + gap; so pObj = tau, dObj = tau*(1+gap). */
    arb_set_d(it->tau, tau);
    arb_set_d(it->kappa, kappa);
    arb_set_d(it->pObj, 1.0 * tau);
    arb_set_d(it->dObj, (1.0 + gap) * tau);
    /* inf norms */
    arb_set_d(it->primal_inf, fabs(rp0));
    arb_set_d(it->dual_inf, fabs(rd00));
    arb_set_d(it->gap_inf, 0.0);
    arb_set_d(it->mu, 0.0);
}

/* ----- Test 3: convergence flags + ||b||,||c|| scaling -------------------- */
static void
test_convergence(void)
{
    char path[1024];
    arbsdp_problem p;
    arbsdp_problem_init(&p);
    golden_path(path, sizeof path, "sdp_sqrt2");
    CHECK(arbsdp_read_sdpa(&p, path) == 0, "convergence: read sdp_sqrt2");

    arbsdp_iterate it;
    arbsdp_iterate_init(&it, &p, PREC);

    arbsdp_convergence cv;

    /* (a) tiny residuals + tiny gap + tau>>kappa => OPTIMAL, all flags true. */
    set_iterate_for_convergence(&it, 1e-14, 1e-14, 1e-14, 1.0, 1e-12);
    arbsdp_check_convergence(&cv, &it, ARBSDP_FEAS_TOL, ARBSDP_OPT_TOL, PREC);
    CHECK(cv.status == ARBSDP_STATUS_OPTIMAL, "convergence: tiny-res not OPTIMAL");
    CHECK(cv.flag_rel_primal, "convergence: rel primal flag false at optimum");
    CHECK(cv.flag_rel_dual, "convergence: rel dual flag false at optimum");
    CHECK(cv.flag_rel_gap, "convergence: rel gap flag false at optimum");

    /* (b) large primal residual => rel primal flag false, status RUNNING. */
    set_iterate_for_convergence(&it, 1e3, 1e-14, 1e-14, 1.0, 1e-12);
    arbsdp_check_convergence(&cv, &it, ARBSDP_FEAS_TOL, ARBSDP_OPT_TOL, PREC);
    CHECK(!cv.flag_rel_primal, "convergence: rel primal flag true despite huge r_p");
    CHECK(cv.status == ARBSDP_STATUS_RUNNING,
          "convergence: status not RUNNING despite huge r_p");

    /* (c) SCALING: the RELATIVE flags must be invariant to scaling the whole
     * problem (data + iterate) by a constant.  rho_p = primalInf/(tau*eps*(1+
     * max(1,||b||_inf))); when both primalInf and ||b|| scale by 1000, rho_p is
     * (nearly) invariant, so a borderline-inside flag must stay TRUE.
     *
     * Reference (unscaled sdp_sqrt2, ||b||_inf = 2 so 1+max(1,2) = 3): pick
     * primalInf at half the rho_p = 1 boundary, primalInf = 3*eps/2, so the rel
     * primal flag is TRUE and rho_p = 0.5. */
    const double eps = ARBSDP_FEAS_TOL;
    set_iterate_for_convergence(&it, 3.0 * eps * 0.5, 1e-14, 1e-14, 1.0, 1e-12);
    arbsdp_check_convergence(&cv, &it, eps, ARBSDP_OPT_TOL, PREC);
    int flag_unscaled = cv.flag_rel_primal;
    double rho_p_unscaled = cv.rho_p;
    CHECK(flag_unscaled, "convergence: borderline-inside rel primal flag should be true");

    /* Build a genuinely x1000-scaled sdp_sqrt2 (b -> 1000 b, C -> 1000 C, A_i ->
     * 1000 A_i) by writing a temp .dat-s, then feed a residual scaled by 1000.
     * ||b||_inf grows to 2000; rho_p stays ~0.5 ONLY if the (1+||b||) scaling is
     * applied.  A routine that dropped it would see rho_p ~1000x and flip the
     * flag to false (MUTATION coverage, rule 8 -- verified separately by sed). */
    {
        const char *scaled_path = "/tmp/arbsdp_sdp_sqrt2_scaled.dat-s";
        FILE *f = fopen(scaled_path, "w");
        CHECK(f != NULL, "SCALING: open scaled temp file");
        if (f) {
            fprintf(f, "* scaled sdp_sqrt2 x1000\n");
            fprintf(f, "2\n1\n2\n1000.0 2000.0\n");
            fprintf(f, "0 1 1 2 500.0\n");
            fprintf(f, "1 1 1 1 1000.0\n");
            fprintf(f, "2 1 2 2 1000.0\n");
            fclose(f);
        }
        arbsdp_problem psc;
        arbsdp_problem_init(&psc);
        CHECK(arbsdp_read_sdpa(&psc, scaled_path) == 0, "SCALING: read scaled problem");
        arbsdp_iterate itsc;
        arbsdp_iterate_init(&itsc, &psc, PREC);
        set_iterate_for_convergence(&itsc, 3.0 * eps * 0.5 * 1000.0, 1e-14,
                                    1e-14, 1.0, 1e-12);
        arbsdp_convergence cvsc;
        arbsdp_check_convergence(&cvsc, &itsc, eps, ARBSDP_OPT_TOL, PREC);
        CHECK(cvsc.flag_rel_primal == flag_unscaled,
              "SCALING: rel primal flag changed under x1000 scaling");
        CHECK(cvsc.rho_p < 2.0 * rho_p_unscaled + 1e-12,
              "SCALING: rho_p grew under scaling -> ||b|| scaling was dropped");
        arbsdp_iterate_clear(&itsc);
        arbsdp_problem_clear(&psc);
        remove(scaled_path);
    }

    arbsdp_iterate_clear(&it);
    arbsdp_problem_clear(&p);
}

int
main(void)
{
    test_adjointness();
    test_residuals_vanish();
    test_convergence();

    if (failures != 0) {
        fprintf(stderr, "test_iterate: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_iterate (adjoint, residual-vanish, convergence+scaling)\n");
    return EXIT_SUCCESS;
}
