/*
 * test/test_infeasible.c -- tests for the POINT-MODE HSDE (tau,kappa)
 * infeasibility classification (c/src/convergence.c, c/src/solve.c,
 * c/src/precision.c; bead arb-prec-IPM-epic.2).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written BEFORE the convergence/solve/
 * precision edits land.  In the RED phase the classification branches are
 * stubbed, so the driver exits an infeasible/unbounded problem as
 * ITER_LIMIT / STALL / NUMERICAL (not PRIMAL/DUAL_INFEASIBLE) and these
 * assertions FAIL.  After the edits the firing tau/kappa-collapse iterate is
 * classified and returned, and they PASS.
 *
 * ==========================================================================
 * POINT MODE -- NOT A RIGOROUS CERTIFICATE (CLAUDE.md rule 1, invariant 8)
 * ==========================================================================
 * The (tau,kappa) classification tested here is POINT MODE: float64 read from
 * the iterate midpoints.  It STEERS the loop and yields the Layer-0 status
 * CLASS, exactly analogous to the existing point-mode ARBSDP_STATUS_OPTIMAL
 * (which b19's bracket independently certifies).  It is NOT the rigorous
 * user-facing certificate.  Invariant 8's "verified Farkas certificate in ball
 * arithmetic" governs the RIGOROUS certify-side status (bead b20, separate);
 * it does NOT forbid this point-mode steering status (solve.h documents the
 * solve_status enum as "POINT-mode classifications steering the loop -- NOT the
 * rigorous Layer-1 status").  For an infeasible status the recovered objective
 * value is NOT meaningful (tau -> 0); the STATUS is the result.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - HsdeNtSdpSolver.ts checkHsdeTermination (the prstatus<-0.5 kappa-dominant
 *     branches; primal-infeasible FIRST, then dual-infeasible/unbounded).
 *   - Andersen-Roos-Terlaky 2003 (HSDE: tau->0, kappa>0 carries the Farkas
 *     certificate; primal-infeasible <-> dual unbounded direction and vice versa).
 *   - c/include/arbsdp/convergence.h (the status enum), solve.h (solve_status),
 *     precision.h (ARBSDP_ADAPTIVE_INFEASIBLE).
 *   - The de-risk problem specs (SDPA .dat-s), reconstructed below:
 *       P2 primal_infeasible_psd: C=E22, A1=E11, b=-1
 *           (max X_22 s.t. X_11 = -1, X>=0 -> X_11>=0 contradiction).
 *           Farkas y_1=-1: -E_11 <= 0, b'y = (-1)(-1) = 1 > 0.
 *       P3 unbounded_eig: C=E11, A1=E22, b=1
 *           (max X_11 s.t. X_22 = 1, X>=0 -> X_11 unbounded).
 *           Recession ray Xhat=E_11: <A_1,Xhat>=0, <C,Xhat>=1>0.
 *       P5 dual_infeasible_diag: C=I_2, A1=E12 (off-diag <A,X>=2 X_12), b=0
 *           (max tr X s.t. X_12 = 0, X>=0 -> tr X unbounded; the BEST-SNAPSHOT
 *            case -- before the firing-iterate fix the best-by-gap snapshot
 *            PREDATED the tau-collapse, so the firing iterate was lost and the
 *            driver returned ITER_LIMIT/NUMERICAL instead of DUAL_INFEASIBLE).
 *
 * Property coverage (bead arb-prec-IPM-epic.2):
 *   - P2: arbsdp_solve -> PRIMAL_INFEASIBLE; solve_adaptive -> ADAPTIVE_INFEASIBLE
 *         with res.res.status == PRIMAL_INFEASIBLE.
 *   - P3: -> DUAL_INFEASIBLE (primal unbounded).
 *   - P5: -> DUAL_INFEASIBLE (the firing-iterate fix; ITER_LIMIT/NUMERICAL means
 *         the best-snapshot restore was NOT skipped for the infeasible exit).
 *   - REGRESSION (no misfire): a feasible golden (max_eigenvalue_2x2, prstatus>0.5)
 *         still classifies ADAPTIVE_OPTIMAL -- the prstatus<-0.5 guard keeps the
 *         optimal and infeasible regimes disjoint.
 *
 * MUTATION (CLAUDE.md rule 8): commenting out the two convergence.c cert branches
 * makes P2/P3/P5 FAIL (they revert to ITER_LIMIT/NUMERICAL) -- the branches are
 * load-bearing.  Restore to GREEN.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every problem/result is init'd and
 * cleared; temp .dat-s files are written and removed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <flint/arb.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/solve.h"
#include "arbsdp/precision.h"

#define PREC      256
#define PREC_MAX 1024

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* The embedded .dat-s problem specs (de-risk P2/P3/P5; see banner). */
static const char *DATS_P2 =
    "* P2 primal_infeasible_psd: max X_22 s.t. X_11 = -1, X>=0 (needs X_11>=0)\n"
    "* Farkas y_1 = -1: -E_11 <= 0, b'y = (-1)(-1) = 1 > 0\n"
    "1\n"
    "1\n"
    "2\n"
    "-1.0\n"
    "0 1 2 2 1.0\n"
    "1 1 1 1 1.0\n";

static const char *DATS_P3 =
    "* P3 unbounded_eig: max X_11 s.t. X_22 = 1, X>=0 -> X_11 unbounded\n"
    "* Recession ray Xhat=E_11: <A_1,Xhat>=0, <C,Xhat>=1>0\n"
    "1\n"
    "1\n"
    "2\n"
    "1.0\n"
    "0 1 1 1 1.0\n"
    "1 1 2 2 1.0\n";

static const char *DATS_P5 =
    "* P5 dual_infeasible_diag: max tr X s.t. X_12 = 0 (via <A_1,X>=2 X_12), X>=0\n"
    "* X stays diagonal, tr X unbounded -> primal unbounded = dual infeasible\n"
    "* Recession ray Xhat=E_11: <A_1,Xhat>=0, <C,Xhat>=1>0\n"
    "1\n"
    "1\n"
    "2\n"
    "0.0\n"
    "0 1 1 1 1.0\n"
    "0 1 2 2 1.0\n"
    "1 1 1 2 1.0\n";

static const char *
solve_status_name(arbsdp_solve_status s)
{
    switch (s) {
    case ARBSDP_SOLVE_OPTIMAL:           return "OPTIMAL";
    case ARBSDP_SOLVE_PRIMAL_INFEASIBLE: return "PRIMAL_INFEASIBLE";
    case ARBSDP_SOLVE_DUAL_INFEASIBLE:   return "DUAL_INFEASIBLE";
    case ARBSDP_SOLVE_ITER_LIMIT:        return "ITER_LIMIT";
    case ARBSDP_SOLVE_STALL:             return "STALL";
    case ARBSDP_SOLVE_NUMERICAL:         return "NUMERICAL";
    default:                             return "?";
    }
}

static const char *
adaptive_status_name(arbsdp_adaptive_status s)
{
    switch (s) {
    case ARBSDP_ADAPTIVE_OPTIMAL:    return "ADAPTIVE_OPTIMAL";
    case ARBSDP_ADAPTIVE_LIMIT:      return "ADAPTIVE_LIMIT";
    case ARBSDP_ADAPTIVE_INFEASIBLE: return "ADAPTIVE_INFEASIBLE";
    default:                         return "?";
    }
}

/*
 * write_temp_dats -- write `contents` to a unique temp path (filled into
 * path[cap]); returns 1 on success.  The caller removes the file afterwards.
 */
static int
write_temp_dats(const char *tag, const char *contents, char *path, size_t cap)
{
    FILE *f;
    snprintf(path, cap, "/tmp/arbsdp_test_inf_%s_%d.dat-s", tag, (int) getpid());
    f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "FAIL: cannot open temp file %s\n", path);
        return 0;
    }
    fputs(contents, f);
    fclose(f);
    return 1;
}

/*
 * load_embedded -- write the embedded .dat-s, parse it into `p`.  Returns 1 on
 * success (p left init'd & loaded; *path holds the temp file to unlink).
 */
static int
load_embedded(arbsdp_problem *p, const char *tag, const char *contents,
              char *path, size_t cap)
{
    arbsdp_problem_init(p);
    if (!write_temp_dats(tag, contents, path, cap))
        return 0;
    if (arbsdp_read_sdpa(p, path) != 0) {
        fprintf(stderr, "FAIL: cannot parse embedded %s (%s)\n", tag, path);
        arbsdp_problem_clear(p);
        remove(path);
        return 0;
    }
    return 1;
}

/* Read a golden problem by name; returns 1 on success (p init'd & loaded). */
static int
load_golden(arbsdp_problem *p, const char *name)
{
    char path[1024];
    arbsdp_problem_init(p);
    snprintf(path, sizeof path, "%s/%s/%s.dat-s", GOLDEN_DIR, name, name);
    if (arbsdp_read_sdpa(p, path) != 0) {
        fprintf(stderr, "FAIL: cannot read %s\n", path);
        arbsdp_problem_clear(p);
        return 0;
    }
    return 1;
}

/*
 * run_one -- solve the embedded infeasible problem `tag` with BOTH the fixed
 * driver (arbsdp_solve @ PREC) and the adaptive driver (arbsdp_solve_adaptive,
 * target a few digits, prec_max PREC_MAX), printing both outcomes.  Fills
 * *fixed_status and *adapt_status / *adapt_res_status.
 */
static void
run_one(const char *tag, const char *contents,
        arbsdp_solve_status *fixed_status,
        arbsdp_adaptive_status *adapt_status,
        arbsdp_solve_status *adapt_res_status)
{
    arbsdp_problem p;
    char path[256];

    *fixed_status     = ARBSDP_SOLVE_NUMERICAL;
    *adapt_status     = ARBSDP_ADAPTIVE_LIMIT;
    *adapt_res_status = ARBSDP_SOLVE_NUMERICAL;

    if (!load_embedded(&p, tag, contents, path, sizeof path)) {
        CHECK(0, "run_one: load embedded problem");
        return;
    }

    /* Fixed driver. */
    {
        arbsdp_result res;
        arbsdp_solve_params sp;
        arbsdp_solve_default_params(&sp);
        arbsdp_solve(&res, &p, PREC, &sp);
        *fixed_status = res.status;
        printf("  %-3s fixed   (prec=%d): status=%s, iters=%d\n",
               tag, PREC, solve_status_name(res.status), res.iters);
        arbsdp_result_clear(&res);
    }

    /* Adaptive driver. */
    {
        arbsdp_adaptive_result r;
        arbsdp_precision_params pp;
        arbsdp_precision_default_params(&pp);
        pp.prec0         = 128;
        pp.prec_max      = PREC_MAX;
        pp.target_digits = 6;
        arbsdp_solve_adaptive(&r, &p, &pp);
        *adapt_status     = r.status;
        *adapt_res_status = r.res.status;
        printf("  %-3s adaptive        : status=%s, res.status=%s, "
               "final_prec=%ld, history_len=%d\n",
               tag, adaptive_status_name(r.status),
               solve_status_name(r.res.status),
               (long) r.final_prec, r.prec_history_len);
        arbsdp_adaptive_result_clear(&r);
    }

    arbsdp_problem_clear(&p);
    remove(path);
}

/* ------------------------------------------------------------------------- *
 * P2: primal-infeasible.                                                      *
 * ------------------------------------------------------------------------- */
static void
test_p2_primal_infeasible(void)
{
    arbsdp_solve_status fixed;
    arbsdp_adaptive_status adapt;
    arbsdp_solve_status adapt_res;

    printf("\n  P2 primal_infeasible_psd (C=E22, A1=E11, b=-1):\n");
    run_one("P2", DATS_P2, &fixed, &adapt, &adapt_res);

    CHECK(fixed == ARBSDP_SOLVE_PRIMAL_INFEASIBLE,
          "P2: arbsdp_solve must classify PRIMAL_INFEASIBLE");
    CHECK(adapt == ARBSDP_ADAPTIVE_INFEASIBLE,
          "P2: solve_adaptive must return ADAPTIVE_INFEASIBLE");
    CHECK(adapt_res == ARBSDP_SOLVE_PRIMAL_INFEASIBLE,
          "P2: solve_adaptive res.status must be PRIMAL_INFEASIBLE");
}

/* ------------------------------------------------------------------------- *
 * P3: dual-infeasible (primal unbounded).                                     *
 * ------------------------------------------------------------------------- */
static void
test_p3_dual_infeasible(void)
{
    arbsdp_solve_status fixed;
    arbsdp_adaptive_status adapt;
    arbsdp_solve_status adapt_res;

    printf("\n  P3 unbounded_eig (C=E11, A1=E22, b=1):\n");
    run_one("P3", DATS_P3, &fixed, &adapt, &adapt_res);

    CHECK(fixed == ARBSDP_SOLVE_DUAL_INFEASIBLE,
          "P3: arbsdp_solve must classify DUAL_INFEASIBLE (primal unbounded)");
    CHECK(adapt == ARBSDP_ADAPTIVE_INFEASIBLE,
          "P3: solve_adaptive must return ADAPTIVE_INFEASIBLE");
    CHECK(adapt_res == ARBSDP_SOLVE_DUAL_INFEASIBLE,
          "P3: solve_adaptive res.status must be DUAL_INFEASIBLE");
}

/* ------------------------------------------------------------------------- *
 * P5: dual-infeasible -- the BEST-SNAPSHOT / firing-iterate case.             *
 * ------------------------------------------------------------------------- */
static void
test_p5_firing_iterate(void)
{
    arbsdp_solve_status fixed;
    arbsdp_adaptive_status adapt;
    arbsdp_solve_status adapt_res;

    printf("\n  P5 dual_infeasible_diag (C=I, A1=E12, b=0; firing-iterate case):\n");
    run_one("P5", DATS_P5, &fixed, &adapt, &adapt_res);

    /* If P5 returns ITER_LIMIT/NUMERICAL the best-snapshot restore was NOT
     * skipped for the infeasible exit (the firing iterate that collapsed tau
     * was overwritten by the best-by-gap snapshot that predates the collapse). */
    CHECK(fixed == ARBSDP_SOLVE_DUAL_INFEASIBLE,
          "P5: arbsdp_solve must classify DUAL_INFEASIBLE (firing-iterate fix)");
    CHECK(adapt == ARBSDP_ADAPTIVE_INFEASIBLE,
          "P5: solve_adaptive must return ADAPTIVE_INFEASIBLE");
    CHECK(adapt_res == ARBSDP_SOLVE_DUAL_INFEASIBLE,
          "P5: solve_adaptive res.status must be DUAL_INFEASIBLE");
}

/* ------------------------------------------------------------------------- *
 * REGRESSION: a feasible golden must NOT misfire as infeasible.               *
 * ------------------------------------------------------------------------- */
static void
test_feasible_regression(void)
{
    arbsdp_problem p;
    arbsdp_adaptive_result r;
    arbsdp_precision_params pp;

    printf("\n  REGRESSION feasible max_eigenvalue_2x2 (must stay OPTIMAL):\n");
    if (!load_golden(&p, "max_eigenvalue_2x2")) {
        CHECK(0, "regression: read max_eigenvalue_2x2");
        return;
    }

    arbsdp_precision_default_params(&pp);
    pp.prec0         = 128;
    pp.prec_max      = PREC_MAX;
    pp.target_digits = 10;

    arbsdp_solve_adaptive(&r, &p, &pp);
    printf("    status=%s, res.status=%s, final_prec=%ld\n",
           adaptive_status_name(r.status), solve_status_name(r.res.status),
           (long) r.final_prec);

    CHECK(r.status == ARBSDP_ADAPTIVE_OPTIMAL,
          "regression: feasible problem must stay ADAPTIVE_OPTIMAL (no misfire)");
    CHECK(r.res.status == ARBSDP_SOLVE_OPTIMAL,
          "regression: feasible problem final fixed status must be OPTIMAL");

    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

int
main(void)
{
    printf("test_infeasible: POINT-MODE (tau,kappa) infeasibility classification "
           "(epic.2)\n");

    test_p2_primal_infeasible();
    test_p3_dual_infeasible();
    test_p5_firing_iterate();
    test_feasible_regression();

    if (failures == 0) {
        printf("\nAll test_infeasible checks passed.\n");
        return 0;
    }
    fprintf(stderr, "\ntest_infeasible: %d check(s) FAILED.\n", failures);
    return 1;
}
