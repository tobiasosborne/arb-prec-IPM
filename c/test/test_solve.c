/*
 * test/test_solve.c -- tests for the end-to-end fixed-precision HSDE-NT Mehrotra
 * solve (c/src/solve.c, bead arb-prec-IPM-b15).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written BEFORE c/src/solve.c carries an
 * implementation; the target first fails to LINK (undefined symbols
 * arbsdp_solve / arbsdp_result_init / arbsdp_result_clear /
 * arbsdp_solve_default_params), then passes once solve.c lands.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - c/include/arbsdp/solve.h (the contract: status enum, params, algorithm
 *     steps, OBJECTIVE RECOVERY banner, starting point).
 *   - docs/MATH_SPEC.md §3; HsdeNtSdpSolver.ts solveHsdeSdpNt().
 *   - golden/<name>/provenance.json (the 65-digit analytic optimal_value, baked
 *     below as decimal-string constants).
 *
 * POINT MODE (CLAUDE.md rule 1): the recovered objective is a midpoint; it
 * certifies NOTHING.  Assertions compare the recovered point-mode value to the
 * analytic golden within a justified working tolerance (as test_schur /
 * test_direction do for their point-mode kernels).
 *
 * Property coverage (bead arb-prec-IPM-b15):
 *   1. OBJECTIVE MATCH: solve each golden problem at prec=256 and compare the
 *      recovered value to its analytic golden.  >=3 MUST pass.
 *   2. STATUS == OPTIMAL for the converging problems; iters reported and < limit.
 *   3. BEAT-OR-BRACKET (CLAUDE.md rule 4): report achieved digits vs the analytic
 *      golden; sdp_sqrt2 (irrational rank-deficient optimum) reaches OPTIMAL at
 *      both prec=256 and prec=1024 (~10 digits), demonstrating the driver is
 *      correct on the rank-deficient cone boundary.
 *   4. MUTATION (CLAUDE.md rule 8): the OBJECTIVE RECOVERY sign flip (invariant 7)
 *      is load-bearing -- the correctly-recovered sqrt(2) is POSITIVE, while the
 *      un-flipped internal objective (pObj/tau) is -sqrt(2); a missing/flipped
 *      sign would yield -sqrt(2) and the test catches it.
 *
 * BOUNDARY CASE (CLAUDE.md rule 5): sdp_sqrt2 has an IRRATIONAL optimum on the
 * RANK-DEFICIENT cone boundary.  PRE-epic.6 the point-mode iterate carried an
 * untrusted radius that grew like kappa*2^-prec near convergence; at fixed
 * prec=256 it tripped iterate_is_valid (mu -> 0^-) before the 1e-8 6-flag test
 * fired, so the driver exited NUMERICAL with a best-effort value good to ~4
 * digits and was treated as a precision FLOOR.  POST-epic.6 (untrusted iterate
 * radii zeroed each IPM step, c/src/solve.c) sdp_sqrt2 reaches OPTIMAL at
 * prec=256, matching the analytic golden to ~10 digits (error ~1e-10, vs the
 * 1e-6 match tol) -- so it is now a must_converge case alongside the others.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear; the
 * result is always arbsdp_result_clear'd regardless of status.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/solve.h"

#define PREC 256

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/*
 * The 65-digit analytic optima, baked from golden/<name>/provenance.json
 * (CLAUDE.md rule 3: the analytic closed forms are the ground truth, not a prior
 * solver run).  Compared via arb_set_str so all 65 digits are honoured.
 */
#define GOLD_SQRT2     "1.4142135623730950488016887242096980785696718753769480731766797379"
#define GOLD_NEG_SQRT2 "-1.4142135623730950488016887242096980785696718753769480731766797379"
#define GOLD_TRIVIAL   "1.0000000000000000000000000000000000000000000000000000000000000000"
#define GOLD_MAXEIG2   "3.0000000000000000000000000000000000000000000000000000000000000000"
#define GOLD_TRIDIAG3  "3.7320508075688772935274463415058723669428052538103806280558069794"
#define GOLD_LPDIAG    "7.0000000000000000000000000000000000000000000000000000000000000000"
#define GOLD_MIXED     "1.7071067811865475244008443621048490392848359376884740365883398689"

/*
 * digits_matched -- |recovered - golden| reduced to a number of matching decimal
 * digits: -log10(|err| / max(1,|golden|)).  Returns a large sentinel when the
 * error underflows to zero at this precision.  POINT-mode diagnostic (rule 4).
 */
static double
digits_matched(const arb_t recovered, const char *golden_str)
{
    arb_t g, err, rel, one;
    double d;
    arb_init(g); arb_init(err); arb_init(rel); arb_init(one);

    arb_get_mid_arb(err, recovered);    /* point mode: midpoint only */
    arb_set_str(g, golden_str, PREC);
    arb_sub(err, err, g, PREC);
    arb_abs(err, err);

    arb_abs(rel, g);
    arb_one(one);
    if (arb_lt(rel, one)) arb_set(rel, one);
    arb_div(err, err, rel, PREC);

    if (arb_is_zero(err) || !arb_is_finite(err)) {
        d = 1e9;
    } else {
        double e = arf_get_d(arb_midref(err), ARF_RND_NEAR);
        d = (e <= 0.0) ? 1e9 : -log10(e);
    }

    arb_clear(g); arb_clear(err); arb_clear(rel); arb_clear(one);
    return d;
}

/* |mid(recovered) - golden_str| <= tol (absolute), POINT-mode closeness.
 * Compares MIDPOINTS only (CLAUDE.md rule 1: the point-mode value carries an
 * untrusted radius that grows over the iteration; only the midpoint is the
 * approximate solution -- exactly as test_direction.c's close_d does). */
static int
close_str(const arb_t recovered, const char *golden_str, double tol)
{
    arb_t g, dd, t;
    int ok;
    arb_init(g); arb_init(dd); arb_init(t);
    arb_get_mid_arb(dd, recovered);     /* drop the radius (point mode) */
    arb_set_str(g, golden_str, PREC);
    arb_sub(dd, dd, g, PREC);
    arb_abs(dd, dd);
    arb_set_d(t, tol);
    ok = arb_le(dd, t);
    arb_clear(g); arb_clear(dd); arb_clear(t);
    return ok;
}

static const char *
status_name(arbsdp_solve_status s)
{
    switch (s) {
    case ARBSDP_SOLVE_OPTIMAL:    return "OPTIMAL";
    case ARBSDP_SOLVE_ITER_LIMIT: return "ITER_LIMIT";
    case ARBSDP_SOLVE_STALL:      return "STALL";
    case ARBSDP_SOLVE_NUMERICAL:  return "NUMERICAL";
    default:                      return "?";
    }
}

/*
 * solve_one -- solve golden problem `name` at precision `prec` and report.
 * Fills *value (recovered objective, caller-init'd arb), *status, *iters,
 * *digits.  Returns 1 if arbsdp_read_sdpa succeeded.
 */
static int
solve_one(const char *name, slong prec, arb_t value, arbsdp_solve_status *status,
          int *iters, double *digits, const char *golden_str)
{
    arbsdp_problem p;
    arbsdp_result res;
    arbsdp_solve_params params;
    char path[1024];
    int rc;

    arbsdp_problem_init(&p);
    snprintf(path, sizeof path, "%s/%s/%s.dat-s", GOLDEN_DIR, name, name);
    rc = arbsdp_read_sdpa(&p, path);
    if (rc != 0) {
        fprintf(stderr, "FAIL: solve_one: cannot read %s\n", path);
        arbsdp_problem_clear(&p);
        return 0;
    }

    arbsdp_solve_default_params(&params);
    arbsdp_solve(&res, &p, prec, &params);

    arb_set(value, res.value);
    *status = res.status;
    *iters = res.iters;
    *digits = digits_matched(res.value, golden_str);

    arbsdp_result_clear(&res);
    arbsdp_problem_clear(&p);
    return 1;
}

/* ------------------------------------------------------------------------- *
 * Test 1+2: solve each golden problem at prec=256, match objective, status,   *
 * iters.  Prints the per-problem results TABLE.                               *
 * ------------------------------------------------------------------------- */
static void
test_solve_golden(void)
{
    struct {
        const char *name;
        const char *golden;
        int must_converge;   /* must reach OPTIMAL + match at prec=256          */
    } cases[] = {
        { "sdp_sqrt2",          GOLD_SQRT2,    1 },  /* post-epic.6: OPTIMAL @ prec=256  */
        { "trivial_2x2",        GOLD_TRIVIAL,  1 },  /* rank-deficient, converges        */
        { "max_eigenvalue_2x2", GOLD_MAXEIG2,  1 },
        { "max_eig_tridiag_3x3",GOLD_TRIDIAG3, 1 },
        { "lp_diagonal_block",  GOLD_LPDIAG,   1 },
        { "mixed_blocks",       GOLD_MIXED,    1 },
    };
    int ncases = (int) (sizeof cases / sizeof cases[0]);
    int passed = 0;
    /* Justified working tolerance: a point-mode driver exiting once the SCALED
     * residuals fall below ~1e-8 (the 6-flag test) recovers the objective good to
     * roughly the gap tolerance.  1e-6 is loose enough to clear the converging
     * cases honestly at prec=256, tight enough that a sign flip or a one-digit
     * error fails.  Driving the objective to full precision is the precision
     * controller's job (b16); we do NOT loosen this to force a boundary pass. */
    double tol = 1e-6;

    printf("\n  PER-PROBLEM RESULTS (prec=%d, feas/opt tol=1e-8):\n\n", PREC);
    printf("  %-22s | %-20s | %-20s | %-7s | %-5s | %s\n",
           "problem", "golden", "recovered", "digits", "iters", "status");
    printf("  %s\n",
           "----------------------------------------------------------"
           "----------------------------------------------------------");

    for (int c = 0; c < ncases; c++) {
        arb_t value;
        arbsdp_solve_status status = ARBSDP_SOLVE_NUMERICAL;
        int iters = -1;
        double digits = 0.0;
        char gbuf[40], rbuf[40];
        int ok;

        arb_init(value);
        if (!solve_one(cases[c].name, PREC, value, &status, &iters, &digits,
                       cases[c].golden)) {
            arb_clear(value);
            CHECK(0, "test_solve_golden: read failure");
            continue;
        }

        {
            arb_t g, vmid;
            arb_init(g);
            arb_init(vmid);
            arb_set_str(g, cases[c].golden, PREC);
            arb_get_mid_arb(vmid, value);    /* point mode: print the midpoint */
            char *gs = arb_get_str(g, 14, ARB_STR_NO_RADIUS);
            char *rs = arb_get_str(vmid, 14, ARB_STR_NO_RADIUS);
            snprintf(gbuf, sizeof gbuf, "%s", gs);
            snprintf(rbuf, sizeof rbuf, "%s", rs);
            flint_free(gs);
            flint_free(rs);
            arb_clear(g);
            arb_clear(vmid);
        }

        ok = close_str(value, cases[c].golden, tol);

        printf("  %-22s | %-20s | %-20s | %6.1f  | %5d | %s\n",
               cases[c].name, gbuf, rbuf, digits, iters, status_name(status));

        if (ok && status == ARBSDP_SOLVE_OPTIMAL) passed++;

        if (cases[c].must_converge) {
            char m1[160], m2[160], m3[160];
            snprintf(m1, sizeof m1,
                     "test_solve_golden: %s must match golden to %.0e",
                     cases[c].name, tol);
            snprintf(m2, sizeof m2,
                     "test_solve_golden: %s status must be OPTIMAL",
                     cases[c].name);
            snprintf(m3, sizeof m3,
                     "test_solve_golden: %s iters in (0, iter_limit)",
                     cases[c].name);
            CHECK(ok, m1);
            CHECK(status == ARBSDP_SOLVE_OPTIMAL, m2);
            CHECK(iters > 0 && iters < ARBSDP_ITER_LIMIT, m3);
        }

        arb_clear(value);
    }

    printf("\n  %d / %d golden problems reached OPTIMAL and matched to %.0e at "
           "prec=%d.\n", passed, ncases, tol, PREC);
    printf("  (sdp_sqrt2: irrational rank-deficient boundary optimum -- post-epic.6 "
           "reaches\n   OPTIMAL at prec=256, matching the golden to ~10 digits.)\n");

    /* >= 3 must pass (bead requirement). */
    CHECK(passed >= 3, "test_solve_golden: at least 3 golden problems must match");
}

/* ------------------------------------------------------------------------- *
 * Test 3: BEAT-OR-BRACKET (CLAUDE.md rule 4).  sdp_sqrt2 reaches OPTIMAL at     *
 * BOTH prec=256 and prec=1024 (post-epic.6) on the irrational rank-deficient    *
 * cone boundary -- proving the algorithm is correct there.  Higher precision    *
 * sharpens the recovered value; it is not required to clear NUMERICAL.          *
 * ------------------------------------------------------------------------- */
static void
test_beat_or_bracket(void)
{
    arb_t value;
    arbsdp_solve_status status;
    int iters;
    double digits;

    arb_init(value);

    /* prec=256: post-epic.6 this reaches OPTIMAL (~10 digits) on the boundary. */
    if (solve_one("sdp_sqrt2", 256, value, &status, &iters, &digits, GOLD_SQRT2)) {
        printf("\n  BEAT-OR-BRACKET (rule 4):\n");
        printf("    sdp_sqrt2 @ prec=256 : ~%.1f digits, %d iters, status=%s.\n",
               digits, iters, status_name(status));
        /* The recovered value must be a sane approximation to sqrt(2) (positive,
         * within ~1e-3) -- a sign or recovery bug fails this. */
        CHECK(close_str(value, GOLD_SQRT2, 1e-3),
              "beat_or_bracket: sdp_sqrt2 @256 must approximate +sqrt(2)");
    }

    /* prec=1024: the SAME driver converges to OPTIMAL on the irrational boundary
     * optimum -- the beat-or-bracket evidence (rule 4) and proof of correctness. */
    if (solve_one("sdp_sqrt2", 1024, value, &status, &iters, &digits, GOLD_SQRT2)) {
        printf("    sdp_sqrt2 @ prec=1024: ~%.1f digits, %d iters, status=%s "
               "(converged).\n", digits, iters, status_name(status));
        printf("    analytic golden sqrt(2) = %s\n", GOLD_SQRT2);
        CHECK(status == ARBSDP_SOLVE_OPTIMAL,
              "beat_or_bracket: sdp_sqrt2 must reach OPTIMAL at prec=1024");
        CHECK(close_str(value, GOLD_SQRT2, 1e-6),
              "beat_or_bracket: sdp_sqrt2 @1024 must match golden to 1e-6");
        CHECK(digits >= 6.0,
              "beat_or_bracket: sdp_sqrt2 @1024 must match golden to >= 6 digits");
    }

    arb_clear(value);
}

/* ------------------------------------------------------------------------- *
 * Test 4: MUTATION (CLAUDE.md rule 8) -- the OBJECTIVE RECOVERY sign flip       *
 * (invariant 7) is load-bearing.  We solve sdp_sqrt2 (maximize) and check:      *
 *   - the correctly-recovered value is +sqrt(2)  (NOT -sqrt(2));                 *
 *   - the UN-flipped internal objective pObj/tau IS -sqrt(2).                    *
 * A solver that omitted the maximize sign flip would return -sqrt(2); this test  *
 * discriminates that mutation directly.  (Done at prec=1024 so both quantities   *
 * are sharp to many digits and the sign comparison is unambiguous.)              *
 * ------------------------------------------------------------------------- */
static void
test_mutation_sign(void)
{
    arbsdp_problem p;
    arbsdp_result res;
    arbsdp_solve_params params;
    char path[1024];
    int rc;
    slong prec = 1024;

    arbsdp_problem_init(&p);
    snprintf(path, sizeof path, "%s/sdp_sqrt2/sdp_sqrt2.dat-s", GOLDEN_DIR);
    rc = arbsdp_read_sdpa(&p, path);
    CHECK(rc == 0, "mutation_sign: cannot read sdp_sqrt2.dat-s");
    if (rc != 0) { arbsdp_problem_clear(&p); return; }

    arbsdp_solve_default_params(&params);
    arbsdp_solve(&res, &p, prec, &params);

    /* (a) the correctly-recovered, sign-flipped value is +sqrt(2). */
    {
        char *vs = arb_get_str(res.value, 20, ARB_STR_NO_RADIUS);
        printf("\n  MUTATION (rule 8): sdp_sqrt2 (maximize) recovered value = %s.\n",
               vs);
        flint_free(vs);
    }
    CHECK(close_str(res.value, GOLD_SQRT2, 1e-6),
          "mutation_sign: maximize recovery must be +sqrt(2)");
    CHECK(!close_str(res.value, GOLD_NEG_SQRT2, 1e-6),
          "mutation_sign: maximize recovery must NOT be -sqrt(2) (sign flip is real)");

    /* (b) the UN-flipped internal objective pObj/tau IS -sqrt(2): this is what a
     *     missing maximize sign flip would have returned.  Computed from the
     *     result's final iterate (public handle, solve.h). */
    {
        arb_t internal;
        arb_init(internal);
        arb_div(internal, res.it.pObj, res.it.tau, prec);  /* NOT flipped */
        char *is = arb_get_str(internal, 20, ARB_STR_NO_RADIUS);
        printf("    un-flipped internal pObj/tau = %s (this is what a missing "
               "sign flip yields).\n", is);
        flint_free(is);
        CHECK(close_str(internal, GOLD_NEG_SQRT2, 1e-6),
              "mutation_sign: un-flipped internal objective must be -sqrt(2)");
        CHECK(!close_str(internal, GOLD_SQRT2, 1e-6),
              "mutation_sign: internal objective must NOT be +sqrt(2) (test discriminates)");
        arb_clear(internal);
    }

    arbsdp_result_clear(&res);
    arbsdp_problem_clear(&p);
}

/* ------------------------------------------------------------------------- *
 * Test 5 (bead arb-prec-IPM-epic.6): LOW-PREC OPTIMAL on a rank-deficient      *
 * boundary optimum.  trivial_2x2 (max x_11 s.t. x_11=1, X>=0; X*=diag(1,0) is  *
 * SINGULAR) at prec=256 with a TIGHT feas/opt tol of 1e-15 must reach          *
 * ARBSDP_SOLVE_OPTIMAL with the recovered value matching 1.0 to >= 15 digits.  *
 *                                                                              *
 * RED (unfixed solver): epic.6 / b30 diagnosis -- the Layer-0 point-mode loop  *
 * runs on arb BALLS whose untrusted radii (CLAUDE.md rule 1: NOT trusted in    *
 * L0) propagate; near convergence the Schur cond ~1/mu grows lambda_min's ball *
 * until it straddles zero, so arb_rsqrt/arb_inv in NT scaling return a NaN     *
 * MIDPOINT -> iterate_is_valid fails -> ARBSDP_SOLVE_NUMERICAL with a value    *
 * good to ~14 digits.  Confirmed: mu radius 1.1e-75 -> 3.8e-24 over 5 iters,   *
 * then iter 6 NaNs.  This assertion FAILS at status==NUMERICAL pre-fix.        *
 *                                                                              *
 * GREEN (after iterate_clear_radii): pure-midpoint arithmetic keeps every      *
 * lambda_min ball clean so rsqrt/inv never NaN; the strict 6-flag test fires   *
 * and the solver returns OPTIMAL at prec=256.                                  *
 * ------------------------------------------------------------------------- */
static void
test_epic6_lowprec_optimal(void)
{
    arbsdp_problem p;
    arbsdp_result res;
    arbsdp_solve_params params;
    char path[1024];
    int rc;
    slong prec = 256;

    arbsdp_problem_init(&p);
    snprintf(path, sizeof path, "%s/trivial_2x2/trivial_2x2.dat-s", GOLDEN_DIR);
    rc = arbsdp_read_sdpa(&p, path);
    CHECK(rc == 0, "epic6_lowprec_optimal: cannot read trivial_2x2.dat-s");
    if (rc != 0) { arbsdp_problem_clear(&p); return; }

    /* Direct (NON-adaptive) fixed-prec solve at prec=256, demanding 15 digits
     * (feas_tol = opt_tol = 1e-15) -- tighter than the 1e-8 defaults so the
     * 6-flag test actually requires full convergence at this precision. */
    arbsdp_solve_default_params(&params);
    params.feas_tol = 1e-15;
    params.opt_tol  = 1e-15;
    arbsdp_solve(&res, &p, prec, &params);

    printf("\n  EPIC.6 (low-prec OPTIMAL): trivial_2x2 @ prec=256, "
           "feas/opt tol=1e-15:\n");
    {
        char *vs = arb_get_str(res.value, 20, ARB_STR_NO_RADIUS);
        printf("    status=%s, iters=%d, recovered value=%s.\n",
               status_name(res.status), res.iters, vs);
        flint_free(vs);
    }

    /* (3) The unfixed solver returns NUMERICAL here; the fix makes it OPTIMAL. */
    CHECK(res.status == ARBSDP_SOLVE_OPTIMAL,
          "epic6_lowprec_optimal: trivial_2x2 @256 must reach OPTIMAL (epic.6 fix)");

    /* (4) Recovered value (already sign/maximize-flipped by recover_value) must
     *     match the analytic optimum 1.0 to >= 15 digits on the midpoint. */
    CHECK(close_str(res.value, GOLD_TRIVIAL, 1e-15),
          "epic6_lowprec_optimal: trivial_2x2 @256 value must match 1.0 to 1e-15");

    arbsdp_result_clear(&res);
    arbsdp_problem_clear(&p);
}

int
main(void)
{
    test_solve_golden();
    test_beat_or_bracket();
    test_mutation_sign();
    test_epic6_lowprec_optimal();

    if (failures != 0) {
        fprintf(stderr, "\ntest_solve: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("\nPASS: test_solve (end-to-end HSDE-NT Mehrotra solve)\n");
    return EXIT_SUCCESS;
}
