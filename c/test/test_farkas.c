/*
 * test/test_farkas.c -- BALL-MODE (RIGOROUS, Layer-1) tests for the verified
 * Farkas INFEASIBILITY certificate API (bead arb-prec-IPM-b20):
 *     arbsdp_verify_primal_infeasible, arbsdp_verify_dual_infeasible,
 *     and the top-level dispatcher arbsdp_certify (certify.h).
 *
 * ==========================================================================
 * RED-GREEN TDD (CLAUDE.md rule 9) -- THIS IS THE RED STEP
 * ==========================================================================
 * Written BEFORE the verifiers are implemented.  In the RED phase
 * arbsdp_verify_primal_infeasible / arbsdp_verify_dual_infeasible are STUBS that
 * return 0 ("not certified"), and arbsdp_certify on the infeasibility path is
 * wired to honest [-inf,+inf] + INCONCLUSIVE -- so the per-golden verify/certify
 * assertions FAIL LOUDLY.  After the GREEN step (the real Farkas verifiers) they
 * PASS.  The point-mode solve.status assertions and the feasible no-misfire
 * assertions already PASS in RED (the point-mode (tau,kappa) classification of
 * epic.2 is in place, and the stubs never misfire on a feasible problem).
 *
 * ==========================================================================
 * RIGOR, NOT A HEURISTIC (CLAUDE.md rule 2, invariant 8)
 * ==========================================================================
 * When the HSDE solve drives tau -> 0 the iterate is on the infeasibility path:
 * (tau, kappa) carry an infeasibility certificate, NOT an approximate optimum.
 * The USER-FACING status must then be a VERIFIED Farkas certificate checked in
 * BALL arithmetic (invariant 8), distinct from the POINT-mode (tau,kappa) steering
 * status that arbsdp_solve returns (test_infeasible.c covers that point-mode
 * class).  This test exercises the RIGOROUS verifiers and the unified certifier.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - certify.h: arbsdp_verify_primal_infeasible (MATH_SPEC §5.7.1; JCK 2007 §5
 *     eq 5.1, Jansson 2009 Prop 3.1: D^b = sum_i y_i A_i^b NSD for all b AND
 *     b^T y > 0 => primal infeasible); arbsdp_verify_dual_infeasible (§5.7.2;
 *     JCK 2007 §5 eq 5.3, Jansson 2009 Prop 3.2: a coordinate recession ray
 *     X_hat = E_kk with <A_i,X_hat>=0 for all i and <C,X_hat> > 0 => primal
 *     unbounded = dual infeasible); arbsdp_certify (§5.7.3: tau-dispatch, primal
 *     Farkas FIRST then dual).
 *   - The MEASURED verdicts (golden provenance.json + a direct probe of the solved
 *     firing iterates) are the table below; they are asserted as facts.
 *
 * Property coverage / DISCRIMINATORS (rule 9 -- prove the test BITES):
 *   - primal_infeasible_psd  is genuinely BOTH-infeasible: verify_primal AND
 *     verify_dual both 1 (E_22 is a coordinate recession ray of the relaxation
 *     AND y=-E_11 is a dual improving ray); arbsdp_certify reports PRIMAL_INFEASIBLE
 *     because the primal Farkas test is tried FIRST (§5.7.3).
 *   - SIGN DISCRIMINATOR (primal Farkas): the 4 dual goldens have b^T y of the
 *     WRONG sign for a primal Farkas ray, so verify_primal MUST be 0 on them -- a
 *     primal detector that ignored the b^T y > 0 sign would (wrongly) fire here.
 *   - COORDINATE DISCRIMINATOR (dual Farkas): primal_infeasible_minor is genuinely
 *     ONLY primal-infeasible (no coordinate recession ray: the off-diagonal pin
 *     X_12=2 leaves no E_kk feasible direction), so verify_dual MUST be 0 on it --
 *     a dual coordinate detector that was too loose would (wrongly) fire here.
 *   - FEASIBLE NO-MISFIRE: max_eigenvalue_2x2 (a healthy OPTIMAL solve) must yield
 *     verify_primal == verify_dual == 0 and arbsdp_certify NOT an infeasibility
 *     status -- no fabricated infeasibility certificate on a feasible problem.
 *   - FEASIBLE NO-MISFIRE BATTERY (embedded .dat-s, no new goldens): four
 *     feasible/bounded discriminators promoted from a throwaway probe to permanent
 *     regression guards (all must give vp=0 vd=0):
 *       (1) feasible_lp                   -- bounded LP baseline (no qualifying ray);
 *       (2) dual_filesign_discriminator   -- THE file-sign-C guard: the dual
 *           detector reads (C_file)_{22} = -1 < 0 (NOT the solver's C_int = +1), so
 *           a free coordinate does NOT fire; a regression to C_int WOULD misfire;
 *       (3) dual_Ckk_zero_boundary        -- the strict ">0" boundary: (C_file)_{22}
 *           lbound = 0 must NOT qualify;
 *       (4) feasible_offdiag_plus_trace   -- the trace constraint touches every
 *           diagonal, so no coordinate survives zero_ok (vd=0) and it is feasible
 *           (vp=0).
 *
 * Arb memory discipline (CLAUDE.md rule 7): every arbsdp_problem / arbsdp_result /
 * arbsdp_apriori and every arb_t is init'd and cleared; the problem OUTLIVES the
 * borrowing result/iterate.  The suite runs under ASan/valgrind.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <flint/arb.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/solve.h"
#include "arbsdp/certify.h"

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "../../golden"
#endif

#define SOLVE_PREC   256   /* fixed-precision Layer-0 solve                       */
#define VERIFY_PREC  384   /* Layer-1 certification precision (task spec)         */

static int failures = 0;

/* -------------------------------------------------------------------------
 * Status-name helpers (legible RED output).
 * ---------------------------------------------------------------------- */
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
certify_status_name(arbsdp_certify_status s)
{
    switch (s) {
    case ARBSDP_CERTIFY_OPTIMAL_BRACKET:   return "OPTIMAL_BRACKET";
    case ARBSDP_CERTIFY_INCONCLUSIVE:      return "INCONCLUSIVE";
    case ARBSDP_CERTIFY_PRIMAL_INFEASIBLE: return "PRIMAL_INFEASIBLE";
    case ARBSDP_CERTIFY_DUAL_INFEASIBLE:   return "DUAL_INFEASIBLE";
    default:                               return "?";
    }
}

/* -------------------------------------------------------------------------
 * expect_int -- assert got == want; print golden + field + expected vs got.
 * Increments `failures` and returns 0 on mismatch (legible RED output).
 * ---------------------------------------------------------------------- */
static int
expect_int(int got, int want, const char *golden, const char *field)
{
    if (got != want) {
        fprintf(stderr, "FAIL: %-24s %-18s expected %d, got %d (%s)\n",
                golden, field, want, got, __FILE__);
        failures++;
        return 0;
    }
    printf("  ok : %-24s %-18s = %d\n", golden, field, got);
    return 1;
}

/* Read a golden problem by name; returns 1 on success (p init'd & loaded).
 * (Mirrors test_infeasible.c load_golden exactly.) */
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

/* -------------------------------------------------------------------------
 * load_and_solve -- load golden `name` into `p`, run the fixed-precision driver
 * (arbsdp_solve @ SOLVE_PREC, default params) into `res`.  Both are caller-owned:
 * `res` borrows `p`, so the caller clears `res` BEFORE `p`.  Returns 1 on success
 * (p init'd+loaded, res init'd by arbsdp_solve).
 * ---------------------------------------------------------------------- */
static int
load_and_solve(arbsdp_problem *p, arbsdp_result *res, const char *name)
{
    arbsdp_solve_params sp;
    if (!load_golden(p, name))
        return 0;
    arbsdp_solve_default_params(&sp);
    arbsdp_solve(res, p, SOLVE_PREC, &sp);   /* INITIALIZES res */
    return 1;
}

/* -------------------------------------------------------------------------
 * Embedded-.dat-s helpers (mirror test_infeasible.c write_temp_dats /
 * load_embedded): write a PID-tagged tempfile under /tmp, parse it.  These let
 * the no-misfire battery below carry its problems INLINE (no new golden dirs).
 * ---------------------------------------------------------------------- */

/* write_temp_dats -- write `contents` to a unique temp path (filled into
 * path[cap]); returns 1 on success.  The caller removes the file afterwards. */
static int
write_temp_dats(const char *tag, const char *contents, char *path, size_t cap)
{
    FILE *f;
    snprintf(path, cap, "/tmp/arbsdp_test_farkas_%s_%d.dat-s", tag, (int) getpid());
    f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "FAIL: cannot open temp file %s\n", path);
        return 0;
    }
    fputs(contents, f);
    fclose(f);
    return 1;
}

/* load_and_solve_embedded -- write the embedded .dat-s `contents`, parse it into
 * `p`, then solve (arbsdp_solve @ `prec`, default params) into `res`.  Both are
 * caller-owned; `res` borrows `p`, so the caller clears `res` BEFORE `p` and
 * unlinks *path.  Returns 1 on success (p init'd+loaded, res init'd by solve). */
static int
load_and_solve_embedded(arbsdp_problem *p, arbsdp_result *res, const char *tag,
                        const char *contents, char *path, size_t cap, slong prec)
{
    arbsdp_solve_params sp;
    arbsdp_problem_init(p);
    if (!write_temp_dats(tag, contents, path, cap)) {
        arbsdp_problem_clear(p);
        return 0;
    }
    if (arbsdp_read_sdpa(p, path) != 0) {
        fprintf(stderr, "FAIL: cannot parse embedded %s (%s)\n", tag, path);
        arbsdp_problem_clear(p);
        remove(path);
        return 0;
    }
    arbsdp_solve_default_params(&sp);
    arbsdp_solve(res, p, prec, &sp);   /* INITIALIZES res */
    return 1;
}

/* tau midpoint as a double (diagnostic only).  Uses a local arb copy to avoid
 * the arb_midref(it->tau) alias that trips a GCC -O2 -Wstringop-overread false
 * positive (the same idiom certify.c uses). */
static double
tau_mid_d(const arbsdp_iterate *it)
{
    arb_t t; double d;
    arb_init(t);
    arb_get_mid_arb(t, it->tau);
    d = arf_get_d(arb_midref(t), ARF_RND_DOWN);
    arb_clear(t);
    return d;
}

/* ==========================================================================
 * The 6 infeasible goldens + their MEASURED rigorous verdicts (the contract).
 * ======================================================================== */
typedef struct {
    const char           *name;
    arbsdp_solve_status   solve_class;     /* point-mode (tau,kappa) class (epic.2) */
    int                   verify_primal;   /* expected arbsdp_verify_primal_infeasible */
    int                   verify_dual;     /* expected arbsdp_verify_dual_infeasible   */
    arbsdp_certify_status certify;         /* expected arbsdp_certify status           */
} farkas_case;

static const farkas_case CASES[] = {
    /* BOTH-infeasible: vp=1 AND vd=1; certify reports PRIMAL (tried first, §5.7.3). */
    { "primal_infeasible_psd",   ARBSDP_SOLVE_PRIMAL_INFEASIBLE, 1, 1,
      ARBSDP_CERTIFY_PRIMAL_INFEASIBLE },
    /* ONLY primal-infeasible: vd=0 is the COORDINATE DISCRIMINATOR (no E_kk ray). */
    { "primal_infeasible_minor", ARBSDP_SOLVE_PRIMAL_INFEASIBLE, 1, 0,
      ARBSDP_CERTIFY_PRIMAL_INFEASIBLE },
    /* dual-infeasible (primal unbounded): vp=0 is the SIGN DISCRIMINATOR (b^T y). */
    { "unbounded_eig",           ARBSDP_SOLVE_DUAL_INFEASIBLE,   0, 1,
      ARBSDP_CERTIFY_DUAL_INFEASIBLE },
    { "unbounded_lp",            ARBSDP_SOLVE_DUAL_INFEASIBLE,   0, 1,
      ARBSDP_CERTIFY_DUAL_INFEASIBLE },
    { "dual_infeasible_diag",    ARBSDP_SOLVE_DUAL_INFEASIBLE,   0, 1,
      ARBSDP_CERTIFY_DUAL_INFEASIBLE },
    { "dual_infeasible_3x3",     ARBSDP_SOLVE_DUAL_INFEASIBLE,   0, 1,
      ARBSDP_CERTIFY_DUAL_INFEASIBLE },
};

/* ==========================================================================
 * test_farkas_table -- the 6-golden contract.  For each golden: solve, assert the
 * point-mode infeasible class (already GREEN), then assert the RIGOROUS verifiers
 * and the unified certifier match the table (RED until the verifiers land).
 * ======================================================================== */
static void
test_farkas_table(void)
{
    int ncases = (int) (sizeof CASES / sizeof CASES[0]);

    printf("\nFARKAS TABLE (rigorous verifiers + arbsdp_certify; BALL, prec=%d):\n",
           VERIFY_PREC);
    printf("  %-24s solve.status   vp vd  certify\n", "golden");

    for (int k = 0; k < ncases; k++) {
        const farkas_case *fc = &CASES[k];
        arbsdp_problem p;
        arbsdp_result  res;

        if (!load_and_solve(&p, &res, fc->name)) {
            fprintf(stderr, "FAIL: %s load+solve\n", fc->name);
            failures++;
            continue;
        }

        int vp = arbsdp_verify_primal_infeasible(&p, &res.it, VERIFY_PREC);
        int vd = arbsdp_verify_dual_infeasible(&p, &res.it, VERIFY_PREC);

        arbsdp_apriori ab;
        arbsdp_apriori_init(&ab, p.nblocks);   /* no xbar: infeasibility path */
        arb_t lb, ub;
        arb_init(lb); arb_init(ub);
        arbsdp_certify_status cs =
            arbsdp_certify(lb, ub, &p, &res.it, &ab, VERIFY_PREC);

        /* Diagnostic line (measured) BEFORE the assertions, so the RED output is
         * legible against the expected table (CLAUDE.md rule 11). */
        printf("  %-24s %-14s %d  %d  %-16s (tau=%.3e)\n",
               fc->name, solve_status_name(res.status), vp, vd,
               certify_status_name(cs), tau_mid_d(&res.it));

        /* Point-mode steering class (epic.2 -- already GREEN). */
        expect_int((int) res.status, (int) fc->solve_class,
                   fc->name, "solve.status");
        /* RIGOROUS verifiers (RED: stubs return 0). */
        expect_int(vp, fc->verify_primal, fc->name, "verify_primal");
        expect_int(vd, fc->verify_dual,   fc->name, "verify_dual");
        /* Unified certifier (RED: infeasibility path returns INCONCLUSIVE stub). */
        expect_int((int) cs, (int) fc->certify, fc->name, "certify.status");

        arb_clear(lb); arb_clear(ub);
        arbsdp_apriori_clear(&ab);
        arbsdp_result_clear(&res);   /* res borrows p -> clear res BEFORE p */
        arbsdp_problem_clear(&p);
    }
}

/* ==========================================================================
 * test_feasible_no_misfire -- a FEASIBLE problem must NOT yield any infeasibility
 * certificate (CLAUDE.md rule 2: never a fabricated/false certificate).
 *
 * max_eigenvalue_2x2 solves to a healthy OPTIMAL (tau ~ 12, well above the
 * infeasibility path).  Both rigorous verifiers must return 0, and arbsdp_certify
 * must NOT return PRIMAL_INFEASIBLE / DUAL_INFEASIBLE (it is OPTIMAL_BRACKET or,
 * with xbar unset, INCONCLUSIVE -- either is acceptable; only an infeasibility
 * verdict would be a misfire).  These assertions already PASS in RED (the stubs
 * return 0 and arbsdp_certify delegates a healthy tau to the bracket path).
 * ======================================================================== */
static void
test_feasible_no_misfire(void)
{
    const char *name = "max_eigenvalue_2x2";
    arbsdp_problem p;
    arbsdp_result  res;

    printf("\nFEASIBLE NO-MISFIRE (%s must NOT certify infeasible):\n", name);

    if (!load_and_solve(&p, &res, name)) {
        fprintf(stderr, "FAIL: %s load+solve\n", name);
        failures++;
        return;
    }

    int vp = arbsdp_verify_primal_infeasible(&p, &res.it, VERIFY_PREC);
    int vd = arbsdp_verify_dual_infeasible(&p, &res.it, VERIFY_PREC);

    arbsdp_apriori ab;
    arbsdp_apriori_init(&ab, p.nblocks);   /* no xbar */
    arb_t lb, ub;
    arb_init(lb); arb_init(ub);
    arbsdp_certify_status cs =
        arbsdp_certify(lb, ub, &p, &res.it, &ab, VERIFY_PREC);

    printf("  %-24s %-14s vp=%d vd=%d certify=%s (tau=%.3e)\n",
           name, solve_status_name(res.status), vp, vd,
           certify_status_name(cs), tau_mid_d(&res.it));

    expect_int((int) res.status, (int) ARBSDP_SOLVE_OPTIMAL, name, "solve.status");
    expect_int(vp, 0, name, "verify_primal");
    expect_int(vd, 0, name, "verify_dual");

    int is_infeasible_verdict = (cs == ARBSDP_CERTIFY_PRIMAL_INFEASIBLE ||
                                 cs == ARBSDP_CERTIFY_DUAL_INFEASIBLE);
    if (is_infeasible_verdict) {
        fprintf(stderr, "FAIL: %-24s %-18s MISFIRE: certify=%s on a FEASIBLE problem (%s)\n",
                name, "certify.status", certify_status_name(cs), __FILE__);
        failures++;
    } else {
        printf("  ok : %-24s %-18s = %s (not an infeasibility verdict)\n",
               name, "certify.status", certify_status_name(cs));
    }

    arb_clear(lb); arb_clear(ub);
    arbsdp_apriori_clear(&ab);
    arbsdp_result_clear(&res);
    arbsdp_problem_clear(&p);
}

/* ==========================================================================
 * test_feasible_no_misfire_battery -- a battery of FEASIBLE/BOUNDED problems
 * carried as embedded .dat-s, each a no-misfire / discriminator guard that the
 * adversarial review found only in a throwaway probe (bead arb-prec-IPM-b20).
 * For EVERY case BOTH rigorous verifiers MUST return 0: a verified Farkas
 * certificate on a feasible/bounded problem would be a fabricated infeasibility
 * proof (CLAUDE.md rule 2 -- the worst possible output).  vp/vd are properties of
 * the problem DATA (verify_dual) / the dual ray (verify_primal) and the Farkas
 * alternative is EXCLUSIVE of feasibility, so a rigorous-and-conservative verifier
 * CANNOT return 1 here; these cases prove it does not, on shapes that bite.
 *
 * SDPA .dat-s (max <C,X>): line1=m, line2=nblocks, line3=block sizes (neg=LP/
 * diagonal), line4=b vector, then `matno blockno i j value` (matno 0 = C, 1..m =
 * A_i; 1<=i<=j).  Whitespace/format modelled on golden/dual_infeasible_diag.
 * ======================================================================== */

/* (1) feasible_lp: bounded LP.  max x1 s.t. x1+x2 = 1, x>=0 (opt 1).  A_1 =
 * diag(1,1), C = diag(1,0) on a size -2 (LP/diagonal) block.  A plain feasible
 * baseline -- neither verifier has any qualifying coordinate/ray. */
static const char DATS_FEAS_LP[] =
    "* feasible_lp: bounded LP. max x1 s.t. x1+x2 = 1, x>=0 (opt 1).\n"
    "1\n"
    "1\n"
    "-2\n"
    "1.0\n"
    "0 1 1 1 1.0\n"
    "1 1 1 1 1.0\n"
    "1 1 2 2 1.0\n";

/* (2) dual_filesign_discriminator -- THE KEY GUARD.  max -X_22 s.t. X_11 = 1 on
 * a 2x2 PSD block (FEASIBLE, BOUNDED; opt 0).  A_1 = E_11, C = diag(0,-1).
 *
 * Coordinate k=1 (0-indexed) is free of A_1's diagonal ((A_1)_{22} = 0), so the
 * dual detector's zero_ok[1] SURVIVES the constraint sweep.  The ONLY thing that
 * keeps verify_dual from (wrongly) firing DUAL_INFEASIBLE here is that step (b)
 * reads C at its FILE sign: (C_file)_{22} = -1 < 0, so lbound > 0 is FALSE and no
 * coordinate qualifies -> vd MUST be 0.  If a future regression makes the detector
 * read the solver's INTERNAL C_int = -C_file (= +1 at (2,2)), coordinate k=1 WOULD
 * qualify and verify_dual WOULD fire a SPURIOUS DUAL_INFEASIBLE on this feasible
 * problem.  This is the file-sign-C rigor guard (certify.c reads block_mat matno 0
 * = C_file, NOT C_int); vd=0 here is load-bearing. */
static const char DATS_DUAL_FILESIGN[] =
    "* dual_filesign_discriminator: max -X_22 s.t. X_11 = 1 (2x2 PSD; opt 0).\n"
    "* file-sign-C guard: (C_file)_22 = -1 < 0 keeps verify_dual from firing.\n"
    "1\n"
    "1\n"
    "2\n"
    "1.0\n"
    "0 1 2 2 -1.0\n"
    "1 1 1 1 1.0\n";

/* (3) dual_Ckk_zero_boundary: max 0 s.t. X_11 = 1 (C = 0) on a 2x2 PSD block
 * (FEASIBLE, BOUNDED; opt 0).  A_1 = E_11, no C entries.  Free coordinate k=1 has
 * (C_file)_{22} lbound = 0, NOT strictly > 0 -> the dual detector's strict ">0"
 * test must REJECT it (probes the exact ">0" boundary).  vd MUST be 0. */
static const char DATS_DUAL_CKK_ZERO[] =
    "* dual_Ckk_zero_boundary: max 0 s.t. X_11 = 1, C = 0 (2x2 PSD; opt 0).\n"
    "* strict-positivity boundary: (C_file)_22 lbound = 0, NOT > 0 -> no fire.\n"
    "1\n"
    "1\n"
    "2\n"
    "1.0\n"
    "1 1 1 1 1.0\n";

/* (4) feasible_offdiag_plus_trace: max X_11 s.t. 2*X_12 = 0 AND tr X = 1 on a
 * 2x2 PSD block (FEASIBLE, BOUNDED; tr X = 1 caps X_11, opt 1).  A_1 = [[0,1],
 * [1,0]], A_2 = I, C = E_11.  Coordinate k=0 is killed by (A_2)_{00} = 1 != 0
 * (the trace constraint touches every diagonal), so NO coordinate qualifies for
 * the dual detector -> vd=0; and the problem is feasible -> vp=0. */
static const char DATS_FEAS_OFFDIAG_TRACE[] =
    "* feasible_offdiag_plus_trace: max X_11 s.t. 2*X_12 = 0 AND tr X = 1 (opt 1).\n"
    "2\n"
    "1\n"
    "2\n"
    "0.0 1.0\n"
    "0 1 1 1 1.0\n"
    "1 1 1 2 1.0\n"
    "2 1 1 1 1.0\n"
    "2 1 2 2 1.0\n";

typedef struct {
    const char *name;   /* tempfile tag + report label                          */
    const char *dats;   /* embedded SDPA-sparse .dat-s                           */
} feasible_case;

static const feasible_case FEAS_CASES[] = {
    { "feasible_lp",                  DATS_FEAS_LP },
    { "dual_filesign_discriminator",  DATS_DUAL_FILESIGN },
    { "dual_Ckk_zero_boundary",       DATS_DUAL_CKK_ZERO },
    { "feasible_offdiag_plus_trace",  DATS_FEAS_OFFDIAG_TRACE },
};

static void
test_feasible_no_misfire_battery(void)
{
    int ncases = (int) (sizeof FEAS_CASES / sizeof FEAS_CASES[0]);

    printf("\nFEASIBLE NO-MISFIRE BATTERY (embedded .dat-s; vp=vd=0 guards; "
           "BALL prec=%d):\n", VERIFY_PREC);
    printf("  %-30s solve.status   vp vd\n", "case");

    for (int k = 0; k < ncases; k++) {
        const feasible_case *fc = &FEAS_CASES[k];
        arbsdp_problem p;
        arbsdp_result  res;
        char path[256];
        slong prec = SOLVE_PREC;

        if (!load_and_solve_embedded(&p, &res, fc->name, fc->dats,
                                     path, sizeof path, prec)) {
            fprintf(stderr, "FAIL: %s load+solve\n", fc->name);
            failures++;
            continue;
        }

        /* Sanity: the solve should converge.  If prec 256 did not reach OPTIMAL,
         * re-solve at 512 (task spec) -- the verifier no-misfire below is the real
         * assertion, the solve.status is only a convergence sanity check. */
        if (res.status != ARBSDP_SOLVE_OPTIMAL) {
            arbsdp_solve_params sp;
            arbsdp_result_clear(&res);
            prec = 512;
            arbsdp_solve_default_params(&sp);
            arbsdp_solve(&res, &p, prec, &sp);
        }

        int vp = arbsdp_verify_primal_infeasible(&p, &res.it, VERIFY_PREC);
        int vd = arbsdp_verify_dual_infeasible(&p, &res.it, VERIFY_PREC);

        printf("  %-30s %-14s %d  %d  (prec=%ld, tau=%.3e)\n",
               fc->name, solve_status_name(res.status), vp, vd,
               (long) prec, tau_mid_d(&res.it));

        /* THE REAL ASSERTION: no spurious infeasibility certificate (rule 2). */
        expect_int(vp, 0, fc->name, "verify_primal");
        expect_int(vd, 0, fc->name, "verify_dual");

        /* Convergence sanity.  OPTIMAL is expected; if not OPTIMAL even at prec
         * 512, relax to "not an infeasibility class" (a misfire of the point-mode
         * steering status would itself be a regression worth catching). */
        if (res.status == ARBSDP_SOLVE_OPTIMAL) {
            expect_int((int) res.status, (int) ARBSDP_SOLVE_OPTIMAL,
                       fc->name, "solve.status");
        } else {
            int is_inf_class = (res.status == ARBSDP_SOLVE_PRIMAL_INFEASIBLE ||
                                res.status == ARBSDP_SOLVE_DUAL_INFEASIBLE);
            if (is_inf_class) {
                fprintf(stderr, "FAIL: %-30s %-18s MISFIRE: %s on a FEASIBLE "
                        "problem (%s)\n", fc->name, "solve.status",
                        solve_status_name(res.status), __FILE__);
                failures++;
            } else {
                printf("  ok : %-30s %-18s = %s (not OPTIMAL at prec %ld but not an "
                       "infeasibility class; verifier no-misfire is the real "
                       "assertion)\n", fc->name, "solve.status",
                       solve_status_name(res.status), (long) prec);
            }
        }

        arbsdp_result_clear(&res);   /* res borrows p -> clear res BEFORE p */
        arbsdp_problem_clear(&p);
        remove(path);
    }
}

/* ==========================================================================
 * test_minimize_refused -- bead arb-prec-IPM-agc (REVIEW F3): the MAXIMIZE
 * GUARD.  Layer 1 is derived for the EXTERNAL MAX problem only (MATH_SPEC §5.4,
 * §5.5, §5.7.2; certify.h).  A minimize problem (p->maximize == 0) must NEVER
 * receive a certifying status from an objective-sense-dependent entry point --
 * a wrong-objective "rigorous" bracket / recession-ray verdict is the worst
 * possible output (CLAUDE.md rule 2/5).  The real minimize certification is
 * bead arb-prec-IPM-fjw; until it lands the honest answer is INCONCLUSIVE.
 *
 * (A) min_trace_diag13: C = diag(1,3), tr X = 1 (2x2 PSD).  min = 1 (X = E_11),
 *     max = 3.  Solved as MINIMIZE.  Without the guard the oc7 primal Rayleigh
 *     tightening (a FEASIBLE point's objective, a lower bound only for a MAX)
 *     sets lb = <C, X_hat> >= 1 = the true min, so the "bracket" can EXCLUDE
 *     the minimize optimum.  Asserted: certify_bracket and certify return
 *     INCONCLUSIVE with [-inf, +inf]; primal_lower_bound returns 0.
 *     CONTROL (the test is not vacuous): the same data with maximize = 1 gives
 *     OPTIMAL_BRACKET containing 3.
 * (B) min_free_coord: min X_22 s.t. X_11 = 1 (2x2 PSD; C = diag(0,1)).  BOUNDED
 *     (opt 0) and dual-feasible (y = 0 gives C >= 0).  E_22 is a recession ray
 *     with <C,E_22> = +1 > 0 -- an UNBOUNDED direction for a MAX, an IMPROVING-
 *     AWAY direction for a MIN.  Without the guard verify_dual fires 1 and
 *     certify reports DUAL_INFEASIBLE: a FALSE infeasibility certificate.
 *     Asserted: verify_dual == 0, certify == INCONCLUSIVE.  CONTROL: the same
 *     data with maximize = 1 (max X_22, genuinely unbounded) gives verify_dual 1.
 * (C) SENSE-INDEPENDENCE of the primal Farkas test (§5.7.1): primal
 *     infeasibility is a property of {X >= 0 : <A_i,X> = b_i} alone; the
 *     certificate (sum y_i A_i <= 0, b^T y > 0) never reads C or the sense.  So
 *     primal_infeasible_psd with maximize flipped to 0 must STILL verify, and
 *     arbsdp_certify must still report PRIMAL_INFEASIBLE (the guard must not
 *     over-refuse a valid sense-free theorem).
 * ======================================================================== */
static const char DATS_MIN_TRACE[] =
    "* min_trace_diag13: opt C=diag(1,3) s.t. tr X = 1 (min 1, max 3).\n"
    "1\n"
    "1\n"
    "2\n"
    "1.0\n"
    "0 1 1 1 1.0\n"
    "0 1 2 2 3.0\n"
    "1 1 1 1 1.0\n"
    "1 1 2 2 1.0\n";

static const char DATS_MIN_FREE_COORD[] =
    "* min_free_coord: opt X_22 s.t. X_11 = 1 (min 0 bounded; max unbounded).\n"
    "1\n"
    "1\n"
    "2\n"
    "1.0\n"
    "0 1 2 2 1.0\n"
    "1 1 1 1 1.0\n";

/* Load embedded `contents` into p with the given sense, solve at SOLVE_PREC. */
static int
load_and_solve_sense(arbsdp_problem *p, arbsdp_result *res, const char *tag,
                     const char *contents, int maximize, char *path, size_t cap)
{
    arbsdp_solve_params sp;
    arbsdp_problem_init(p);
    if (!write_temp_dats(tag, contents, path, cap)) {
        arbsdp_problem_clear(p);
        return 0;
    }
    if (arbsdp_read_sdpa(p, path) != 0) {
        fprintf(stderr, "FAIL: cannot parse embedded %s (%s)\n", tag, path);
        arbsdp_problem_clear(p);
        remove(path);
        return 0;
    }
    p->maximize = maximize;               /* read_sdpa hardcodes 1 */
    arbsdp_solve_default_params(&sp);
    arbsdp_solve(res, p, SOLVE_PREC, &sp);   /* INITIALIZES res */
    return 1;
}

static void
test_minimize_refused(void)
{
    char path[256];

    printf("\nMINIMIZE GUARD (bead agc: maximize=0 must NOT be certified):\n");

    /* ---- (A) bracket path ------------------------------------------------ */
    for (int maximize = 0; maximize <= 1; maximize++) {
        arbsdp_problem p;
        arbsdp_result  res;
        const char *lbl = maximize ? "min_trace(CONTROL max)" : "min_trace(min)";

        if (!load_and_solve_sense(&p, &res, "min_trace", DATS_MIN_TRACE, maximize,
                                  path, sizeof path)) {
            failures++;
            continue;
        }

        arbsdp_apriori ab;
        arb_t lb, ub, one, lbp, opt;
        arbsdp_apriori_init(&ab, p.nblocks);
        arb_init(lb); arb_init(ub); arb_init(one); arb_init(lbp); arb_init(opt);
        arb_one(one);
        arbsdp_apriori_set_xbar(&ab, 0, one);   /* tr X = 1 exactly */

        arbsdp_certify_status csb =
            arbsdp_certify_bracket(lb, ub, &p, &res.it, &ab, VERIFY_PREC);
        {
            char *slb = arb_get_str(lb, 20, 0);
            char *sub = arb_get_str(ub, 20, 0);
            printf("  %-24s solve=%s certify_bracket=%s lb=%s ub=%s\n", lbl,
                   solve_status_name(res.status), certify_status_name(csb),
                   slb, sub);
            flint_free(slb);
            flint_free(sub);
        }
        arbsdp_certify_status cs =
            arbsdp_certify(lb, ub, &p, &res.it, &ab, VERIFY_PREC);
        int plb = arbsdp_primal_lower_bound(lbp, &p, &res.it, VERIFY_PREC);

        if (!maximize) {
            expect_int((int) res.status, (int) ARBSDP_SOLVE_OPTIMAL, lbl,
                       "solve.status");
            expect_int((int) csb, (int) ARBSDP_CERTIFY_INCONCLUSIVE, lbl,
                       "certify_bracket");
            expect_int((int) cs, (int) ARBSDP_CERTIFY_INCONCLUSIVE, lbl,
                       "certify.status");
            expect_int(arb_is_finite(lb) || arb_is_finite(ub), 0, lbl,
                       "lb/ub finite");
            expect_int(plb, 0, lbl, "primal_lower_bnd");
        } else {
            /* CONTROL: max opt = 3 must be bracketed (the guard is sense-gated,
             * not a blanket refusal). */
            arb_set_si(opt, 3);
            expect_int((int) cs, (int) ARBSDP_CERTIFY_OPTIMAL_BRACKET, lbl,
                       "certify.status");
            expect_int(arb_le(lb, opt) && arb_le(opt, ub), 1, lbl,
                       "lb<=3<=ub");
            expect_int(plb, 1, lbl, "primal_lower_bnd");
        }

        arb_clear(lb); arb_clear(ub); arb_clear(one); arb_clear(lbp);
        arb_clear(opt);
        arbsdp_apriori_clear(&ab);
        arbsdp_result_clear(&res);
        arbsdp_problem_clear(&p);
        remove(path);
    }

    /* ---- (B) dual-infeasibility (recession-ray) path -------------------- */
    {
        arbsdp_problem p;
        arbsdp_result  res;
        const char *lbl = "min_free_coord";

        if (load_and_solve_sense(&p, &res, "min_free_coord", DATS_MIN_FREE_COORD,
                                 0, path, sizeof path)) {
            arbsdp_apriori ab;
            arb_t lb, ub;
            arbsdp_apriori_init(&ab, p.nblocks);
            arb_init(lb); arb_init(ub);

            int vd = arbsdp_verify_dual_infeasible(&p, &res.it, VERIFY_PREC);
            arbsdp_certify_status cs =
                arbsdp_certify(lb, ub, &p, &res.it, &ab, VERIFY_PREC);
            printf("  %-24s solve=%s vd=%d certify=%s\n", lbl,
                   solve_status_name(res.status), vd, certify_status_name(cs));
            expect_int(vd, 0, lbl, "verify_dual(min)");
            expect_int((int) cs, (int) ARBSDP_CERTIFY_INCONCLUSIVE, lbl,
                       "certify.status");

            /* CONTROL: as a MAX the same data IS unbounded (E_22 ray); the
             * verifier is data-only, so flip the flag on the same iterate. */
            p.maximize = 1;
            vd = arbsdp_verify_dual_infeasible(&p, &res.it, VERIFY_PREC);
            expect_int(vd, 1, "min_free_coord(CONTROL)", "verify_dual(max)");

            arb_clear(lb); arb_clear(ub);
            arbsdp_apriori_clear(&ab);
            arbsdp_result_clear(&res);
            arbsdp_problem_clear(&p);
            remove(path);
        } else {
            failures++;
        }
    }

    /* ---- (C) primal Farkas is sense-independent ------------------------- */
    {
        arbsdp_problem p;
        arbsdp_result  res;
        const char *lbl = "primal_infeasible_psd(min)";

        if (load_and_solve(&p, &res, "primal_infeasible_psd")) {
            arbsdp_apriori ab;
            arb_t lb, ub;
            arbsdp_apriori_init(&ab, p.nblocks);
            arb_init(lb); arb_init(ub);

            p.maximize = 0;   /* same feasible set, same Farkas ray */
            int vp = arbsdp_verify_primal_infeasible(&p, &res.it, VERIFY_PREC);
            arbsdp_certify_status cs =
                arbsdp_certify(lb, ub, &p, &res.it, &ab, VERIFY_PREC);
            printf("  %-24s vp=%d certify=%s\n", lbl, vp, certify_status_name(cs));
            expect_int(vp, 1, lbl, "verify_primal");
            expect_int((int) cs, (int) ARBSDP_CERTIFY_PRIMAL_INFEASIBLE, lbl,
                       "certify.status");

            arb_clear(lb); arb_clear(ub);
            arbsdp_apriori_clear(&ab);
            arbsdp_result_clear(&res);
            arbsdp_problem_clear(&p);
        } else {
            failures++;
        }
    }
}

int
main(void)
{
    printf("test_farkas: RIGOROUS Farkas infeasibility certificates (bead b20)\n");

    test_farkas_table();
    test_feasible_no_misfire();
    test_feasible_no_misfire_battery();
    test_minimize_refused();

    /* Free FLINT's thread-local caches (fmpz pool, arf scratch, registered
     * cleanup hooks) so valgrind reports no possibly-lost / reachable
     * blocks at exit (bead arb-prec-IPM-7o8 memcheck gate). */
    flint_cleanup_master();

    if (failures == 0) {
        printf("\nAll test_farkas checks passed.\n");
        return EXIT_SUCCESS;
    }
    fprintf(stderr, "\ntest_farkas: %d check(s) FAILED (expected in the RED step).\n",
            failures);
    return EXIT_FAILURE;
}
