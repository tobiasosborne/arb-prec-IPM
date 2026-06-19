/*
 * test/test_precision.c -- tests for the adaptive precision-escalation driver
 * (c/src/precision.c, bead arb-prec-IPM-b16).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written BEFORE c/src/precision.c carries an
 * implementation; the target first fails to LINK (undefined symbols
 * arbsdp_solve_adaptive / arbsdp_precision_default_params /
 * arbsdp_adaptive_result_clear), then passes once precision.c lands.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - c/include/arbsdp/precision.h (the contract this file pins).
 *   - docs/MATH_SPEC.md §6 (precision controller), §10.5 (stall = muNew>0.99*mu,
 *     K=10).  PRD.md §4.3 (adaptive precision: start / escalate / caps).
 *   - The fixed-precision driver it wraps: c/src/solve.c (b15), c/include/.../solve.h.
 *   - golden/sdp_sqrt2/provenance.json -- the 65-digit analytic optimum sqrt(2).
 *
 * AUDITION (recorded in precision.h banner): geometric DOUBLING was chosen over
 * the model-based and hybrid escalation strategies.  On sdp_sqrt2 reaching a
 * 20-digit OPTIMAL from prec0=128, doubling cost 6 solves / 8064 bits / 45 iters;
 * the pure model-based strategy cost 79 solves / 159030 bits / 815 iters (20x);
 * the hybrid tied doubling (6 / 7878 / 44) for a fragile model term.  Doubling
 * wins on robustness with no meaningful cost penalty (see precision.h).
 *
 * HONEST CEILING (CLAUDE.md rule 5, measured in the audition): the point-mode
 * driver on sdp_sqrt2 caps OPTIMAL accuracy at ~28-29 digits (the irrational
 * rank-deficient boundary optimum drives mu out of the cone near mu~1e-28 before
 * a tighter gap test fires).  We therefore target a HIGH but reachable accuracy
 * (>=20 digits, OPTIMAL at prec~3072-4096) for the motivating case; we never
 * loosen a claim and never report a false OPTIMAL when the target is unreachable.
 *
 * POINT MODE (CLAUDE.md rule 1): the recovered value is a midpoint; the adaptive
 * layer only improves the APPROXIMATE solve.  All rigor is Layer 1 (b18/b19).
 *
 * Property coverage (bead arb-prec-IPM-b16):
 *   1. MOTIVATING CASE: sdp_sqrt2 from prec0=128 reaches OPTIMAL and matches
 *      sqrt(2) to >= the target digits BY ESCALATION (prec_history len > 1).
 *   2. EASY CASE no over-escalation: trivial_2x2 converges at/near prec0 with a
 *      short prec_history.
 *   3. CAP -> HONEST LIMIT: prec_max absurdly low (32) so sdp_sqrt2 cannot
 *      converge -> honest LIMIT status (NOT a false OPTIMAL); history stops at cap.
 *   4. TARGET DIGITS MET: request N digits; achieved accuracy >= N.
 *   5. MUTATION (rule 8): catches "never escalates" (returns prec0 NUMERICAL) and
 *      "claims OPTIMAL at the cap without meeting target".
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear; the
 * adaptive result is always arbsdp_adaptive_result_clear'd regardless of status.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <flint/arb.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/solve.h"
#include "arbsdp/precision.h"

#define WORKPREC 256  /* precision for the test's own arb comparisons (point mode) */

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

#define GOLD_SQRT2   "1.4142135623730950488016887242096980785696718753769480731766797379"
#define GOLD_TRIVIAL "1.0000000000000000000000000000000000000000000000000000000000000000"

/*
 * digits_matched -- |mid(recovered) - golden| as matching decimal digits
 * (POINT-mode diagnostic, mirrors test_solve.c).  -log10(|err|/max(1,|golden|)).
 */
static double
digits_matched(const arb_t recovered, const char *golden_str)
{
    arb_t g, err, rel, one;
    double d;
    arb_init(g); arb_init(err); arb_init(rel); arb_init(one);

    arb_get_mid_arb(err, recovered);
    arb_set_str(g, golden_str, WORKPREC);
    arb_sub(err, err, g, WORKPREC);
    arb_abs(err, err);

    arb_abs(rel, g);
    arb_one(one);
    if (arb_lt(rel, one)) arb_set(rel, one);
    arb_div(err, err, rel, WORKPREC);

    if (arb_is_zero(err) || !arb_is_finite(err)) {
        d = 1e9;
    } else {
        double e = arf_get_d(arb_midref(err), ARF_RND_NEAR);
        d = (e <= 0.0) ? 1e9 : -log10(e);
    }
    arb_clear(g); arb_clear(err); arb_clear(rel); arb_clear(one);
    return d;
}

static const char *
solve_status_name(arbsdp_solve_status s)
{
    switch (s) {
    case ARBSDP_SOLVE_OPTIMAL:    return "OPTIMAL";
    case ARBSDP_SOLVE_ITER_LIMIT: return "ITER_LIMIT";
    case ARBSDP_SOLVE_STALL:      return "STALL";
    case ARBSDP_SOLVE_NUMERICAL:  return "NUMERICAL";
    default:                      return "?";
    }
}

static const char *
adaptive_status_name(arbsdp_adaptive_status s)
{
    switch (s) {
    case ARBSDP_ADAPTIVE_OPTIMAL: return "ADAPTIVE_OPTIMAL";
    case ARBSDP_ADAPTIVE_LIMIT:   return "ADAPTIVE_LIMIT";
    default:                      return "?";
    }
}

/* Read a golden problem; returns 1 on success (p left init'd & loaded). */
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

/* Pretty-print a result's escalation trace. */
static void
print_trace(const char *label, const arbsdp_adaptive_result *r, const char *gold)
{
    printf("\n  %s trace (%d precisions tried):\n", label, r->prec_history_len);
    printf("    %-3s | %-7s | %-11s | %-12s | %s\n",
           "#", "prec", "status", "mu", "digits");
    printf("    -----------------------------------------------------------\n");
    for (int i = 0; i < r->prec_history_len; i++) {
        printf("    %-3d | %-7ld | %-11s | %-12.3e | %.1f\n",
               i, (long) r->prec_history[i].prec,
               solve_status_name(r->prec_history[i].status),
               r->prec_history[i].mu, r->prec_history[i].digits);
    }
    {
        double d = digits_matched(r->res.value, gold);
        char *vs = arb_get_str(r->res.value, 30, ARB_STR_NO_RADIUS);
        printf("    final: status=%s value=%s (%.1f digits vs golden)\n",
               adaptive_status_name(r->status), vs, d);
        flint_free(vs);
    }
}

/* ------------------------------------------------------------------------- *
 * Test 1: MOTIVATING CASE -- sdp_sqrt2 from prec0=128 auto-escalates to        *
 * OPTIMAL matching sqrt(2) to >= target digits; prec_history shows escalation. *
 * ------------------------------------------------------------------------- */
static void
test_motivating_sqrt2(void)
{
    arbsdp_problem p;
    arbsdp_adaptive_result r;
    arbsdp_precision_params pp;
    int target = 20;   /* HIGH accuracy, reachable OPTIMAL (audition ceiling ~28d) */
    double d;

    if (!load_golden(&p, "sdp_sqrt2")) { CHECK(0, "motivating: read"); return; }

    arbsdp_precision_default_params(&pp);
    pp.prec0         = 128;
    pp.prec_max      = 8192;
    pp.target_digits = target;

    arbsdp_solve_adaptive(&r, &p, &pp);
    print_trace("MOTIVATING sdp_sqrt2", &r, GOLD_SQRT2);

    d = digits_matched(r.res.value, GOLD_SQRT2);

    /* Reached OPTIMAL by escalation. */
    CHECK(r.status == ARBSDP_ADAPTIVE_OPTIMAL,
          "motivating: sdp_sqrt2 must reach ADAPTIVE_OPTIMAL by escalation");
    CHECK(r.res.status == ARBSDP_SOLVE_OPTIMAL,
          "motivating: final fixed-solve status must be OPTIMAL");
    /* Escalation actually happened: more than one precision was tried, and the
     * final precision exceeds prec0 (this is the anti-"never escalates" mutation
     * guard, rule 8). */
    CHECK(r.prec_history_len > 1,
          "motivating: prec_history must show escalation (len > 1)");
    CHECK(r.prec_history[0].prec == 128,
          "motivating: first precision tried must be prec0=128");
    CHECK(r.final_prec > 128,
          "motivating: final precision must exceed prec0 (escalation occurred)");
    /* The prec0=128 solve must NOT have been OPTIMAL at the target (else the test
     * would not be exercising escalation) -- it is a documented NUMERICAL floor. */
    CHECK(r.prec_history[0].status != ARBSDP_SOLVE_OPTIMAL,
          "motivating: the prec0 solve must be a non-OPTIMAL floor (escalation needed)");
    /* Accuracy target met. */
    CHECK(d >= (double) target,
          "motivating: sdp_sqrt2 must match sqrt(2) to >= target digits");

    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

/* ------------------------------------------------------------------------- *
 * Test 2: EASY CASE -- trivial_2x2 converges at/near prec0 with NO needless    *
 * escalation (short prec_history).                                             *
 * ------------------------------------------------------------------------- */
static void
test_easy_no_overescalation(void)
{
    arbsdp_problem p;
    arbsdp_adaptive_result r;
    arbsdp_precision_params pp;

    if (!load_golden(&p, "trivial_2x2")) { CHECK(0, "easy: read"); return; }

    arbsdp_precision_default_params(&pp);
    pp.prec0         = 128;
    pp.prec_max      = 8192;
    pp.target_digits = 10;   /* modest target the easy problem clears at prec0 */

    arbsdp_solve_adaptive(&r, &p, &pp);
    print_trace("EASY trivial_2x2", &r, GOLD_TRIVIAL);

    CHECK(r.status == ARBSDP_ADAPTIVE_OPTIMAL,
          "easy: trivial_2x2 must reach ADAPTIVE_OPTIMAL");
    /* HONEST CONVERGENCE (bead b30, p68).  This test previously asserted
     * "converges at prec0=128, history len == 1".  That rested on a FRAGILE p68
     * false-positive: the prec=128 soft-optimal snapshot was lifted to OPTIMAL
     * while its recovered value was only ~3 correct digits ([1.000 +/- 6.4e-4]),
     * far short of the 10-digit target.  It survived only by bit-exact arb
     * arithmetic -- any refactor of the cone kernels (here: the b30 strict-PD
     * gate replacing the silent 1e-300 floor) tips it, and the honest result is
     * NUMERICAL-then-escalate to a prec that genuinely reaches the target
     * ([1.000000 +/- 3.1e-7] already at prec=128, tighter than the old "OPTIMAL").
     * We therefore assert the HONEST contract: reach the target with a SMALL,
     * BOUNDED escalation (b30's bits/digit cut keeps it to 2 solves: 128->256),
     * not a bit-fragile one-shot.  Removing the p68 over-claim is exactly the
     * accuracy-contract work the HANDOFF assigns to p68; b30 forces it here. */
    CHECK(r.prec_history_len <= 2,
          "easy: trivial_2x2 must reach the target with <=1 escalation (b30/p68)");
    CHECK(r.final_prec <= 256,
          "easy: trivial_2x2 final precision must stay small (<=256) (b30/p68)");

    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

/* ------------------------------------------------------------------------- *
 * Test 3: CAP -> HONEST LIMIT (CLAUDE.md rule 5).  prec_max absurdly low (32)  *
 * so sdp_sqrt2 cannot converge -> honest LIMIT (NOT a false OPTIMAL); history  *
 * stopped at the cap.                                                          *
 * ------------------------------------------------------------------------- */
static void
test_cap_honest_limit(void)
{
    arbsdp_problem p;
    arbsdp_adaptive_result r;
    arbsdp_precision_params pp;

    if (!load_golden(&p, "sdp_sqrt2")) { CHECK(0, "cap: read"); return; }

    arbsdp_precision_default_params(&pp);
    pp.prec0         = 32;
    pp.prec_max      = 32;     /* absurdly low: even prec0==prec_max, one shot */
    pp.target_digits = 30;

    arbsdp_solve_adaptive(&r, &p, &pp);
    print_trace("CAP sdp_sqrt2 (prec_max=32)", &r, GOLD_SQRT2);

    /* Honest limit, never a false OPTIMAL. */
    CHECK(r.status == ARBSDP_ADAPTIVE_LIMIT,
          "cap: must report honest ADAPTIVE_LIMIT at the cap");
    CHECK(r.status != ARBSDP_ADAPTIVE_OPTIMAL,
          "cap: must NOT report a false OPTIMAL when target unreachable (rule 5)");
    /* Stopped at the cap: the final (and largest) precision tried is prec_max. */
    CHECK(r.final_prec == 32,
          "cap: must stop at prec_max=32");
    CHECK(r.prec_history_len >= 1 &&
          r.prec_history[r.prec_history_len - 1].prec == 32,
          "cap: last history entry must be at the cap (32)");

    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

/* ------------------------------------------------------------------------- *
 * Test 4: TARGET DIGITS MET -- request a feasible N; achieved accuracy >= N.   *
 * Uses a smaller target than test 1 to exercise a different escalation depth.  *
 * ------------------------------------------------------------------------- */
static void
test_target_digits_met(void)
{
    arbsdp_problem p;
    arbsdp_adaptive_result r;
    arbsdp_precision_params pp;
    int target = 12;
    double d;

    if (!load_golden(&p, "sdp_sqrt2")) { CHECK(0, "target: read"); return; }

    arbsdp_precision_default_params(&pp);
    pp.prec0         = 128;
    pp.prec_max      = 8192;
    pp.target_digits = target;

    arbsdp_solve_adaptive(&r, &p, &pp);
    print_trace("TARGET sdp_sqrt2 (12 digits)", &r, GOLD_SQRT2);

    d = digits_matched(r.res.value, GOLD_SQRT2);
    CHECK(r.status == ARBSDP_ADAPTIVE_OPTIMAL,
          "target: must reach ADAPTIVE_OPTIMAL for a feasible target");
    CHECK(d >= (double) target,
          "target: achieved accuracy must be >= requested digits");

    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

/* ------------------------------------------------------------------------- *
 * Test 5: MUTATION (CLAUDE.md rule 8).  Two mutations the suite must catch:    *
 *   (m1) "never escalates" -- a driver returning the prec0 NUMERICAL result    *
 *        would have prec_history_len == 1 with a NUMERICAL final status and ~4  *
 *        digits on sdp_sqrt2; test 1's (len>1) + (OPTIMAL) + (>=20 digits)      *
 *        checks each discriminate it.  Here we additionally assert that the     *
 *        prec0 solve on sdp_sqrt2 is genuinely a NON-optimal floor, so the      *
 *        escalation path is the ONLY way test 1 can pass.                       *
 *   (m2) "claims OPTIMAL at the cap without meeting target" -- a driver that    *
 *        stamped OPTIMAL at prec_max regardless would pass test 3's existence   *
 *        but fail the (status==LIMIT) + (status!=OPTIMAL) checks.  Here we      *
 *        confirm that at a cap where the achievable accuracy is BELOW the       *
 *        target, the status is LIMIT and the achieved digits are < target       *
 *        (so a false-OPTIMAL stamp is detectable).                              *
 * ------------------------------------------------------------------------- */
static void
test_mutation_guards(void)
{
    /* (m1) the prec0 floor on sdp_sqrt2 is non-optimal at a high target. */
    {
        arbsdp_problem p;
        arbsdp_result res;
        arbsdp_solve_params sp;
        if (load_golden(&p, "sdp_sqrt2")) {
            arbsdp_solve_default_params(&sp);
            sp.opt_tol = 1e-20; sp.feas_tol = 1e-20;   /* the 20-digit target */
            arbsdp_solve(&res, &p, 128, &sp);          /* a single prec0 solve */
            double d = digits_matched(res.value, GOLD_SQRT2);
            printf("\n  MUTATION m1: prec0=128 single solve on sdp_sqrt2 @ tol=1e-20"
                   " -> status=%s, %.1f digits.\n", solve_status_name(res.status), d);
            CHECK(res.status != ARBSDP_SOLVE_OPTIMAL,
                  "mutation m1: prec0 solve must be NON-optimal (escalation is required)");
            CHECK(d < 20.0,
                  "mutation m1: prec0 solve must fall short of 20 digits");
            arbsdp_result_clear(&res);
            arbsdp_problem_clear(&p);
        }
    }

    /* (m2) a cap below the achievable accuracy yields LIMIT + digits < target,
     * so a false-OPTIMAL-at-cap mutation is detectable.  prec_max=512 on
     * sdp_sqrt2 with a 30-digit target: ~7.6 digits achievable, NOT optimal. */
    {
        arbsdp_problem p;
        arbsdp_adaptive_result r;
        arbsdp_precision_params pp;
        if (load_golden(&p, "sdp_sqrt2")) {
            arbsdp_precision_default_params(&pp);
            pp.prec0 = 128; pp.prec_max = 512; pp.target_digits = 30;
            arbsdp_solve_adaptive(&r, &p, &pp);
            double d = digits_matched(r.res.value, GOLD_SQRT2);
            printf("  MUTATION m2: prec_max=512, target=30 on sdp_sqrt2 -> status=%s,"
                   " %.1f digits (cap below achievable).\n",
                   adaptive_status_name(r.status), d);
            CHECK(r.status == ARBSDP_ADAPTIVE_LIMIT,
                  "mutation m2: capped-below-target must be LIMIT, not OPTIMAL");
            CHECK(d < 30.0,
                  "mutation m2: achieved digits must be < target at the cap");
            CHECK(r.final_prec == 512,
                  "mutation m2: must have escalated up to the cap (512)");
            arbsdp_adaptive_result_clear(&r);
            arbsdp_problem_clear(&p);
        }
    }
}

/* ------------------------------------------------------------------------- *
 * Test 6: CONFIRM GATE (bead arb-prec-IPM-p68).  Pins arbsdp_adaptive_confirmed *
 * with CONSTRUCTED arb values (no solver -- deterministic + fast).  The gate    *
 * confirms a candidate iff the confirming solve is OPTIMAL AND its value agrees  *
 * with the candidate to >= target_digits.  Case 1 is the p68 discriminator: a   *
 * candidate that DISAGREES with the higher-precision confirming solve beyond the *
 * target is NOT confirmed (the precision-starved-OPTIMAL class).                 *
 * ------------------------------------------------------------------------- */
static void
test_confirm_gate(void)
{
    arb_t v_cand, v_conf;
    arb_init(v_cand); arb_init(v_conf);

    /* Case 1: agreement ~6.7 digits (|diff|=2e-7, rel=1) < target 12 -> 0.
     * This is the p68 discriminator -- a stub that ignores agreement returns 1. */
    arb_set_str(v_cand, "1.0000002", WORKPREC);
    arb_set_str(v_conf, "1.0",       WORKPREC);
    CHECK(arbsdp_adaptive_confirmed(ARBSDP_SOLVE_OPTIMAL, v_cand, v_conf, 12, WORKPREC) == 0,
          "confirm: candidate disagreeing beyond target must NOT be confirmed (p68)");

    /* Case 2: agreement ~15 digits (|diff|=1e-15, rel=1) >= target 12 -> 1. */
    arb_set_str(v_cand, "1.000000000000001", WORKPREC);
    arb_set_str(v_conf, "1.0",               WORKPREC);
    CHECK(arbsdp_adaptive_confirmed(ARBSDP_SOLVE_OPTIMAL, v_cand, v_conf, 12, WORKPREC) == 1,
          "confirm: candidate agreeing to >= target must be confirmed");

    /* Case 3: exact agreement at a high target -> 1 (+inf agreement). */
    arb_set_str(v_cand, "1.2345678901234567890123456789012345", WORKPREC);
    arb_set(v_conf, v_cand);
    CHECK(arbsdp_adaptive_confirmed(ARBSDP_SOLVE_OPTIMAL, v_cand, v_conf, 30, WORKPREC) == 1,
          "confirm: exact agreement must be confirmed (even at a high target)");

    /* Case 4: confirming solve is NOT OPTIMAL -> 0 regardless of agreement. */
    arb_set_str(v_cand, "1.0", WORKPREC);
    arb_set(v_conf, v_cand);
    CHECK(arbsdp_adaptive_confirmed(ARBSDP_SOLVE_NUMERICAL, v_cand, v_conf, 2, WORKPREC) == 0,
          "confirm: a non-OPTIMAL confirming solve must NOT confirm (point-mode stop)");

    arb_clear(v_cand); arb_clear(v_conf);
    printf("\n  CONFIRM GATE: 4 constructed cases pinned (p68 cross-precision stability).\n");
}

/* ill_conditioned_3x3 analytic optimum (golden/ill_conditioned_3x3/provenance.json
 * key "optimal_value", 65 digits): 1 + 10^-6 = 1000001/1000000 exact. */
#define GOLD_ILL3X3 "1.0000010000000000000000000000000000000000000000000000000000000000"

/* ------------------------------------------------------------------------- *
 * Test 7: P68 VICTIMS CONFIRMED END-TO-END (bead arb-prec-IPM-p68).             *
 *                                                                              *
 * The two goldens that originally exposed p68 -- trivial_2x2 (optimum exactly  *
 * 1) and ill_conditioned_3x3 (optimum 1+1e-6) -- driven through the full       *
 * arbsdp_solve_adaptive controller at the CALIBRATED target_digits=12.  Before *
 * the p68 fix, a prec=128 soft-OPTIMAL snapshot was lifted to OPTIMAL on its    *
 * status alone, delivering an objective short of 12 digits.  The fix is a       *
 * cross-precision STABILITY gate: an OPTIMAL fixed solve at precision P is only  *
 * a CANDIDATE; the next doubling (2P) CONFIRMS it iff that solve is also OPTIMAL *
 * AND its recovered value agrees with the candidate to >= target_digits; on     *
 * confirmation the controller RETURNS THE CANDIDATE (final_prec stays at P).     *
 *                                                                              *
 * prec_history_len >= 2 is the LOAD-BEARING assertion: it proves a CONFIRMING   *
 * solve ran after the candidate.  A status-only gate (the pre-p68 bug) would    *
 * accept the candidate one-shot at len == 1, so this check bites exactly if the *
 * loop is reverted to status-only.  final_prec <= 256 pins the return-lower     *
 * behaviour (the controller returns the low candidate precision, no bits/digit  *
 * regression from over-escalation).                                            *
 * ------------------------------------------------------------------------- */
static void
test_p68_victims_confirmed(void)
{
    /* Verify the hardcoded ill_conditioned_3x3 optimum against its provenance
     * file (CLAUDE.md rule 8: do not trust a transcribed constant). */
    {
        char path[1024];
        FILE *f;
        snprintf(path, sizeof path,
                 "%s/ill_conditioned_3x3/provenance.json", GOLDEN_DIR);
        f = fopen(path, "r");
        CHECK(f != NULL, "p68: cannot open ill_conditioned_3x3 provenance.json");
        if (f) {
            char buf[8192];
            size_t n = fread(buf, 1, sizeof buf - 1, f);
            buf[n] = '\0';
            fclose(f);
            CHECK(strstr(buf, "\"" GOLD_ILL3X3 "\"") != NULL,
                  "p68: GOLD_ILL3X3 must match provenance optimal_value verbatim");
        }
    }

    struct { const char *name; const char *gold; }
    victims[2] = {
        { "trivial_2x2",        GOLD_TRIVIAL },
        { "ill_conditioned_3x3", GOLD_ILL3X3 },
    };

    for (int k = 0; k < 2; k++) {
        arbsdp_problem p;
        arbsdp_adaptive_result r;
        arbsdp_precision_params pp;
        double d;

        if (!load_golden(&p, victims[k].name)) {
            CHECK(0, "p68: read victim golden");
            continue;
        }

        arbsdp_precision_default_params(&pp);
        pp.prec0         = 128;
        pp.prec_max      = 1024;   /* ill_conditioned_3x3 provenance prec_cap */
        pp.target_digits = 12;     /* calibrated target_digits */

        arbsdp_solve_adaptive(&r, &p, &pp);
        print_trace(victims[k].name, &r, victims[k].gold);

        d = digits_matched(r.res.value, victims[k].gold);

        /* (1) the controller reports OPTIMAL. */
        CHECK(r.status == ARBSDP_ADAPTIVE_OPTIMAL,
              "p68: victim must reach ADAPTIVE_OPTIMAL");
        /* (2) the DELIVERED objective meets the 12-digit target (the accuracy
         * property p68 is about -- a status-only accept failed this). */
        CHECK(d >= 12.0,
              "p68: delivered value must match the optimum to >= 12 digits");
        /* (3) LOAD-BEARING: the CONFIRMATION path ran (candidate + >=1 confirming
         * solve).  A status-only gate would accept one-shot at len == 1; this is
         * the regression guard that bites if the loop reverts to status-only. */
        CHECK(r.prec_history_len >= 2,
              "p68: confirmation path must run (prec_history_len >= 2)");
        /* (4) return-lower: the controller returns the low candidate precision,
         * no bits/digit regression. */
        CHECK(r.final_prec <= 256,
              "p68: controller must return the low candidate precision (<=256)");

        printf("    p68 victim %s: status=%s final_prec=%ld"
               " prec_history_len=%d digits=%.3f\n",
               victims[k].name, adaptive_status_name(r.status),
               (long) r.final_prec, r.prec_history_len, d);

        arbsdp_adaptive_result_clear(&r);
        arbsdp_problem_clear(&p);
    }
}

int
main(void)
{
    test_motivating_sqrt2();
    test_easy_no_overescalation();
    test_cap_honest_limit();
    test_target_digits_met();
    test_mutation_guards();
    test_confirm_gate();
    test_p68_victims_confirmed();

    if (failures != 0) {
        fprintf(stderr, "\ntest_precision: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("\nPASS: test_precision (adaptive precision escalation)\n");
    return EXIT_SUCCESS;
}
