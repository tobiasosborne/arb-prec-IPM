/*
 * test/test_golden.c -- the GATED golden-master runner (bead arb-prec-IPM-epic.1).
 * The primary Layer-0 robustness gate: discover every golden problem under
 * GOLDEN_DIR, solve it with the adaptive precision driver, and CHARACTERIZE the
 * point-mode recovered objective against the analytic golden -- per assertion
 * class (correctness, efficiency, status) with explicit xfail/xpass semantics.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * The recovered objective r.res.value is a POINT-mode midpoint; it CERTIFIES
 * NOTHING.  This runner compares that midpoint to the analytic golden value --
 * this is fine: it is a Layer-0 CHARACTERIZATION test (does the approximate
 * solver reach its reachable accuracy at the budgeted cost?), NOT a rigor
 * claim.  All rigor is Layer 1 (certify.c, b18/b19), which derives [lb,ub]
 * independently in ball arithmetic and is gated elsewhere.
 *
 * ==========================================================================
 * GROUND TRUTH (CLAUDE.md rule 3)
 * ==========================================================================
 *   - This bead (arb-prec-IPM-epic.1): the assertion-class + xfail/xpass design.
 *   - The TS reference's KNOWN_CONVERGENCE_GAPS pattern (HsdeNtSdpSolver test
 *     harness): a per-case, kind-tagged registry of TOLERATED gaps so a known
 *     shortfall is GREEN (xfail) but stays VISIBLE, and the gate goes RED the
 *     instant a gap CLOSES (xpass -> the stale marker must be deleted) or a
 *     previously-passing class REGRESSES (fail).
 *   - golden_provenance.h / .c (schema v2): the per-case golden_case carrying
 *     target_digits, prec_cap, correctness_tol_digits, expected_max_bits_per_digit,
 *     expected_max_iters, and the gap_{status,correctness,efficiency} flags.
 *   - c/include/arbsdp/precision.h (the adaptive solve contract this drives).
 *   - c/include/arbsdp/solve.h (the embedded fixed-prec result + status enum).
 *
 * ==========================================================================
 * ASSERTION CLASSES + VERDICT TABLE (the heart -- get this wrong and the gate
 * lies)
 * ==========================================================================
 * Each (problem, class) pair yields one boolean `ok` (did the measured property
 * meet the calibrated expectation?) and one boolean `gap` (is there a known_gaps
 * entry of that kind tolerating a shortfall?).  The verdict is:
 *
 *     ok    gap   verdict   color   meaning
 *     ----  ----  --------  ------  --------------------------------------------
 *      1     0    PASS      green   met expectation, no excuse needed
 *      0     1    XFAIL     green   fell short, but a known_gap tolerates it
 *      0     0    FAIL      RED     regression: a real, untracked shortfall
 *      1     0    --        --      (see PASS)
 *      1     1    XPASS     RED     the gap CLOSED -- delete the stale known_gap
 *
 * The GATE fails (process exit nonzero) IFF any (problem,class) pair is FAIL or
 * XPASS.  PASS and XFAIL are green.  XFAILs are printed with their gap bead +
 * reason so a tolerated shortfall is never hidden.
 *
 * Classes by expected_status:
 *   - "optimal": CORRECTNESS (adaptive OPTIMAL AND digits >= target - tol) and
 *     EFFICIENCY (bits/digit <= budget AND iters <= budget).
 *   - other (none exist yet; forward-compatible): STATUS (got_status == mapped).
 *
 * ==========================================================================
 * EXPECTED RESULT ON TODAY's CODE (recorded per CLAUDE.md rule 11, concrete
 * numbers)
 * ==========================================================================
 * Every golden is corr=PASS (reaches its calibrated reachable target) and
 * eff=XFAIL (bits/digit ~70-160 >> budget 8, tolerated via the b30 efficiency
 * known_gap).  GATE: PASS with 7 XFAILs visible.  A corr=FAIL would mean the
 * calibrated target is unreachable -> a REAL finding (the provenance needs a
 * correctness gap); the runner surfaces it, it must NOT be papered over.
 *
 * Arb memory discipline (CLAUDE.md rule 7): per case, problem_init -> read_sdpa
 * -> solve_adaptive -> evaluate -> adaptive_result_clear -> problem_clear (the
 * problem OUTLIVES the borrowing result).  digits_matched init's and clears its
 * own four temporaries.  ASan/LeakSanitizer-clean.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <flint/arb.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/solve.h"
#include "arbsdp/precision.h"
#include "golden_provenance.h"

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "../../golden"
#endif

/* Ample precision for the test's own arb comparison of the recovered midpoint
 * against the 65-digit golden string (POINT-mode diagnostic). */
#define PREC 8192

/* Upper bound on the number of golden cases discovered (fail loud if exceeded;
 * golden_discover returns -1 if count would exceed this). */
#define MAX_CASES 64

/* -------------------------------------------------------------------------
 * digits_matched -- |mid(recovered) - golden| as matching decimal digits
 * (POINT-mode diagnostic, copied verbatim from the bead spec / test_solve.c).
 * -log10(|err| / max(1,|golden|)).  Returns 1e9 for an exact / zero error.
 * ---------------------------------------------------------------------- */
static double digits_matched(const arb_t recovered, const char *golden_str) {
    arb_t g, err, rel, one; double d;
    arb_init(g); arb_init(err); arb_init(rel); arb_init(one);
    arb_get_mid_arb(err, recovered);
    arb_set_str(g, golden_str, PREC);
    arb_sub(err, err, g, PREC); arb_abs(err, err);
    arb_abs(rel, g); arb_one(one); if (arb_lt(rel, one)) arb_set(rel, one);
    arb_div(err, err, rel, PREC);
    if (arb_is_zero(err) || !arb_is_finite(err)) d = 1e9;
    else { double e = arf_get_d(arb_midref(err), ARF_RND_NEAR); d = (e <= 0.0) ? 1e9 : -log10(e); }
    arb_clear(g); arb_clear(err); arb_clear(rel); arb_clear(one);
    return d;
}

/* -------------------------------------------------------------------------
 * Status-name helpers (small, per the bead).
 * ---------------------------------------------------------------------- */
static const char *solve_status_name(arbsdp_solve_status s) {
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

static const char *adaptive_status_name(arbsdp_adaptive_status s) {
    switch (s) {
    case ARBSDP_ADAPTIVE_OPTIMAL: return "OPTIMAL";
    case ARBSDP_ADAPTIVE_LIMIT:   return "LIMIT";
    default:                      return "?";
    }
}

/* -------------------------------------------------------------------------
 * Verdict semantics.  A class is evaluated to one of these from (ok, gap).
 * VERDICT_NA is the non-evaluated class for a given expected_status (printed as
 * "n/a" and never counted).
 * ---------------------------------------------------------------------- */
typedef enum { V_PASS, V_XFAIL, V_FAIL, V_XPASS, V_NA } verdict;

static verdict verdict_of(int ok, int gap) {
    if (ok && !gap) return V_PASS;
    if (!ok && gap) return V_XFAIL;
    if (!ok && !gap) return V_FAIL;
    return V_XPASS; /* ok && gap: the gap closed -> stale marker, RED */
}

static const char *verdict_name(verdict v) {
    switch (v) {
    case V_PASS:  return "PASS";
    case V_XFAIL: return "XFAIL";
    case V_FAIL:  return "FAIL";
    case V_XPASS: return "XPASS";
    default:      return "n/a";
    }
}

/* Map a non-"optimal" expected_status string to the corresponding solve status
 * (forward-compatible; no such goldens exist yet).  HSDE: primal-unbounded <=>
 * dual-infeasible.  Unknown / "inconclusive" -> NUMERICAL. */
static arbsdp_solve_status expected_solve_status(const char *es) {
    if (strcmp(es, "primal_infeasible") == 0) return ARBSDP_SOLVE_PRIMAL_INFEASIBLE;
    if (strcmp(es, "dual_infeasible")   == 0) return ARBSDP_SOLVE_DUAL_INFEASIBLE;
    if (strcmp(es, "unbounded")         == 0) return ARBSDP_SOLVE_DUAL_INFEASIBLE;
    return ARBSDP_SOLVE_NUMERICAL; /* "inconclusive" and any unknown */
}

/* Running tallies across all (problem, class) pairs. */
typedef struct {
    int pass, xfail, fail, xpass;
} tally;

static void tally_add(tally *t, verdict v) {
    switch (v) {
    case V_PASS:  t->pass++;  break;
    case V_XFAIL: t->xfail++; break;
    case V_FAIL:  t->fail++;  break;
    case V_XPASS: t->xpass++; break;
    default: break; /* V_NA: not counted */
    }
}

int main(void) {
    golden_case cases[MAX_CASES];
    int n = golden_discover(GOLDEN_DIR, cases, MAX_CASES);
    if (n < 0) {
        fprintf(stderr, "FATAL: golden_discover(%s) failed (see above)\n", GOLDEN_DIR);
        return EXIT_FAILURE;
    }
    if (n == 0) {
        fprintf(stderr, "FATAL: no golden cases discovered under %s\n", GOLDEN_DIR);
        return EXIT_FAILURE;
    }

    printf("GOLDEN-MASTER GATE (Layer-0, POINT mode -- characterization, NOT rigor)\n");
    printf("  golden_dir=%s  cases=%d  cmp_prec=%d\n\n", GOLDEN_DIR, n, PREC);
    printf("  %-20s %-8s %-7s %-18s %-13s %-15s %-14s %s\n",
           "name", "expect", "adapt", "got_status", "digits(a/t)",
           "bits/dig(a/b)", "iters(a/b)", "verdicts");
    printf("  %-20s %-8s %-7s %-18s %-13s %-15s %-14s %s\n",
           "--------------------", "--------", "-------",
           "------------------", "-------------", "---------------",
           "--------------", "-----------------------");

    tally t = {0, 0, 0, 0};
    int hard_fail = 0;            /* read_sdpa failures (counted as hard FAILs)   */
    int xfail_lines = 0;          /* count to gate the "visible XFAIL" footer     */

    /* Collected XFAIL notes (printed after the table so the gaps stay visible). */
    typedef struct { char name[128]; char cls[8]; char bead[64]; char reason[256]; } xnote;
    xnote *notes = calloc((size_t) n * 3u, sizeof *notes); /* <=3 classes/problem */
    if (!notes) { fprintf(stderr, "FATAL: oom (xfail notes)\n"); return EXIT_FAILURE; }
    int notes_n = 0;

    for (int k = 0; k < n; k++) {
        golden_case *c = &cases[k];

        arbsdp_problem p;
        arbsdp_problem_init(&p);
        if (arbsdp_read_sdpa(&p, c->dats_path) != 0) {
            fprintf(stderr, "  HARD-FAIL %-18s read_sdpa(%s) failed\n", c->name, c->dats_path);
            hard_fail++;
            arbsdp_problem_clear(&p);
            continue;
        }

        arbsdp_precision_params pp;
        arbsdp_precision_default_params(&pp);
        pp.target_digits = c->target_digits;
        pp.prec_max      = c->prec_cap;
        /* leave prec0=128, escalation=2, iter_limit/stall_cap at defaults */

        arbsdp_adaptive_result r;
        arbsdp_solve_adaptive(&r, &p, &pp);

        double digits = digits_matched(r.res.value, c->optimal_value);
        double bits_per_digit = (digits > 0.0) ? (double) r.final_prec / digits : 1e18;

        verdict v_corr = V_NA, v_eff = V_NA, v_stat = V_NA;
        char vstr[96];

        if (strcmp(c->expected_status, "optimal") == 0) {
            /* CORRECTNESS: reached adaptive OPTIMAL AND matched target (within tol). */
            int ok_corr = (r.status == ARBSDP_ADAPTIVE_OPTIMAL) &&
                          (digits >= (double) (c->target_digits - c->correctness_tol_digits));
            v_corr = verdict_of(ok_corr, c->gap_correctness);

            /* EFFICIENCY: within the bits/digit budget AND the iteration budget. */
            int ok_eff = (bits_per_digit <= c->expected_max_bits_per_digit) &&
                         (r.res.iters <= c->expected_max_iters);
            v_eff = verdict_of(ok_eff, c->gap_efficiency);

            snprintf(vstr, sizeof vstr, "corr=%s eff=%s",
                     verdict_name(v_corr), verdict_name(v_eff));
        } else {
            /* STATUS (forward-compatible; no such goldens exist yet). */
            arbsdp_solve_status mapped = expected_solve_status(c->expected_status);
            int ok_status = (r.res.status == mapped);
            v_stat = verdict_of(ok_status, c->gap_status);
            snprintf(vstr, sizeof vstr, "status=%s", verdict_name(v_stat));
        }

        /* One row per problem. */
        {
            char dcol[16], bcol[18], icol[16];
            snprintf(dcol, sizeof dcol, "%.1f/%d", digits, c->target_digits);
            snprintf(bcol, sizeof bcol, "%.0f/%.0f", bits_per_digit,
                     c->expected_max_bits_per_digit);
            snprintf(icol, sizeof icol, "%d/%d", r.res.iters, c->expected_max_iters);
            printf("  %-20s %-8s %-7s %-18s %-13s %-15s %-14s %s\n",
                   c->name, c->expected_status,
                   adaptive_status_name(r.status),
                   solve_status_name(r.res.status),
                   dcol, bcol, icol, vstr);
        }

        /* Tally + record XFAIL notes for visibility. */
        verdict vs[3] = { v_corr, v_eff, v_stat };
        const char *cls[3] = { "corr", "eff", "status" };
        for (int ci = 0; ci < 3; ci++) {
            if (vs[ci] == V_NA) continue;
            tally_add(&t, vs[ci]);
            if (vs[ci] == V_XFAIL) {
                xfail_lines++;
                xnote *xn = &notes[notes_n++];
                snprintf(xn->name, sizeof xn->name, "%s", c->name);
                snprintf(xn->cls, sizeof xn->cls, "%s", cls[ci]);
                snprintf(xn->bead, sizeof xn->bead, "%s",
                         c->gap_bead[0] ? c->gap_bead : "(no bead)");
                snprintf(xn->reason, sizeof xn->reason, "%s",
                         c->gap_reason[0] ? c->gap_reason : "(no reason)");
            }
        }

        arbsdp_adaptive_result_clear(&r);
        arbsdp_problem_clear(&p);
    }

    /* Visible XFAILs: bead + short reason so a tolerated gap is never hidden. */
    if (xfail_lines > 0) {
        printf("\n  TOLERATED GAPS (XFAIL -- visible, not hidden):\n");
        for (int i = 0; i < notes_n; i++) {
            char shortreason[81];
            snprintf(shortreason, sizeof shortreason, "%s", notes[i].reason);
            printf("    %-20s %-6s [%s] %s\n",
                   notes[i].name, notes[i].cls, notes[i].bead, shortreason);
        }
    }
    free(notes);

    /* Summary across all (problem, class) pairs. */
    int total = t.pass + t.xfail + t.fail + t.xpass;
    printf("\n  SUMMARY (%d problems, %d (problem,class) pairs):\n", n, total);
    printf("    PASS=%d  XFAIL=%d  FAIL=%d  XPASS=%d  (hard read-fail=%d)\n",
           t.pass, t.xfail, t.fail, t.xpass, hard_fail);

    /* The gate fails IFF any class is FAIL or XPASS, or any problem hard-failed
     * to read.  PASS and XFAIL are green. */
    int gate_fail = (t.fail > 0) || (t.xpass > 0) || (hard_fail > 0);
    if (gate_fail) {
        printf("\n  GATE: FAIL -- ");
        if (hard_fail > 0)
            printf("%d problem(s) failed to read (hard fail); ", hard_fail);
        if (t.fail > 0)
            printf("%d class(es) REGRESSED (FAIL: a real, untracked shortfall -- "
                   "do NOT loosen; if a calibrated target is unreachable, add a "
                   "correctness gap to the provenance); ", t.fail);
        if (t.xpass > 0)
            printf("%d class(es) XPASS (a known_gap CLOSED -- delete the stale "
                   "known_gaps marker from the provenance); ", t.xpass);
        printf("\n");
        return EXIT_FAILURE;
    }

    printf("\n  GATE: PASS -- %d PASS, %d XFAIL (tolerated, visible above), "
           "0 FAIL, 0 XPASS.\n", t.pass, t.xfail);
    return EXIT_SUCCESS;
}
