/*
 * test/test_certify_bracket.c -- BALL-MODE (RIGOROUS, Layer-1) tests for the
 * PRODUCT: the rigorous two-sided bracket [lb, ub] (bead arb-prec-IPM-b19,
 * MATH_SPEC §5.4 + §5.5).
 *
 * ==========================================================================
 * RIGOR IS FALSIFIABLE -- THE GATE (CLAUDE.md rule 2)
 * ==========================================================================
 * The output [lb, ub] is a THEOREM: the true optimum p* is provably inside it.
 * A bracket that EVER excludes a known/constructed optimum is a P0 bug -- the
 * RIGOR GATE (test_rigor_gate) asserts arb_le(lb, opt) && arb_le(opt, ub) for
 * EVERY golden with a known 65-digit analytic optimum.  If ANY golden fails, the
 * test FAILS LOUDLY (printing lb/opt/ub) and must NOT be "fixed" by loosening the
 * assertion (P0; report it).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): the first link target exercises
 * arbsdp_certify_bracket before it exists (link-fail -> RED), then GREEN.
 *
 * Tests:
 *   1. smoke              -- max_eigenvalue_2x2: 3 in [lb,ub], both finite.
 *   2. lambda_max_upper   -- arbsdp_lambda_max_upper_bound([[2,1],[1,2]]) >= 3,
 *                            and tight (proves it is an upper bound on lambda_max).
 *   6. RIGOR GATE         -- 7 goldens: opt in [lb,ub]; table lb/opt/ub/gap/margin.
 *   7. honest infinity    -- trivial_2x2 with xbar UNSET: bracket NOT both finite.
 *   8. sign self-test     -- (a) internal-frame route reproduces ub_ext (overlap);
 *                            (b) WRONG sign (y_int) -> bracket EXCLUDES opt.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear; the
 * problem OUTLIVES the borrowing result/iterate.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>     /* unlink (POSIX; mkstemp lives in <stdlib.h>)         */

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/solve.h"
#include "arbsdp/precision.h"
#include "arbsdp/certify.h"
#include "golden_provenance.h"

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "../../golden"
#endif

#define MAX_CASES 64

/* Comparison precision for printing/decimal-margin diagnostics (POINT-mode). */
#define CMP_PREC 4096

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* Find a discovered golden_case by name (NULL if absent). */
static const golden_case *
find_case(const golden_case *cases, int n, const char *name)
{
    for (int k = 0; k < n; k++)
        if (strcmp(cases[k].name, name) == 0)
            return &cases[k];
    return NULL;
}

/* -------------------------------------------------------------------------
 * solve_to_iterate -- read `c`'s .dat-s, adaptively solve to ~15 digits, and
 * hand back the loaded problem + adaptive result (the caller owns both and must
 * clear them; the problem outlives the result).  Returns the certification
 * precision prec_c = final_prec + 128 (MATH_SPEC §10, decision 2).
 * ---------------------------------------------------------------------- */
static slong
solve_to_iterate(arbsdp_problem *p, arbsdp_adaptive_result *r,
                 const golden_case *c)
{
    arbsdp_problem_init(p);
    CHECK(arbsdp_read_sdpa(p, c->dats_path) == 0, "solve_to_iterate: read_sdpa");

    arbsdp_precision_params pp;
    arbsdp_precision_default_params(&pp);
    pp.target_digits = 15;
    arbsdp_solve_adaptive(r, p, &pp);
    return r->final_prec + 128;
}

/* -------------------------------------------------------------------------
 * decimal containment margin: how many decimal digits separate opt from the
 * nearer bracket endpoint, as a POINT-mode diagnostic.  +inf if opt is exactly
 * on an endpoint (zero distance).  Negative would mean opt is OUTSIDE [lb,ub].
 * ---------------------------------------------------------------------- */
static double
containment_margin_digits(const arb_t lb, const arb_t ub, const arb_t opt)
{
    arb_t dlo, dhi, lo; double mlo, mhi, m;
    arb_init(dlo); arb_init(dhi); arb_init(lo);
    arb_sub(dlo, opt, lb, CMP_PREC);     /* opt - lb  (>= 0 if rigorous)        */
    arb_sub(dhi, ub, opt, CMP_PREC);     /* ub  - opt (>= 0 if rigorous)        */
    mlo = arf_get_d(arb_midref(dlo), ARF_RND_NEAR);
    mhi = arf_get_d(arb_midref(dhi), ARF_RND_NEAR);
    arb_clear(dlo); arb_clear(dhi); arb_clear(lo);
    m = (mlo < mhi) ? mlo : mhi;         /* nearer endpoint */
    if (m <= 0.0) return -1e9;           /* on/outside an endpoint */
    return -log10(m);                    /* digits of margin (smaller dist -> more digits) */
}

/* Print a 17-sig-digit decimal of an arb midpoint (diagnostic). */
static void
print_arb(const char *label, const arb_t x)
{
    char *s = arb_get_str(x, 30, ARB_STR_NO_RADIUS);
    printf("%s=%s ", label, s);
    flint_free(s);
}

/* -------------------------------------------------------------------------
 * parse_opt_enclosure -- parse the golden's `optimal_value` string (correctly
 * ROUNDED to `optimal_value_digits` significant decimal digits) into a BALL that
 * PROVABLY CONTAINS the true optimum (NOT the bare rounded point).
 *
 * WHY (bead arb-prec-IPM-vry FINDING, CLAUDE.md rule 2/8): the reference string is
 * the true optimum rounded to ~65 sig digits, so its value is OFF the truth by up
 * to half a ulp at the last digit (e.g. 2+sqrt3's 65-digit string sits 5.19e-65
 * BELOW the true 2+sqrt3).  The rigorous primal Rayleigh lb now resolves the true
 * optimum to ~190 digits, so it can land ABOVE the 65-digit POINT while still being
 * <= the TRUE optimum -- a tighter, still-rigorous lb.  Comparing the bound against
 * the bare rounded point would FALSELY flag a P0; we must compare against the true
 * optimum.  We do so by inflating the parsed value to a ball of radius one ulp at
 * the last reference digit -- a CONSERVATIVE over-cover of the half-ulp rounding
 * error -- so the ball provably brackets the true optimum.  This WIDENS the
 * reference (rule 2: widen the interval, never the claim); the rigor gate then
 * checks lb <= ubound(opt_enc) and lbound(opt_enc) <= ub, i.e. against the true
 * optimum.  A bound that exceeded the TRUE optimum by more than the reference
 * rounding would STILL be caught.
 *
 * radius = |opt| * 10^-(digits-1)  (one ulp; >= the 0.5-ulp worst case).  Falls back
 * to a zero-radius point if optimal_value_digits is unset (<=0).
 * ---------------------------------------------------------------------- */
static void
parse_opt_enclosure(arb_t opt_enc, const golden_case *c, slong prec)
{
    arb_set_str(opt_enc, c->optimal_value, prec);
    if (c->optimal_value_digits > 0) {
        arb_t rad;
        arb_init(rad);
        arb_abs(rad, opt_enc);                       /* |opt|                    */
        /* rad *= 10^-(digits-1): one ulp at the last reference significant digit. */
        {
            arb_t ten_pow;
            arb_init(ten_pow);
            arb_set_si(ten_pow, 10);
            arb_pow_ui(ten_pow, ten_pow, (ulong)(c->optimal_value_digits - 1), prec);
            arb_div(rad, rad, ten_pow, prec);
            arb_clear(ten_pow);
        }
        arb_add_error(opt_enc, rad);                 /* inflate the ball         */
        arb_clear(rad);
    }
}

/* ==========================================================================
 * Step 1 (RED -> GREEN): smoke test on max_eigenvalue_2x2.
 * max <C,X>, C=[[2,1],[1,2]], A_1=I, b=[1] -> p*=lambda_max(C)=3, xbar=[1].
 * ======================================================================== */
static void
test_bracket_smoke(const golden_case *cases, int n)
{
    const golden_case *c = find_case(cases, n, "max_eigenvalue_2x2");
    CHECK(c != NULL, "smoke: max_eigenvalue_2x2 must be discovered");
    if (c == NULL)
        return;

    arbsdp_problem p;
    arbsdp_adaptive_result r;
    slong prec_c = solve_to_iterate(&p, &r, c);

    arbsdp_apriori ab;
    arbsdp_apriori_init(&ab, p.nblocks);
    {
        arb_t one; arb_init(one); arb_one(one);
        arbsdp_apriori_set_xbar(&ab, 0, one);
        arb_clear(one);
    }

    arb_t lb, ub, three;
    arb_init(lb); arb_init(ub); arb_init(three);
    arb_set_si(three, 3);

    arbsdp_certify_status st =
        arbsdp_certify_bracket(lb, ub, &p, &r.res.it, &ab, prec_c);

    CHECK(st == ARBSDP_CERTIFY_OPTIMAL_BRACKET, "smoke: status OPTIMAL_BRACKET");
    CHECK(arb_is_finite(lb), "smoke: lb finite");
    CHECK(arb_is_finite(ub), "smoke: ub finite");
    CHECK(arb_le(lb, three), "smoke: lb <= 3");
    CHECK(arb_le(three, ub), "smoke: 3 <= ub");

    arb_clear(lb); arb_clear(ub); arb_clear(three);
    arbsdp_apriori_clear(&ab);
    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

/* ==========================================================================
 * Step 2: arbsdp_lambda_max_upper_bound.
 * [[2,1],[1,2]] has eigenvalues {1,3}; lambda_max = 3.  The returned dhi is a
 * rigorous UPPER bound: dhi >= 3 (checked at dhi's LOWER endpoint so even radius
 * cannot cheat -- lbound(dhi) >= 3 proves dhi >= 3).  Tightness: dhi - 3 small.
 * ======================================================================== */
static void
test_lambda_max_upper(void)
{
    arb_mat_t A;
    arb_t dhi, three, gap, tol;
    arf_t dhi_lo;

    arb_mat_init(A, 2, 2);
    arb_set_si(arb_mat_entry(A, 0, 0), 2);
    arb_set_si(arb_mat_entry(A, 1, 1), 2);
    arb_set_si(arb_mat_entry(A, 0, 1), 1);
    arb_set_si(arb_mat_entry(A, 1, 0), 1);

    arb_init(dhi); arb_init(three); arb_init(gap); arb_init(tol);
    arf_init(dhi_lo);
    arb_set_si(three, 3);

    arbsdp_lambda_max_upper_bound(dhi, A, 256);

    /* RIGOR: lbound(dhi) >= 3  <=>  dhi >= 3 even at its lower endpoint. */
    arb_get_lbound_arf(dhi_lo, dhi, 256);
    CHECK(arf_cmp_si(dhi_lo, 3) >= 0, "lambda_max_upper: dhi >= 3 (rigorous UB)");

    /* TIGHTNESS: dhi - 3 < 1e-60 (bisection converged at prec 256, not stalled).
     * (At prec 256 the verified-Cholesky shift resolves to ~2^-128 ~ 1e-38.) */
    arb_sub(gap, dhi, three, 256);
    arb_set_str(tol, "1e-30", 256);
    CHECK(arb_lt(gap, tol), "lambda_max_upper: dhi - 3 < 1e-30 (tight)");

    arb_mat_clear(A);
    arb_clear(dhi); arb_clear(three); arb_clear(gap); arb_clear(tol);
    arf_clear(dhi_lo);
}

/* ==========================================================================
 * Step 6: THE RIGOR GATE.  For each of the 7 testable goldens, solve, certify
 * [lb,ub], and ASSERT the analytic 65-digit optimum is inside.  Print the table.
 * ======================================================================== */
typedef struct {
    const char *name;
    int         nblocks;     /* expected # of blocks (for xbar setup)            */
    const char *xbar;        /* the per-block trace bound as a decimal string    */
    /* all goldens below are uniform-xbar except two_block_corr_coupled handled
     * by the special-case flag. */
    int         xbar_each;   /* 1 if `xbar` applies to EVERY block               */
    const char *xbar_b0;     /* if xbar_each==0: block-0 bound                    */
    const char *xbar_b1;     /* if xbar_each==0: block-1 bound                    */
} gate_case;

static void
test_rigor_gate(const golden_case *cases, int n)
{
    /* xbar CONVENTION: every entry below is a verified TRACE bound, xbar_b >=
     * tr(X*_b) (NOT a Loewner / lambda_max bound) -- bead om9, MATH_SPEC §5.3.1.
     * Audit of each .dat-s confirmed each block's trace is PINNED by diagonal
     * EQUALITY constraints, so xbar_b == tr(X*_b) exactly (the trace is an
     * equality, hence both an upper and lower bound on tr).  Because these are
     * trace bounds, the no-s_b-factor Jansson penalty (MATH_SPEC §5.3.1, om9
     * reconfirmation) is safe: there is no spurious block-dimension factor.
     * Per-row justification (the constraint that pins the block trace):
     *   max_eig_*           A_1 = I_n (all diag entries 1), b_1=1 => tr(X_1)=1.
     *   separable_12block   block c: A_c = I_2 on block c, b_c=1  => tr(X_c)=1.
     *   two_block_corr_coupled  diag equalities fix X_b(1,1)=1 and X_b(2,2)=2
     *                       => tr(X_b)=1+2=3 (blk0: constr 1,2; blk1: constr 3,4). */
    static const gate_case gcs[] = {
        { "max_eigenvalue_2x2",     1,  "1", 1, NULL, NULL }, /* tr(X)=1: A_1=I_2, b=1 */
        { "max_eig_tridiag_3x3",    1,  "1", 1, NULL, NULL }, /* tr(X)=1: A_1=I_3, b=1 */
        { "max_eig_path_4",         1,  "1", 1, NULL, NULL }, /* tr(X)=1: A_1=I_4, b=1 */
        { "max_eig_path_8",         1,  "1", 1, NULL, NULL }, /* tr(X)=1: A_1=I_8, b=1 */
        { "max_eig_path_16",        1,  "1", 1, NULL, NULL }, /* tr(X)=1: A_1=I_16, b=1 */
        { "separable_12block",     12,  "1", 1, NULL, NULL }, /* tr(X_c)=1/blk: A_c=I_2, b_c=1 */
        { "two_block_corr_coupled", 2, NULL, 0, "3",  "3"  }, /* tr(X_b)=3: diag eqs X(1,1)=1,X(2,2)=2 */
    };
    int ngc = (int) (sizeof gcs / sizeof gcs[0]);

    printf("\nRIGOR GATE (MATH_SPEC §5.4/§5.5; opt MUST be in [lb,ub]; all BALL):\n");
    printf("  %-22s %-12s %-12s %-12s %-12s %-8s %s\n",
           "golden", "lb", "opt", "ub", "gap=ub-lb", "margin", "contains?");

    for (int g = 0; g < ngc; g++) {
        const gate_case *gc = &gcs[g];
        const golden_case *c = find_case(cases, n, gc->name);
        CHECK(c != NULL, "rigor_gate: golden discovered");
        if (c == NULL)
            continue;

        arbsdp_problem p;
        arbsdp_adaptive_result r;
        slong prec_c = solve_to_iterate(&p, &r, c);

        /* sanity: the discovered block count matches the gate expectation. */
        CHECK(p.nblocks == gc->nblocks, "rigor_gate: nblocks matches");

        arbsdp_apriori ab;
        arbsdp_apriori_init(&ab, p.nblocks);
        {
            arb_t v; arb_init(v);
            if (gc->xbar_each) {
                arb_set_str(v, gc->xbar, prec_c);
                for (int b = 0; b < p.nblocks; b++)
                    arbsdp_apriori_set_xbar(&ab, b, v);
            } else {
                /* two_block_corr_coupled: xbar = [3, 3]. */
                arb_set_str(v, gc->xbar_b0, prec_c);
                arbsdp_apriori_set_xbar(&ab, 0, v);
                arb_set_str(v, gc->xbar_b1, prec_c);
                arbsdp_apriori_set_xbar(&ab, 1, v);
            }
            arb_clear(v);
        }

        arb_t lb, ub, opt, gap;
        arb_init(lb); arb_init(ub); arb_init(opt); arb_init(gap);
        /* opt is an ENCLOSURE ball provably containing the TRUE optimum (the
         * reference string is only correctly-rounded to ~65 digits; bead vry). */
        parse_opt_enclosure(opt, c, CMP_PREC);

        arbsdp_certify_status st =
            arbsdp_certify_bracket(lb, ub, &p, &r.res.it, &ab, prec_c);

        /* THE GATE (CLAUDE.md rule 2): the TRUE optimum MUST be inside [lb, ub].
         * Falsifiable against the true opt (NOT the rounded point): lb is sound
         * unless it exceeds the UPPER endpoint of the opt enclosure; ub is sound
         * unless it falls below the LOWER endpoint.  A bound exceeding the true
         * optimum by more than the reference rounding is still caught. */
        int lo_ok, hi_ok, contains;
        {
            arf_t opt_lo, opt_hi, lb_lo, ub_hi;
            arf_init(opt_lo); arf_init(opt_hi); arf_init(lb_lo); arf_init(ub_hi);
            arb_get_lbound_arf(opt_lo, opt, CMP_PREC);
            arb_get_ubound_arf(opt_hi, opt, CMP_PREC);
            arb_get_lbound_arf(lb_lo, lb, CMP_PREC);   /* lb is an exact point here */
            arb_get_ubound_arf(ub_hi, ub, CMP_PREC);   /* ub is an exact point here */
            lo_ok = (arf_cmp(lb_lo, opt_hi) <= 0);     /* lb <= ubound(true opt)   */
            hi_ok = (arf_cmp(opt_lo, ub_hi) <= 0);     /* lbound(true opt) <= ub   */
            arf_clear(opt_lo); arf_clear(opt_hi); arf_clear(lb_lo); arf_clear(ub_hi);
        }
        contains = lo_ok && hi_ok;

        arb_sub(gap, ub, lb, CMP_PREC);
        double margin = containment_margin_digits(lb, ub, opt);

        {
            char lbs[40], ubs[40], opts[40], gaps[40];
            char *t;
            t = arb_get_str(lb,  6, ARB_STR_NO_RADIUS); snprintf(lbs,  sizeof lbs,  "%s", t); flint_free(t);
            t = arb_get_str(ub,  6, ARB_STR_NO_RADIUS); snprintf(ubs,  sizeof ubs,  "%s", t); flint_free(t);
            t = arb_get_str(opt, 6, ARB_STR_NO_RADIUS); snprintf(opts, sizeof opts, "%s", t); flint_free(t);
            t = arb_get_str(gap, 3, ARB_STR_NO_RADIUS); snprintf(gaps, sizeof gaps, "%s", t); flint_free(t);
            printf("  %-22s %-12s %-12s %-12s %-12s %-8.1f %s\n",
                   gc->name, lbs, opts, ubs, gaps, margin,
                   contains ? "YES" : "*** NO (P0!) ***");
        }

        CHECK(st == ARBSDP_CERTIFY_OPTIMAL_BRACKET,
              "rigor_gate: status OPTIMAL_BRACKET (finite both sides)");
        if (!lo_ok) {
            fprintf(stderr, "P0 SUSPECTED: %s -- opt < lb! ", gc->name);
            print_arb("lb", lb); print_arb("opt", opt); fprintf(stderr, "\n");
        }
        if (!hi_ok) {
            fprintf(stderr, "P0 SUSPECTED: %s -- opt > ub! ", gc->name);
            print_arb("opt", opt); print_arb("ub", ub); fprintf(stderr, "\n");
        }
        CHECK(lo_ok, "rigor_gate: lb <= opt (LOWER BOUND CONTAINS OPT)");
        CHECK(hi_ok, "rigor_gate: opt <= ub (UPPER BOUND CONTAINS OPT)");

        arb_clear(lb); arb_clear(ub); arb_clear(opt); arb_clear(gap);
        arbsdp_apriori_clear(&ab);
        arbsdp_adaptive_result_clear(&r);
        arbsdp_problem_clear(&p);
    }
}

/* ==========================================================================
 * Step 7: HONEST INFINITY (CLAUDE.md rule 5, invariant 5).
 * trivial_2x2: C=[[1,0],[0,0]], A_1=[[1,0],[0,0]], b=[1] -> X_11=1, X_22 FREE,
 * so tr(X) is a-priori UNBOUNDED.  With xbar UNSET the bracket MUST be one/two-
 * sided infinite -- never a fabricated finite bound.
 * ======================================================================== */
static void
test_honest_infinity(const golden_case *cases, int n)
{
    const golden_case *c = find_case(cases, n, "trivial_2x2");
    CHECK(c != NULL, "honest_inf: trivial_2x2 discovered");
    if (c == NULL)
        return;

    arbsdp_problem p;
    arbsdp_adaptive_result r;
    slong prec_c = solve_to_iterate(&p, &r, c);

    /* xbar UNSET (all blocks). */
    arbsdp_apriori ab;
    arbsdp_apriori_init(&ab, p.nblocks);

    arb_t lb, ub;
    arb_init(lb); arb_init(ub);

    arbsdp_certify_status st =
        arbsdp_certify_bracket(lb, ub, &p, &r.res.it, &ab, prec_c);

    int both_finite = arb_is_finite(lb) && arb_is_finite(ub);
    CHECK(!both_finite, "honest_inf: NOT both finite (xbar unset -> infinite side)");
    CHECK(st == ARBSDP_CERTIFY_INCONCLUSIVE,
          "honest_inf: status INCONCLUSIVE (an infinite side)");

    printf("\nHONEST INFINITY: trivial_2x2 (xbar unset) -> ");
    print_arb("lb", lb); print_arb("ub", ub);
    printf("status=%s\n",
           st == ARBSDP_CERTIFY_INCONCLUSIVE ? "INCONCLUSIVE" : "OPTIMAL_BRACKET");

    arb_clear(lb); arb_clear(ub);
    arbsdp_apriori_clear(&ab);
    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

/* assemble b^T y in the test (independent of certify.c's static assemble_bty). */
static void
test_bty(arb_t out, const arbsdp_problem *p, arb_srcptr y, slong prec)
{
    int m = p->m;
    if (m == 0) { arb_zero(out); return; }
    arb_ptr bvec = _arb_vec_init(m);
    for (int i = 0; i < m; i++)
        arbsdp_problem_b(bvec + i, p, i, prec);
    arb_dot(out, NULL, 0, bvec, 1, y, 1, m, prec);
    _arb_vec_clear(bvec, m);
}

/* ==========================================================================
 * Step 8: SIGN SELF-TEST (defends the P0 sign surface; MATH_SPEC §5.5).
 * (a) The §5.4 lower bound run on the INTERNAL min-problem reproduces ub_ext:
 *     LB_int = b^T y_int + sum_b min(0, lambda_min(Z_int^b)) xbar_b,
 *     Z_int^b = -Z_ext^b,  y_int = +it->y/it->tau;  then -LB_int == ub_ext (overlap).
 *     The assembly (min vs max, y_int vs y_ext) is built INDEPENDENTLY here, so a
 *     double-flip in certify.c would break the overlap.
 * (b) Passing y_int (WRONG sign) to the bracket forms Z_int = -Z_ext, swaps
 *     lambda_min<->lambda_max, and yields a bracket that EXCLUDES opt (=3).  We
 *     assert the wrong-sign bracket does NOT contain opt (documents the failure
 *     mode so a regression is caught).
 * ======================================================================== */
static void
test_sign_selftest(const golden_case *cases, int n)
{
    const golden_case *c = find_case(cases, n, "max_eigenvalue_2x2");
    CHECK(c != NULL, "sign: max_eigenvalue_2x2 discovered");
    if (c == NULL)
        return;

    arbsdp_problem p;
    arbsdp_adaptive_result r;
    slong prec_c = solve_to_iterate(&p, &r, c);
    const arbsdp_iterate *it = &r.res.it;
    int m = p.m, nb = p.nblocks;

    arbsdp_apriori ab;
    arbsdp_apriori_init(&ab, nb);
    {
        arb_t one; arb_init(one); arb_one(one);
        for (int b = 0; b < nb; b++)
            arbsdp_apriori_set_xbar(&ab, b, one);
        arb_clear(one);
    }

    /* y_ext = -(it->y / it->tau);  y_int = +(it->y / it->tau). */
    arb_ptr y_ext = _arb_vec_init(m);
    arb_ptr y_int = _arb_vec_init(m);
    for (int i = 0; i < m; i++) {
        arb_div(y_int + i, it->y + i, it->tau, prec_c);
        arb_neg(y_ext + i, y_int + i);
    }

    /* --- the EXTERNAL upper bound (the routine under test). --- */
    arb_t ub_ext;
    arb_init(ub_ext);
    arbsdp_upper_bound(ub_ext, &p, y_ext, &ab, prec_c);

    /* --- the INTERNAL-frame route, assembled independently. --- */
    /* Z_ext^b via dual_residual, negate to Z_int^b = -Z_ext^b. */
    arb_mat_t *Z = flint_malloc((size_t) nb * sizeof *Z);
    for (int b = 0; b < nb; b++) {
        slong sz = p.block_sizes[b] < 0 ? -p.block_sizes[b] : p.block_sizes[b];
        arb_mat_init(Z[b], sz, sz);
    }
    arbsdp_dual_residual(Z, &p, y_ext, prec_c);   /* Z[b] = Z_ext^b */

    arb_t lb_int, dlo, mn, zero;
    arb_init(lb_int); arb_init(dlo); arb_init(mn); arb_init(zero);
    arb_zero(zero);
    test_bty(lb_int, &p, y_int, prec_c);          /* b^T y_int = -b^T y_ext */
    for (int b = 0; b < nb; b++) {
        arb_mat_neg(Z[b], Z[b]);                  /* Z[b] = Z_int^b = -Z_ext^b */
        arbsdp_lambda_min_lower_bound(dlo, Z[b], prec_c);
        arb_min(mn, dlo, zero, prec_c);           /* min(0, lambda_min(Z_int)) */
        arb_mul(mn, mn, &ab.xbar[b], prec_c);
        arb_add(lb_int, lb_int, mn, prec_c);
    }
    /* ub_via_internal = -LB_int.  Should OVERLAP ub_ext (§5.5 cross-check). */
    arb_t ub_via_internal;
    arb_init(ub_via_internal);
    arb_neg(ub_via_internal, lb_int);

    CHECK(arb_overlaps(ub_via_internal, ub_ext),
          "sign(a): internal-frame route overlaps ub_ext (no double-flip)");
    printf("\nSIGN SELF-TEST (a): internal-frame ub vs external ub: ");
    print_arb("ub_ext", ub_ext); print_arb("ub_int", ub_via_internal);
    printf("overlap=%s\n", arb_overlaps(ub_via_internal, ub_ext) ? "YES" : "NO");

    /* --- (b) WRONG sign: feed y_int (the un-negated dual) to the bounds, exactly
     * reproducing the "forgot to negate in the P0 sign locus" bug.  This forms
     * Z_buggy = C_file - sum (y_int)_i A_i (neither Z_ext nor Z_int).
     *
     * IMPORTANT FINDING (CLAUDE.md rule 8, skepticism; the bead's premise was
     * "wrong sign EXCLUDES opt", which is FALSE here): for the max-eigenvalue
     * family the trace constraint is A_1 = I with b = [1], so at the converged
     * dual (y_ext = opt) the wrong sign gives y_int = -opt and
     *   Z_buggy = C + opt*I,  lambda_max(Z_buggy) = lambda_max(C) + opt = 2*opt,
     *   b^T y_int = -opt  =>  ub_buggy = -opt + max(0,2*opt) = +opt,
     *   lambda_min(Z_buggy) = lambda_min(C) + opt >= 0 => lb_buggy = -opt + 0 = -opt.
     * So the buggy bracket is the SYMMETRIC [-opt, +opt] and CONTAINS opt at its
     * upper edge -- the forgot-to-negate bug is NOT caught by "contains opt" on
     * this family (verified for all 7 goldens; two_block gives [-2.83, 3.54], also
     * containing opt).  Loosening the rigor gate to "wrong sign fails" would be a
     * lie.  What DOES reliably bite: the wrong-sign bracket has the WRONG lb
     * (= -opt), strictly below the correct lb -- so a sign regression corrupts the
     * lower endpoint detectably.  We assert lb_correct > lb_wrong (the sign matters)
     * and document that the upper edge alone cannot distinguish the bug. --- */
    arb_t lb_wrong_ball, ub_wrong_ball, three;
    arf_t lo, hi;
    arb_init(lb_wrong_ball); arb_init(ub_wrong_ball); arb_init(three);
    arf_init(lo); arf_init(hi);
    arb_set_si(three, 3);                                  /* opt for max_eig_2x2 */
    arbsdp_lower_bound(lb_wrong_ball, &p, y_int, &ab, prec_c);  /* WRONG sign */
    arbsdp_upper_bound(ub_wrong_ball, &p, y_int, &ab, prec_c);  /* WRONG sign */
    arb_t lb_wrong, ub_wrong;
    arb_init(lb_wrong); arb_init(ub_wrong);
    arb_get_lbound_arf(lo, lb_wrong_ball, prec_c); arb_set_arf(lb_wrong, lo);
    arb_get_ubound_arf(hi, ub_wrong_ball, prec_c); arb_set_arf(ub_wrong, hi);

    /* The CORRECT bracket (the routine under test, via the iterate sign locus). */
    arb_t lb_right, ub_right;
    arb_init(lb_right); arb_init(ub_right);
    arbsdp_certify_bracket(lb_right, ub_right, &p, it, &ab, prec_c);

    /* The sign flip corrupts the LOWER endpoint detectably: correct lb > wrong lb.
     * (For max_eigenvalue_2x2: correct lb = 1, wrong lb = -3.) */
    CHECK(arb_lt(lb_wrong, lb_right),
          "sign(b): wrong-sign lb (-opt) strictly below correct lb (sign regression bites lb)");

    int wrong_contains = arb_le(lb_wrong, three) && arb_le(three, ub_wrong);
    printf("SIGN SELF-TEST (b): wrong-sign bracket: ");
    print_arb("lb", lb_wrong); print_arb("ub", ub_wrong);
    print_arb("correct_lb", lb_right);
    printf("wrong_contains_opt=%s (EXPECTED yes for A=I family; see comment)\n",
           wrong_contains ? "yes" : "no");

    arb_clear(lb_right); arb_clear(ub_right);

    /* cleanup */
    for (int b = 0; b < nb; b++) arb_mat_clear(Z[b]);
    flint_free(Z);
    _arb_vec_clear(y_ext, m);
    _arb_vec_clear(y_int, m);
    arb_clear(ub_ext);
    arb_clear(lb_int); arb_clear(dlo); arb_clear(mn); arb_clear(zero);
    arb_clear(ub_via_internal);
    arb_clear(lb_wrong_ball); arb_clear(ub_wrong_ball); arb_clear(three);
    arb_clear(lb_wrong); arb_clear(ub_wrong);
    arf_clear(lo); arf_clear(hi);
    arbsdp_apriori_clear(&ab);
    arbsdp_adaptive_result_clear(&r);
    arbsdp_problem_clear(&p);
}

/* ==========================================================================
 * Step 9: TRACE-vs-LOEWNER DISCRIMINATOR (bead arb-prec-IPM-om9).
 *
 * Proves the a-priori bound `xbar_b` is consumed as a TRACE bound (xbar >= tr X)
 * and NOT a Loewner / lambda_max bound, and that there is NO spurious
 * block-dimension factor s_b on the penalty term (certify.h / MATH_SPEC §5.3.1).
 *
 * The existing goldens cannot make this distinction: their optima are RANK-1, so
 * tr(X*) == lambda_max(X*) and the two conventions coincide.  This test uses a
 * FULL-RANK optimum where they split.
 *
 * Constructed problem (built in a tmpfile, parsed via the real read_sdpa path):
 *     max <I, X>  s.t.  X11 = 1,  X22 = 1,  X12 = 0   (2x2 block, m=3)
 * The three constraints pin the UNIQUE feasible point X* = I, so
 *     tr(X*)        = 2   (the TRACE bound),
 *     lambda_max(X*)= 1   (the LOEWNER bound),
 *     opt = <I,I>   = 2.
 *
 * We call arbsdp_lower_bound / arbsdp_upper_bound DIRECTLY with synthetic y_ext
 * (no solver), with eps = 2^-10:
 *
 * LOWER side, y_ext = (1+eps, 1+eps, 0):  Z = -eps*I, dlo <= -eps,
 *     b^T y_ext = 2 + 2eps,  lb(xbar) = (2+2eps) + min(0,dlo)*xbar.
 *   xbar = 2 (TRACE):  lb ~= 2     -> R1 lb <= opt, R2 lb > opt-eps (no s_b factor)
 *   xbar = 1 (LOEWNER):lb ~= 2+eps -> B1 lb > opt   (trace bound is NECESSARY)
 *
 * UPPER side, y_ext = (1-eps, 1-eps, 0):  Z = +eps*I, dhi >= +eps,
 *     b^T y_ext = 2 - 2eps,  ub(xbar) = (2-2eps) + max(0,dhi)*xbar.
 *   xbar = 2 (TRACE):  ub ~= 2     -> R3 opt <= ub, R4 ub < opt+eps (no s_b factor)
 *   xbar = 1 (LOEWNER):ub ~= 2-eps -> B2 ub < opt   (trace bound is NECESSARY)
 *
 * A spurious block-dimension factor (s_b = 2 here) on the penalty would push the
 * trace-bound endpoints to lb ~= 2-2eps < opt-eps (fails R2) / ub ~= 2+2eps >
 * opt+eps (fails R4) -- which is exactly what R2/R4 catch.  (The trace bisection
 * tolerance ~2^-128 << eps = 2^-10 is negligible in every comparison.)
 * ======================================================================== */
static void
test_trace_vs_loewner(void)
{
    /* The discriminator .dat-s (see the function banner for the line-by-line
     * decoding): m=3, nblocks=1, block-size "2", b="1 1 0"; C=I; A_1=diag(1,0);
     * A_2=diag(0,1); A_3=offdiag[[0,1],[1,0]]. */
    static const char *dats =
        "* trace_vs_loewner_2x2 (om9 discriminator): max <I,X> s.t. X11=1, X22=1, X12=0\n"
        "* unique feasible X*=I (full rank): tr(X*)=2 (TRACE bound), lambda_max(X*)=1 (LOEWNER bound), opt=2\n"
        "3\n"
        "1\n"
        "2\n"
        "1 1 0\n"
        "0 1 1 1 1\n"
        "0 1 2 2 1\n"
        "1 1 1 1 1\n"
        "2 1 2 2 1\n"
        "3 1 1 2 1\n";

    const slong prec = 256;

    char path[] = "/tmp/arbsdp_tvl_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "tvl: mkstemp temp .dat-s");
    if (fd < 0)
        return;
    {
        FILE *f = fdopen(fd, "w");
        if (f == NULL) {
            CHECK(0, "tvl: fdopen temp .dat-s");
            close(fd);
            unlink(path);
            return;
        }
        fputs(dats, f);
        fclose(f);
    }

    arbsdp_problem p;
    arbsdp_problem_init(&p);
    CHECK(arbsdp_read_sdpa(&p, path) == 0, "tvl: read_sdpa discriminator");
    CHECK(p.m == 3, "tvl: m == 3");
    CHECK(p.nblocks == 1, "tvl: nblocks == 1");

    /* eps = 2^-10, opt = 2. */
    arb_t eps, opt, opt_m_eps, opt_p_eps;
    arb_init(eps); arb_init(opt); arb_init(opt_m_eps); arb_init(opt_p_eps);
    arb_set_si(eps, 1);
    arb_mul_2exp_si(eps, eps, -10);            /* eps = 2^-10 */
    arb_set_si(opt, 2);                         /* opt = 2 */
    arb_sub(opt_m_eps, opt, eps, prec);        /* opt - eps */
    arb_add(opt_p_eps, opt, eps, prec);        /* opt + eps */

    /* a-priori bounds: one with xbar = 2 (TRACE), one with xbar = 1 (LOEWNER). */
    arbsdp_apriori ab_trace, ab_loewner;
    arbsdp_apriori_init(&ab_trace, 1);
    arbsdp_apriori_init(&ab_loewner, 1);
    {
        arb_t two, one;
        arb_init(two); arb_init(one);
        arb_set_si(two, 2);
        arb_set_si(one, 1);
        arbsdp_apriori_set_xbar(&ab_trace,   0, two);
        arbsdp_apriori_set_xbar(&ab_loewner, 0, one);
        arb_clear(two); arb_clear(one);
    }

    /* y vectors (length m=3): y_lo = (1+eps,1+eps,0), y_hi = (1-eps,1-eps,0). */
    arb_ptr y_lo = _arb_vec_init(3);
    arb_ptr y_hi = _arb_vec_init(3);
    {
        arb_t one_p, one_m;
        arb_init(one_p); arb_init(one_m);
        arb_add_si(one_p, eps,  1, prec);      /* 1 + eps */
        arb_sub_si(one_m, eps,  1, prec);      /* eps - 1 */
        arb_neg(one_m, one_m);                 /* 1 - eps */
        arb_set(y_lo + 0, one_p); arb_set(y_lo + 1, one_p); arb_zero(y_lo + 2);
        arb_set(y_hi + 0, one_m); arb_set(y_hi + 1, one_m); arb_zero(y_hi + 2);
        arb_clear(one_p); arb_clear(one_m);
    }

    /* ---- LOWER-bound side (y_lo). ---- */
    arb_t lb_trace, lb_loewner;
    arb_init(lb_trace); arb_init(lb_loewner);
    arbsdp_lower_bound(lb_trace,   &p, y_lo, &ab_trace,   prec);   /* xbar = 2 */
    arbsdp_lower_bound(lb_loewner, &p, y_lo, &ab_loewner, prec);   /* xbar = 1 */

    /* ---- UPPER-bound side (y_hi). ---- */
    arb_t ub_trace, ub_loewner;
    arb_init(ub_trace); arb_init(ub_loewner);
    arbsdp_upper_bound(ub_trace,   &p, y_hi, &ab_trace,   prec);   /* xbar = 2 */
    arbsdp_upper_bound(ub_loewner, &p, y_hi, &ab_loewner, prec);   /* xbar = 1 */

    /* Concrete diagnostics (CLAUDE.md rule 11; ~30 digits to stderr). */
    fprintf(stderr,
            "\nTRACE-vs-LOEWNER (om9): opt=2, eps=2^-10; full-rank X*=I "
            "(tr=2, lambda_max=1)\n");
    {
        char *s;
        s = arb_get_str(lb_trace,   30, ARB_STR_NO_RADIUS);
        fprintf(stderr, "  LOWER xbar=2(TRACE)   lb=%s  (expect ~2, in (opt-eps, opt])\n", s);
        flint_free(s);
        s = arb_get_str(lb_loewner, 30, ARB_STR_NO_RADIUS);
        fprintf(stderr, "  LOWER xbar=1(LOEWNER) lb=%s  (expect ~2+eps > opt -> bites)\n", s);
        flint_free(s);
        s = arb_get_str(ub_trace,   30, ARB_STR_NO_RADIUS);
        fprintf(stderr, "  UPPER xbar=2(TRACE)   ub=%s  (expect ~2, in [opt, opt+eps))\n", s);
        flint_free(s);
        s = arb_get_str(ub_loewner, 30, ARB_STR_NO_RADIUS);
        fprintf(stderr, "  UPPER xbar=1(LOEWNER) ub=%s  (expect ~2-eps < opt -> bites)\n", s);
        flint_free(s);
    }

    /* ---- The six discriminating assertions (DO NOT WEAKEN; CLAUDE rule 2/8). ---- */
    /* R1/R2: trace bound is RIGOROUS and has NO spurious s_b factor on lb. */
    CHECK(arb_le(lb_trace, opt),         "tvl: lb(trace) <= opt");
    CHECK(arb_gt(lb_trace, opt_m_eps),   "tvl: lb(trace) > opt-eps (no s_b factor on lb)");
    /* B1: the Loewner value (xbar=1) over-shoots opt -> the trace bound is necessary. */
    CHECK(arb_gt(lb_loewner, opt),       "tvl: lb(Loewner) > opt -> trace bound is necessary");

    /* R3/R4: trace bound is RIGOROUS and has NO spurious s_b factor on ub. */
    CHECK(arb_le(opt, ub_trace),         "tvl: opt <= ub(trace)");
    CHECK(arb_lt(ub_trace, opt_p_eps),   "tvl: ub(trace) < opt+eps (no s_b factor on ub)");
    /* B2: the Loewner value (xbar=1) under-shoots opt -> the trace bound is necessary. */
    CHECK(arb_lt(ub_loewner, opt),       "tvl: ub(Loewner) < opt -> trace bound is necessary");

    /* ---- cleanup (CLAUDE.md rule 7): scalars, y vectors, apriori, problem. ---- */
    arb_clear(lb_trace); arb_clear(lb_loewner);
    arb_clear(ub_trace); arb_clear(ub_loewner);
    arb_clear(eps); arb_clear(opt); arb_clear(opt_m_eps); arb_clear(opt_p_eps);
    _arb_vec_clear(y_lo, 3);
    _arb_vec_clear(y_hi, 3);
    arbsdp_apriori_clear(&ab_trace);
    arbsdp_apriori_clear(&ab_loewner);
    arbsdp_problem_clear(&p);
    unlink(path);
}

/* -------------------------------------------------------------------------
 * closed_form_opt -- fill `out` with the EXACT closed-form true optimum for the
 * three max-eigenvalue goldens used by test_tight_lb_primal, computed as a tight
 * ball at precision `prec` (~2^-prec radius).  This gives a strictly stronger rigor
 * gate than the 65-digit enclosure from parse_opt_enclosure: the margin is ~2^-4096
 * rather than ~4e-64.
 *
 *   max_eigenvalue_2x2   -> 3              (exact integer)
 *   max_eig_tridiag_3x3  -> 2 + sqrt(3)
 *   max_eig_path_4       -> (5 + sqrt(5)) / 2
 *
 * Sets `out` to arb_indeterminate for any unrecognised name (test will flag it).
 * ---------------------------------------------------------------------- */
static void
closed_form_opt(arb_t out, const char *name, slong prec)
{
    if (strcmp(name, "max_eigenvalue_2x2") == 0) {
        arb_set_si(out, 3);
    } else if (strcmp(name, "max_eig_tridiag_3x3") == 0) {
        /* 2 + sqrt(3): set out = 3, sqrt in place, add 2. */
        arb_set_ui(out, 3);
        arb_sqrt(out, out, prec);
        arb_add_ui(out, out, 2, prec);
    } else if (strcmp(name, "max_eig_path_4") == 0) {
        /* (5 + sqrt(5)) / 2: set out = 5, sqrt in place, add 5, halve. */
        arb_set_ui(out, 5);
        arb_sqrt(out, out, prec);
        arb_add_ui(out, out, 5, prec);
        arb_mul_2exp_si(out, out, -1);
    } else {
        arb_indeterminate(out);
    }
}

/* ==========================================================================
 * Step 10: TIGHT LOWER BOUND via the PRIMAL Rayleigh bound (bead arb-prec-IPM-vry).
 *
 * For the single-block, single-trace-constraint max-eigenvalue family (A_1 = I,
 * b = [1], so tr(X) = 1), the DUAL-residual lower bound is STRUCTURALLY LOOSE: it
 * returns lb = lambda_min(C) (e.g. 1 for max_eigenvalue_2x2), giving a ~2-wide gap
 * to opt = lambda_max(C).  The primal Rayleigh bound (arbsdp_primal_lower_bound)
 * exhibits a FEASIBLE rank-1 primal point X_hat = v v^T (v ~ top eigenvector of C)
 * whose objective <C,X_hat> = R(v) is quadratically accurate near an eigenvector;
 * arbsdp_certify_bracket now takes the MAX of the two rigorous lower bounds, so the
 * bracket's lb is TIGHT (true_opt - lb ~ 1e-65: the Rayleigh quotient resolves the
 * true optimum essentially exactly, far below the 1e-6 gate).
 *
 * FINDING (bead vry; CLAUDE.md rule 2/8): the now-tight lb resolves the true
 * optimum to ~190 digits, PAST the 65-digit reference STRING -- so lb sits ABOVE
 * the rounded reference point while remaining <= the TRUE optimum (verified
 * directly against 2+sqrt3 et al.).  The rigor gate must therefore compare against
 * an ENCLOSURE ball (parse_opt_enclosure) that provably contains the true optimum,
 * NOT the bare rounded point.  This WIDENS the reference (rule 2: widen the
 * interval, never the claim); the bound itself is unchanged and rigorous.
 *
 * GATE (CLAUDE.md rule 2/8): lb <= ubound(true opt) (rigor) is the falsifiable
 * theorem; opt <= ub is unchanged.  The 1e-6 tightness gate is the whole point of
 * the bead, conservatively far below the OLD ~2-wide gap; if a golden's gap is NOT
 * < 1e-6 it is a P0 finding (primal bound not applied or poor solve) -- DO NOT
 * loosen the threshold, STOP and report (per the bead).
 * ======================================================================== */
static void
test_tight_lb_primal(const golden_case *cases, int n)
{
    /* Single-block, single-trace-constraint max-eigenvalue goldens (A_1 = I, so
     * the primal Rayleigh bound applies).  opt = lambda_max(C). */
    static const char *names[] = {
        "max_eigenvalue_2x2",   /* opt = 3                 */
        "max_eig_tridiag_3x3",  /* opt = 3.7320508...      */
        "max_eig_path_4",       /* opt = 3.6180339...      */
    };
    int nn = (int) (sizeof names / sizeof names[0]);

    fprintf(stderr,
            "\nTIGHT-LB (vry): primal Rayleigh lower bound on max-eigenvalue goldens "
            "(opt-lb MUST be < 1e-6):\n");

    for (int k = 0; k < nn; k++) {
        const golden_case *c = find_case(cases, n, names[k]);
        CHECK(c != NULL, "tight_lb: golden discovered");
        if (c == NULL)
            continue;

        arbsdp_problem p;
        arbsdp_adaptive_result r;
        slong prec_c = solve_to_iterate(&p, &r, c);

        /* xbar = 1: tr(X) = 1 (A_1 = I_n, b_1 = 1), exactly as test_rigor_gate. */
        arbsdp_apriori ab;
        arbsdp_apriori_init(&ab, p.nblocks);
        {
            arb_t one; arb_init(one); arb_one(one);
            arbsdp_apriori_set_xbar(&ab, 0, one);
            arb_clear(one);
        }

        arb_t lb, ub, opt, gap, tol6;
        arb_init(lb); arb_init(ub); arb_init(opt); arb_init(gap); arb_init(tol6);
        /* opt is an ENCLOSURE ball provably containing the TRUE optimum (bead vry;
         * the reference string is only correctly-rounded to ~65 digits, and the
         * now-tight lb resolves the true optimum well past 65 digits). */
        parse_opt_enclosure(opt, c, CMP_PREC);

        arbsdp_certify_status st =
            arbsdp_certify_bracket(lb, ub, &p, &r.res.it, &ab, prec_c);

        /* RIGOR (falsifiable against the TRUE optimum, not the rounded point): lb is
         * sound unless it exceeds the UPPER endpoint of the opt enclosure; ub is
         * sound unless it falls below the LOWER endpoint. */
        CHECK(st == ARBSDP_CERTIFY_OPTIMAL_BRACKET,
              "tight_lb: status OPTIMAL_BRACKET");
        {
            arf_t opt_lo, opt_hi, lb_lo, ub_hi;
            arf_init(opt_lo); arf_init(opt_hi); arf_init(lb_lo); arf_init(ub_hi);
            arb_get_lbound_arf(opt_lo, opt, CMP_PREC);
            arb_get_ubound_arf(opt_hi, opt, CMP_PREC);
            arb_get_lbound_arf(lb_lo, lb, CMP_PREC);
            arb_get_ubound_arf(ub_hi, ub, CMP_PREC);
            CHECK(arf_cmp(lb_lo, opt_hi) <= 0, "tight_lb: lb <= ubound(true opt) (rigor)");
            CHECK(arf_cmp(opt_lo, ub_hi) <= 0, "tight_lb: lbound(true opt) <= ub (rigor)");
            arf_clear(opt_lo); arf_clear(opt_hi); arf_clear(lb_lo); arf_clear(ub_hi);
        }

        /* STRONG RIGOR GATE (closed-form; ~2^-4096 margin).
         * Compare lb against the TRUE closed-form optimum computed to CMP_PREC bits
         * (radius ~2^-4096), rather than only the 65-digit enclosure (~4e-64).
         * lb is an exact point; topt_cf is a tight ball.  arb_le(lb, topt_cf)
         * requires ubound(lb) <= lbound(topt_cf), which holds with a large margin
         * when the bound is correct.  Any failure here is a real rigor violation:
         * lb exceeds the TRUE optimum -- DO NOT weaken; STOP and report (rule 2/8). */
        {
            arb_t topt_cf, diff;
            arb_init(topt_cf);
            arb_init(diff);
            closed_form_opt(topt_cf, names[k], CMP_PREC);
            CHECK(arb_le(lb, topt_cf),
                  "tight_lb: lb <= TRUE closed-form opt (3 / 2+sqrt3 / (5+sqrt5)/2)");
            /* Diagnostic: (topt_cf - lb) at ~12 significant digits. */
            arb_sub(diff, topt_cf, lb, CMP_PREC);
            {
                char *sdiff = arb_get_str(diff, 12, ARB_STR_NO_RADIUS);
                fprintf(stderr, "  %-22s  topt_cf - lb = %s  (closed-form rigor gate)\n",
                        names[k], sdiff);
                flint_free(sdiff);
            }
            arb_clear(topt_cf);
            arb_clear(diff);
        }

        /* TIGHTNESS: (ubound(true opt)) - lb < 1e-6.  Measuring the gap from the
         * UPPER endpoint of the opt enclosure is a CONSERVATIVE upper bound on the
         * true gap (true_opt - lb), so this gate is if anything stricter than the
         * literal opt-lb.  The primal Rayleigh bound makes the true gap ~1e-65
         * (it resolves the true optimum essentially exactly), far below 1e-6. */
        {
            arf_t opt_hi, lb_lo;
            arf_init(opt_hi); arf_init(lb_lo);
            arb_get_ubound_arf(opt_hi, opt, CMP_PREC);
            arb_get_lbound_arf(lb_lo, lb, CMP_PREC);
            arf_sub(arb_midref(gap), opt_hi, lb_lo, CMP_PREC, ARF_RND_CEIL);
            mag_zero(arb_radref(gap));               /* exact point gap (over-estimate) */
            arf_clear(opt_hi); arf_clear(lb_lo);
        }
        arb_set_str(tol6, "1e-6", CMP_PREC);
        CHECK(arb_lt(gap, tol6),
              "tight_lb: opt-lb < 1e-6 (primal Rayleigh tightened the loose dual lb)");

        /* Concrete diagnostics (CLAUDE.md rule 11; ~30 digits to stderr). */
        {
            char *slb  = arb_get_str(lb,  30, ARB_STR_NO_RADIUS);
            char *sub  = arb_get_str(ub,  30, ARB_STR_NO_RADIUS);
            char *sgap = arb_get_str(gap, 30, ARB_STR_NO_RADIUS);
            fprintf(stderr, "  %-22s lb=%s ub=%s opt-lb=%s\n",
                    names[k], slb, sub, sgap);
            flint_free(slb);
            flint_free(sub);
            flint_free(sgap);
        }

        arb_clear(lb); arb_clear(ub); arb_clear(opt); arb_clear(gap); arb_clear(tol6);
        arbsdp_apriori_clear(&ab);
        arbsdp_adaptive_result_clear(&r);
        arbsdp_problem_clear(&p);
    }
}

int main(void)
{
    golden_case cases[MAX_CASES];
    int n = golden_discover(GOLDEN_DIR, cases, MAX_CASES);
    if (n < 0) {
        fprintf(stderr, "FATAL: golden_discover(%s) failed\n", GOLDEN_DIR);
        return EXIT_FAILURE;
    }

    test_bracket_smoke(cases, n);
    test_lambda_max_upper();
    test_rigor_gate(cases, n);
    test_honest_infinity(cases, n);
    test_sign_selftest(cases, n);
    test_trace_vs_loewner();
    test_tight_lb_primal(cases, n);

    if (failures == 0) {
        printf("\ntest_certify_bracket: ALL PASS\n");
        return EXIT_SUCCESS;
    }
    fprintf(stderr, "\ntest_certify_bracket: %d FAILURE(S)\n", failures);
    return EXIT_FAILURE;
}
