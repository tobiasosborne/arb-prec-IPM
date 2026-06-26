/*
 * test/test_serialize.c -- JSON result serializer + the rigour-critical directed
 * decimal helper (bead arb-prec-IPM-b21).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written against a STUB io.c first (the stub
 * returns "0" / "{}" so test 1's directed-rounding assertions FAIL loudly = RED),
 * then the real impl makes it GREEN.
 *
 * ==========================================================================
 * RIGOR IS FALSIFIABLE -- THE GATE (CLAUDE.md rule 2)
 * ==========================================================================
 * Test 1 is the rigour-critical unit test: arbsdp_arb_decimal(x, 0, d) re-parsed
 * MUST be <= x and arbsdp_arb_decimal(x, 1, d) re-parsed MUST be >= x, so the
 * printed [obj_lb, obj_ub] still ENCLOSES the optimum after decimal rounding.
 * If that property fails, STOP and fix arbsdp_arb_decimal -- do not weaken it.
 *
 * Coverage:
 *   1. RIGOR UNIT: directed decimals of 2/3, -2/3, exact 1.5, +/-inf.
 *   2. INTEGRATION optimal: solve+certify max_eigenvalue_2x2, serialize, assert
 *      certificate status "optimal", obj_lb/obj_ub present and enclose opt=3,
 *      and (if python3 present) the JSON parses.
 *   3. INTEGRATION infeasible: dual_infeasible_diag -> status "dual_infeasible",
 *      obj_lb "-inf", obj_ub "+inf".
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear and
 * every malloc'd string is freed.
 */

/* popen/pclose are POSIX; declare them before any standard header. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flint/flint.h>
#include <flint/arb.h>

#include "arbsdp/io.h"
#include "arbsdp/solve.h"
#include "arbsdp/precision.h"
#include "arbsdp/certify.h"

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "../../golden"
#endif

/* Re-parse precision for the rigour checks.  High enough that the re-parse ball
 * radius (~2^-REPARSE_PREC) is far below any realistic gap between a directed
 * decimal and x, so arb_le decides cleanly. */
#define REPARSE_PREC 4096

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

/* ----- Test 1: RIGOR UNIT TEST of arbsdp_arb_decimal ----------------------- */

/* For a finite x: assert the directed decimals ENCLOSE x after re-parsing, and
 * (if must_differ) that directed rounding actually changed the last digit. */
static void
check_directed(const arb_t x, const char *label, int must_differ)
{
    const slong digits = 5;
    char *lo = arbsdp_arb_decimal(x, 0, digits); /* toward -inf: lo <= x */
    char *hi = arbsdp_arb_decimal(x, 1, digits); /* toward +inf: hi >= x */

    arb_t blo, bhi;
    arb_init(blo);
    arb_init(bhi);
    CHECK(arb_set_str(blo, lo, REPARSE_PREC) == 0, "directed lo parses");
    CHECK(arb_set_str(bhi, hi, REPARSE_PREC) == 0, "directed hi parses");

    /* THE RIGOR GATE: lo <= x <= hi (certainly, in ball arithmetic). */
    CHECK(arb_le(blo, x), "directed lower decimal <= x (RIGOR)");
    CHECK(arb_le(x, bhi), "x <= directed upper decimal (RIGOR)");

    if (must_differ)
        CHECK(strcmp(lo, hi) != 0,
              "directed rounding bit the last digit (lo != hi)");

    fprintf(stderr, "  %-6s lo=%s hi=%s%s\n", label, lo, hi,
            must_differ ? "" : " (exact; may match)");

    free(lo);
    free(hi);
    arb_clear(blo);
    arb_clear(bhi);
}

static void
test_decimal_rigor(void)
{
    const slong prec = 256;
    arb_t x;
    arb_init(x);

    fprintf(stderr, "\nRIGOR UNIT (b21): directed decimals must enclose x:\n");

    /* x = 2/3 (irrational decimal): the two directions MUST differ. */
    arb_set_ui(x, 2);
    arb_div_ui(x, x, 3, prec);
    check_directed(x, "2/3", 1);

    /* x = -2/3: directed rounding must respect the sign (RNDD/RNDU vs -inf/+inf). */
    arb_neg(x, x);
    check_directed(x, "-2/3", 1);

    /* exact 1.5: both directions enclose (need not differ). */
    CHECK(arb_set_str(x, "1.5", prec) == 0, "parse 1.5");
    check_directed(x, "1.5", 0);

    arb_clear(x);

    /* +/- infinity. */
    {
        arb_t inf;
        char *s;
        arb_init(inf);

        arb_pos_inf(inf);
        s = arbsdp_arb_decimal(inf, 1, 5);
        CHECK(strcmp(s, "+inf") == 0, "+inf (round_up) -> \"+inf\"");
        free(s);
        s = arbsdp_arb_decimal(inf, 0, 5);
        CHECK(strcmp(s, "+inf") == 0, "+inf (round_down) -> \"+inf\"");
        free(s);

        arb_neg_inf(inf);
        s = arbsdp_arb_decimal(inf, 0, 5);
        CHECK(strcmp(s, "-inf") == 0, "-inf (round_down) -> \"-inf\"");
        free(s);
        s = arbsdp_arb_decimal(inf, 1, 5);
        CHECK(strcmp(s, "-inf") == 0, "-inf (round_up) -> \"-inf\"");
        free(s);

        arb_clear(inf);
    }
}

/* ----- python3 JSON validation (optional sub-check) ------------------------ */
static int
python3_available(void)
{
    return system("command -v python3 >/dev/null 2>&1") == 0;
}

static void
assert_json_parses(const char *json)
{
    if (!python3_available()) {
        printf("NOTE: python3 not on PATH; skipping JSON-parse sub-check\n");
        return;
    }
    FILE *pp = popen("python3 -c 'import json,sys; json.load(sys.stdin)'", "w");
    CHECK(pp != NULL, "popen python3");
    if (pp == NULL)
        return;
    fputs(json, pp);
    int rc = pclose(pp);
    CHECK(rc == 0, "emitted JSON parses via python3 json.load");
}

/* ----- Test 2: INTEGRATION (optimal) -------------------------------------- */
static void
test_json_optimal(void)
{
    char path[1024];
    arbsdp_problem p;
    arbsdp_adaptive_result ar;
    arbsdp_apriori ab;
    arbsdp_precision_params pp;
    arb_t lb, ub, three, rlb, rub;

    golden_path(path, sizeof(path), "max_eigenvalue_2x2");
    arbsdp_problem_init(&p);
    CHECK(arbsdp_read_sdpa(&p, path) == 0, "optimal: read max_eigenvalue_2x2");

    arbsdp_precision_default_params(&pp);
    pp.target_digits = 20;
    arbsdp_solve_adaptive(&ar, &p, &pp);

    /* max-eig: A_1 = I, b = [1] -> tr(X) = 1, so xbar[0] = 1. */
    arbsdp_apriori_init(&ab, p.nblocks);
    {
        arb_t one;
        arb_init(one);
        arb_one(one);
        arbsdp_apriori_set_xbar(&ab, 0, one);
        arb_clear(one);
    }

    slong prec_c = ar.final_prec + 128;
    arb_init(lb);
    arb_init(ub);
    arbsdp_certify_status cs = arbsdp_certify(lb, ub, &p, &ar.res.it, &ab, prec_c);
    CHECK(cs == ARBSDP_CERTIFY_OPTIMAL_BRACKET, "optimal: certify OPTIMAL_BRACKET");

    char *json = arbsdp_result_to_json(&p, &ar, cs, lb, ub, prec_c, &ab);
    CHECK(json != NULL, "optimal: to_json non-NULL");

    /* Schema substrings (assert on what we actually emit: ": " spacing). */
    CHECK(strstr(json, "\"status\": \"optimal\"") != NULL,
          "optimal: certificate status \"optimal\"");
    CHECK(strstr(json, "\"obj_lb\"") != NULL, "optimal: has obj_lb");
    CHECK(strstr(json, "\"obj_ub\"") != NULL, "optimal: has obj_ub");
    CHECK(strstr(json, "\"arbsdp_version\"") != NULL, "optimal: has arbsdp_version");
    CHECK(strstr(json, "\"trace_bounds\"") != NULL, "optimal: has trace_bounds");

    /* The serialized obj_lb / obj_ub must still ENCLOSE the true optimum 3.
     * Recompute the same directed decimals (same args to_json uses), confirm they
     * appear verbatim in the JSON, then re-parse and check 3 is inside. */
    slong digits_c = (slong)((double) prec_c * 0.301) + 1;
    char *slb = arbsdp_arb_decimal(lb, 0, digits_c);
    char *sub = arbsdp_arb_decimal(ub, 1, digits_c);
    CHECK(strstr(json, slb) != NULL, "optimal: obj_lb value present in JSON");
    CHECK(strstr(json, sub) != NULL, "optimal: obj_ub value present in JSON");

    arb_init(three);
    arb_init(rlb);
    arb_init(rub);
    arb_set_si(three, 3);
    CHECK(arb_set_str(rlb, slb, REPARSE_PREC) == 0, "optimal: obj_lb re-parses");
    CHECK(arb_set_str(rub, sub, REPARSE_PREC) == 0, "optimal: obj_ub re-parses");
    CHECK(arb_le(rlb, three), "optimal: obj_lb <= 3 (opt enclosed)");
    CHECK(arb_le(three, rub), "optimal: 3 <= obj_ub (opt enclosed)");

    fprintf(stderr, "\nINTEGRATION optimal (max_eigenvalue_2x2): obj_lb=%s obj_ub=%s\n",
            slb, sub);

    assert_json_parses(json);

    free(slb);
    free(sub);
    free(json);
    arb_clear(three);
    arb_clear(rlb);
    arb_clear(rub);
    arb_clear(lb);
    arb_clear(ub);
    arbsdp_apriori_clear(&ab);
    arbsdp_adaptive_result_clear(&ar);
    arbsdp_problem_clear(&p);
}

/* ----- Test 3: INTEGRATION (infeasible / unbounded) ----------------------- */
static void
test_json_infeasible(void)
{
    char path[1024];
    arbsdp_problem p;
    arbsdp_adaptive_result ar;
    arbsdp_apriori ab;
    arbsdp_precision_params pp;
    arb_t lb, ub;

    golden_path(path, sizeof(path), "dual_infeasible_diag");
    arbsdp_problem_init(&p);
    CHECK(arbsdp_read_sdpa(&p, path) == 0, "infeasible: read dual_infeasible_diag");

    arbsdp_precision_default_params(&pp);
    pp.target_digits = 20;
    arbsdp_solve_adaptive(&ar, &p, &pp);

    arbsdp_apriori_init(&ab, p.nblocks); /* xbar UNSET (irrelevant here) */

    slong prec_c = ar.final_prec + 128;
    arb_init(lb);
    arb_init(ub);
    arbsdp_certify_status cs = arbsdp_certify(lb, ub, &p, &ar.res.it, &ab, prec_c);
    CHECK(cs == ARBSDP_CERTIFY_DUAL_INFEASIBLE,
          "infeasible: certify DUAL_INFEASIBLE");

    char *json = arbsdp_result_to_json(&p, &ar, cs, lb, ub, prec_c, &ab);
    CHECK(json != NULL, "infeasible: to_json non-NULL");

    CHECK(strstr(json, "\"status\": \"dual_infeasible\"") != NULL,
          "infeasible: certificate status \"dual_infeasible\"");
    CHECK(strstr(json, "\"obj_lb\": \"-inf\"") != NULL,
          "infeasible: obj_lb \"-inf\"");
    CHECK(strstr(json, "\"obj_ub\": \"+inf\"") != NULL,
          "infeasible: obj_ub \"+inf\"");

    fprintf(stderr, "\nINTEGRATION infeasible (dual_infeasible_diag): "
                    "status=dual_infeasible, obj_lb=-inf, obj_ub=+inf\n");

    assert_json_parses(json);

    free(json);
    arb_clear(lb);
    arb_clear(ub);
    arbsdp_apriori_clear(&ab);
    arbsdp_adaptive_result_clear(&ar);
    arbsdp_problem_clear(&p);
}

int
main(void)
{
    test_decimal_rigor();
    test_json_optimal();
    test_json_infeasible();

    /* Free FLINT's thread-local mpz/fmpz recycling pool so valgrind does not
     * report it as "possibly lost" at exit (CLAUDE.md rule 7: a clean run). */
    flint_cleanup();

    if (failures != 0) {
        fprintf(stderr, "test_serialize: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_serialize (directed-decimal rigor + JSON optimal/infeasible)\n");
    return EXIT_SUCCESS;
}
