/*
 * test/test_cli.c -- CLI smoke test for the real `arbsdp` binary (bead
 * arb-prec-IPM-b21).
 *
 * Unlike the other test executables, this one links NOTHING from libarbsdp: it
 * execs the actual built CLI via popen() and inspects its stdout + exit status.
 * That is the point -- it exercises the full read -> adaptive solve -> certify ->
 * JSON-serialize pipeline through the same surface a user (or ArbSDP.jl, or a
 * shell script) sees, catching wiring regressions that the in-process unit tests
 * (test_serialize, test_certify_bracket, ...) cannot: arg parsing, exit codes,
 * stdout-only-JSON, and end-to-end byte output.
 *
 * GROUND TRUTH (CLAUDE.md rule 3): the CLI contract is documented in src/cli.c --
 * exit 0 on a COMPLETED run (optimal / infeasible / inconclusive are RESULTS, not
 * errors), nonzero ONLY on usage/IO/parse errors (CLI_USAGE_ERR=2, CLI_IO_ERR=3,
 * CLI_PARSE_ERR=4); STDOUT carries ONLY the JSON object (errors go to stderr).
 * The exact JSON spelling is fixed by arbsdp_result_to_json in src/io.c: the
 * serializer emits `"status": "<...>"` (one space after the colon) inside the
 * nested "certificate" object, and prints +/-inf endpoints as "-inf" / "+inf".
 * The substrings asserted below match that serializer byte-for-byte.
 *
 * Inputs come from compile definitions:
 *   ARBSDP_BIN  -- absolute path to the built `arbsdp` binary ($<TARGET_FILE:...>)
 *   GOLDEN_DIR  -- absolute path to <repo>/golden
 *
 * No arb/FLINT calls here (it only runs the binary + inspects text), so there are
 * no ball/precision concerns -- but every popen() is matched by a pclose().
 *
 * popen/pclose are POSIX (not C11), so the feature-test macro must be defined
 * before any standard header; WEXITSTATUS lives in <sys/wait.h>.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "arbsdp/api.h" /* ARBSDP_VERSION (macro only; no library link needed) */

#ifndef ARBSDP_BIN
#error "ARBSDP_BIN compile definition is required (path to the built CLI binary)"
#endif
#ifndef GOLDEN_DIR
#define GOLDEN_DIR "../../golden"
#endif

#define CMD_MAX 2048
#define OUT_MAX 65536 /* the JSON object is < 2 KB; this is comfortable headroom */

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* -------------------------------------------------------------------------
 * run_capture -- run `cmd` via popen("r"), read all of its stdout into buf
 * (NUL-terminated, truncated safely to buflen), drain any overflow so the child
 * never blocks on a full pipe, then pclose and extract WEXITSTATUS into
 * *exit_status.  Returns 0 on success, -1 if popen itself failed.
 * ---------------------------------------------------------------------- */
static int
run_capture(const char *cmd, char *buf, size_t buflen, int *exit_status)
{
    FILE *fp;
    size_t total = 0;
    int status;

    *exit_status = -1;
    if (buflen > 0)
        buf[0] = '\0';

    fp = popen(cmd, "r");
    if (fp == NULL)
        return -1;

    if (buflen > 0) {
        size_t n;
        while (total + 1 < buflen &&
               (n = fread(buf + total, 1, buflen - 1 - total, fp)) > 0)
            total += n;
        buf[total] = '\0';
    }

    /* Drain anything beyond buflen so the child can exit cleanly. */
    {
        char sink[4096];
        while (fread(sink, 1, sizeof sink, fp) > 0)
            ;
    }

    status = pclose(fp);
    if (status != -1)
        *exit_status = WEXITSTATUS(status);
    return 0;
}

/* ----- Test 1: --version -------------------------------------------------- */
static void
test_version(void)
{
    char cmd[CMD_MAX];
    char out[OUT_MAX];
    int rc, ex;

    snprintf(cmd, sizeof cmd, "%s --version", ARBSDP_BIN);
    rc = run_capture(cmd, out, sizeof out, &ex);
    CHECK(rc == 0, "version: popen failed");
    CHECK(ex == 0, "version: exit status should be 0");
    CHECK(strstr(out, "arbsdp ") != NULL, "version: stdout should contain 'arbsdp '");
    CHECK(strstr(out, ARBSDP_VERSION) != NULL,
          "version: stdout should contain the version string");
}

/* ----- Test 2: solve a FEASIBLE golden -> optimal bracket ----------------- */
static void
test_solve_optimal(void)
{
    char cmd[CMD_MAX];
    char out[OUT_MAX];
    int rc, ex;

    snprintf(cmd, sizeof cmd,
             "%s solve %s/max_eigenvalue_2x2/max_eigenvalue_2x2.dat-s "
             "--target-digits 15 --trace-bound 1",
             ARBSDP_BIN, GOLDEN_DIR);
    rc = run_capture(cmd, out, sizeof out, &ex);
    CHECK(rc == 0, "solve-optimal: popen failed");
    CHECK(ex == 0, "solve-optimal: a completed run must exit 0");
    /* Exact serializer spelling (src/io.c): one space after the colon. */
    CHECK(strstr(out, "\"status\": \"optimal\"") != NULL,
          "solve-optimal: stdout should contain \"status\": \"optimal\"");
    CHECK(strstr(out, "\"obj_lb\"") != NULL,
          "solve-optimal: stdout should contain \"obj_lb\"");
    CHECK(strstr(out, "\"obj_ub\"") != NULL,
          "solve-optimal: stdout should contain \"obj_ub\"");
    CHECK(strstr(out, "\"adaptive_status\"") != NULL,
          "solve-optimal: stdout should contain \"adaptive_status\"");
}

/* ----- Test 3: solve a DUAL-INFEASIBLE golden -> -inf/+inf bracket -------- */
static void
test_solve_dual_infeasible(void)
{
    char cmd[CMD_MAX];
    char out[OUT_MAX];
    int rc, ex;

    snprintf(cmd, sizeof cmd,
             "%s solve %s/dual_infeasible_diag/dual_infeasible_diag.dat-s "
             "--target-digits 8",
             ARBSDP_BIN, GOLDEN_DIR);
    rc = run_capture(cmd, out, sizeof out, &ex);
    CHECK(rc == 0, "solve-dual-infeasible: popen failed");
    CHECK(ex == 0, "solve-dual-infeasible: a completed run must exit 0");
    CHECK(strstr(out, "\"status\": \"dual_infeasible\"") != NULL,
          "solve-dual-infeasible: stdout should contain \"status\": \"dual_infeasible\"");
    CHECK(strstr(out, "\"-inf\"") != NULL,
          "solve-dual-infeasible: stdout should contain the -inf lower bound");
    CHECK(strstr(out, "\"+inf\"") != NULL,
          "solve-dual-infeasible: stdout should contain the +inf upper bound");
}

/* ----- Test 4: error paths must exit NONZERO ------------------------------ */
static void
test_error_paths(void)
{
    char cmd[CMD_MAX];
    char out[OUT_MAX];
    int rc, ex;

    /* Unreadable problem file -> CLI_IO_ERR (3); stderr suppressed. */
    snprintf(cmd, sizeof cmd, "%s solve /no/such/file.dat-s 2>/dev/null",
             ARBSDP_BIN);
    rc = run_capture(cmd, out, sizeof out, &ex);
    CHECK(rc == 0, "error-nofile: popen failed");
    CHECK(ex != 0, "error-nofile: missing file must exit nonzero");

    /* No arguments at all -> CLI_USAGE_ERR (2); stderr suppressed. */
    snprintf(cmd, sizeof cmd, "%s 2>/dev/null", ARBSDP_BIN);
    rc = run_capture(cmd, out, sizeof out, &ex);
    CHECK(rc == 0, "error-noargs: popen failed");
    CHECK(ex != 0, "error-noargs: no arguments must exit nonzero");
}

/* ----- Test 4b: --trace-bound token validation (bead arb-prec-IPM-2ny) -----
 * The trace bound xbar is a RIGOR-RELEVANT a-priori input (CLAUDE.md invariant
 * 5): the certified [lb, ub] is a theorem only if xbar_b >= tr(X_b).  FLINT
 * 3.0.1 arb_set_str ACCEPTS "inf", "nan", "1+/-0.1", "[1]" and "--1" (read as
 * +1, a silent sign flip), so a raw arb_set_str parse lets a malformed token
 * silently become a different bound.  Every token below must be REJECTED
 * fail-loud: exit CLI_PARSE_ERR (4), a stderr message naming --trace-bound, and
 * NO JSON on stdout.  Negative values are rejected too: tr(X_b) >= 0 on a PSD or
 * diagonal block, so xbar < 0 asserts an empty feasible set -- a sign typo, not
 * a usable bound (see src/cli.c).  Both the all-blocks form V and the per-block
 * form B:V are exercised. */
static void
check_bad_trace_token(const char *tok)
{
    char cmd[CMD_MAX];
    char out[OUT_MAX];
    char msg[256];
    int rc, ex;

    /* stdout only (stderr discarded): must be empty of JSON, exit 4. */
    snprintf(cmd, sizeof cmd,
             "%s solve %s/max_eigenvalue_2x2/max_eigenvalue_2x2.dat-s "
             "--target-digits 8 --trace-bound '%s' 2>/dev/null",
             ARBSDP_BIN, GOLDEN_DIR, tok);
    rc = run_capture(cmd, out, sizeof out, &ex);
    snprintf(msg, sizeof msg, "bad-trace '%s': popen failed", tok);
    CHECK(rc == 0, msg);
    snprintf(msg, sizeof msg, "bad-trace '%s': must exit CLI_PARSE_ERR (4), got %d",
             tok, ex);
    CHECK(ex == 4, msg);
    snprintf(msg, sizeof msg, "bad-trace '%s': stdout must carry no JSON", tok);
    CHECK(strchr(out, '{') == NULL, msg);

    /* stderr only (stdout discarded): must name the offending option. */
    snprintf(cmd, sizeof cmd,
             "%s solve %s/max_eigenvalue_2x2/max_eigenvalue_2x2.dat-s "
             "--target-digits 8 --trace-bound '%s' 2>&1 >/dev/null",
             ARBSDP_BIN, GOLDEN_DIR, tok);
    rc = run_capture(cmd, out, sizeof out, &ex);
    snprintf(msg, sizeof msg, "bad-trace '%s': popen failed (stderr)", tok);
    CHECK(rc == 0, msg);
    snprintf(msg, sizeof msg, "bad-trace '%s': stderr must mention --trace-bound", tok);
    CHECK(strstr(out, "--trace-bound") != NULL, msg);
}

static void
test_trace_bound_validation(void)
{
    static const char *bad[] = {
        /* non-finite spellings arb_set_str accepts */
        "inf", "+inf", "-inf", "Inf", "nan", "NaN", "[inf]",
        /* ball syntax: a radius is not an a-priori bound */
        "1+/-0.1", "[1+/-0.1]", "[1]", "[+/-1]",
        /* doubled sign: arb_set_str reads "--1" as +1 */
        "--1", "+-1", "-+1",
        /* negative: tr(X_b) >= 0, so xbar < 0 is a sign error */
        "-1", "-0.5", "-1e-30",
        /* plain garbage / malformed literals */
        "", "abc", "1e", "1.2.3", ".", "1 ", "0x10",
        /* the same through the per-block B:V form */
        "0:inf", "0:nan", "0:--1", "0:1+/-0.1", "0:[1]", "0:-1", "0:", "0:abc",
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++)
        check_bad_trace_token(bad[i]);
}

/* Valid trace-bound literals must still be accepted (no over-rejection):
 * a completed run, exit 0, an optimal certificate. */
static void
test_trace_bound_valid_forms(void)
{
    static const char *good[] = { "1", "1.0", "1e0", "+1", "10E-1", ".1e1",
                                  "0:1", "0:+1.000" };
    for (size_t i = 0; i < sizeof good / sizeof good[0]; i++) {
        char cmd[CMD_MAX];
        char out[OUT_MAX];
        char msg[256];
        int rc, ex;
        snprintf(cmd, sizeof cmd,
                 "%s solve %s/max_eigenvalue_2x2/max_eigenvalue_2x2.dat-s "
                 "--target-digits 8 --trace-bound '%s' 2>/dev/null",
                 ARBSDP_BIN, GOLDEN_DIR, good[i]);
        rc = run_capture(cmd, out, sizeof out, &ex);
        snprintf(msg, sizeof msg, "good-trace '%s': popen failed", good[i]);
        CHECK(rc == 0, msg);
        snprintf(msg, sizeof msg, "good-trace '%s': must exit 0, got %d", good[i], ex);
        CHECK(ex == 0, msg);
        snprintf(msg, sizeof msg, "good-trace '%s': must certify optimal", good[i]);
        CHECK(strstr(out, "\"status\": \"optimal\"") != NULL, msg);
    }
}

/* ----- Test 5: JSON round-trip through a real parser (bead acceptance) ----
 * Pipe the CLI's stdout straight into python3's json.load; a zero exit proves
 * the emitted bytes are valid JSON (the strongest end-to-end check).  python3 is
 * optional -- if absent, SKIP without failing. */
static void
test_json_roundtrip(void)
{
    char cmd[CMD_MAX];
    char out[OUT_MAX];
    int rc, ex;
    int have_python3;

    {
        int sysrc = system("command -v python3 >/dev/null 2>&1");
        have_python3 = (sysrc != -1 && WEXITSTATUS(sysrc) == 0);
    }

    if (!have_python3) {
        printf("SKIP: python3 not found (round-trip check skipped)\n");
        return;
    }

    snprintf(cmd, sizeof cmd,
             "%s solve %s/max_eigenvalue_2x2/max_eigenvalue_2x2.dat-s "
             "--target-digits 15 --trace-bound 1 "
             "| python3 -c 'import json,sys; json.load(sys.stdin)'",
             ARBSDP_BIN, GOLDEN_DIR);
    rc = run_capture(cmd, out, sizeof out, &ex);
    CHECK(rc == 0, "json-roundtrip: popen failed");
    CHECK(ex == 0, "json-roundtrip: CLI output must parse as valid JSON");
}

int
main(void)
{
    test_version();
    test_solve_optimal();
    test_solve_dual_infeasible();
    test_error_paths();
    test_trace_bound_validation();
    test_trace_bound_valid_forms();
    test_json_roundtrip();

    if (failures != 0) {
        fprintf(stderr, "test_cli: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_cli (version/solve/infeasible/errors/json-roundtrip)\n");
    return EXIT_SUCCESS;
}
