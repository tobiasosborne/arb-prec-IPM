/*
 * src/cli.c -- the `arbsdp` command-line driver (bead arb-prec-IPM-b21).
 *
 * A thin front-end over the library: parse args, read a .dat-s problem
 * (io.c / b07), run the adaptive point-mode solve (precision.c / b16), build the
 * a-priori trace bounds the certifier needs (certify.c / b18), certify a rigorous
 * [lb, ub] bracket in ball arithmetic (certify.c / b19/b20), and serialize the
 * whole result to a single JSON object on stdout (io.c / b21).  All rigor lives in
 * the library; this file only wires it together and handles the CLI surface.
 *
 * CONTRACT (CLAUDE.md rule 5, fail loud): a usage / IO / parse error prints a
 * message to STDERR and exits NONZERO.  A COMPLETED run exits 0 regardless of the
 * mathematical verdict -- optimal / infeasible / inconclusive are RESULTS, not
 * errors, and are reported in the JSON "certificate".status field.  STDOUT carries
 * ONLY the JSON object (errors go to stderr) so it pipes cleanly to a parser.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear on
 * EVERY path, including the error exits (the CLI is valgrind-gated).  The solve+
 * certify region uses a single `cleanup` label that tears down everything it has
 * init'd; the pure arg/param error paths free only what they allocated.
 */

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flint/flint.h>
#include <flint/arb.h>

#include "arbsdp/api.h"
#include "arbsdp/io.h"
#include "arbsdp/problem.h"
#include "arbsdp/precision.h"
#include "arbsdp/certify.h"
#include "arbsdp/solve.h"

/* Exit codes: 0 = a completed run; nonzero ONLY on usage / IO / parse errors. */
enum {
    CLI_OK         = 0,
    CLI_USAGE_ERR  = 2, /* bad/unknown/missing argument                       */
    CLI_IO_ERR     = 3, /* the .dat-s file could not be read                  */
    CLI_PARSE_ERR  = 4  /* a --trace-bound value/block index failed to parse  */
};

/* -------------------------------------------------------------------------
 * Small helpers
 * ---------------------------------------------------------------------- */

static void
usage(FILE *out)
{
    fputs(
"usage:\n"
"  arbsdp solve <file.dat-s> [options]\n"
"  arbsdp --version | -v\n"
"  arbsdp --help | -h\n"
"\n"
"solve options:\n"
"  --prec N           starting working precision in bits (default 128)\n"
"  --prec-max M       maximum working precision in bits  (default 8192)\n"
"  --target-digits D  requested decimal digits of accuracy (default 30)\n"
"  --trace-bound V    a-priori bound tr(X_b) <= V for ALL blocks\n"
"  --trace-bound B:V  a-priori bound tr(X_b) <= V for block B (0-indexed);\n"
"                     repeatable; overrides the all-blocks value for that block\n"
"                     V must be a finite decimal literal >= 0 (e.g. 1, 2.5e3);\n"
"                     omit the option for an unbounded block (no 'inf')\n",
        out);
}

/* arg_error -- print a usage message to stderr (optionally naming the offending
 * argument), emit the usage banner, and return CLI_USAGE_ERR. */
static int
arg_error(const char *msg, const char *arg)
{
    if (arg != NULL)
        fprintf(stderr, "arbsdp: %s '%s'\n", msg, arg);
    else
        fprintf(stderr, "arbsdp: %s\n", msg);
    usage(stderr);
    return CLI_USAGE_ERR;
}

/* parse_slong / parse_int -- strtol with FULL-string consumption + range checks
 * (fail loud on any trailing garbage or overflow).  Return 0 on success. */
static int
parse_slong(const char *s, slong *out)
{
    char *end;
    long v;
    errno = 0;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0)
        return 1;
    *out = (slong) v;
    return 0;
}

static int
parse_int(const char *s, int *out)
{
    char *end;
    long v;
    errno = 0;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0)
        return 1;
    if (v < INT_MIN || v > INT_MAX)
        return 1;
    *out = (int) v;
    return 0;
}

/* parse_trace_value -- parse ONE --trace-bound value token into `out` (bead
 * arb-prec-IPM-2ny).  Returns 0 on success, nonzero on rejection.
 *
 * The trace bound xbar_b is a RIGOR-RELEVANT a-priori input (CLAUDE.md invariant
 * 5): the certified [lb, ub] is a theorem only if xbar_b >= tr(X_b).  A raw
 * arb_set_str is NOT a sufficient parser: FLINT 3.0.1 accepts "inf", "nan",
 * ball syntax "1+/-0.1" / "[1]" and "--1" (read as +1), so a typo would silently
 * become a DIFFERENT bound.  We therefore require:
 *   1. SYNTAX -- the strict finite decimal-literal grammar shared with the SDPA
 *      reader (arbsdp_decimal_is_valid, io.c, bead 37w).  This also rejects an
 *      explicit "inf": "no bound" is spelled by OMITTING the flag (an unset
 *      xbar_set flag), the single source of truth in arbsdp_apriori.
 *   2. SEMANTICS -- xbar >= 0.  Every block is PSD or diagonal-nonnegative, so
 *      tr(X_b) >= 0 and a negative xbar asserts the feasible set is EMPTY.  The
 *      Jansson consumers (arbsdp_lower_bound / arbsdp_upper_bound, certify.c)
 *      multiply min(0, dlo_b) / max(0, dhi_b) by xbar_b, so a negative xbar flips
 *      the sign of the penalty: lb moves UP and ub moves DOWN, producing a
 *      narrow "optimal" bracket that is only vacuously true (of an empty set) --
 *      i.e. a sign typo would yield a confident, wrong certificate for the
 *      user's actual problem.  Rejected fail-loud (CLAUDE.md rule 5).
 *      xbar = 0 is accepted: it is a legitimate (if strong) claim X_b = 0.
 * The literal is parsed at `prec` bits; arb_set_str then yields a ball that
 * ENCLOSES the decimal value (e.g. 0.1), which the certifier's ball products
 * consume rigorously. */
static int
parse_trace_value(arb_t out, const char *tok, slong prec)
{
    if (arbsdp_decimal_is_valid(tok) != 0)
        return 1;
    if (arb_set_str(out, tok, prec) != 0 || !arb_is_finite(out))
        return 1;
    if (!arb_is_nonnegative(out))
        return 1;
    return 0;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int
main(int argc, char **argv)
{
    /* ----- top-level dispatch (version / help / solve) -------------------- */
    if (argc < 2) {
        usage(stderr);
        return CLI_USAGE_ERR;
    }

    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0) {
        printf("arbsdp %s\n", arbsdp_version());
        return CLI_OK;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout);
        return CLI_OK;
    }
    if (strcmp(argv[1], "solve") != 0)
        return arg_error("unknown command", argv[1]);

    /* ----- parse `solve` arguments --------------------------------------- */
    const char *file = NULL;

    int   have_prec = 0,     have_prec_max = 0, have_target = 0;
    slong prec0 = 0,         prec_max = 0;
    int   target_digits = 0;

    /* The raw --trace-bound argument strings (argv lives for the run, so we
     * only store pointers).  At most argc-2 of them. */
    const char **trace_args = malloc((size_t) argc * sizeof(*trace_args));
    if (trace_args == NULL) {
        fprintf(stderr, "arbsdp: out of memory\n");
        return CLI_USAGE_ERR;
    }
    int n_trace = 0;

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];

        if (strcmp(a, "--prec") == 0) {
            if (i + 1 >= argc) { free(trace_args); return arg_error("--prec requires a value", NULL); }
            if (parse_slong(argv[++i], &prec0) != 0) { free(trace_args); return arg_error("invalid --prec value", argv[i]); }
            have_prec = 1;
        } else if (strcmp(a, "--prec-max") == 0) {
            if (i + 1 >= argc) { free(trace_args); return arg_error("--prec-max requires a value", NULL); }
            if (parse_slong(argv[++i], &prec_max) != 0) { free(trace_args); return arg_error("invalid --prec-max value", argv[i]); }
            have_prec_max = 1;
        } else if (strcmp(a, "--target-digits") == 0) {
            if (i + 1 >= argc) { free(trace_args); return arg_error("--target-digits requires a value", NULL); }
            if (parse_int(argv[++i], &target_digits) != 0) { free(trace_args); return arg_error("invalid --target-digits value", argv[i]); }
            have_target = 1;
        } else if (strcmp(a, "--trace-bound") == 0) {
            if (i + 1 >= argc) { free(trace_args); return arg_error("--trace-bound requires a value", NULL); }
            trace_args[n_trace++] = argv[++i];
        } else if (a[0] == '-') {
            free(trace_args);
            return arg_error("unknown option", a);
        } else if (file == NULL) {
            file = a;
        } else {
            free(trace_args);
            return arg_error("unexpected extra argument", a);
        }
    }

    if (file == NULL) {
        free(trace_args);
        return arg_error("solve requires a problem file", NULL);
    }

    /* ----- precision params: defaults + overrides + validation ----------- */
    arbsdp_precision_params pp;
    arbsdp_precision_default_params(&pp);
    if (have_prec)     pp.prec0 = prec0;
    if (have_prec_max) pp.prec_max = prec_max;
    if (have_target)   pp.target_digits = target_digits;

    if (pp.prec0 < 2) { free(trace_args); return arg_error("--prec must be >= 2", NULL); }
    if (pp.prec_max < pp.prec0) { free(trace_args); return arg_error("--prec-max must be >= --prec", NULL); }
    if (pp.target_digits < 1) { free(trace_args); return arg_error("--target-digits must be >= 1", NULL); }

    /* ----- read the problem (CLAUDE.md rule 5: fail loud on IO error) ----- */
    arbsdp_problem p;
    arbsdp_problem_init(&p);
    if (arbsdp_read_sdpa(&p, file) != 0) {
        fprintf(stderr, "arbsdp: failed to read SDPA problem '%s'\n", file);
        arbsdp_problem_clear(&p); /* already clean on failure; safe + explicit */
        free(trace_args);
        flint_cleanup();
        return CLI_IO_ERR;
    }

    /* From here on a single `cleanup` label tears everything down.  The trace
     * bounds are parsed BEFORE the solve (fail fast: a malformed bound never costs
     * a solve), so the adaptive result is init'd only once they are accepted. */
    int exit_code = CLI_OK;
    char *json = NULL;
    int have_ar = 0;

    arbsdp_adaptive_result ar;

    arbsdp_apriori ab;
    arbsdp_apriori_init(&ab, p.nblocks);

    arb_t tbval, lb, ub;
    arb_init(tbval);
    arb_init(lb);
    arb_init(ub);

    /* ----- apply --trace-bound: all-blocks first, then per-block override
     * (CLAUDE.md invariant 5: NO silent boundedness -- an unset block stays
     * unbounded and the certifier reports -inf for it honestly).  Omitting the
     * flag is the ONLY way to say "no bound"; an explicit "inf" is rejected. */
    {
        slong parse_prec = pp.prec_max > 256 ? pp.prec_max : 256;

        /* Pass A -- all-blocks (no ':'). */
        for (int k = 0; k < n_trace; k++) {
            if (strchr(trace_args[k], ':') != NULL)
                continue; /* per-block; handled in pass B */
            if (parse_trace_value(tbval, trace_args[k], parse_prec) != 0) {
                fprintf(stderr, "arbsdp: invalid --trace-bound value '%s' "
                        "(need a finite decimal literal >= 0)\n", trace_args[k]);
                exit_code = CLI_PARSE_ERR;
                goto cleanup;
            }
            for (int b = 0; b < p.nblocks; b++)
                arbsdp_apriori_set_xbar(&ab, b, tbval);
        }

        /* Pass B -- per-block B:V (overrides any all-blocks value for block B). */
        for (int k = 0; k < n_trace; k++) {
            const char *colon = strchr(trace_args[k], ':');
            char *end;
            long b;
            if (colon == NULL)
                continue; /* all-blocks; handled in pass A */
            errno = 0;
            b = strtol(trace_args[k], &end, 10);
            if (end != colon || end == trace_args[k] || errno != 0 ||
                b < 0 || b >= p.nblocks) {
                fprintf(stderr,
                        "arbsdp: invalid --trace-bound block index in '%s' "
                        "(valid 0..%d)\n",
                        trace_args[k], p.nblocks - 1);
                exit_code = CLI_PARSE_ERR;
                goto cleanup;
            }
            if (parse_trace_value(tbval, colon + 1, parse_prec) != 0) {
                fprintf(stderr, "arbsdp: invalid --trace-bound value in '%s' "
                        "(need a finite decimal literal >= 0)\n", trace_args[k]);
                exit_code = CLI_PARSE_ERR;
                goto cleanup;
            }
            arbsdp_apriori_set_xbar(&ab, (int) b, tbval);
        }
    }

    /* ----- solve (Layer 0, POINT MODE -- certifies nothing) -------------- */
    arbsdp_solve_adaptive(&ar, &p, &pp);
    have_ar = 1;

    /* ----- certify (Layer 1, ball arithmetic) + serialize ---------------- */
    {
        slong prec_c = ar.final_prec + 128; /* MATH_SPEC §10 */
        arbsdp_certify_status cs =
            arbsdp_certify(lb, ub, &p, &ar.res.it, &ab, prec_c);

        json = arbsdp_result_to_json(&p, &ar, cs, lb, ub, prec_c, &ab);
        fputs(json, stdout);
        putchar('\n');
    }

cleanup:
    free(json);
    arb_clear(ub);
    arb_clear(lb);
    arb_clear(tbval);
    arbsdp_apriori_clear(&ab);
    if (have_ar)
        arbsdp_adaptive_result_clear(&ar);
    arbsdp_problem_clear(&p);
    free(trace_args);
    flint_cleanup(); /* free FLINT's thread-local pools (valgrind cleanliness) */
    return exit_code;
}
