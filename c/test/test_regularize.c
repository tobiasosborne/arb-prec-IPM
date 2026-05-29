/*
 * test/test_regularize.c -- tests for the adaptive 3-way Tikhonov regularization
 * of the Schur Cholesky factorization (c/src/regularize.c, bead arb-prec-IPM-b13).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written BEFORE c/src/regularize.c and
 * include/arbsdp/regularize.h exist; the target fails to compile/link first
 * (no header / undefined symbols arbsdp_regstate_* / arbsdp_factor_with_reg),
 * then passes once the impl lands.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §4 (three-way Tikhonov: M_reg = M + (d_p+d_d+d_g)I;
 *     §4.2 diagnose by per-constraint Frobenius row norm vs 1e-10*global;
 *     §4.3 retry up to maxRefactor; §10.4 carry d across iterations;
 *     §10.6 maxRefactor = 20).
 *   - Regularization.ts:96-115 (caps cap_p=1e-2, cap_d=1e+2, cap_g=1e+2; bump
 *     factors primal x10, dual x100, gap x100), :174-192 (makeSdpDiagnose),
 *     :231-269 (factorWith3Way retry + gap->dual->primal fall-through),
 *     :278-305 (tryBump: cap clamp, max(jitter,1e-12) base for first bump).
 *   - Defaults.ts:40-46 (initialJitter = 1e-12 on the GAP tier; caps; bumps).
 *   - docs/FLINT_NOTES.md: arb_mat_cho returns 1 iff PROVEN PD.
 *
 * POINT MODE (CLAUDE.md rule 1, invariant 4): the regularized M is what gets
 * factored for Layer 0 directions; Layer 1 certifies the ORIGINAL M.  The
 * carried d values are recorded so the perturbation is auditable.  Here we test
 * the regularization machinery in isolation against constructed M matrices.
 *
 * Property coverage (bead arb-prec-IPM-b13):
 *   1. NO-OP ON PD       a well-conditioned M factors on the first try; d
 *                        unchanged from its init values; attempts == 1.
 *   2. RESCUE            an M that fails plain arb_mat_cho (tiny/zero pivot) is
 *                        rescued: arbsdp_factor_with_reg returns 1 with d bumped
 *                        within caps, and the regularized M factors.
 *   3. ESCALATION        repeated failures grow d by the documented factor and
 *                        clamp at the cap; attempts <= maxRefactor (20).
 *   4. CARRY-OVER        two successive calls share one regstate; the second
 *                        starts from the CARRIED d (not reset).
 *   5. MUTATION (rule 8) a reference that resets d each call -- OR factors the
 *                        UNregularized M -- produces a DIFFERENT observable than
 *                        the real routine, proving the tests discriminate both
 *                        mutations.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"
#include "arbsdp/regularize.h"

#define PREC 256

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* |mid(x) - val| <= tol on the midpoint. */
static int
close_d(const arb_t x, double val, double tol)
{
    arb_t d, t;
    int ok;
    arb_init(d);
    arb_init(t);
    arb_get_mid_arb(d, x);
    arb_set_d(t, val);
    arb_sub(d, d, t, PREC);
    arb_abs(d, d);
    arb_set_d(t, tol);
    ok = arb_le(d, t);
    arb_clear(d);
    arb_clear(t);
    return ok;
}

static double
to_double(const arb_t x)
{
    arf_t m;
    double v;
    arf_init(m);
    arf_set(m, arb_midref(x));
    v = arf_get_d(m, ARF_RND_NEAR);
    arf_clear(m);
    return v;
}

/* Heap-duplicate a string with malloc (matches arbsdp_problem_clear's free()). */
static char *
dup_str(const char *s)
{
    size_t len = strlen(s) + 1;
    char *out = malloc(len);
    memcpy(out, s, len);
    return out;
}

/* -------------------------------------------------------------------------
 * Minimal in-memory problem with a SINGLE 1x1-per-constraint diagonal layout
 * used only as the diagnose input (per-constraint Frobenius row norms).  We
 * build m constraints, one block of side n; constraint i's A_i^0 is the dense
 * symmetric integer matrix A_dense[i].  C (matno 0) is left empty.
 * ---------------------------------------------------------------------- */
static void
build_problem(arbsdp_problem *p, int m, int n, const long (*A_dense)[16])
{
    char buf[64];
    arbsdp_problem_init(p);
    p->m = m;
    p->nblocks = 1;
    p->maximize = 1;
    p->block_sizes = malloc(sizeof(int));
    p->block_sizes[0] = n;
    p->b = malloc((size_t) m * sizeof(char *));
    for (int i = 0; i < m; i++)
        p->b[i] = dup_str("0");

    slong ncells = (slong) (m + 1) * 1;
    p->mats = calloc((size_t) ncells, sizeof(arbsdp_matblock));
    for (int mi = 1; mi <= m; mi++) {
        arbsdp_matblock *mb = &p->mats[mi];
        const long *Ad = A_dense[mi - 1];
        int cap = n * (n + 1) / 2;
        mb->entries = malloc((size_t) cap * sizeof(arbsdp_entry));
        mb->capacity = cap;
        mb->count = 0;
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++) {
                long v = Ad[i * n + j];
                if (v == 0) continue;
                snprintf(buf, sizeof buf, "%ld", v);
                mb->entries[mb->count].i = i;
                mb->entries[mb->count].j = j;
                mb->entries[mb->count].value = dup_str(buf);
                mb->count++;
            }
    }
}

/* A problem whose 3 constraints all have a HEALTHY (non-tiny) Frobenius row
 * norm -- diagnose will route generic failures to "gap". */
static void
build_healthy_problem(arbsdp_problem *p)
{
    const long A[3][16] = {
        { 1,0,0, 0,0,0, 0,0,0 },
        { 0,1,0, 1,1,0, 0,0,0 },
        { 0,0,1, 0,0,1, 1,1,1 },
    };
    build_problem(p, 3, 3, A);
}

/* ----- Test 1: no-op on a well-conditioned PD M ----------------------------
 * A diagonal M = diag(3,4,5) is PD; arb_mat_cho succeeds on the first try.
 * d_p, d_d unchanged (0); d_g unchanged from its init (1e-12); attempts == 1. */
static void
test_noop_on_pd(void)
{
    arbsdp_problem p;
    arbsdp_regstate rs;
    arb_mat_t M, L;
    int ok, attempts = -1;

    build_healthy_problem(&p);
    arbsdp_regstate_init(&rs, PREC);
    arb_mat_init(M, 3, 3);
    arb_mat_init(L, 3, 3);
    arb_set_si(arb_mat_entry(M, 0, 0), 3);
    arb_set_si(arb_mat_entry(M, 1, 1), 4);
    arb_set_si(arb_mat_entry(M, 2, 2), 5);

    ok = arbsdp_factor_with_reg(L, M, &rs, &p, PREC, &attempts);

    CHECK(ok == 1, "noop: well-conditioned PD M must factor");
    CHECK(attempts == 1, "noop: PD M must factor on the FIRST attempt");
    CHECK(close_d(rs.delta_p, 0.0, 1e-300), "noop: delta_p must stay 0");
    CHECK(close_d(rs.delta_d, 0.0, 1e-300), "noop: delta_d must stay 0");
    /* delta_g stays at its init value (initialJitter = 1e-12). */
    CHECK(close_d(rs.delta_g, 1e-12, 1e-24), "noop: delta_g must stay at init 1e-12");
    CHECK(rs.refactors == 0, "noop: no refactors on a first-try success");

    arb_mat_clear(M);
    arb_mat_clear(L);
    arbsdp_regstate_clear(&rs);
    arbsdp_problem_clear(&p);
}

/* ----- Test 2: rescue a non-PD M -------------------------------------------
 * M = diag(1, 1, -1e-3) is indefinite, so plain arb_mat_cho FAILS.  The carried
 * initial jitter delta_g = 1e-12 is INSUFFICIENT (needs lift > 1e-3), so the
 * routine must actually BUMP the gap tier up to 1e-2 before M+lift*I is provably
 * PD.  All 3 constraint rows have healthy norms => diagnose routes to "gap".
 * The routine must succeed with delta_g bumped above its init and within cap. */
static void
test_rescue(void)
{
    arbsdp_problem p;
    arbsdp_regstate rs;
    arb_mat_t M, L, Lplain;
    int ok, plain, attempts = -1;

    build_healthy_problem(&p);
    arbsdp_regstate_init(&rs, PREC);
    arb_mat_init(M, 3, 3);
    arb_mat_init(L, 3, 3);
    arb_mat_init(Lplain, 3, 3);
    arb_set_si(arb_mat_entry(M, 0, 0), 1);
    arb_set_si(arb_mat_entry(M, 1, 1), 1);
    arb_set_d(arb_mat_entry(M, 2, 2), -1e-3); /* indefinite => not PD */

    /* sanity: plain cho on M fails (the thing we must rescue). */
    plain = arb_mat_cho(Lplain, M, PREC);
    CHECK(plain == 0, "rescue: precondition -- plain cho on singular M must fail");

    ok = arbsdp_factor_with_reg(L, M, &rs, &p, PREC, &attempts);
    CHECK(ok == 1, "rescue: regularized factorization must succeed");
    CHECK(attempts >= 2, "rescue: must take at least one refactor (attempt 1 fails)");
    CHECK(attempts <= 20, "rescue: attempts must be <= maxRefactor (20)");
    /* gap tier was bumped from 1e-12 up to 1e-2; primal/dual untouched. */
    CHECK(to_double(rs.delta_g) > 1.5e-12, "rescue: delta_g must have been bumped");
    CHECK(to_double(rs.delta_g) <= 1e2, "rescue: delta_g must stay within cap 1e2");
    CHECK(close_d(rs.delta_p, 0.0, 1e-300), "rescue: delta_p untouched (healthy rows)");
    CHECK(close_d(rs.delta_d, 0.0, 1e-300), "rescue: delta_d untouched (healthy rows)");
    CHECK(rs.bumps_g >= 1, "rescue: at least one gap bump recorded");

    /* The factor L returned must actually be a valid factor of M + lift*I, i.e.
     * cho on the regularized matrix succeeds.  Reconstruct the lift and verify. */
    {
        arb_mat_t Mreg, Lchk;
        arb_t lift;
        int chk;
        arb_mat_init(Mreg, 3, 3);
        arb_mat_init(Lchk, 3, 3);
        arb_init(lift);
        arb_add(lift, rs.delta_p, rs.delta_d, PREC);
        arb_add(lift, lift, rs.delta_g, PREC);
        arb_mat_set(Mreg, M);
        for (int i = 0; i < 3; i++)
            arb_add(arb_mat_entry(Mreg, i, i), arb_mat_entry(Mreg, i, i), lift, PREC);
        chk = arb_mat_cho(Lchk, Mreg, PREC);
        CHECK(chk == 1, "rescue: M + lift*I must be provably PD (factor valid)");
        arb_mat_clear(Mreg);
        arb_mat_clear(Lchk);
        arb_clear(lift);
    }

    arb_mat_clear(M);
    arb_mat_clear(L);
    arb_mat_clear(Lplain);
    arbsdp_regstate_clear(&rs);
    arbsdp_problem_clear(&p);
}

/* ----- Test 3: escalation within caps + honest failure ---------------------
 * Drive a HARD failure: M = -I (negative definite) can never be made PD by any
 * lift <= cap_g = 1e2 (it needs > 1).  Wait -- a lift of (cap_p+cap_d+cap_g) ~
 * 1e-2+1e2+1e2 >> 1 WOULD fix -I, so -I is recoverable.  To force the honest
 * total-failure path we use M = diag(-1e6,...) which exceeds the total cap.
 *
 * We assert: (a) on a recoverable-but-hard M the gap tier escalates by x100 each
 * bump and CLAMPS at cap_g = 1e2; (b) on an unrecoverable M the routine returns
 * 0 (honest failure, CLAUDE rule 5) with attempts <= maxRefactor.
 */
static void
test_escalation(void)
{
    /* (a) escalation factor + clamp: M = diag(-50, 1, 1).  Gap tier from 1e-12
     * grows x100 per bump (1e-12, 1e-10, ..., 1e2 clamp).  M+lift*I becomes PD
     * once lift > 50, i.e. at lift = 1e2 (the clamp).  Verifies the x100 factor
     * AND the clamp at exactly cap_g. */
    {
        arbsdp_problem p;
        arbsdp_regstate rs;
        arb_mat_t M, L;
        int ok, attempts = -1;

        build_healthy_problem(&p);
        arbsdp_regstate_init(&rs, PREC);
        arb_mat_init(M, 3, 3);
        arb_mat_init(L, 3, 3);
        arb_set_si(arb_mat_entry(M, 0, 0), -50);
        arb_set_si(arb_mat_entry(M, 1, 1), 1);
        arb_set_si(arb_mat_entry(M, 2, 2), 1);

        ok = arbsdp_factor_with_reg(L, M, &rs, &p, PREC, &attempts);
        CHECK(ok == 1, "escalation: M=diag(-50,1,1) recoverable at lift=cap_g");
        CHECK(attempts <= 20, "escalation: attempts <= maxRefactor (20)");
        /* delta_g must have CLAMPED at the cap (since lift must exceed 50). */
        CHECK(close_d(rs.delta_g, 1e2, 1e-6), "escalation: delta_g clamped at cap_g=1e2");

        arb_mat_clear(M);
        arb_mat_clear(L);
        arbsdp_regstate_clear(&rs);
        arbsdp_problem_clear(&p);
    }

    /* (b) honest failure: M = diag(-1e6,1,1).  The total available lift
     * cap_p+cap_d+cap_g = 1e-2 + 1e2 + 1e2 ~ 200.02 < 1e6, so M + lift*I is
     * never PD.  The routine must return 0 (FAIL FAST, rule 5) -- never a
     * silently-fudged factor -- and must not exceed maxRefactor attempts. */
    {
        arbsdp_problem p;
        arbsdp_regstate rs;
        arb_mat_t M, L;
        int ok, attempts = -1;

        build_healthy_problem(&p);
        arbsdp_regstate_init(&rs, PREC);
        arb_mat_init(M, 3, 3);
        arb_mat_init(L, 3, 3);
        arb_set_d(arb_mat_entry(M, 0, 0), -1e6);
        arb_set_si(arb_mat_entry(M, 1, 1), 1);
        arb_set_si(arb_mat_entry(M, 2, 2), 1);

        ok = arbsdp_factor_with_reg(L, M, &rs, &p, PREC, &attempts);
        CHECK(ok == 0, "escalation: unrecoverable M must fail honestly (return 0)");
        CHECK(attempts == 20, "escalation: failure exhausts maxRefactor (20) attempts");
        /* The gap tier is bumped first (healthy rows => diagnose "gap") and caps
         * after 8 x100 bumps, well within the 20-attempt budget; then the
         * fall-through bumps dual.  Capping ALL three tiers would need 27 bumps
         * (8 gap + 8 dual + 11 primal) > maxRefactor, so the routine fails
         * HONESTLY (return 0) at attempt 20 -- matching the TS loop, which also
         * stops at maxRefactor regardless (Regularization.ts:240,268).  We assert
         * the guaranteed-within-budget facts: gap and dual both reached their cap. */
        CHECK(close_d(rs.delta_g, 1e2, 1e-6), "escalation: delta_g at cap on failure");
        CHECK(close_d(rs.delta_d, 1e2, 1e-6), "escalation: delta_d at cap on failure");
        CHECK(to_double(rs.delta_p) > 0.0,
              "escalation: delta_p partially bumped via fall-through on failure");

        arb_mat_clear(M);
        arb_mat_clear(L);
        arbsdp_regstate_clear(&rs);
        arbsdp_problem_clear(&p);
    }
}

/* ----- Test 4: carry-over across calls (MATH_SPEC §10.4) -------------------
 * One regstate, two successive factorizations.  Call 1 rescues a non-PD M and
 * leaves delta_g > 1e-12.  Call 2 uses the SAME regstate on a PD M -- since the
 * carried delta_g is non-zero and the PD M factors immediately, delta_g must be
 * UNCHANGED from where call 1 left it (carry-over, not reset).  The carried
 * value persisting (not snapping back to 1e-12) is the assertion. */
static void
test_carry_over(void)
{
    arbsdp_problem p;
    arbsdp_regstate rs;
    arb_mat_t M1, M2, L;
    int ok, attempts = -1;
    double carried;

    build_healthy_problem(&p);
    arbsdp_regstate_init(&rs, PREC);
    arb_mat_init(M1, 3, 3);
    arb_mat_init(M2, 3, 3);
    arb_mat_init(L, 3, 3);

    /* Call 1: indefinite M1 = diag(1,1,-1e-3) -> gap bumped to 1e-2 (the carried
     * 1e-12 init is insufficient, forcing real bumps). */
    arb_set_si(arb_mat_entry(M1, 0, 0), 1);
    arb_set_si(arb_mat_entry(M1, 1, 1), 1);
    arb_set_d(arb_mat_entry(M1, 2, 2), -1e-3);
    ok = arbsdp_factor_with_reg(L, M1, &rs, &p, PREC, &attempts);
    CHECK(ok == 1, "carry: call 1 must rescue M1");
    carried = to_double(rs.delta_g);
    CHECK(carried > 1.5e-12, "carry: call 1 must leave delta_g bumped");

    /* Call 2: well-conditioned M2 = diag(3,4,5) (PD even at delta=0).  With the
     * carried delta_g it factors on the first attempt; delta_g must be EXACTLY
     * the carried value -- no reset, no further bump. */
    arb_set_si(arb_mat_entry(M2, 0, 0), 3);
    arb_set_si(arb_mat_entry(M2, 1, 1), 4);
    arb_set_si(arb_mat_entry(M2, 2, 2), 5);
    attempts = -1;
    ok = arbsdp_factor_with_reg(L, M2, &rs, &p, PREC, &attempts);
    CHECK(ok == 1, "carry: call 2 must factor PD M2");
    CHECK(attempts == 1, "carry: call 2 PD M2 factors on first attempt");
    CHECK(close_d(rs.delta_g, carried, carried * 1e-12 + 1e-300),
          "carry: delta_g must PERSIST across calls (carry-over, not reset)");

    arb_mat_clear(M1);
    arb_mat_clear(M2);
    arb_mat_clear(L);
    arbsdp_regstate_clear(&rs);
    arbsdp_problem_clear(&p);
}

/* ----- Test 5: mutation check (CLAUDE.md rule 8) ---------------------------
 * Two mutations the suite must be able to catch:
 *
 *  (M-A) RESET-PER-ITER: a buggy variant that resets d to its init each call.
 *        We simulate it by re-initializing the regstate between the two calls
 *        of the carry-over scenario; the carried-value assertion (test 4) would
 *        then see delta_g == 1e-12, NOT the carried value.  We assert here that
 *        these two observables DIFFER, proving the carry-over test discriminates
 *        a reset mutation.
 *
 *  (M-B) FACTOR-UNREGULARIZED-M: a buggy variant that factors the ORIGINAL M
 *        (no lift) and returns its cho result.  On the singular M = diag(1,1,0)
 *        that returns 0 (cho fails), whereas the real routine returns 1 (rescued
 *        via the lift).  We assert the two observables DIFFER.
 */
static void
test_mutation(void)
{
    arbsdp_problem p;
    build_healthy_problem(&p);

    /* (M-A) reset-per-iter would lose the carry-over. */
    {
        arbsdp_regstate rs;
        arb_mat_t M1, L;
        int attempts = -1;
        double real_carried, mutant_after_reset;

        arbsdp_regstate_init(&rs, PREC);
        arb_mat_init(M1, 3, 3);
        arb_mat_init(L, 3, 3);
        arb_set_si(arb_mat_entry(M1, 0, 0), 1);
        arb_set_si(arb_mat_entry(M1, 1, 1), 1);
        arb_set_d(arb_mat_entry(M1, 2, 2), -1e-3);

        arbsdp_factor_with_reg(L, M1, &rs, &p, PREC, &attempts);
        real_carried = to_double(rs.delta_g); /* what the REAL (carry) state holds */

        /* MUTANT behaviour: reset back to init before the next iteration. */
        arbsdp_regstate_reset(&rs, PREC);
        mutant_after_reset = to_double(rs.delta_g); /* would be 1e-12 again */

        CHECK(real_carried != mutant_after_reset,
              "MUTATION (reset): carried delta must differ from a reset delta");
        CHECK(close_d(rs.delta_g, 1e-12, 1e-24),
              "MUTATION (reset): reset returns delta_g to init 1e-12");

        arb_mat_clear(M1);
        arb_mat_clear(L);
        arbsdp_regstate_clear(&rs);
    }

    /* (M-B) factoring the UNregularized M would fail to rescue. */
    {
        arbsdp_regstate rs;
        arb_mat_t M, L, Lplain;
        int real_ok, mutant_ok, attempts = -1;

        arbsdp_regstate_init(&rs, PREC);
        arb_mat_init(M, 3, 3);
        arb_mat_init(L, 3, 3);
        arb_mat_init(Lplain, 3, 3);
        arb_set_si(arb_mat_entry(M, 0, 0), 1);
        arb_set_si(arb_mat_entry(M, 1, 1), 1);
        arb_set_si(arb_mat_entry(M, 2, 2), 0);

        real_ok = arbsdp_factor_with_reg(L, M, &rs, &p, PREC, &attempts);
        /* MUTANT behaviour: factor M directly with no regularization lift. */
        mutant_ok = arb_mat_cho(Lplain, M, PREC);

        CHECK(real_ok != mutant_ok,
              "MUTATION (unregularized): rescued (1) must differ from plain cho (0)");
        CHECK(real_ok == 1 && mutant_ok == 0,
              "MUTATION (unregularized): real rescues, plain-on-M fails");

        arb_mat_clear(M);
        arb_mat_clear(L);
        arb_mat_clear(Lplain);
        arbsdp_regstate_clear(&rs);
    }

    arbsdp_problem_clear(&p);
}

int
main(void)
{
    test_noop_on_pd();
    test_rescue();
    test_escalation();
    test_carry_over();
    test_mutation();

    if (failures != 0) {
        fprintf(stderr, "test_regularize: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_regularize (3-way Tikhonov: diagnose/bump/cap/carry-over)\n");
    return EXIT_SUCCESS;
}
