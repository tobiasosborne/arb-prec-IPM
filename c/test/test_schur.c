/*
 * test/test_schur.c -- tests for the NT-scaled Schur complement assembly and the
 * factor-once/solve-many interface (c/src/schur.c, bead arb-prec-IPM-b12).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written before c/src/schur.c and
 * include/arbsdp/schur.h exist; the target fails to compile/link first
 * (no header / undefined symbols arbsdp_schur_assemble / _factor / _solve),
 * then passes once the impl lands.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §3.4: the NT-scaled Schur complement
 *         M_ik = sum_b <A_i^b, W^b A_k^b W^b>,
 *     symmetric PSD in exact arithmetic; symmetrized in the TS reference.
 *   - HsdeNtSdpSolver.ts:541-558 (M.fill(0); M_ik += frobInner(A_i, WAW_k);
 *     then symmetrise by averaging).
 *   - docs/FLINT_NOTES.md: arb_mat_cho returns 1 iff A proven PD;
 *     arb_mat_solve_cho_precomp reuses ONE factor for many RHS (the Mehrotra
 *     predictor+corrector share one Schur factorisation).
 *
 * POINT MODE (CLAUDE.md rule 1): the Schur assembly is a Layer-0 kernel; W is a
 * POINT (midpoint) from arbsdp_nt_scaling and M is consumed as midpoints.  The
 * symmetry / accuracy assertions compare MIDPOINTS within a justified tolerance,
 * as in test_ntscaling.c / test_linalg.c.  arb_mat_cho/_solve_cho_precomp are
 * themselves rigorous (ball) routines; here we use cho only to confirm PD and to
 * obtain a usable factor, and check the SOLVE residual on midpoints.
 *
 * Property coverage (bead arb-prec-IPM-b12):
 *   1. M SYMMETRIC          M_ik == M_ki on a constructed (problem, W).
 *   2. M PD + FACTOR        arbsdp_schur_factor returns 1 on a well-conditioned
 *                           fixture (W = nt_scaling(X,S) for PD X,S).
 *   3. SOLVE / FACTOR-REUSE factor M once, arbsdp_schur_solve for TWO rhs;
 *                           assert M·x ~ rhs (small residual) for both -- this
 *                           validates solve_cho_precomp reuse (predictor+corrector).
 *   4. CLOSED-FORM          (a) A_i = E_ii (diagonal unit), W = I  =>  M = I exactly;
 *                           (b) W = 2*I  =>  M_ik = <A_i, (2I)A_k(2I)> = 4*<A_i,A_k>
 *                               so for A_i=E_ii, M = 4*I.  Pins the W·A·W triple:
 *                               a DROPPED W gives 2*I (one factor) or I (both
 *                               dropped), and a wrong index is caught by the
 *                               off-diagonal being exactly 0.
 *   5. MUTATION CHECK (CLAUDE.md rule 8): a hand-rolled assembly that DROPS one W
 *                           from the triple product must produce a DIFFERENT M
 *                           than arbsdp_schur_assemble -- proving the test
 *                           discriminates a missing W.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"
#include "arbsdp/ntscaling.h"
#include "arbsdp/svec.h"
#include "arbsdp/schur.h"

#define PREC 256
#define TOL 1e-50

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* Point-mode closeness on midpoints: |mid(a) - mid(b)| <= tol. */
static int
close_tol(const arb_t a, const arb_t b, double tol)
{
    arb_t d, t;
    int ok;
    arb_init(d);
    arb_init(t);
    arb_get_mid_arb(d, a);
    arb_get_mid_arb(t, b);
    arb_sub(d, d, t, PREC);
    arb_abs(d, d);
    arb_set_d(t, tol);
    ok = arb_le(d, t);
    arb_clear(d);
    arb_clear(t);
    return ok;
}

static int
mat_close(const arb_mat_t A, const arb_mat_t B, double tol)
{
    slong n = arb_mat_nrows(A), m = arb_mat_ncols(A);
    if (arb_mat_nrows(B) != n || arb_mat_ncols(B) != m) return 0;
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < m; j++)
            if (!close_tol(arb_mat_entry(A, i, j), arb_mat_entry(B, i, j), tol))
                return 0;
    return 1;
}

static void
set_sym(arb_mat_t A, slong i, slong j, slong v)
{
    arb_set_si(arb_mat_entry(A, i, j), v);
    arb_set_si(arb_mat_entry(A, j, i), v);
}

/* Heap-duplicate a NUL-terminated string with malloc (matches what
 * arbsdp_problem_clear frees with free()).  Avoids POSIX strdup, which is not
 * declared under -std=c11 -Wpedantic (as in io.c, which copies via malloc). */
static char *
dup_str(const char *s)
{
    size_t len = strlen(s) + 1;
    char *out = malloc(len);
    memcpy(out, s, len);
    return out;
}

/* -------------------------------------------------------------------------
 * In-memory problem builder.
 *
 * Constructs an arbsdp_problem with a SINGLE block of side n and m constraints,
 * whose constraint matrices A_i (i=1..m) are given as dense symmetric integer
 * matrices A_dense[i].  C (matno 0) is left empty (the Schur assembly does not
 * read C).  Allocations use malloc/strdup so arbsdp_problem_clear (which frees
 * with free()) tears it down leak-clean -- matching io.c exactly.
 *
 * Only the UPPER triangle (i<=j) is stored as entries, exactly as the .dat-s
 * format and arbsdp_problem_block_mat expect (the materializer mirrors i<j to
 * (j,i)).  Off-diagonal entries are stored verbatim (v), no factor of 2 (the "2v"
 * lives in the Frobenius inner product; see problem.h).
 * ---------------------------------------------------------------------- */
static void
build_problem(arbsdp_problem *p, int m, int n, /* m constraints, 1 block side n */
              const long (*A_dense)[16] /* [m][n*n] row-major */)
{
    char buf[64];
    arbsdp_problem_init(p);
    p->m = m;
    p->nblocks = 1;
    p->maximize = 1;
    p->block_sizes = malloc(sizeof(int));
    p->block_sizes[0] = n; /* positive => dense PSD block */
    p->b = malloc((size_t) m * sizeof(char *));
    for (int i = 0; i < m; i++)
        p->b[i] = dup_str("0"); /* b is not read by the Schur assembly */

    slong ncells = (slong) (m + 1) * 1;
    p->mats = calloc((size_t) ncells, sizeof(arbsdp_matblock));
    /* matno 0 = C: leave empty (count = 0). matno 1..m = A_i. */
    for (int mi = 1; mi <= m; mi++) {
        arbsdp_matblock *mb = &p->mats[mi]; /* block 0 of matno mi */
        const long *Ad = A_dense[mi - 1];
        /* store upper triangle (i<=j) nonzeros */
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

/* ----- Test 1: M symmetric -------------------------------------------------
 * Single 3x3 block, m=3 constraints with mixed dense A_i, W = nt_scaling(X,S)
 * for distinct PD X,S.  M must be symmetric. */
static void
test_symmetric(void)
{
    const int n = 3, m = 3;
    /* A_1 = E_00; A_2 = E_11 + (E_01+E_10); A_3 = E_22 + (E_02+E_20)+(E_12+E_21). */
    const long A[3][16] = {
        { 1,0,0, 0,0,0, 0,0,0 },
        { 0,1,0, 1,1,0, 0,0,0 },
        { 0,0,1, 0,0,1, 1,1,1 },
    };
    arbsdp_problem p;
    arb_mat_t X, S, W, M, Mt;
    arb_mat_t *Wb;

    build_problem(&p, m, n, A);

    arb_mat_init(X, n, n);
    arb_mat_init(S, n, n);
    arb_mat_init(W, n, n);
    arb_mat_init(M, m, m);
    arb_mat_init(Mt, m, m);

    /* PD X,S (same fixtures as test_ntscaling n=3). */
    arb_set_si(arb_mat_entry(X, 0, 0), 5);
    arb_set_si(arb_mat_entry(X, 1, 1), 4);
    arb_set_si(arb_mat_entry(X, 2, 2), 6);
    set_sym(X, 0, 1, 1); set_sym(X, 0, 2, 2); set_sym(X, 1, 2, 1);
    arb_set_si(arb_mat_entry(S, 0, 0), 6);
    arb_set_si(arb_mat_entry(S, 1, 1), 7);
    arb_set_si(arb_mat_entry(S, 2, 2), 4);
    set_sym(S, 0, 1, 2); set_sym(S, 0, 2, 1); set_sym(S, 1, 2, -1);

    arbsdp_nt_scaling(W, X, S, PREC);

    Wb = &W; /* single block */
    arbsdp_schur_assemble(M, &p, (const arb_mat_t *) Wb, PREC);

    arb_mat_transpose(Mt, M);
    CHECK(mat_close(M, Mt, TOL), "schur_assemble: M != M^T");

    arb_mat_clear(X);
    arb_mat_clear(S);
    arb_mat_clear(W);
    arb_mat_clear(M);
    arb_mat_clear(Mt);
    arbsdp_problem_clear(&p);
}

/* ----- Test 2: M PD + factor succeeds --------------------------------------
 * Same well-conditioned fixture; arbsdp_schur_factor must return 1 (proven PD).
 * The A_i above are linearly independent so M is PD. */
static void
test_pd_factor(void)
{
    const int n = 3, m = 3;
    const long A[3][16] = {
        { 1,0,0, 0,0,0, 0,0,0 },
        { 0,1,0, 1,1,0, 0,0,0 },
        { 0,0,1, 0,0,1, 1,1,1 },
    };
    arbsdp_problem p;
    arb_mat_t X, S, W, M, L;
    arb_mat_t *Wb;
    int pd;

    build_problem(&p, m, n, A);
    arb_mat_init(X, n, n);
    arb_mat_init(S, n, n);
    arb_mat_init(W, n, n);
    arb_mat_init(M, m, m);
    arb_mat_init(L, m, m);

    arb_set_si(arb_mat_entry(X, 0, 0), 5);
    arb_set_si(arb_mat_entry(X, 1, 1), 4);
    arb_set_si(arb_mat_entry(X, 2, 2), 6);
    set_sym(X, 0, 1, 1); set_sym(X, 0, 2, 2); set_sym(X, 1, 2, 1);
    arb_set_si(arb_mat_entry(S, 0, 0), 6);
    arb_set_si(arb_mat_entry(S, 1, 1), 7);
    arb_set_si(arb_mat_entry(S, 2, 2), 4);
    set_sym(S, 0, 1, 2); set_sym(S, 0, 2, 1); set_sym(S, 1, 2, -1);

    arbsdp_nt_scaling(W, X, S, PREC);
    Wb = &W;
    arbsdp_schur_assemble(M, &p, (const arb_mat_t *) Wb, PREC);

    pd = arbsdp_schur_factor(L, M, PREC);
    CHECK(pd == 1, "schur_factor: well-conditioned M not proven PD");

    arb_mat_clear(X);
    arb_mat_clear(S);
    arb_mat_clear(W);
    arb_mat_clear(M);
    arb_mat_clear(L);
    arbsdp_problem_clear(&p);
}

/* ----- Test 3: solve accuracy / factor-reuse -------------------------------
 * Factor M ONCE, then arbsdp_schur_solve for TWO distinct rhs.  For each, check
 * M·x ~ rhs (small residual) -- validates solve_cho_precomp reuse (the Mehrotra
 * predictor + corrector share one factor). */
static void
test_solve_reuse(void)
{
    const int n = 3, m = 3;
    const long A[3][16] = {
        { 1,0,0, 0,0,0, 0,0,0 },
        { 0,1,0, 1,1,0, 0,0,0 },
        { 0,0,1, 0,0,1, 1,1,1 },
    };
    arbsdp_problem p;
    arb_mat_t X, S, W, M, L, rhs1, rhs2, x1, x2, chk;
    arb_mat_t *Wb;
    int pd;

    build_problem(&p, m, n, A);
    arb_mat_init(X, n, n);
    arb_mat_init(S, n, n);
    arb_mat_init(W, n, n);
    arb_mat_init(M, m, m);
    arb_mat_init(L, m, m);
    arb_mat_init(rhs1, m, 1);
    arb_mat_init(rhs2, m, 1);
    arb_mat_init(x1, m, 1);
    arb_mat_init(x2, m, 1);
    arb_mat_init(chk, m, 1);

    arb_set_si(arb_mat_entry(X, 0, 0), 5);
    arb_set_si(arb_mat_entry(X, 1, 1), 4);
    arb_set_si(arb_mat_entry(X, 2, 2), 6);
    set_sym(X, 0, 1, 1); set_sym(X, 0, 2, 2); set_sym(X, 1, 2, 1);
    arb_set_si(arb_mat_entry(S, 0, 0), 6);
    arb_set_si(arb_mat_entry(S, 1, 1), 7);
    arb_set_si(arb_mat_entry(S, 2, 2), 4);
    set_sym(S, 0, 1, 2); set_sym(S, 0, 2, 1); set_sym(S, 1, 2, -1);

    arbsdp_nt_scaling(W, X, S, PREC);
    Wb = &W;
    arbsdp_schur_assemble(M, &p, (const arb_mat_t *) Wb, PREC);

    pd = arbsdp_schur_factor(L, M, PREC);
    CHECK(pd == 1, "solve_reuse: M not proven PD");

    /* two distinct rhs (predictor + corrector) */
    arb_set_si(arb_mat_entry(rhs1, 0, 0), 1);
    arb_set_si(arb_mat_entry(rhs1, 1, 0), 2);
    arb_set_si(arb_mat_entry(rhs1, 2, 0), 3);
    arb_set_si(arb_mat_entry(rhs2, 0, 0), -5);
    arb_set_si(arb_mat_entry(rhs2, 1, 0), 7);
    arb_set_si(arb_mat_entry(rhs2, 2, 0), 0);

    arbsdp_schur_solve(x1, L, rhs1, PREC);
    arbsdp_schur_solve(x2, L, rhs2, PREC);

    arb_mat_mul(chk, M, x1, PREC);
    CHECK(mat_close(chk, rhs1, TOL), "solve_reuse: M*x1 != rhs1 (rhs1)");

    arb_mat_mul(chk, M, x2, PREC);
    CHECK(mat_close(chk, rhs2, TOL), "solve_reuse: M*x2 != rhs2 (rhs2, factor reuse)");

    arb_mat_clear(X);
    arb_mat_clear(S);
    arb_mat_clear(W);
    arb_mat_clear(M);
    arb_mat_clear(L);
    arb_mat_clear(rhs1);
    arb_mat_clear(rhs2);
    arb_mat_clear(x1);
    arb_mat_clear(x2);
    arb_mat_clear(chk);
    arbsdp_problem_clear(&p);
}

/* ----- Test 4: closed-form Schur -------------------------------------------
 * Single block n=3, A_i = E_ii (the three diagonal unit matrices), m=3.
 *   (a) W = I:   M_ik = <E_ii, I E_kk I> = <E_ii, E_kk> = delta_ik  =>  M = I.
 *   (b) W = 2I:  M_ik = <E_ii, (2I)E_kk(2I)> = 4<E_ii,E_kk> = 4 delta_ik => M=4I.
 * This pins the W A W triple AND the index: a missing W in (b) gives 2*I (one
 * dropped) or I (both dropped); a transposed index leaves the diagonal but the
 * exact-zero off-diagonals catch a wrong i/k pairing. */
static void
test_closed_form(void)
{
    const int n = 3, m = 3;
    const long A[3][16] = {
        { 1,0,0, 0,0,0, 0,0,0 },   /* E_00 */
        { 0,0,0, 0,1,0, 0,0,0 },   /* E_11 */
        { 0,0,0, 0,0,0, 0,0,1 },   /* E_22 */
    };
    arbsdp_problem p;
    arb_mat_t W, M, Iexp, FourI;
    arb_mat_t *Wb;

    build_problem(&p, m, n, A);
    arb_mat_init(W, n, n);
    arb_mat_init(M, m, m);
    arb_mat_init(Iexp, m, m);
    arb_mat_init(FourI, m, m);
    arb_mat_one(Iexp);
    for (int i = 0; i < m; i++) arb_set_si(arb_mat_entry(FourI, i, i), 4);

    /* (a) W = I  =>  M = I */
    arb_mat_one(W);
    Wb = &W;
    arbsdp_schur_assemble(M, &p, (const arb_mat_t *) Wb, PREC);
    CHECK(mat_close(M, Iexp, TOL), "closed-form: W=I, A_i=E_ii must give M=I");

    /* (b) W = 2I  =>  M = 4I */
    for (int i = 0; i < n; i++) arb_set_si(arb_mat_entry(W, i, i), 2);
    arbsdp_schur_assemble(M, &p, (const arb_mat_t *) Wb, PREC);
    CHECK(mat_close(M, FourI, TOL),
          "closed-form: W=2I, A_i=E_ii must give M=4I (W A W triple)");

    arb_mat_clear(W);
    arb_mat_clear(M);
    arb_mat_clear(Iexp);
    arb_mat_clear(FourI);
    arbsdp_problem_clear(&p);
}

/* ----- Test 5: mutation check (CLAUDE.md rule 8) ---------------------------
 * A reference assembly that DROPS one W (computes <A_i, A_k W> instead of
 * <A_i, W A_k W>) must produce a M that DIFFERS from arbsdp_schur_assemble on a
 * fixture where W != I.  This proves the assembly's two-W triple is load-bearing
 * and the test would catch a dropped factor.  We use the closed-form A_i = E_ii
 * with W = 2I: the correct M = 4I; the dropped-W variant gives <E_ii,(2I)E_kk> =
 * 2 delta_ik = 2I, which the test must see as DIFFERENT. */
static void
test_mutation(void)
{
    const int n = 3, m = 3;
    const long A[3][16] = {
        { 1,0,0, 0,0,0, 0,0,0 },
        { 0,0,0, 0,1,0, 0,0,0 },
        { 0,0,0, 0,0,0, 0,0,1 },
    };
    arbsdp_problem p;
    arb_mat_t W, M, Mmut, tmp;
    arb_mat_t Ai, Ak;
    arb_t v;
    arb_mat_t *Wb;

    build_problem(&p, m, n, A);
    arb_mat_init(W, n, n);
    arb_mat_init(M, m, m);
    arb_mat_init(Mmut, m, m);
    arb_mat_init(tmp, n, n);
    arb_mat_init(Ai, n, n);
    arb_mat_init(Ak, n, n);
    arb_init(v);

    for (int i = 0; i < n; i++) arb_set_si(arb_mat_entry(W, i, i), 2); /* W = 2I */
    Wb = &W;

    /* correct assembly */
    arbsdp_schur_assemble(M, &p, (const arb_mat_t *) Wb, PREC);

    /* MUTANT: drop one W -- M'_ik = <A_i, A_k W> (single factor). */
    for (int k = 0; k < m; k++) {
        arbsdp_problem_block_mat(Ak, &p, k + 1, 0, PREC);
        arb_mat_mul(tmp, Ak, W, PREC); /* A_k W  (NO left W) */
        for (int i = 0; i < m; i++) {
            arbsdp_problem_block_mat(Ai, &p, i + 1, 0, PREC);
            arbsdp_frob_inner(v, Ai, tmp, PREC);
            arb_set(arb_mat_entry(Mmut, i, k), v);
        }
    }

    CHECK(!mat_close(M, Mmut, TOL),
          "MUTATION: dropping a W must change M (test would be blind otherwise)");

    arb_mat_clear(W);
    arb_mat_clear(M);
    arb_mat_clear(Mmut);
    arb_mat_clear(tmp);
    arb_mat_clear(Ai);
    arb_mat_clear(Ak);
    arb_clear(v);
    arbsdp_problem_clear(&p);
}

int
main(void)
{
    test_symmetric();
    test_pd_factor();
    test_solve_reuse();
    test_closed_form();
    test_mutation();

    if (failures != 0) {
        fprintf(stderr, "test_schur: %d check(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS: test_schur (M=sum_b <A_i,W A_k W>, factor-once/solve-many)\n");
    return EXIT_SUCCESS;
}
