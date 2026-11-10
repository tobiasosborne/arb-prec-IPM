/*
 * test/test_pcache.c -- the per-(solve, prec) constraint-data materialization
 * cache (include/arbsdp/pcache.h, bead arb-prec-IPM-6b6).
 *
 * RED-GREEN TDD (CLAUDE.md rule 9): written before pcache.c existed; it failed to
 * build (no pcache.h / undefined arbsdp_pcache_* and *_pc symbols), and the
 * parse-count property (5) was confirmed RED against the pre-cache library with
 * the parse-count instrument alone: the matrix-parse count of a solve grew with
 * the iteration count.
 *
 * The cache is a pure performance change and MUST be numerically neutral
 * (bead 6b6 acceptance: bit-identical goldens).  Every comparison below is
 * therefore EXACT: arb_equal (identical midpoint AND identical radius), not a
 * tolerance.  The reference is the pre-cache on-demand path (problem.h
 * materializers; the p-based arbsdp_apply_A / arbsdp_apply_At /
 * arbsdp_schur_assemble / arbsdp_factor_with_reg, which still materialize per
 * call; and an iterate with pc == NULL).
 *
 * Properties:
 *   1. CACHE == MATERIALIZER  every cached A_i^b / b_i is arb_equal to
 *      arbsdp_problem_block_mat / arbsdp_problem_b at the same prec, on multi-
 *      block goldens (PSD + LP blocks, 2v off-diagonals) at several precs incl.
 *      a non-limb-multiple prec.  Also checks a nonzero data radius is present
 *      somewhere, so the radius-preservation half is actually exercised.
 *   2. KERNELS BIT-IDENTICAL  apply_A_pc / apply_At_pc / schur_assemble_pc /
 *      factor_with_reg_pc == their uncached references on pseudo-random
 *      symmetric inputs (with nonzero radii on X, to exercise radius flow).
 *   3. RESIDUALS BIT-IDENTICAL  arbsdp_iterate_residuals with it->pc set ==
 *      with it->pc == NULL, every output field.
 *   4. DIRECTION BIT-IDENTICAL  arbsdp_direction_compute with it->pc set ==
 *      with it->pc == NULL (dX, dS, dy, dtau, dkappa, alpha, sigma, mu_aff,
 *      factor attempts, carried regstate), on a mid-solve iterate.
 *   5. PARSE COUNT IS O(nnz) PER SOLVE, NOT O(iters)  a fixed-prec solve
 *      performs exactly nnz(A) + 3 nnz(C) matrix-entry parses (cache build once
 *      + C for the result iterate, the best snapshot, initial_scale), the SAME
 *      for iter_limit = 2 and the full solve; and the returned iterate does not
 *      hold a (dangling) cache pointer.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every init has a matching clear.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <flint/flint.h>
#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"
#include "arbsdp/pcache.h"
#include "arbsdp/iterate.h"
#include "arbsdp/schur.h"
#include "arbsdp/regularize.h"
#include "arbsdp/ntscaling.h"
#include "arbsdp/direction.h"
#include "arbsdp/solve.h"
#include "arbsdp/svec.h"

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);  \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* The multi-block goldens' A / b data is dyadic (exact in binary: 1, 0.5, 2,
 * ...), so on them arb_set_str returns radius-0 balls and the radius half of the
 * bit-identity contract would go untested (rule 8; measured: the any_radius
 * check below fails without this fixture).  This multi-block fixture (PSD 2x2 +
 * LP 2 + PSD 3x3, m = 3, tr X = 1.7 so bounded) uses NON-dyadic decimals (0.1,
 * 0.3, 1/3 to 34 digits, 1.7, ...) whose arb_set_str balls carry a nonzero
 * radius.  Written to the build directory at test start. */
#define FIXTURE "@fixture"
static const char *const FIXTURE_PATH = FIXTURE_DIR "/test_pcache_fixture.dat-s";
static const char *const FIXTURE_TEXT =
    "* test_pcache fixture (bead 6b6): non-dyadic multi-block data\n"
    "3\n3\n2 -2 3\n"
    "1.7 0.1 0.2\n"
    "0 1 1 1 1.1\n0 1 1 2 0.25\n0 2 1 1 0.6\n0 2 2 2 -0.4\n"
    "0 3 1 2 0.9\n0 3 2 2 0.7\n0 3 3 3 -0.2\n"
    "1 1 1 1 1\n1 1 2 2 1\n1 2 1 1 1\n1 2 2 2 1\n1 3 1 1 1\n1 3 2 2 1\n1 3 3 3 1\n"
    "2 1 1 2 0.1\n2 3 1 1 0.3\n2 3 2 3 -0.7\n"
    "3 2 1 1 0.3\n3 2 2 2 -0.1\n3 3 3 3 0.3333333333333333333333333333333333\n";

static const char *const MULTI_BLOCK[] = {
    FIXTURE,                   /* non-dyadic: nonzero data radii       */
    "mixed_blocks",            /* PSD 2x2 + LP 1x1                     */
    "two_block_corr_coupled",  /* 2 PSD blocks, coupling constraints   */
    "separable_12block",       /* 12 blocks, m = 12                    */
    "lp_diagonal_block",       /* LP block                             */
};
#define N_MULTI ((int) (sizeof MULTI_BLOCK / sizeof MULTI_BLOCK[0]))

static int
write_fixture(void)
{
    FILE *f = fopen(FIXTURE_PATH, "w");
    if (f == NULL || fputs(FIXTURE_TEXT, f) < 0) {
        fprintf(stderr, "FAIL: cannot write %s\n", FIXTURE_PATH);
        failures++;
        if (f != NULL)
            fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

static int
load(arbsdp_problem *p, const char *name)
{
    char path[1024];
    if (strcmp(name, FIXTURE) == 0)
        snprintf(path, sizeof path, "%s", FIXTURE_PATH);
    else
        snprintf(path, sizeof path, "%s/%s/%s.dat-s", GOLDEN_DIR, name, name);
    if (arbsdp_read_sdpa(p, path) != 0) {
        fprintf(stderr, "FAIL: cannot read %s\n", path);
        failures++;
        return 1;
    }
    return 0;
}

static slong
side(const arbsdp_problem *p, int b)
{
    int s = p->block_sizes[b];
    return s < 0 ? -(slong) s : (slong) s;
}

static int
mat_equal_exact(const arb_mat_t A, const arb_mat_t B)
{
    if (arb_mat_nrows(A) != arb_mat_nrows(B) || arb_mat_ncols(A) != arb_mat_ncols(B))
        return 0;
    for (slong i = 0; i < arb_mat_nrows(A); i++)
        for (slong j = 0; j < arb_mat_ncols(A); j++)
            if (!arb_equal(arb_mat_entry(A, i, j), arb_mat_entry(B, i, j)))
                return 0;
    return 1;
}

static int
vec_equal_exact(arb_srcptr a, arb_srcptr b, slong n)
{
    for (slong i = 0; i < n; i++)
        if (!arb_equal(a + i, b + i))
            return 0;
    return 1;
}

static ulong
nnz(const arbsdp_problem *p, int matno_lo, int matno_hi)
{
    ulong c = 0;
    for (int k = matno_lo; k <= matno_hi; k++)
        for (int b = 0; b < p->nblocks; b++)
            c += (ulong) p->mats[(slong) k * p->nblocks + b].count;
    return c;
}

/* Deterministic pseudo-random symmetric matrix with entries in (-1,1) plus
 * `diag` on the diagonal; radius `rad` (0 for none) on every entry. */
static void
fill_sym(arb_mat_t A, flint_rand_t st, double diag, double rad, slong prec)
{
    slong n = arb_mat_nrows(A);
    for (slong i = 0; i < n; i++)
        for (slong j = i; j < n; j++) {
            double v = (double) n_randint(st, 2000001) / 1e6 - 1.0;
            arb_set_d(arb_mat_entry(A, i, j), v);
            arb_div_ui(arb_mat_entry(A, i, j), arb_mat_entry(A, i, j), 3, prec);
            if (i == j)
                arb_add_si(arb_mat_entry(A, i, j), arb_mat_entry(A, i, j),
                           (slong) diag, prec);
            if (rad > 0)
                mag_set_d(arb_radref(arb_mat_entry(A, i, j)), rad);
            else
                mag_zero(arb_radref(arb_mat_entry(A, i, j)));
            arb_set(arb_mat_entry(A, j, i), arb_mat_entry(A, i, j));
        }
}

/* ---- 1. cache == materializer ------------------------------------------- */
static void
test_cache_equals_materializer(void)
{
    const slong precs[] = { 64, 128, 333 };
    int any_radius = 0;
    for (int c = 0; c < N_MULTI; c++) {
        arbsdp_problem p;
        if (load(&p, MULTI_BLOCK[c]))
            continue;
        for (int k = 0; k < 3; k++) {
            slong prec = precs[k];
            arbsdp_pcache pc;
            arb_t bi;
            arbsdp_pcache_init(&pc, &p, prec);
            arb_init(bi);
            CHECK(pc.prec == prec && pc.m == p.m && pc.nblocks == p.nblocks,
                  "pcache header fields");
            CHECK(pc.primal_degen == -1, "diagnose memo starts unset");
            for (int i = 0; i < p.m; i++) {
                for (int b = 0; b < p.nblocks; b++) {
                    arb_mat_t R;
                    arb_mat_init(R, side(&p, b), side(&p, b));
                    arbsdp_problem_block_mat(R, &p, i + 1, b, prec);
                    CHECK(mat_equal_exact(arbsdp_pcache_A(&pc, i, b), R),
                          "cached A_i^b == block_mat (mid AND rad)");
                    for (slong r = 0; r < side(&p, b); r++)
                        for (slong s = 0; s < side(&p, b); s++)
                            if (!mag_is_zero(arb_radref(arb_mat_entry(R, r, s))))
                                any_radius = 1;
                    arb_mat_clear(R);
                }
                arbsdp_problem_b(bi, &p, i, prec);
                CHECK(arb_equal(pc.b + i, bi), "cached b_i == problem_b");
                if (!mag_is_zero(arb_radref(bi)))
                    any_radius = 1;
            }
            arb_clear(bi);
            arbsdp_pcache_clear(&pc);
            CHECK(pc.A == NULL && pc.b == NULL, "clear re-zeroes");
        }
        arbsdp_problem_clear(&p);
    }
    /* Rule 8: did we exercise the radius-preservation property at all? */
    CHECK(any_radius, "some materialized datum carries a nonzero radius");
}

/* ---- 2. kernels bit-identical ------------------------------------------- */
static void
test_kernels_bit_identical(void)
{
    const slong prec = 128;
    flint_rand_t st;
    flint_randinit(st);
    for (int c = 0; c < N_MULTI; c++) {
        arbsdp_problem p;
        if (load(&p, MULTI_BLOCK[c]))
            continue;
        const int m = p.m, nb = p.nblocks;
        arbsdp_pcache pc;
        arb_mat_t *X = flint_malloc((size_t) nb * sizeof *X);
        arb_mat_t *W = flint_malloc((size_t) nb * sizeof *W);
        arb_mat_t *T1 = flint_malloc((size_t) nb * sizeof *T1);
        arb_mat_t *T2 = flint_malloc((size_t) nb * sizeof *T2);
        arb_ptr o1 = _arb_vec_init(m), o2 = _arb_vec_init(m), y = _arb_vec_init(m);
        arb_mat_t M1, M2, L1, L2;

        arbsdp_pcache_init(&pc, &p, prec);
        for (int b = 0; b < nb; b++) {
            slong n = side(&p, b);
            arb_mat_init(X[b], n, n);
            arb_mat_init(W[b], n, n);
            arb_mat_init(T1[b], n, n);
            arb_mat_init(T2[b], n, n);
            fill_sym(X[b], st, 3.0, 1e-30, prec);   /* radius flows through */
            fill_sym(W[b], st, 2.0, 0.0, prec);
        }
        for (int i = 0; i < m; i++) {
            arb_set_d(y + i, (double) n_randint(st, 2001) / 1000.0 - 1.0);
            arb_div_ui(y + i, y + i, 7, prec);
        }

        arbsdp_apply_A(o1, &p, (const arb_mat_t *) X, prec);
        arbsdp_apply_A_pc(o2, &pc, (const arb_mat_t *) X, prec);
        CHECK(vec_equal_exact(o1, o2, m), "apply_A_pc == apply_A (exact)");

        arbsdp_apply_At(T1, &p, y, prec);
        arbsdp_apply_At_pc(T2, &pc, y, prec);
        for (int b = 0; b < nb; b++)
            CHECK(mat_equal_exact(T1[b], T2[b]), "apply_At_pc == apply_At (exact)");

        arb_mat_init(M1, m, m);
        arb_mat_init(M2, m, m);
        arb_mat_init(L1, m, m);
        arb_mat_init(L2, m, m);
        arbsdp_schur_assemble(M1, &p, (const arb_mat_t *) W, prec);
        arbsdp_schur_assemble_pc(M2, &pc, (const arb_mat_t *) W, prec);
        CHECK(mat_equal_exact(M1, M2), "schur_assemble_pc == schur_assemble (exact)");

        /* factor_with_reg: an indefinite shift forces the bump/diagnose path
         * (M - 1e-3*I on a symmetric W-scaled M may or may not be PD; also run
         * the plain M).  Compare L, the carried regstate and attempts. */
        for (int shifted = 0; shifted < 2; shifted++) {
            arbsdp_regstate r1, r2;
            int a1 = 0, a2 = 0, ok1, ok2;
            arbsdp_regstate_init(&r1, prec);
            arbsdp_regstate_init(&r2, prec);
            if (shifted) {
                arb_t sh;
                arb_init(sh);
                arb_set_d(sh, 1e-3);
                for (int i = 0; i < m; i++)
                    arb_sub(arb_mat_entry(M1, i, i), arb_mat_entry(M1, i, i), sh, prec);
                arb_clear(sh);
            }
            arb_mat_zero(L1);
            arb_mat_zero(L2);
            ok1 = arbsdp_factor_with_reg(L1, M1, &r1, &p, prec, &a1);
            ok2 = arbsdp_factor_with_reg_pc(L2, M1, &r2, &pc, prec, &a2);
            CHECK(ok1 == ok2 && a1 == a2, "factor_with_reg_pc: same ok/attempts");
            CHECK(mat_equal_exact(L1, L2), "factor_with_reg_pc: same L (exact)");
            CHECK(arb_equal(r1.delta_p, r2.delta_p) && arb_equal(r1.delta_d, r2.delta_d)
                  && arb_equal(r1.delta_g, r2.delta_g) && r1.refactors == r2.refactors,
                  "factor_with_reg_pc: same carried regstate");
            CHECK(pc.primal_degen == 0 || pc.primal_degen == 1,
                  "diagnose memo filled by factor_with_reg_pc");
            arbsdp_regstate_clear(&r1);
            arbsdp_regstate_clear(&r2);
        }

        arb_mat_clear(M1);
        arb_mat_clear(M2);
        arb_mat_clear(L1);
        arb_mat_clear(L2);
        for (int b = 0; b < nb; b++) {
            arb_mat_clear(X[b]);
            arb_mat_clear(W[b]);
            arb_mat_clear(T1[b]);
            arb_mat_clear(T2[b]);
        }
        flint_free(X);
        flint_free(W);
        flint_free(T1);
        flint_free(T2);
        _arb_vec_clear(o1, m);
        _arb_vec_clear(o2, m);
        _arb_vec_clear(y, m);
        arbsdp_pcache_clear(&pc);
        arbsdp_problem_clear(&p);
    }
    flint_randclear(st);
}

/* ---- 3 + 4. residuals and direction bit-identical ----------------------- */
static int
iterate_outputs_equal(const arbsdp_iterate *a, const arbsdp_iterate *b)
{
    int ok = vec_equal_exact(a->r_p, b->r_p, a->m)
          && arb_equal(a->r_g, b->r_g) && arb_equal(a->mu, b->mu)
          && arb_equal(a->pObj, b->pObj) && arb_equal(a->dObj, b->dObj)
          && arb_equal(a->primal_inf, b->primal_inf)
          && arb_equal(a->dual_inf, b->dual_inf)
          && arb_equal(a->gap_inf, b->gap_inf);
    for (int k = 0; k < a->nblocks; k++)
        ok = ok && mat_equal_exact(a->R_d[k], b->R_d[k]);
    return ok;
}

static void
test_residuals_and_direction_bit_identical(void)
{
    const slong prec = 192;
    for (int c = 0; c < N_MULTI; c++) {
        arbsdp_problem p;
        if (load(&p, MULTI_BLOCK[c]))
            continue;
        const int nb = p.nblocks;
        arbsdp_result res;
        arbsdp_solve_params sp;
        arbsdp_pcache pc;
        arb_mat_t *W;
        int nt_ok = 1;

        /* a genuine MID-SOLVE iterate: stop after 4 iterations */
        arbsdp_solve_default_params(&sp);
        sp.iter_limit = 4;
        arbsdp_solve(&res, &p, prec, &sp);
        CHECK(res.it.pc == NULL, "solve detaches its cache from the result iterate");

        arbsdp_pcache_init(&pc, &p, prec);

        /* residuals: pc == NULL (scoped temporary) vs cache */
        {
            arbsdp_iterate ref;
            arbsdp_iterate_init(&ref, &p, prec);
            for (int b = 0; b < nb; b++) {
                arb_mat_set(ref.X[b], res.it.X[b]);
                arb_mat_set(ref.S[b], res.it.S[b]);
            }
            _arb_vec_set(ref.y, res.it.y, p.m);
            arb_set(ref.tau, res.it.tau);
            arb_set(ref.kappa, res.it.kappa);
            arbsdp_iterate_residuals(&ref, prec);           /* pc == NULL */
            res.it.pc = &pc;
            arbsdp_iterate_residuals(&res.it, prec);        /* cached     */
            CHECK(iterate_outputs_equal(&ref, &res.it),
                  "residuals: cached == uncached (exact)");
            arbsdp_iterate_clear(&ref);
        }

        W = flint_malloc((size_t) nb * sizeof *W);
        for (int b = 0; b < nb; b++) {
            arb_mat_init(W[b], res.it.block_n[b], res.it.block_n[b]);
            if (arbsdp_nt_scaling(W[b], res.it.X[b], res.it.S[b], prec))
                nt_ok = 0;
        }
        CHECK(nt_ok, "mid-solve iterate admits an NT scaling");
        if (nt_ok) {
            arbsdp_direction d1, d2;
            arbsdp_regstate r1, r2;
            int ok1, ok2, same;
            arbsdp_direction_init(&d1, &p);
            arbsdp_direction_init(&d2, &p);
            arbsdp_regstate_init(&r1, prec);
            arbsdp_regstate_init(&r2, prec);

            res.it.pc = NULL;
            ok1 = arbsdp_direction_compute(&d1, &res.it, (const arb_mat_t *) W, &r1, prec);
            res.it.pc = &pc;
            ok2 = arbsdp_direction_compute(&d2, &res.it, (const arb_mat_t *) W, &r2, prec);
            CHECK(ok1 == ok2, "direction: same ok");
            same = arb_equal(d1.dtau, d2.dtau) && arb_equal(d1.dkappa, d2.dkappa)
                && arb_equal(d1.alpha, d2.alpha) && arb_equal(d1.sigma, d2.sigma)
                && arb_equal(d1.mu_aff, d2.mu_aff)
                && vec_equal_exact(d1.dy, d2.dy, p.m)
                && d1.factor_attempts == d2.factor_attempts
                && arb_equal(r1.delta_g, r2.delta_g) && arb_equal(r1.delta_p, r2.delta_p);
            for (int b = 0; b < nb; b++)
                same = same && mat_equal_exact(d1.dX[b], d2.dX[b])
                            && mat_equal_exact(d1.dS[b], d2.dS[b]);
            CHECK(!ok1 || same, "direction: cached == uncached (exact)");
            arbsdp_regstate_clear(&r1);
            arbsdp_regstate_clear(&r2);
            arbsdp_direction_clear(&d1);
            arbsdp_direction_clear(&d2);
        }
        res.it.pc = NULL;   /* never leave a borrowed pointer past the cache */

        for (int b = 0; b < nb; b++)
            arb_mat_clear(W[b]);
        flint_free(W);
        arbsdp_pcache_clear(&pc);
        arbsdp_result_clear(&res);
        arbsdp_problem_clear(&p);
    }
}

/* ---- 5. parse count is O(nnz) per solve, independent of iterations ------- */
static void
test_parse_count_independent_of_iters(void)
{
    const char *const names[] = { FIXTURE, "separable_12block", "two_block_corr_coupled",
                                  "max_eig_path_8" };
    const slong prec = 128;
    for (int c = 0; c < 4; c++) {
        arbsdp_problem p;
        if (load(&p, names[c]))
            continue;
        const ulong want = nnz(&p, 1, p.m) + 3 * nnz(&p, 0, 0);
        ulong cnt[2];
        int iters[2];
        const int limits[2] = { 2, ARBSDP_ITER_LIMIT };
        for (int k = 0; k < 2; k++) {
            arbsdp_result res;
            arbsdp_solve_params sp;
            arbsdp_solve_default_params(&sp);
            sp.iter_limit = limits[k];
            arbsdp_parse_count_reset();
            arbsdp_solve(&res, &p, prec, &sp);
            cnt[k] = arbsdp_parse_count_mat();
            iters[k] = res.iters;
            CHECK(res.it.pc == NULL, "result iterate holds no cache pointer");
            arbsdp_result_clear(&res);
        }
        printf("  %-24s m=%d nnz(A)=%lu nnz(C)=%lu  mat parses: %lu (<=2 it), "
               "%lu (%d it); want %lu\n",
               names[c], p.m, nnz(&p, 1, p.m), nnz(&p, 0, 0), cnt[0], cnt[1],
               iters[1], want);
        CHECK(iters[1] > 2, "the full solve really runs more iterations");
        CHECK(cnt[0] == want, "short solve: nnz(A) + 3 nnz(C) matrix parses");
        CHECK(cnt[1] == want, "full solve: SAME parse count (no per-iter parse)");
        arbsdp_problem_clear(&p);
    }
}

int
main(void)
{
    if (write_fixture())
        return 1;
    test_cache_equals_materializer();
    test_kernels_bit_identical();
    test_residuals_and_direction_bit_identical();
    test_parse_count_independent_of_iters();

    if (failures) {
        fprintf(stderr, "test_pcache: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("test_pcache: all checks passed\n");
    flint_cleanup();
    return 0;
}
