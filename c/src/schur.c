/*
 * src/schur.c -- the NT-scaled Schur complement assembly and the
 * factor-once/solve-many interface (POINT MODE, Layer 0).
 *
 * GROUND TRUTH (CLAUDE.md rule 3) -- see include/arbsdp/schur.h banner for the
 * full citation list and the symmetric-W vs G-factor AUDITION (decision: use the
 * symmetric W; the G-factor yields the SAME M, measured in
 * docs/probes/schur_nt_audition.c).
 *   - docs/MATH_SPEC.md §3.4: M_ik = sum_b <A_i^b, W^b A_k^b W^b>.
 *   - HsdeNtSdpSolver.ts:511-558 (WAW cache + frobInner accumulation + symmetrise).
 *   - docs/FLINT_NOTES.md: arb_mat_cho (1 => proven PD), arb_mat_solve_cho_precomp
 *     (factor-once / solve-many for predictor+corrector).
 *
 * REUSE (no duplication): arbsdp_problem_block_mat (problem.h),
 * arbsdp_frob_inner / arbsdp_symmetrize (svec.h).  W is supplied by the caller
 * (arbsdp_nt_scaling, ntscaling.h) -- this module does NOT recompute it.
 */

#include <assert.h>

#include <flint/flint.h>
#include <flint/arb_mat.h>

#include "arbsdp/schur.h"
#include "arbsdp/svec.h"

/* side length of block b (|block_sizes[b]|; negative = diagonal/LP block). */
static slong
block_side(const arbsdp_problem *p, int b)
{
    int s = p->block_sizes[b];
    return (s < 0) ? -(slong) s : (slong) s;
}

void
arbsdp_schur_assemble(arb_mat_t M, const arbsdp_problem *p,
                      const arb_mat_t *W_blocks, slong prec)
{
    int m, nb;

    assert(M != NULL);
    assert(p != NULL);
    assert(W_blocks != NULL);

    m = p->m;
    nb = p->nblocks;
    assert(arb_mat_nrows(M) == m);
    assert(arb_mat_ncols(M) == m);

    arb_mat_zero(M);
    if (m == 0)
        return;

    /*
     * Per-block inner-loop structure (MATH_SPEC §3.4, HsdeNtSdpSolver.ts:541-558),
     * with symmetry exploited: for each block b and each column k, build the cache
     *     WAW = sym(W^b A_k^b W^b)        (two mat-muls)
     * then accumulate ONLY the upper triangle  M_ik += <A_i^b, WAW>  for i <= k
     * (the Frobenius inner product is symmetric in its two arguments and M is
     * symmetric in exact arithmetic, so the strict-lower entries are redundant).
     * The lower triangle is mirrored from the upper after all blocks -- this both
     * fills M and removes the residual point-mode asymmetry (matching the TS
     * averaging, which here reduces to a copy since we computed each pair once).
     */
    for (int b = 0; b < nb; b++) {
        slong n = block_side(p, b);
        const arb_mat_t *Wb = &W_blocks[b];
        arb_mat_t Ak, Ai, WA, WAW;
        arb_t v;

        assert(arb_mat_nrows(*Wb) == n);
        assert(arb_mat_ncols(*Wb) == n);

        arb_mat_init(Ak, n, n);
        arb_mat_init(Ai, n, n);
        arb_mat_init(WA, n, n);
        arb_mat_init(WAW, n, n);
        arb_init(v);

        for (int k = 0; k < m; k++) {
            /* A_k^b (matno = k+1), then WAW = sym(W A_k W). */
            arbsdp_problem_block_mat(Ak, p, k + 1, b, prec);
            arb_mat_mul(WA, *Wb, Ak, prec);
            arb_mat_mul(WAW, WA, *Wb, prec);
            arbsdp_symmetrize(WAW, prec);

            /* upper triangle i <= k: M_ik += <A_i^b, WAW>. */
            for (int i = 0; i <= k; i++) {
                arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);
                arbsdp_frob_inner(v, Ai, WAW, prec);
                arb_add(arb_mat_entry(M, i, k), arb_mat_entry(M, i, k), v, prec);
            }
        }

        arb_mat_clear(Ak);
        arb_mat_clear(Ai);
        arb_mat_clear(WA);
        arb_mat_clear(WAW);
        arb_clear(v);
    }

    /* mirror upper -> lower (M symmetric). */
    for (int k = 0; k < m; k++)
        for (int i = 0; i < k; i++)
            arb_set(arb_mat_entry(M, k, i), arb_mat_entry(M, i, k));
}

int
arbsdp_schur_factor(arb_mat_t L, const arb_mat_t M, slong prec)
{
    assert(L != NULL);
    assert(M != NULL);
    assert(arb_mat_nrows(M) == arb_mat_ncols(M));
    assert(arb_mat_nrows(L) == arb_mat_nrows(M));
    assert(arb_mat_ncols(L) == arb_mat_ncols(M));

    /* arb_mat_cho returns 1 iff M is PROVEN positive-definite (FLINT_NOTES.md).
     * Report the result honestly; no regularization-on-failure here (b13). */
    return arb_mat_cho(L, M, prec);
}

void
arbsdp_schur_solve(arb_mat_t x, const arb_mat_t L, const arb_mat_t rhs,
                   slong prec)
{
    assert(x != NULL);
    assert(L != NULL);
    assert(rhs != NULL);
    assert(arb_mat_nrows(L) == arb_mat_ncols(L));
    assert(arb_mat_nrows(rhs) == arb_mat_nrows(L));
    assert(arb_mat_nrows(x) == arb_mat_nrows(rhs));
    assert(arb_mat_ncols(x) == arb_mat_ncols(rhs));

    /* factor-once / solve-many: reuse L for each RHS (predictor + corrector). */
    arb_mat_solve_cho_precomp(x, L, rhs, prec);
}
