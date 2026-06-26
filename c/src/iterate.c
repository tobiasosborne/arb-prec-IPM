/*
 * src/iterate.c -- the HSDE iterate state, the constraint operator A / A^T, and
 * the 3-row HSDE residuals.  POINT MODE (CLAUDE.md rule 1): all arithmetic at a
 * fixed working `prec`, consumed as midpoints; radii not trusted.
 *
 * GROUND TRUTH (CLAUDE.md rule 3):
 *   - docs/MATH_SPEC.md §3.1 (HSDE system, mu denom n+1), §3.2 (residuals).
 *   - HsdeNtSdpSolver.ts:371-407 (residual assembly + mu + objectives).
 *   - CLAUDE.md invariant 7 (internal min <C_int,X>; C_int = -C_file when
 *     maximize, +C_file when minimize -- bead n1x, see iterate.h banner),
 *     invariant 8 (HSDE tau,kappa).
 * REUSE (no duplication): arbsdp_problem_block_mat / arbsdp_problem_b (problem.h),
 * arbsdp_frob_inner (svec.h).
 */

#include <assert.h>

#include <flint/flint.h>

#include "arbsdp/iterate.h"
#include "arbsdp/svec.h"

/* side length of block b (|block_sizes[b]|; negative = diagonal/LP block). */
static slong
block_side(const arbsdp_problem *p, int b)
{
    int s = p->block_sizes[b];
    return (s < 0) ? -(slong) s : (slong) s;
}

void
arbsdp_iterate_init(arbsdp_iterate *it, const arbsdp_problem *p, slong prec)
{
    assert(it != NULL);
    assert(p != NULL);
    assert(p->m >= 0);
    assert(p->nblocks >= 0);

    it->prob = p;
    it->m = p->m;
    it->nblocks = p->nblocks;

    it->block_n = flint_malloc((size_t) p->nblocks * sizeof *it->block_n);
    it->total_cone_dim = 0;
    for (int b = 0; b < p->nblocks; b++) {
        it->block_n[b] = block_side(p, b);
        it->total_cone_dim += it->block_n[b];
    }

    it->X = flint_malloc((size_t) p->nblocks * sizeof *it->X);
    it->S = flint_malloc((size_t) p->nblocks * sizeof *it->S);
    it->C_int = flint_malloc((size_t) p->nblocks * sizeof *it->C_int);
    it->R_d = flint_malloc((size_t) p->nblocks * sizeof *it->R_d);

    for (int b = 0; b < p->nblocks; b++) {
        slong n = it->block_n[b];
        arb_mat_init(it->X[b], n, n);
        arb_mat_init(it->S[b], n, n);
        arb_mat_init(it->R_d[b], n, n);
        /* C_int^b: the internal cost block (matno 0 = C).  The internal loop
         * always MINIMIZES <C_int,X> (invariant 7); recover_value (solve.c)
         * applies the matching output sign (maximize ? -pObj/tau : +pObj/tau).
         * For max <C_file,X> we need C_int = -C_file (min of the negative = max);
         * for min <C_file,X> we need C_int = +C_file.  So negate ONLY when
         * maximize (MATH_SPEC §1.2; invariant 7; bead arb-prec-IPM-n1x -- the
         * unconditional negate solved every minimize problem as a maximize and
         * returned -(max<C,X>) instead of min<C,X>).  See iterate.h banner. */
        arb_mat_init(it->C_int[b], n, n);
        arbsdp_problem_block_mat(it->C_int[b], p, 0, b, prec);   /* C_file^b */
        if (p->maximize)
            arb_mat_neg(it->C_int[b], it->C_int[b]);   /* C_int = -C_file (max) */
    }

    it->y = _arb_vec_init(p->m);
    it->r_p = _arb_vec_init(p->m);

    arb_init(it->tau);
    arb_init(it->kappa);
    arb_init(it->r_g);
    arb_init(it->mu);
    arb_init(it->pObj);
    arb_init(it->dObj);
    arb_init(it->primal_inf);
    arb_init(it->dual_inf);
    arb_init(it->gap_inf);

    /* HSDE default homogenization (MATH_SPEC §3.9): tau = kappa = 1. */
    arb_one(it->tau);
    arb_one(it->kappa);
}

void
arbsdp_iterate_clear(arbsdp_iterate *it)
{
    if (it == NULL)
        return;
    for (int b = 0; b < it->nblocks; b++) {
        arb_mat_clear(it->X[b]);
        arb_mat_clear(it->S[b]);
        arb_mat_clear(it->R_d[b]);
        arb_mat_clear(it->C_int[b]);
    }
    flint_free(it->X);
    flint_free(it->S);
    flint_free(it->R_d);
    flint_free(it->C_int);
    flint_free(it->block_n);

    _arb_vec_clear(it->y, it->m);
    _arb_vec_clear(it->r_p, it->m);

    arb_clear(it->tau);
    arb_clear(it->kappa);
    arb_clear(it->r_g);
    arb_clear(it->mu);
    arb_clear(it->pObj);
    arb_clear(it->dObj);
    arb_clear(it->primal_inf);
    arb_clear(it->dual_inf);
    arb_clear(it->gap_inf);

    it->prob = NULL;
    it->X = NULL;
    it->S = NULL;
    it->R_d = NULL;
    it->C_int = NULL;
    it->block_n = NULL;
    it->y = NULL;
    it->r_p = NULL;
    it->m = 0;
    it->nblocks = 0;
    it->total_cone_dim = 0;
}

/* out_i = <A_i, X> = sum_b frobInner(A_i^b, X^b)  (HsdeNtSdpSolver.ts:375-379) */
void
arbsdp_apply_A(arb_ptr out, const arbsdp_problem *p, const arb_mat_t *X_blocks,
               slong prec)
{
    assert(out != NULL);
    assert(p != NULL);
    assert(X_blocks != NULL);

    arb_t acc, Ai_dot;
    arb_init(acc);
    arb_init(Ai_dot);

    for (int i = 0; i < p->m; i++) {
        arb_zero(acc);
        for (int b = 0; b < p->nblocks; b++) {
            slong n = block_side(p, b);
            assert(arb_mat_nrows(X_blocks[b]) == n);
            assert(arb_mat_ncols(X_blocks[b]) == n);
            arb_mat_t Ai;
            arb_mat_init(Ai, n, n);
            /* matno = i+1 (1..m are A_1..A_m; 0 is C). */
            arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);
            arbsdp_frob_inner(Ai_dot, Ai, X_blocks[b], prec);
            arb_add(acc, acc, Ai_dot, prec);
            arb_mat_clear(Ai);
        }
        arb_set(&out[i], acc);
    }

    arb_clear(acc);
    arb_clear(Ai_dot);
}

/* (A^T y)^b = sum_i y_i A_i^b  (the adjoint of apply_A). */
void
arbsdp_apply_At(arb_mat_t *out_blocks, const arbsdp_problem *p, arb_srcptr y,
                slong prec)
{
    assert(out_blocks != NULL);
    assert(p != NULL);
    assert(y != NULL);

    for (int b = 0; b < p->nblocks; b++) {
        slong n = block_side(p, b);
        assert(arb_mat_nrows(out_blocks[b]) == n);
        assert(arb_mat_ncols(out_blocks[b]) == n);
        arb_mat_zero(out_blocks[b]);

        arb_mat_t Ai;
        arb_mat_init(Ai, n, n);
        for (int i = 0; i < p->m; i++) {
            arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);
            /* out_blocks[b] += y_i * Ai */
            arb_mat_scalar_addmul_arb(out_blocks[b], Ai, &y[i], prec);
        }
        arb_mat_clear(Ai);
    }
}

/* max-abs entry of a vector (point-mode; midpoints).  No prec: arb_abs and the
 * comparison are exact (sign/copy operations). */
static void
vec_inf_norm(arb_t out, arb_srcptr v, slong len)
{
    arb_t a;
    arb_init(a);
    arb_zero(out);
    for (slong i = 0; i < len; i++) {
        arb_abs(a, &v[i]);
        if (arb_gt(a, out))
            arb_set(out, a);
    }
    arb_clear(a);
}

/* max-abs entry of a matrix (point-mode; midpoints).  No prec (as above). */
static void
mat_inf_norm(arb_t out, const arb_mat_t A)
{
    slong n = arb_mat_nrows(A), m = arb_mat_ncols(A);
    arb_t a;
    arb_init(a);
    arb_zero(out);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < m; j++) {
            arb_abs(a, arb_mat_entry(A, i, j));
            if (arb_gt(a, out))
                arb_set(out, a);
        }
    arb_clear(a);
}

void
arbsdp_iterate_residuals(arbsdp_iterate *it, slong prec)
{
    assert(it != NULL);
    const arbsdp_problem *p = it->prob;
    const int m = it->m, nb = it->nblocks;

    arb_t tmp, bi;
    arb_init(tmp);
    arb_init(bi);

    /* --- r_p_i = sum_b <A_i^b, X^b> - b_i*tau  (HsdeNtSdpSolver.ts:375-379) --- */
    arbsdp_apply_A(it->r_p, p, (const arb_mat_t *) it->X, prec);
    for (int i = 0; i < m; i++) {
        arbsdp_problem_b(bi, p, i, prec);
        arb_mul(tmp, bi, it->tau, prec);            /* b_i * tau */
        arb_sub(&it->r_p[i], &it->r_p[i], tmp, prec);
    }

    /* --- R_d^b = sum_i y_i A_i^b + S^b - C_int^b*tau (HsdeNtSdpSolver.ts:380-390)
     * Build sum_i y_i A_i^b = (A^T y)^b first, then add S^b, subtract C_int^b*tau. */
    arbsdp_apply_At(it->R_d, p, it->y, prec);
    for (int b = 0; b < nb; b++) {
        arb_mat_add(it->R_d[b], it->R_d[b], it->S[b], prec);     /* + S^b */
        /* - C_int^b * tau : addmul with scalar -tau */
        arb_neg(tmp, it->tau);
        arb_mat_scalar_addmul_arb(it->R_d[b], it->C_int[b], tmp, prec);
    }

    /* --- pObj = sum_b <C_int^b, X^b>;  dObj = b^T y  (HsdeNtSdpSolver.ts:393-396) */
    arb_zero(it->pObj);
    for (int b = 0; b < nb; b++) {
        arbsdp_frob_inner(tmp, it->C_int[b], it->X[b], prec);
        arb_add(it->pObj, it->pObj, tmp, prec);
    }
    arb_zero(it->dObj);
    for (int i = 0; i < m; i++) {
        arbsdp_problem_b(bi, p, i, prec);
        arb_mul(tmp, bi, &it->y[i], prec);
        arb_add(it->dObj, it->dObj, tmp, prec);
    }

    /* --- r_g = -pObj + dObj - kappa  (HsdeNtSdpSolver.ts:397) --- */
    arb_sub(it->r_g, it->dObj, it->pObj, prec);     /* dObj - pObj */
    arb_sub(it->r_g, it->r_g, it->kappa, prec);     /* - kappa */

    /* --- mu = (sum_b <X^b,S^b> + tau*kappa)/(total_cone_dim+1)
     *     (HsdeNtSdpSolver.ts:398-400, denom n+1 per HsdeIterate.ts:22) --- */
    arb_zero(it->mu);
    for (int b = 0; b < nb; b++) {
        arbsdp_frob_inner(tmp, it->X[b], it->S[b], prec);
        arb_add(it->mu, it->mu, tmp, prec);
    }
    arb_mul(tmp, it->tau, it->kappa, prec);
    arb_add(it->mu, it->mu, tmp, prec);
    arb_div_si(it->mu, it->mu, it->total_cone_dim + 1, prec);

    /* --- inf-norms (HsdeNtSdpSolver.ts:401-407) --- */
    vec_inf_norm(it->primal_inf, it->r_p, m);
    arb_zero(it->dual_inf);
    for (int b = 0; b < nb; b++) {
        mat_inf_norm(tmp, it->R_d[b]);
        if (arb_gt(tmp, it->dual_inf))
            arb_set(it->dual_inf, tmp);
    }
    arb_abs(it->gap_inf, it->r_g);

    arb_clear(tmp);
    arb_clear(bi);
}
