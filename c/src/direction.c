/*
 * src/direction.c -- the Mehrotra predictor-corrector search direction + HSDE
 * step length for the NT-scaled HSDE IPM (POINT MODE, Layer 0).
 *
 * GROUND TRUTH (CLAUDE.md rule 3) -- see include/arbsdp/direction.h banner for the
 * full citation list.  Line numbers below are HsdeNtSdpSolver.ts unless noted.
 *   - docs/MATH_SPEC.md §3.5 (RHS + recovery), §3.6 (sigma), §3.7 (step + safeguard).
 *   - :126-163 buildNtFactor (G, sv -- ported locally as nt_gfactor);
 *     :511-539 W*C*W / W*r_d*W caches; :574-611 data direction; :613-660 affine;
 *     :672-695 mu_aff + sigma; :697-806 corrector (E_x_comb, Rq, scalar dtau/dkappa,
 *     combine); :820-827 step + safeguard; :1168-1175 clip / clipStep.
 *   - PsdCone.ts:186-202 psdMaxStep; HsdeStepLength.ts:48-64 tauKappaMaxStep.
 *
 * REUSE (no duplication): arbsdp_schur_assemble / arbsdp_schur_solve (schur.h),
 * arbsdp_factor_with_reg (regularize.h), arbsdp_apply_At (iterate.h),
 * arbsdp_psd_invsqrt / arbsdp_eigh (linalg.h), arbsdp_frob_inner /
 * arbsdp_symmetrize (svec.h), arbsdp_problem_block_mat / arbsdp_problem_b
 * (problem.h).  The W*A*W Schur assembly is reused from schur.h; only the b14
 * caches (W*C*W, W*r_d*W) and the direction recovery live here.
 *
 * Arb memory discipline (CLAUDE.md rule 7): matching init/clear everywhere; only
 * scoped temporaries allocated.  Preconditions asserted (rule 5).
 */

#include <assert.h>

#include <flint/flint.h>
#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/direction.h"
#include "arbsdp/schur.h"
#include "arbsdp/regularize.h"
#include "arbsdp/iterate.h"
#include "arbsdp/ntscaling.h"
#include "arbsdp/linalg.h"
#include "arbsdp/svec.h"
#include "arbsdp/problem.h"

/* side length of block b (|block_sizes[b]|; negative = diagonal/LP block). */
static slong
block_side(const arbsdp_problem *p, int b)
{
    int s = p->block_sizes[b];
    return (s < 0) ? -(slong) s : (slong) s;
}

/*
 * apply_At_block -- out <- (A^T dy)^b = sum_i dy_i A_i^b  for the SINGLE block b
 * (dy is a column m-vector arb_mat).  This is the per-block specialization of
 * arbsdp_apply_At (iterate.h), used so the direction recovery does not have to
 * materialize the whole nb-block array on every solve.  `out` is a pre-inited
 * n x n arb_mat (n = block_side(p,b)); it is zeroed then accumulated.
 */
static void
apply_At_block(arb_mat_t out, const arbsdp_problem *p, int b,
               const arb_mat_t dy, slong prec)
{
    slong n = block_side(p, b);
    arb_mat_t Ai;
    assert(arb_mat_nrows(out) == n && arb_mat_ncols(out) == n);
    arb_mat_zero(out);
    arb_mat_init(Ai, n, n);
    for (int i = 0; i < p->m; i++) {
        arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);   /* A_{i+1}^b */
        arb_mat_scalar_addmul_arb(out, Ai, arb_mat_entry(dy, i, 0), prec);
    }
    arb_mat_clear(Ai);
}

/* ========================================================================= *
 * Scalar Mehrotra helpers (MATH_SPEC §3.6, §3.7).                            *
 * ========================================================================= */

/* sigma = clip((mu_aff / max(mu, 1e-300))^3, 1e-8, 0.9)  (:694-695). */
void
arbsdp_sigma(arb_t out, const arb_t mu_aff, const arb_t mu, slong prec)
{
    arb_t denom, ratio, lo, hi, floor_mu;

    arb_init(denom);
    arb_init(ratio);
    arb_init(lo);
    arb_init(hi);
    arb_init(floor_mu);

    /* denom = max(mu, 1e-300) */
    arb_set_d(floor_mu, 1e-300);
    arb_max(denom, mu, floor_mu, prec);

    arb_div(ratio, mu_aff, denom, prec);     /* mu_aff / denom */
    arb_pow_ui(ratio, ratio, 3, prec);       /* (...)^3 */

    /* clip to [1e-8, 0.9] */
    arb_set_d(lo, 1e-8);
    arb_set_d(hi, 0.9);
    arb_max(ratio, ratio, lo, prec);
    arb_min(out, ratio, hi, prec);

    arb_clear(denom);
    arb_clear(ratio);
    arb_clear(lo);
    arb_clear(hi);
    arb_clear(floor_mu);
}

/* clip_step(a) = clip(max(0.95*a, 2*a - 1), 0, 0.999999)  (:827 + :1168-1172). */
void
arbsdp_clip_step(arb_t out, const arb_t a, slong prec)
{
    arb_t t1, t2, lo, hi;

    arb_init(t1);
    arb_init(t2);
    arb_init(lo);
    arb_init(hi);

    /* t1 = 0.95*a ; t2 = 2*a - 1 */
    arb_set_d(t1, 0.95);
    arb_mul(t1, t1, a, prec);
    arb_mul_si(t2, a, 2, prec);
    arb_sub_si(t2, t2, 1, prec);
    arb_max(t1, t1, t2, prec);              /* max(0.95a, 2a-1) */

    /* clip to [0, 0.999999] */
    arb_zero(lo);
    arb_set_d(hi, 0.999999);
    arb_max(t1, t1, lo, prec);
    arb_min(out, t1, hi, prec);

    arb_clear(t1);
    arb_clear(t2);
    arb_clear(lo);
    arb_clear(hi);
}

/* ========================================================================= *
 * Step-to-boundary (PsdCone.ts:186-202; HsdeStepLength.ts:48-64).            *
 * ========================================================================= */

/*
 * psd_max_step: alpha keeping X + alpha*dX PSD.  PsdCone.ts uses L L^T = X then
 * lmin(L^{-1} dX L^{-T}); we use the congruent symmetric form lmin(X^{-1/2} dX
 * X^{-1/2}) which has the SAME spectrum (B^{-1} = X^{-1/2}, and X^{-1/2} dX X^{-1/2}
 * = (X^{-1/2}) dX (X^{-1/2})^T is similar to L^{-1} dX L^{-T} by congruence; both
 * equal the generalized eigenvalues of (dX, X)).  alpha = lmin>=0 ? Inf : 1/(-lmin).
 */
int
arbsdp_psd_max_step(arb_t alpha_out, const arb_mat_t X, const arb_mat_t dX,
                    slong prec)
{
    slong n = arb_mat_nrows(X);
    arb_mat_t Xinvh, tmp, M, Q;
    arb_ptr lam;
    int bounded;

    assert(arb_mat_nrows(X) == arb_mat_ncols(X));
    assert(arb_mat_nrows(dX) == n && arb_mat_ncols(dX) == n);

    arb_mat_init(Xinvh, n, n);
    arb_mat_init(tmp, n, n);
    arb_mat_init(M, n, n);
    arb_mat_init(Q, n, n);
    lam = _arb_vec_init(n);

    /* M = X^{-1/2} dX X^{-1/2}  (symmetric; congruent to L^{-1} dX L^{-T}). */
    arbsdp_psd_invsqrt(Xinvh, X, prec);
    arb_mat_mul(tmp, Xinvh, dX, prec);
    arb_mat_mul(M, tmp, Xinvh, prec);
    arbsdp_symmetrize(M, prec);

    arbsdp_eigh(lam, Q, M, prec);            /* ascending; lam[0] = lmin */

    /* lmin >= 0 -> UNBOUNDED (return 0); else alpha = 1/(-lmin). */
    if (arf_sgn(arb_midref(lam + 0)) >= 0) {
        bounded = 0;
    } else {
        arb_neg(alpha_out, lam + 0);         /* -lmin (> 0) */
        arb_inv(alpha_out, alpha_out, prec); /* 1 / (-lmin) */
        bounded = 1;
    }

    _arb_vec_clear(lam, n);
    arb_mat_clear(Xinvh);
    arb_mat_clear(tmp);
    arb_mat_clear(M);
    arb_mat_clear(Q);
    return bounded;
}

/* tauKappaMaxStep: min over the two scalar ratios; UNBOUNDED if neither binds. */
int
arbsdp_tau_kappa_max_step(arb_t alpha_out, const arb_t tau, const arb_t dtau,
                          const arb_t kappa, const arb_t dkappa, slong prec)
{
    arb_t a;
    int bounded = 0;
    arb_init(a);

    if (arf_sgn(arb_midref(dtau)) < 0) {     /* dtau < 0 -> a = -tau/dtau */
        arb_div(a, tau, dtau, prec);
        arb_neg(a, a);
        arb_set(alpha_out, a);
        bounded = 1;
    }
    if (arf_sgn(arb_midref(dkappa)) < 0) {   /* dkappa < 0 -> a = -kappa/dkappa */
        arb_div(a, kappa, dkappa, prec);
        arb_neg(a, a);
        if (!bounded || arb_lt(a, alpha_out))
            arb_set(alpha_out, a);
        bounded = 1;
    }

    arb_clear(a);
    return bounded;
}

/* ========================================================================= *
 * Per-block NT G-factor (buildNtFactor, :126-163) -- needed by the corrector  *
 * for S^{-1} (= G^T diag(1/sv) G) and Rq (hdZ = G dS G^T).  The Schur uses the *
 * symmetric W = G^T G (schur.h audition); here we also need G and sv, which W   *
 * alone does not provide, so we build the G-factor locally per block.          *
 * ========================================================================= */

typedef struct {
    slong     n;
    arb_mat_t G;     /* n x n; G^T diag(sv) G = X, G^T diag(1/sv) G = S^{-1}    */
    arb_ptr   sv;    /* length n; singular values (sqrt of eig of L^T X L)      */
} nt_gfactor;

/*
 * nt_gfactor_build -- L L^T = S (chol, lower); MLXL = sym(L^T X L);
 * [V, sv2] = eigh(MLXL); sv = sqrt(max(sv2, 1e-300));
 * G = diag(sqrt(sv)) * V^T * L^{-1}      (:140-156).
 * L^{-1} is obtained by solving L * invL = I (triangular).
 */
static void
nt_gfactor_build(nt_gfactor *f, const arb_mat_t X, const arb_mat_t S, slong prec)
{
    slong n = arb_mat_nrows(S);
    arb_mat_t Ssym, L, invL, Id, Xsym, Lt, LtX, MLXL, V, Vt, VtinvL;
    arb_ptr sv2;

    f->n = n;
    arb_mat_init(f->G, n, n);
    f->sv = _arb_vec_init(n);

    arb_mat_init(Ssym, n, n);
    arb_mat_init(L, n, n);
    arb_mat_init(invL, n, n);
    arb_mat_init(Id, n, n);
    arb_mat_init(Xsym, n, n);
    arb_mat_init(Lt, n, n);
    arb_mat_init(LtX, n, n);
    arb_mat_init(MLXL, n, n);
    arb_mat_init(V, n, n);
    arb_mat_init(Vt, n, n);
    arb_mat_init(VtinvL, n, n);
    sv2 = _arb_vec_init(n);

    /* L = chol(S) (lower); arb_mat_cho returns L with L L^T = S. */
    arb_mat_set(Ssym, S);
    arbsdp_symmetrize(Ssym, prec);
    /* small jitter (1e-14) to keep cho robust on near-singular S (:129). */
    {
        arb_t jit;
        arb_init(jit);
        arb_set_d(jit, 1e-14);
        for (slong i = 0; i < n; i++)
            arb_add(arb_mat_entry(Ssym, i, i), arb_mat_entry(Ssym, i, i), jit, prec);
        arb_clear(jit);
    }
    arb_mat_cho(L, Ssym, prec);              /* L lower-triangular, L L^T = Ssym */

    /* invL = L^{-1} via L * invL = I (triangular solve, unit-diag = 0). */
    arb_mat_one(Id);
    arb_mat_solve_tril(invL, L, Id, 0, prec);

    /* MLXL = sym(L^T X L)  (= L^T X L; :141-144).  No output/input aliasing. */
    arb_mat_set(Xsym, X);
    arbsdp_symmetrize(Xsym, prec);
    arb_mat_transpose(Lt, L);                /* L^T */
    arb_mat_mul(LtX, Lt, Xsym, prec);        /* L^T X */
    arb_mat_mul(MLXL, LtX, L, prec);         /* L^T X L */
    arbsdp_symmetrize(MLXL, prec);

    /* [V, sv2] = eigh(MLXL);  sv = sqrt(max(sv2, 1e-300))  (:146-148). */
    arbsdp_eigh(sv2, V, MLXL, prec);
    {
        arb_t fl;
        arb_init(fl);
        arb_set_d(fl, 1e-300);
        for (slong i = 0; i < n; i++) {
            if (arf_cmp(arb_midref(sv2 + i), arb_midref(fl)) < 0)
                arb_set(f->sv + i, fl);
            else
                arb_set(f->sv + i, sv2 + i);
            arb_sqrt(f->sv + i, f->sv + i, prec);
        }
        arb_clear(fl);
    }

    /* VtinvL = V^T * invL;  G[i,:] = sqrt(sv_i) * VtinvL[i,:]  (:150-156). */
    arb_mat_transpose(Vt, V);                /* V^T (no self-aliasing) */
    arb_mat_mul(VtinvL, Vt, invL, prec);
    {
        arb_t s;
        arb_init(s);
        for (slong i = 0; i < n; i++) {
            arb_sqrt(s, f->sv + i, prec);    /* sqrt(sv_i) */
            for (slong j = 0; j < n; j++)
                arb_mul(arb_mat_entry(f->G, i, j), arb_mat_entry(VtinvL, i, j), s, prec);
        }
        arb_clear(s);
    }

    _arb_vec_clear(sv2, n);
    arb_mat_clear(Ssym);
    arb_mat_clear(L);
    arb_mat_clear(invL);
    arb_mat_clear(Id);
    arb_mat_clear(Xsym);
    arb_mat_clear(Lt);
    arb_mat_clear(LtX);
    arb_mat_clear(MLXL);
    arb_mat_clear(V);
    arb_mat_clear(Vt);
    arb_mat_clear(VtinvL);
}

static void
nt_gfactor_clear(nt_gfactor *f)
{
    arb_mat_clear(f->G);
    _arb_vec_clear(f->sv, f->n);
    f->n = 0;
}

/* Sinv = G^T diag(1/sv) G  (gtDiagInvSvG, :165-175).  out must be n x n init. */
static void
gfactor_sinv(arb_mat_t out, const nt_gfactor *f, slong prec)
{
    slong n = f->n;
    arb_mat_t tmp, Gt;
    arb_t di;
    arb_mat_init(tmp, n, n);
    arb_mat_init(Gt, n, n);
    arb_init(di);

    /* tmp[i,:] = (1/sv_i) * G[i,:] */
    for (slong i = 0; i < n; i++) {
        arb_inv(di, f->sv + i, prec);
        for (slong j = 0; j < n; j++)
            arb_mul(arb_mat_entry(tmp, i, j), arb_mat_entry(f->G, i, j), di, prec);
    }
    arb_mat_transpose(Gt, f->G);             /* G^T */
    arb_mat_mul(out, Gt, tmp, prec);         /* G^T (diag(1/sv) G) */
    arbsdp_symmetrize(out, prec);

    arb_clear(di);
    arb_mat_clear(tmp);
    arb_mat_clear(Gt);
}

/*
 * gfactor_rq -- Rq = G^T RqSv G with RqSv[i,j] = 2*sym(hdX*hdZ)[i,j]/(sv_i+sv_j),
 * hdZ = G dS_aff G^T, hdX = diag(-sv) - hdZ  (:724-755).  out n x n init.
 */
static void
gfactor_rq(arb_mat_t out, const nt_gfactor *f, const arb_mat_t dS_aff, slong prec)
{
    slong n = f->n;
    arb_mat_t Gt, tmp, hdZ, hdX, cross, symCross, RqSv;
    arb_t v, denom, floor_d;

    arb_mat_init(Gt, n, n);
    arb_mat_init(tmp, n, n);
    arb_mat_init(hdZ, n, n);
    arb_mat_init(hdX, n, n);
    arb_mat_init(cross, n, n);
    arb_mat_init(symCross, n, n);
    arb_mat_init(RqSv, n, n);
    arb_init(v);
    arb_init(denom);
    arb_init(floor_d);

    arb_mat_transpose(Gt, f->G);

    /* hdZ = G * dS_aff * G^T. */
    arb_mat_mul(tmp, f->G, dS_aff, prec);
    arb_mat_mul(hdZ, tmp, Gt, prec);

    /* hdX = diag(-sv) - hdZ  (algebraic identity for the affine direction). */
    arb_mat_neg(hdX, hdZ);
    for (slong i = 0; i < n; i++)
        arb_sub(arb_mat_entry(hdX, i, i), arb_mat_entry(hdX, i, i), f->sv + i, prec);

    /* cross = hdX * hdZ ; symCross = (cross + cross^T)/2. */
    arb_mat_mul(cross, hdX, hdZ, prec);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++) {
            arb_add(v, arb_mat_entry(cross, i, j), arb_mat_entry(cross, j, i), prec);
            arb_mul_2exp_si(v, v, -1);       /* /2 */
            arb_set(arb_mat_entry(symCross, i, j), v);
        }

    /* RqSv[i,j] = 2*symCross[i,j]/(sv_i+sv_j)  (0 if denom ~ 0). */
    arb_set_d(floor_d, 1e-300);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++) {
            arb_add(denom, f->sv + i, f->sv + j, prec);
            if (arf_cmp(arb_midref(denom), arb_midref(floor_d)) > 0) {
                arb_mul_2exp_si(v, arb_mat_entry(symCross, i, j), 1);  /* 2*symCross */
                arb_div(v, v, denom, prec);
                arb_set(arb_mat_entry(RqSv, i, j), v);
            } else {
                arb_zero(arb_mat_entry(RqSv, i, j));
            }
        }

    /* Rq = G^T * RqSv * G. */
    arb_mat_mul(tmp, Gt, RqSv, prec);
    arb_mat_mul(out, tmp, f->G, prec);
    arbsdp_symmetrize(out, prec);

    arb_clear(v);
    arb_clear(denom);
    arb_clear(floor_d);
    arb_mat_clear(Gt);
    arb_mat_clear(tmp);
    arb_mat_clear(hdZ);
    arb_mat_clear(hdX);
    arb_mat_clear(cross);
    arb_mat_clear(symCross);
    arb_mat_clear(RqSv);
}

/* ========================================================================= *
 * Direction lifecycle.                                                       *
 * ========================================================================= */

void
arbsdp_direction_init(arbsdp_direction *d, const arbsdp_problem *p)
{
    assert(d != NULL);
    assert(p != NULL);

    d->m = p->m;
    d->nblocks = p->nblocks;
    d->block_n = flint_malloc((size_t) p->nblocks * sizeof *d->block_n);
    d->dX = flint_malloc((size_t) p->nblocks * sizeof *d->dX);
    d->dS = flint_malloc((size_t) p->nblocks * sizeof *d->dS);
    for (int b = 0; b < p->nblocks; b++) {
        slong n = block_side(p, b);
        d->block_n[b] = n;
        arb_mat_init(d->dX[b], n, n);
        arb_mat_init(d->dS[b], n, n);
    }
    d->dy = _arb_vec_init(p->m);
    arb_init(d->dtau);
    arb_init(d->dkappa);
    arb_init(d->alpha);
    arb_init(d->sigma);
    arb_init(d->mu_aff);
    d->factor_attempts = 0;
    d->factor_calls = 0;
}

void
arbsdp_direction_clear(arbsdp_direction *d)
{
    if (d == NULL)
        return;
    for (int b = 0; b < d->nblocks; b++) {
        arb_mat_clear(d->dX[b]);
        arb_mat_clear(d->dS[b]);
    }
    flint_free(d->dX);
    flint_free(d->dS);
    flint_free(d->block_n);
    _arb_vec_clear(d->dy, d->m);
    arb_clear(d->dtau);
    arb_clear(d->dkappa);
    arb_clear(d->alpha);
    arb_clear(d->sigma);
    arb_clear(d->mu_aff);
    d->dX = NULL;
    d->dS = NULL;
    d->block_n = NULL;
    d->dy = NULL;
    d->m = 0;
    d->nblocks = 0;
}

/* ========================================================================= *
 * The full Mehrotra predictor-corrector step.                                *
 * ========================================================================= */

int
arbsdp_direction_compute(arbsdp_direction *d, const arbsdp_iterate *it,
                         const arb_mat_t *W_blocks, arbsdp_regstate *rs, slong prec)
{
    const arbsdp_problem *p = it->prob;
    const int m = it->m, nb = it->nblocks;
    int ok = 1;

    assert(d->m == m && d->nblocks == nb);

    /* --- per-block caches and NT G-factors ------------------------------- */
    arb_mat_t *WCW  = flint_malloc((size_t) nb * sizeof *WCW);   /* W C_int W   */
    arb_mat_t *WrdW = flint_malloc((size_t) nb * sizeof *WrdW);  /* W r_d W     */
    nt_gfactor *F   = flint_malloc((size_t) nb * sizeof *F);

    /* affine + per-block direction storage (kept across predictor/corrector) */
    arb_mat_t *dX1   = flint_malloc((size_t) nb * sizeof *dX1);
    arb_mat_t *dS1   = flint_malloc((size_t) nb * sizeof *dS1);
    arb_mat_t *dXaff = flint_malloc((size_t) nb * sizeof *dXaff);
    arb_mat_t *dSaff = flint_malloc((size_t) nb * sizeof *dSaff);
    arb_mat_t *dX2   = flint_malloc((size_t) nb * sizeof *dX2);
    arb_mat_t *dS2   = flint_malloc((size_t) nb * sizeof *dS2);
    arb_mat_t *ExC   = flint_malloc((size_t) nb * sizeof *ExC);

    arb_mat_t M, L;
    arb_mat_t rhs, dy1, dy2;
    arb_t v, s, cdx1, bdy1, dtau_denom, kappa_over_tau, cdx2, bdy2;
    arb_t dtau_aff, dkap_aff, eta, sigma_mu, Etau, dtau_num;
    arb_t alpha_raw, big;

    for (int b = 0; b < nb; b++) {
        slong n = block_side(p, b);
        arb_mat_init(WCW[b], n, n);
        arb_mat_init(WrdW[b], n, n);
        arb_mat_init(dX1[b], n, n);
        arb_mat_init(dS1[b], n, n);
        arb_mat_init(dXaff[b], n, n);
        arb_mat_init(dSaff[b], n, n);
        arb_mat_init(dX2[b], n, n);
        arb_mat_init(dS2[b], n, n);
        arb_mat_init(ExC[b], n, n);
        nt_gfactor_build(&F[b], it->X[b], it->S[b], prec);
    }

    arb_mat_init(M, m, m);
    arb_mat_init(L, m, m);
    arb_mat_init(rhs, m, 1);
    arb_mat_init(dy1, m, 1);
    arb_mat_init(dy2, m, 1);
    arb_init(v); arb_init(s);
    arb_init(cdx1); arb_init(bdy1); arb_init(dtau_denom); arb_init(kappa_over_tau);
    arb_init(cdx2); arb_init(bdy2);
    arb_init(dtau_aff); arb_init(dkap_aff); arb_init(eta); arb_init(sigma_mu);
    arb_init(Etau); arb_init(dtau_num);
    arb_init(alpha_raw); arb_init(big);

    /* ---------------------------------------------------------------------
     * Caches W*C_int*W and W*r_d*W per block (:524-538).  (W*A*W is rebuilt
     * inside arbsdp_schur_assemble; we don't duplicate it.)
     * ------------------------------------------------------------------- */
    for (int b = 0; b < nb; b++) {
        slong n = block_side(p, b);
        const arb_mat_t *W = &W_blocks[b];
        arb_mat_t tmpn;
        arb_mat_init(tmpn, n, n);
        /* W C_int W */
        arb_mat_mul(tmpn, *W, it->C_int[b], prec);
        arb_mat_mul(WCW[b], tmpn, *W, prec);
        arbsdp_symmetrize(WCW[b], prec);
        /* W r_d W */
        arb_mat_mul(tmpn, *W, it->R_d[b], prec);
        arb_mat_mul(WrdW[b], tmpn, *W, prec);
        arbsdp_symmetrize(WrdW[b], prec);
        arb_mat_clear(tmpn);
    }

    /* ---------------------------------------------------------------------
     * Schur M = sum_b <A_i, W A_k W>  (schur.h), factor ONCE with 3-way reg.
     * factor-once / solve-many: the SAME L serves data + affine + combined.
     * ------------------------------------------------------------------- */
    arbsdp_schur_assemble(M, p, W_blocks, prec);
    {
        int attempts = 0;
        d->factor_calls = 1;                 /* the ONE factorization invocation */
        if (!arbsdp_factor_with_reg(L, M, rs, p, prec, &attempts)) {
            d->factor_attempts = attempts;
            ok = 0;
            goto cleanup;
        }
        d->factor_attempts = attempts;
    }

    /* ---------------------------------------------------------------------
     * Data direction: M*dy1 = b + <A_i, W C W>;  dS1 = C - sum_k dy1_k A_k;
     * dX1 = -W dS1 W  (:574-604).
     * ------------------------------------------------------------------- */
    for (int i = 0; i < m; i++) {
        arbsdp_problem_b(s, p, i, prec);     /* b_i */
        for (int b = 0; b < nb; b++) {
            slong n = block_side(p, b);
            arb_mat_t Ai;
            arb_mat_init(Ai, n, n);
            arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);
            arbsdp_frob_inner(v, Ai, WCW[b], prec);
            arb_add(s, s, v, prec);
            arb_mat_clear(Ai);
        }
        arb_set(arb_mat_entry(rhs, i, 0), s);
    }
    arbsdp_schur_solve(dy1, L, rhs, prec);

    for (int b = 0; b < nb; b++) {
        slong n = block_side(p, b);
        const arb_mat_t *W = &W_blocks[b];
        arb_mat_t AtDy, tmpn;
        arb_mat_init(AtDy, n, n);
        arb_mat_init(tmpn, n, n);
        /* dS1 = C_int - sum_k dy1_k A_k = C_int - (A^T dy1)^b. */
        apply_At_block(AtDy, p, b, dy1, prec);
        arb_mat_sub(dS1[b], it->C_int[b], AtDy, prec);
        arbsdp_symmetrize(dS1[b], prec);
        /* dX1 = -W dS1 W. */
        arb_mat_mul(tmpn, *W, dS1[b], prec);
        arb_mat_mul(dX1[b], tmpn, *W, prec);
        arb_mat_neg(dX1[b], dX1[b]);
        arbsdp_symmetrize(dX1[b], prec);
        arb_mat_clear(AtDy);
        arb_mat_clear(tmpn);
    }

    /* dtau_denom = kappa/tau - sum_b <C, dX1> + b^T dy1  (:606-611).
     * NOTE C here is C_file = -C_int (the TS blocks[b].C is the internal C; in
     * libarbsdp pObj = <C_int, X> so we use C_int consistently -- the residuals
     * and the gap r_g all use C_int.  The TS frobInner(blocks[b].C, .) is over
     * its internal C, == our C_int.  We match: use C_int. */
    arb_zero(cdx1);
    for (int b = 0; b < nb; b++) {
        arbsdp_frob_inner(v, it->C_int[b], dX1[b], prec);
        arb_add(cdx1, cdx1, v, prec);
    }
    arb_zero(bdy1);
    for (int i = 0; i < m; i++) {
        arbsdp_problem_b(v, p, i, prec);
        arb_mul(v, v, arb_mat_entry(dy1, i, 0), prec);
        arb_add(bdy1, bdy1, v, prec);
    }
    arb_div(kappa_over_tau, it->kappa, it->tau, prec);
    arb_sub(dtau_denom, kappa_over_tau, cdx1, prec);
    arb_add(dtau_denom, dtau_denom, bdy1, prec);

    /* ---------------------------------------------------------------------
     * Affine predictor (sigma=0, eta=1, E_x_aff = -X):
     *   M*dy2 = <A_i, X> - r_p_i - <A_i, W r_d W>   (:616-623)
     *   dS2 = -sum_k dy2_k A_k - r_d;  dX2 = -X - W dS2 W   (:629-649)
     * ------------------------------------------------------------------- */
    for (int i = 0; i < m; i++) {
        arb_neg(s, &it->r_p[i]);             /* -r_p_i */
        for (int b = 0; b < nb; b++) {
            slong n = block_side(p, b);
            arb_mat_t Ai;
            arb_mat_init(Ai, n, n);
            arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);
            arbsdp_frob_inner(v, Ai, it->X[b], prec);     /* + <A_i, X> */
            arb_add(s, s, v, prec);
            arbsdp_frob_inner(v, Ai, WrdW[b], prec);      /* - <A_i, W r_d W> */
            arb_sub(s, s, v, prec);
            arb_mat_clear(Ai);
        }
        arb_set(arb_mat_entry(rhs, i, 0), s);
    }
    arbsdp_schur_solve(dy2, L, rhs, prec);

    for (int b = 0; b < nb; b++) {
        slong n = block_side(p, b);
        const arb_mat_t *W = &W_blocks[b];
        arb_mat_t AtDy, tmpn, tmpn2;
        arb_mat_init(AtDy, n, n);
        arb_mat_init(tmpn, n, n);
        arb_mat_init(tmpn2, n, n);
        apply_At_block(AtDy, p, b, dy2, prec);
        /* dS2 = -(A^T dy2) - r_d. */
        arb_mat_neg(dS2[b], AtDy);
        arb_mat_sub(dS2[b], dS2[b], it->R_d[b], prec);
        arbsdp_symmetrize(dS2[b], prec);
        /* dX2 = -X - W dS2 W. */
        arb_mat_mul(tmpn, *W, dS2[b], prec);
        arb_mat_mul(tmpn2, tmpn, *W, prec);  /* W dS2 W */
        arb_mat_neg(dX2[b], it->X[b]);
        arb_mat_sub(dX2[b], dX2[b], tmpn2, prec);
        arbsdp_symmetrize(dX2[b], prec);
        arb_mat_clear(AtDy);
        arb_mat_clear(tmpn);
        arb_mat_clear(tmpn2);
    }

    /* scalar dtau_aff, dkap_aff (:653-660):
     *   dtau_aff = (-r_g + <C,dX2> - b^T dy2 - kappa) / dtau_denom
     *   dkap_aff = -kappa - (kappa/tau)*dtau_aff. */
    arb_zero(cdx2);
    for (int b = 0; b < nb; b++) {
        arbsdp_frob_inner(v, it->C_int[b], dX2[b], prec);
        arb_add(cdx2, cdx2, v, prec);
    }
    arb_zero(bdy2);
    for (int i = 0; i < m; i++) {
        arbsdp_problem_b(v, p, i, prec);
        arb_mul(v, v, arb_mat_entry(dy2, i, 0), prec);
        arb_add(bdy2, bdy2, v, prec);
    }
    arb_neg(dtau_aff, it->r_g);              /* -r_g */
    arb_add(dtau_aff, dtau_aff, cdx2, prec); /* + <C,dX2> */
    arb_sub(dtau_aff, dtau_aff, bdy2, prec); /* - b^T dy2 */
    arb_sub(dtau_aff, dtau_aff, it->kappa, prec);  /* - kappa (E_tau/tau = -kappa) */
    arb_div(dtau_aff, dtau_aff, dtau_denom, prec);
    /* dkap_aff = -kappa - (kappa/tau)*dtau_aff. */
    arb_mul(dkap_aff, kappa_over_tau, dtau_aff, prec);
    arb_add(dkap_aff, dkap_aff, it->kappa, prec);
    arb_neg(dkap_aff, dkap_aff);

    /* dXaff = dX2 + dtau_aff*dX1 ; dSaff = dS2 + dtau_aff*dS1  (:663-670). */
    for (int b = 0; b < nb; b++) {
        arb_mat_set(dXaff[b], dX2[b]);
        arb_mat_scalar_addmul_arb(dXaff[b], dX1[b], dtau_aff, prec);
        arb_mat_set(dSaff[b], dS2[b]);
        arb_mat_scalar_addmul_arb(dSaff[b], dS1[b], dtau_aff, prec);
    }

    /* ---------------------------------------------------------------------
     * Affine step length (capped at 1) + mu_aff + sigma  (:672-695).
     * alpha_aff = min(aConeP, aConeD, aKap, 1).
     * ------------------------------------------------------------------- */
    {
        arb_t alpha_aff;
        arb_init(alpha_aff);
        arb_set_d(big, 1e300);
        arb_set(alpha_aff, big);             /* start at +Inf surrogate */
        for (int b = 0; b < nb; b++) {
            if (arbsdp_psd_max_step(v, it->X[b], dXaff[b], prec) && arb_lt(v, alpha_aff))
                arb_set(alpha_aff, v);
            if (arbsdp_psd_max_step(v, it->S[b], dSaff[b], prec) && arb_lt(v, alpha_aff))
                arb_set(alpha_aff, v);
        }
        if (arbsdp_tau_kappa_max_step(v, it->tau, dtau_aff, it->kappa, dkap_aff, prec)
            && arb_lt(v, alpha_aff))
            arb_set(alpha_aff, v);
        /* cap at 1.0 (:678). */
        { arb_t one; arb_init(one); arb_one(one); arb_min(alpha_aff, alpha_aff, one, prec); arb_clear(one); }

        /* mu_aff = (sum_b <X+a*dX, S+a*dS> + (tau+a*dtau)(kappa+a*dkap))/(D+1). */
        arb_zero(d->mu_aff);
        arb_mat_t Xa, Sa;
        for (int b = 0; b < nb; b++) {
            slong n = block_side(p, b);
            arb_mat_init(Xa, n, n);
            arb_mat_init(Sa, n, n);
            arb_mat_set(Xa, it->X[b]);
            arb_mat_scalar_addmul_arb(Xa, dXaff[b], alpha_aff, prec);  /* X + a dX */
            arb_mat_set(Sa, it->S[b]);
            arb_mat_scalar_addmul_arb(Sa, dSaff[b], alpha_aff, prec);  /* S + a dS */
            arbsdp_frob_inner(v, Xa, Sa, prec);
            arb_add(d->mu_aff, d->mu_aff, v, prec);
            arb_mat_clear(Xa);
            arb_mat_clear(Sa);
        }
        {
            arb_t ta, ka;
            arb_init(ta); arb_init(ka);
            arb_mul(ta, alpha_aff, dtau_aff, prec); arb_add(ta, ta, it->tau, prec);
            arb_mul(ka, alpha_aff, dkap_aff, prec); arb_add(ka, ka, it->kappa, prec);
            arb_mul(ta, ta, ka, prec);
            arb_add(d->mu_aff, d->mu_aff, ta, prec);
            arb_clear(ta); arb_clear(ka);
        }
        arb_div_si(d->mu_aff, d->mu_aff, it->total_cone_dim + 1, prec);
        arb_clear(alpha_aff);
    }

    /* sigma = clip((mu_aff/mu)^3, 1e-8, 0.9). */
    arbsdp_sigma(d->sigma, d->mu_aff, it->mu, prec);
    arb_set_si(eta, 1);
    arb_sub(eta, eta, d->sigma, prec);       /* eta = 1 - sigma */
    arb_mul(sigma_mu, d->sigma, it->mu, prec);

    /* ---------------------------------------------------------------------
     * Combined corrector  (:697-806).
     *   E_x_comb = -X + sigma*mu*S^{-1} - Rq
     *   M*dy2 = -<A_i,E_x_comb> - eta*r_p_i - eta*<A_i, W r_d W>
     *   dS2 = -(A^T dy2) - eta*r_d ; dX2 = E_x_comb - W dS2 W
     *   E_tau_comb = -tau*kappa + sigma*mu - dtau_aff*dkap_aff
     *   dtau = (-eta*r_g + <C,dX2> - b^T dy2 + E_tau_comb/tau)/dtau_denom
     *   dkappa = (E_tau_comb - kappa*dtau)/tau
     * ------------------------------------------------------------------- */
    for (int b = 0; b < nb; b++) {
        slong n = block_side(p, b);
        arb_mat_t Sinv, Rq;
        arb_mat_init(Sinv, n, n);
        arb_mat_init(Rq, n, n);
        /* E = -X + sigma*mu*S^{-1} - Rq. */
        arb_mat_neg(ExC[b], it->X[b]);
        gfactor_sinv(Sinv, &F[b], prec);
        arb_mat_scalar_addmul_arb(ExC[b], Sinv, sigma_mu, prec);   /* + sigma*mu*Sinv */
        gfactor_rq(Rq, &F[b], dSaff[b], prec);
        arb_mat_sub(ExC[b], ExC[b], Rq, prec);                     /* - Rq */
        arbsdp_symmetrize(ExC[b], prec);
        arb_mat_clear(Sinv);
        arb_mat_clear(Rq);
    }

    for (int i = 0; i < m; i++) {
        arb_mul(s, eta, &it->r_p[i], prec);  /* eta*r_p_i */
        arb_neg(s, s);                        /* -eta*r_p_i */
        for (int b = 0; b < nb; b++) {
            slong n = block_side(p, b);
            arb_mat_t Ai;
            arb_mat_init(Ai, n, n);
            arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);
            arbsdp_frob_inner(v, Ai, ExC[b], prec);     /* - <A_i, E_x_comb> */
            arb_sub(s, s, v, prec);
            arbsdp_frob_inner(v, Ai, WrdW[b], prec);    /* - eta*<A_i, W r_d W> */
            arb_mul(v, v, eta, prec);
            arb_sub(s, s, v, prec);
            arb_mat_clear(Ai);
        }
        arb_set(arb_mat_entry(rhs, i, 0), s);
    }
    arbsdp_schur_solve(dy2, L, rhs, prec);   /* SAME L -- factor reuse */

    for (int b = 0; b < nb; b++) {
        slong n = block_side(p, b);
        const arb_mat_t *W = &W_blocks[b];
        arb_mat_t AtDy, tmpn, tmpn2;
        arb_t neta;
        arb_mat_init(AtDy, n, n);
        arb_mat_init(tmpn, n, n);
        arb_mat_init(tmpn2, n, n);
        arb_init(neta);
        apply_At_block(AtDy, p, b, dy2, prec);
        /* dS2 = -(A^T dy2) - eta*r_d. */
        arb_mat_neg(dS2[b], AtDy);
        arb_neg(neta, eta);
        arb_mat_scalar_addmul_arb(dS2[b], it->R_d[b], neta, prec);
        arbsdp_symmetrize(dS2[b], prec);
        /* dX2 = E_x_comb - W dS2 W. */
        arb_mat_mul(tmpn, *W, dS2[b], prec);
        arb_mat_mul(tmpn2, tmpn, *W, prec);
        arb_mat_sub(dX2[b], ExC[b], tmpn2, prec);
        arbsdp_symmetrize(dX2[b], prec);
        arb_clear(neta);
        arb_mat_clear(AtDy);
        arb_mat_clear(tmpn);
        arb_mat_clear(tmpn2);
    }

    /* scalar dtau, dkappa (combined). */
    arb_zero(cdx2);
    for (int b = 0; b < nb; b++) {
        arbsdp_frob_inner(v, it->C_int[b], dX2[b], prec);
        arb_add(cdx2, cdx2, v, prec);
    }
    arb_zero(bdy2);
    for (int i = 0; i < m; i++) {
        arbsdp_problem_b(v, p, i, prec);
        arb_mul(v, v, arb_mat_entry(dy2, i, 0), prec);
        arb_add(bdy2, bdy2, v, prec);
    }
    /* E_tau_comb = -tau*kappa + sigma*mu - dtau_aff*dkap_aff. */
    arb_mul(Etau, it->tau, it->kappa, prec);
    arb_neg(Etau, Etau);
    arb_add(Etau, Etau, sigma_mu, prec);
    arb_mul(v, dtau_aff, dkap_aff, prec);
    arb_sub(Etau, Etau, v, prec);
    /* dtau_num = -eta*r_g + <C,dX2> - b^T dy2 + E_tau_comb/tau. */
    arb_mul(dtau_num, eta, it->r_g, prec);
    arb_neg(dtau_num, dtau_num);
    arb_add(dtau_num, dtau_num, cdx2, prec);
    arb_sub(dtau_num, dtau_num, bdy2, prec);
    arb_div(v, Etau, it->tau, prec);
    arb_add(dtau_num, dtau_num, v, prec);
    arb_div(d->dtau, dtau_num, dtau_denom, prec);
    /* dkappa = (E_tau_comb - kappa*dtau)/tau. */
    arb_mul(v, it->kappa, d->dtau, prec);
    arb_sub(d->dkappa, Etau, v, prec);
    arb_div(d->dkappa, d->dkappa, it->tau, prec);

    /* combine: dy = dy2 + dtau*dy1 ; dS = dS2 + dtau*dS1 ; dX = dX2 + dtau*dX1. */
    for (int i = 0; i < m; i++) {
        arb_set(&d->dy[i], arb_mat_entry(dy2, i, 0));
        arb_mul(v, d->dtau, arb_mat_entry(dy1, i, 0), prec);
        arb_add(&d->dy[i], &d->dy[i], v, prec);
    }
    for (int b = 0; b < nb; b++) {
        arb_mat_set(d->dX[b], dX2[b]);
        arb_mat_scalar_addmul_arb(d->dX[b], dX1[b], d->dtau, prec);
        arb_mat_set(d->dS[b], dS2[b]);
        arb_mat_scalar_addmul_arb(d->dS[b], dS1[b], d->dtau, prec);
    }

    /* ---------------------------------------------------------------------
     * Combined step length + Mehrotra safeguard (:820-827).
     * ------------------------------------------------------------------- */
    arb_set_d(big, 1e300);
    arb_set(alpha_raw, big);
    for (int b = 0; b < nb; b++) {
        if (arbsdp_psd_max_step(v, it->X[b], d->dX[b], prec) && arb_lt(v, alpha_raw))
            arb_set(alpha_raw, v);
        if (arbsdp_psd_max_step(v, it->S[b], d->dS[b], prec) && arb_lt(v, alpha_raw))
            arb_set(alpha_raw, v);
    }
    if (arbsdp_tau_kappa_max_step(v, it->tau, d->dtau, it->kappa, d->dkappa, prec)
        && arb_lt(v, alpha_raw))
        arb_set(alpha_raw, v);
    arbsdp_clip_step(d->alpha, alpha_raw, prec);

cleanup:
    for (int b = 0; b < nb; b++) {
        arb_mat_clear(WCW[b]);
        arb_mat_clear(WrdW[b]);
        arb_mat_clear(dX1[b]);
        arb_mat_clear(dS1[b]);
        arb_mat_clear(dXaff[b]);
        arb_mat_clear(dSaff[b]);
        arb_mat_clear(dX2[b]);
        arb_mat_clear(dS2[b]);
        arb_mat_clear(ExC[b]);
        nt_gfactor_clear(&F[b]);
    }
    flint_free(WCW); flint_free(WrdW); flint_free(F);
    flint_free(dX1); flint_free(dS1); flint_free(dXaff); flint_free(dSaff);
    flint_free(dX2); flint_free(dS2); flint_free(ExC);

    arb_mat_clear(M); arb_mat_clear(L); arb_mat_clear(rhs);
    arb_mat_clear(dy1); arb_mat_clear(dy2);
    arb_clear(v); arb_clear(s);
    arb_clear(cdx1); arb_clear(bdy1); arb_clear(dtau_denom); arb_clear(kappa_over_tau);
    arb_clear(cdx2); arb_clear(bdy2);
    arb_clear(dtau_aff); arb_clear(dkap_aff); arb_clear(eta); arb_clear(sigma_mu);
    arb_clear(Etau); arb_clear(dtau_num);
    arb_clear(alpha_raw); arb_clear(big);

    return ok;
}
