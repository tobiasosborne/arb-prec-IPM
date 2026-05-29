/*
 * src/linalg.c -- Layer-0 (POINT MODE) real-symmetric spectral kernels:
 * cyclic-Jacobi eigendecomposition, PSD matrix square root and inverse square
 * root.  See include/arbsdp/linalg.h for the POINT-MODE contract and the
 * AUDITION note (rule 3) that selected cyclic-Jacobi over acb_mat_approx_eig_qr.
 *
 * Algorithmic source: PsdCone.ts eighJacobi (lines 63-130), psdSqrt (133-149),
 * psdInvSqrt (152-167).  arb port of the float64 idioms; the math (symmetric A =
 * Q diag(lambda) Q^T via Jacobi rotations) is identical.
 *
 * REUSE (CLAUDE.md, no duplicated implementations): symmetrization uses
 * arbsdp_symmetrize from svec.h -- not re-implemented here.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every arb_init / arb_mat_init /
 * _arb_vec_init below has a matching clear in the same function; no persistent
 * allocation crosses a function boundary.  Preconditions are asserted (rule 5).
 */

#include <assert.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/linalg.h"
#include "arbsdp/svec.h"

/*
 * arbsdp_eigh -- cyclic-Jacobi symmetric eigendecomposition (POINT MODE).
 * Port of PsdCone.ts eighJacobi.  eigenvalues ascending; A = Q diag(lambda) Q^T.
 */
void
arbsdp_eigh(arb_ptr eigvals, arb_mat_t Q, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t M;
    arb_t apq, app, aqq, theta, t, c, s, x, y, nrp, nrq, off, tol2;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A));     /* square */
    assert(arb_mat_nrows(Q) == n && arb_mat_ncols(Q) == n);

    arb_mat_init(M, n, n);
    arb_mat_set(M, A);
    arbsdp_symmetrize(M, prec); /* REUSE svec.h; M <- (M + M^T)/2 */

    arb_mat_one(Q);

    arb_init(apq); arb_init(app); arb_init(aqq); arb_init(theta);
    arb_init(t);   arb_init(c);   arb_init(s);   arb_init(x);
    arb_init(y);   arb_init(nrp); arb_init(nrq); arb_init(off);
    arb_init(tol2);

    /* Convergence floor on sum of squared off-diagonals.  POINT MODE: this is a
     * heuristic stop scaled to the working precision (no rigor claimed).  At
     * prec bits, ~2^(-prec) is the relative floor; square it and scale by n^2 to
     * be safe.  We bound sweeps as a hard backstop. */
    {
        slong e = -2 * prec + 8; /* (2^-prec)^2 with a little slack */
        arb_set_si(tol2, 1);
        arb_mul_2exp_si(tol2, tol2, e);
    }

    for (slong sweep = 0; sweep < 100; sweep++) {
        /* off = sum_{i<j} M_{ij}^2 */
        arb_zero(off);
        for (slong i = 0; i < n; i++)
            for (slong j = i + 1; j < n; j++) {
                arb_mul(x, arb_mat_entry(M, i, j), arb_mat_entry(M, i, j), prec);
                arb_add(off, off, x, prec);
            }
        if (arf_cmp(arb_midref(off), arb_midref(tol2)) < 0)
            break;

        for (slong p = 0; p < n - 1; p++) {
            for (slong q = p + 1; q < n; q++) {
                arb_set(apq, arb_mat_entry(M, p, q));
                if (arf_is_zero(arb_midref(apq)))
                    continue;
                arb_set(app, arb_mat_entry(M, p, p));
                arb_set(aqq, arb_mat_entry(M, q, q));

                /* theta = (aqq - app) / (2 apq) */
                arb_sub(theta, aqq, app, prec);
                arb_mul_2exp_si(x, apq, 1); /* 2*apq */
                arb_div(theta, theta, x, prec);

                /* t = sign(theta) / (|theta| + sqrt(1 + theta^2)) */
                arb_mul(x, theta, theta, prec);
                arb_add_ui(x, x, 1, prec);
                arb_sqrt(x, x, prec);
                arb_abs(y, theta);
                arb_add(x, x, y, prec);
                arb_inv(t, x, prec);
                if (arf_sgn(arb_midref(theta)) < 0)
                    arb_neg(t, t);

                /* c = 1/sqrt(1 + t^2), s = t c */
                arb_mul(x, t, t, prec);
                arb_add_ui(x, x, 1, prec);
                arb_rsqrt(c, x, prec);
                arb_mul(s, t, c, prec);

                /* Diagonal update: M_pp -= t apq ; M_qq += t apq ; M_pq=M_qp=0 */
                arb_mul(x, t, apq, prec);
                arb_sub(arb_mat_entry(M, p, p), app, x, prec);
                arb_add(arb_mat_entry(M, q, q), aqq, x, prec);
                arb_zero(arb_mat_entry(M, p, q));
                arb_zero(arb_mat_entry(M, q, p));

                /* Rotate the remaining rows/cols of M (symmetric). */
                for (slong r = 0; r < n; r++) {
                    if (r == p || r == q)
                        continue;
                    arb_set(x, arb_mat_entry(M, r, p));
                    arb_set(y, arb_mat_entry(M, r, q));
                    /* nrp = c x - s y ; nrq = s x + c y */
                    arb_mul(nrp, c, x, prec);
                    arb_mul(app, s, y, prec); /* app reused as scratch */
                    arb_sub(nrp, nrp, app, prec);
                    arb_mul(nrq, s, x, prec);
                    arb_mul(app, c, y, prec);
                    arb_add(nrq, nrq, app, prec);
                    arb_set(arb_mat_entry(M, r, p), nrp);
                    arb_set(arb_mat_entry(M, p, r), nrp);
                    arb_set(arb_mat_entry(M, r, q), nrq);
                    arb_set(arb_mat_entry(M, q, r), nrq);
                }

                /* Accumulate the rotation into Q (columns p, q). */
                for (slong r = 0; r < n; r++) {
                    arb_set(x, arb_mat_entry(Q, r, p));
                    arb_set(y, arb_mat_entry(Q, r, q));
                    arb_mul(nrp, c, x, prec);
                    arb_mul(app, s, y, prec);
                    arb_sub(arb_mat_entry(Q, r, p), nrp, app, prec);
                    arb_mul(nrq, s, x, prec);
                    arb_mul(app, c, y, prec);
                    arb_add(arb_mat_entry(Q, r, q), nrq, app, prec);
                }
            }
        }
    }

    /* eigenvalues = diag(M) */
    for (slong i = 0; i < n; i++)
        arb_set(eigvals + i, arb_mat_entry(M, i, i));

    /* Selection sort ascending; swap matching Q columns.  Compare midpoints
     * (POINT MODE -- the ordering is on the point estimates). */
    for (slong a = 0; a < n; a++) {
        slong mn = a;
        for (slong b = a + 1; b < n; b++)
            if (arf_cmp(arb_midref(eigvals + b), arb_midref(eigvals + mn)) < 0)
                mn = b;
        if (mn != a) {
            arb_swap(eigvals + a, eigvals + mn);
            for (slong r = 0; r < n; r++)
                arb_swap(arb_mat_entry(Q, r, a), arb_mat_entry(Q, r, mn));
        }
    }

    arb_clear(apq); arb_clear(app); arb_clear(aqq); arb_clear(theta);
    arb_clear(t);   arb_clear(c);   arb_clear(s);   arb_clear(x);
    arb_clear(y);   arb_clear(nrp); arb_clear(nrq); arb_clear(off);
    arb_clear(tol2);
    arb_mat_clear(M);
}

/*
 * Shared core for psd_sqrt / psd_invsqrt: out = Q diag(f(lambda)) Q^T, where the
 * eigenvalue transform f is selected by `invert`:
 *   invert == 0:  f(lam) = sqrt(max(0, lam))         (matrix square root)
 *   invert != 0:  f(lam) = 1/sqrt(max(eps, lam))     (matrix inverse square root)
 * Clamping mirrors PsdCone.ts psdSqrt / psdInvSqrt (Math.max(0,.)/Math.max(1e-300,.)).
 */
static void
psd_pow_half(arb_mat_t out, const arb_mat_t A, int invert, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t Q, QD, Qt;
    arb_ptr lam;
    arb_t d, eps;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A));     /* square */
    assert(arb_mat_nrows(out) == n && arb_mat_ncols(out) == n);
    assert(out != A);                                  /* no aliasing */

    arb_mat_init(Q, n, n);
    arb_mat_init(QD, n, n);
    arb_mat_init(Qt, n, n);
    lam = _arb_vec_init(n);
    arb_init(d);
    arb_init(eps);
    /* eps floor for invsqrt: tiny positive (PsdCone uses 1e-300). */
    arb_set_d(eps, 1e-300);

    arbsdp_eigh(lam, Q, A, prec);

    /* QD = Q * diag(f(lambda)) : scale column i of Q by f(lambda_i). */
    arb_mat_set(QD, Q);
    for (slong i = 0; i < n; i++) {
        if (!invert) {
            /* f = sqrt(max(0, lam)) */
            if (arf_sgn(arb_midref(lam + i)) <= 0)
                arb_zero(d);
            else
                arb_sqrt(d, lam + i, prec);
        } else {
            /* f = 1/sqrt(max(eps, lam)) */
            if (arf_cmp(arb_midref(lam + i), arb_midref(eps)) < 0)
                arb_set(d, eps);
            else
                arb_set(d, lam + i);
            arb_rsqrt(d, d, prec);
        }
        for (slong r = 0; r < n; r++)
            arb_mul(arb_mat_entry(QD, r, i), arb_mat_entry(Q, r, i), d, prec);
    }

    /* out = QD * Q^T */
    arb_mat_transpose(Qt, Q);
    arb_mat_mul(out, QD, Qt, prec);
    arbsdp_symmetrize(out, prec); /* enforce exact symmetry (REUSE svec.h) */

    arb_clear(d);
    arb_clear(eps);
    _arb_vec_clear(lam, n);
    arb_mat_clear(Q);
    arb_mat_clear(QD);
    arb_mat_clear(Qt);
}

void
arbsdp_psd_sqrt(arb_mat_t out, const arb_mat_t A, slong prec)
{
    psd_pow_half(out, A, 0, prec);
}

void
arbsdp_psd_invsqrt(arb_mat_t out, const arb_mat_t A, slong prec)
{
    psd_pow_half(out, A, 1, prec);
}
