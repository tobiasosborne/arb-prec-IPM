/*
 * src/ntscaling.c -- Layer-0 (POINT MODE) Nesterov-Todd scaling.
 * See include/arbsdp/ntscaling.h for the POINT-MODE contract, the ground-truth
 * citations, and the W S W = X verification.
 *
 * Algorithmic source: PsdCone.ts ntScaling (lines 169-183), ported exactly:
 *     Xhalf = psdSqrt(X)
 *     inner = symmetrize(Xhalf * S * Xhalf)
 *     W     = symmetrize(Xhalf * psdInvSqrt(inner) * Xhalf)
 * Math: docs/MATH_SPEC.md §3.3; Todd-Toh-Tutuncu 1998.
 *
 * REUSE (CLAUDE.md, no duplicated implementations): the spectral kernels
 * arbsdp_psd_sqrt / arbsdp_psd_invsqrt (linalg.h) and arbsdp_symmetrize
 * (svec.h); matrix products via arb_mat_mul.  Nothing is re-implemented here.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every arb_mat_init below has a
 * matching arb_mat_clear in the same function.  Preconditions asserted (rule 5).
 */

#include <assert.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/ntscaling.h"
#include "arbsdp/linalg.h"
#include "arbsdp/svec.h"

void
arbsdp_nt_scaling(arb_mat_t W, const arb_mat_t X, const arb_mat_t S, slong prec)
{
    slong n = arb_mat_nrows(X);
    arb_mat_t Xhalf, inner, innerInvSqrt, tmp;

    /* Fail fast (CLAUDE.md rule 5): square, matching size, no aliasing. */
    assert(arb_mat_nrows(X) == arb_mat_ncols(X));     /* X square */
    assert(arb_mat_nrows(S) == arb_mat_ncols(S));     /* S square */
    assert(arb_mat_nrows(S) == n);                     /* same size */
    assert(arb_mat_nrows(W) == n && arb_mat_ncols(W) == n);
    assert(W != X && W != S);                           /* no aliasing */

    arb_mat_init(Xhalf, n, n);
    arb_mat_init(inner, n, n);
    arb_mat_init(innerInvSqrt, n, n);
    arb_mat_init(tmp, n, n);

    /* Xhalf = X^{1/2}. */
    arbsdp_psd_sqrt(Xhalf, X, prec);

    /* inner = X^{1/2} S X^{1/2}, symmetrized (PsdCone.ts symmetrize(inner)). */
    arb_mat_mul(tmp, Xhalf, S, prec);
    arb_mat_mul(inner, tmp, Xhalf, prec);
    arbsdp_symmetrize(inner, prec);

    /* innerInvSqrt = (X^{1/2} S X^{1/2})^{-1/2}. */
    arbsdp_psd_invsqrt(innerInvSqrt, inner, prec);

    /* W = X^{1/2} innerInvSqrt X^{1/2}, symmetrized. */
    arb_mat_mul(tmp, Xhalf, innerInvSqrt, prec);
    arb_mat_mul(W, tmp, Xhalf, prec);
    arbsdp_symmetrize(W, prec);

    arb_mat_clear(Xhalf);
    arb_mat_clear(inner);
    arb_mat_clear(innerInvSqrt);
    arb_mat_clear(tmp);
}
