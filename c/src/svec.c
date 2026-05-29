/*
 * src/svec.c -- symmetric vectorization (svec) with strict-Mosek sqrt(2)
 * off-diagonal scaling.  See include/arbsdp/svec.h for the binding convention
 * (MATH_SPEC §2 + §10.1, CLAUDE.md invariant 6).
 *
 * Index map (0-indexed, i <= j, column-major upper triangle):
 *     k = j*(j+1)/2 + i
 *
 * Arb memory discipline (CLAUDE.md rule 7): the only temporaries below are a
 * single arb_t in arbsdp_svec / arbsdp_smat / arbsdp_symmetrize and a sqrt(2)
 * scalar; each is arb_init'd and arb_clear'd in the same scope.  No persistent
 * allocation crosses a function boundary.  Preconditions are asserted (rule 5).
 */

#include <assert.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/svec.h"

slong
arbsdp_svec_len(slong n)
{
    assert(n >= 0);
    return n * (n + 1) / 2;
}

void
arbsdp_svec(arb_ptr v, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_t sqrt2;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A)); /* square */

    arb_init(sqrt2);
    arb_sqrt_ui(sqrt2, 2, prec);

    /* Column-major over the upper triangle: for each column j, rows i = 0..j.
     * k = j*(j+1)/2 + i increments by 1 in this exact traversal order. */
    for (slong j = 0; j < n; j++) {
        for (slong i = 0; i <= j; i++) {
            slong k = j * (j + 1) / 2 + i;
            if (i == j) {
                arb_set(v + k, arb_mat_entry(A, i, j)); /* diagonal: unscaled */
            } else {
                /* strict upper (i < j): scale by sqrt(2). */
                arb_mul(v + k, arb_mat_entry(A, i, j), sqrt2, prec);
            }
        }
    }

    arb_clear(sqrt2);
}

void
arbsdp_smat(arb_mat_t A, arb_srcptr v, slong n, slong prec)
{
    arb_t inv_sqrt2;

    assert(n >= 0);
    assert(arb_mat_nrows(A) == n && arb_mat_ncols(A) == n); /* square, side n */

    arb_init(inv_sqrt2);
    arb_sqrt_ui(inv_sqrt2, 2, prec);
    arb_inv(inv_sqrt2, inv_sqrt2, prec); /* 1/sqrt(2) */

    for (slong j = 0; j < n; j++) {
        for (slong i = 0; i <= j; i++) {
            slong k = j * (j + 1) / 2 + i;
            if (i == j) {
                arb_set(arb_mat_entry(A, i, j), v + k); /* diagonal: unscaled */
            } else {
                /* Off-diagonal: divide out sqrt(2), write both (i,j),(j,i). */
                arb_mul(arb_mat_entry(A, i, j), v + k, inv_sqrt2, prec);
                arb_set(arb_mat_entry(A, j, i), arb_mat_entry(A, i, j));
            }
        }
    }

    arb_clear(inv_sqrt2);
}

void
arbsdp_symmetrize(arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_t avg;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A)); /* square */

    arb_init(avg);

    /* A_{ij} <- (A_{ij} + A_{ji}) / 2 for i < j; diagonal is unchanged.
     * Reference: PsdCone.ts:8-16. */
    for (slong i = 0; i < n; i++) {
        for (slong j = i + 1; j < n; j++) {
            arb_add(avg, arb_mat_entry(A, i, j), arb_mat_entry(A, j, i), prec);
            arb_mul_2exp_si(avg, avg, -1); /* divide by 2, exact */
            arb_set(arb_mat_entry(A, i, j), avg);
            arb_set(arb_mat_entry(A, j, i), avg);
        }
    }

    arb_clear(avg);
}

void
arbsdp_frob_inner(arb_t out, const arb_mat_t A, const arb_mat_t B, slong prec)
{
    slong nr = arb_mat_nrows(A);
    slong nc = arb_mat_ncols(A);

    assert(arb_mat_nrows(B) == nr && arb_mat_ncols(B) == nc); /* match dims */

    /* out = sum_{ij} A_{ij} B_{ij}.  Accumulate with arb_dot over each row for
     * tight enclosures.  Reference: PsdCone.ts:18-23 (frobInner). */
    arb_zero(out);
    for (slong i = 0; i < nr; i++) {
        arb_dot(out, out, 0,
                arb_mat_entry(A, i, 0), 1,
                arb_mat_entry(B, i, 0), 1,
                nc, prec);
    }
}
