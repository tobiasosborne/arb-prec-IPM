/*
 * include/arbsdp/svec.h -- symmetric vectorization (svec) with the strict-Mosek
 * sqrt(2) off-diagonal scaling.  This module is the SINGLE locus of the svec
 * convention (CLAUDE.md invariant 6); never deviate from it elsewhere.
 *
 * Source / ground truth:
 *   - docs/MATH_SPEC.md §2 (vectorization) and §10.1 (binding decision).
 *   - CLAUDE.md invariant 6 (strict-Mosek sqrt(2) off-diagonal scaling).
 *   - PsdCone.ts (symmetrize / frobInner reference -- note that the TS solver is
 *     DENSE, not svec; here we implement the sqrt(2) svec instead).
 *
 * ==========================================================================
 * THE svec CONVENTION (BINDING -- MATH_SPEC §10.1, orchestrator decision 1)
 * ==========================================================================
 * Upper-triangle, column-major.  For a symmetric n x n matrix A, the packed
 * vector v = svec(A) has length d = n(n+1)/2 and entries indexed by pairs
 * (i, j) with 0 <= i <= j <= n-1 (0-indexed) ordered column-major over the
 * UPPER triangle:
 *
 *     k = j*(j+1)/2 + i        (0-indexed, i <= j)
 *
 *     v[k] =        A_{ij}      if i == j   (diagonal -- UNSCALED)
 *     v[k] = sqrt(2) * A_{ij}   if i <  j   (strict upper -- SCALED by sqrt(2))
 *
 * The inverse, smat(v) -> symmetric A:
 *
 *     A_{ii} = v[k]                          (diagonal)
 *     A_{ij} = A_{ji} = v[k] / sqrt(2)       (i < j; both off-diagonal entries)
 *
 * The sqrt(2) on the off-diagonals makes the Euclidean dot product of two svecs
 * equal the Frobenius inner product of the matrices:
 *
 *     svec(A) . svec(B) = sum_{ij} A_{ij} B_{ij} = <A, B>
 *
 * (the off-diagonal A_{ij}B_{ij} is double-counted in the Frobenius sum over all
 * i,j, and the (sqrt(2))^2 = 2 factor in the svec dot product reproduces it).
 * This identity is the WHOLE POINT of the scaling and must match the TS solver
 * exactly or golden-master comparison is meaningless (CLAUDE.md invariant 6).
 *
 * All functions are precision-parameterized via `slong prec` and operate on
 * arb point OR ball values; they do not assume rigor (CLAUDE.md rule 1).
 * Arb memory discipline (CLAUDE.md rule 7): callers own all init/clear of the
 * arb / arb_mat / arb_ptr arguments; these functions allocate no persistent
 * state.  Square-matrix and matching-dimension preconditions are asserted
 * (CLAUDE.md rule 5, fail fast).
 */

#ifndef ARBSDP_SVEC_H
#define ARBSDP_SVEC_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_svec_len -- packed length of svec for an n x n symmetric block.
 * Returns n*(n+1)/2.  Requires n >= 0.
 */
slong arbsdp_svec_len(slong n);

/*
 * arbsdp_svec -- pack symmetric n x n A into v = svec(A), length n(n+1)/2.
 *
 * Off-diagonal entries are scaled by sqrt(2) (see convention above).  Only the
 * UPPER triangle of A is read (the entry (i,j) with i <= j); the input is NOT
 * required to be symmetric -- the strict-lower triangle is ignored.  If the
 * caller is unsure whether A is symmetric, call arbsdp_symmetrize first.
 *
 * `v` must point to at least arbsdp_svec_len(n) initialized arb elements, where
 * n = arb_mat_nrows(A).  A must be square (asserted).
 */
void arbsdp_svec(arb_ptr v, const arb_mat_t A, slong prec);

/*
 * arbsdp_smat -- inverse of arbsdp_svec: unpack v (length n(n+1)/2) into the
 * symmetric n x n matrix A.  Off-diagonal entries are divided by sqrt(2) and
 * written to BOTH (i,j) and (j,i), so the result is exactly symmetric.
 *
 * A must be an initialized n x n arb_mat (asserted square with side n).
 * `v` must point to at least arbsdp_svec_len(n) arb elements.
 */
void arbsdp_smat(arb_mat_t A, arb_srcptr v, slong n, slong prec);

/*
 * arbsdp_symmetrize -- A <- (A + A^T) / 2, in place.
 *
 * Reference: PsdCone.ts:8-16 (symmetrize).  A must be square (asserted).
 */
void arbsdp_symmetrize(arb_mat_t A, slong prec);

/*
 * arbsdp_frob_inner -- dense Frobenius inner product
 *     out <- <A, B> = sum_{ij} A_{ij} B_{ij}
 * over ALL n*n entries (no svec packing).  Reference: PsdCone.ts:18-23
 * (frobInner).  Provided as an independent cross-check for the svec dot-product
 * identity.  A and B must have identical dimensions (asserted); they need not be
 * square or symmetric.  `out` must be initialized.
 */
void arbsdp_frob_inner(arb_t out, const arb_mat_t A, const arb_mat_t B,
                       slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_SVEC_H */
