/*
 * include/arbsdp/linalg.h -- Layer-0 (POINT MODE) dense real-symmetric spectral
 * kernels for the NT scaling: eigendecomposition, matrix square root and inverse
 * square root.
 *
 * ==========================================================================
 * POINT MODE -- NOT RIGOROUS (CLAUDE.md rule 1, invariant 2)
 * ==========================================================================
 * These routines are used INSIDE the HSDE-NT IPM loop (Layer 0), which runs in
 * Arb *point* arithmetic at adaptive precision.  The results are consumed as
 * POINTS (midpoints); the ball *radii* on the outputs may inflate and are NOT
 * trusted.  This is acceptable and by design: Layer 1 derives ALL rigor
 * independently from verified Cholesky (arb_mat_cho), never from these eig
 * results.  Do NOT use arbsdp_eigh / arbsdp_psd_sqrt / arbsdp_psd_invsqrt in any
 * rigorous (ball) certification path.
 *
 * ==========================================================================
 * AUDITION (CLAUDE.md rule 3) -- why ported cyclic-Jacobi, not acb_mat eig
 * ==========================================================================
 * arb_mat.h has NO real-symmetric eigendecomposition (FLINT_NOTES.md).  Two
 * realistic POINT-MODE routes were auditioned at prec=256 on matrices with KNOWN
 * closed-form spectra (docs/probes/eig_audition.c):
 *
 *   (a) Ported cyclic-Jacobi (pure real arb), port of PsdCone.ts eighJacobi.
 *   (b) acb_mat_approx_eig_qr: pack real-symmetric A -> complex acb_mat, run
 *       Arb's approximate QR eig, take real parts.
 *
 * Result (eigval error vs arb-computed exact spectrum; reconstruction residual
 * ||A - Q diag(lambda) Q^T||_max):
 *
 *   matrix         (a) Jacobi recon   (a) lam rel_acc_bits   (b) acb_qr
 *   diag(1..6)     0                  exact                  ok=0  recon 0
 *   [[2,1],[1,2]]  3.5e-77            ~prec                  ok=0  recon 6.9e-77
 *   laplacian3     9.9e-55            59/251/1018 @64/256/1024 ok=0 recon 1.1e-17
 *
 * WINNER: (a) cyclic-Jacobi.  Its eigenvalue accuracy tracks the working
 * precision (rel_acc_bits ~ prec), giving genuine high-precision spectra, while
 * (b) acb_mat_approx_eig_qr returned its convergence flag = 0 on EVERY case
 * (including a diagonal matrix) and its reconstruction residual stalled at
 * ~1e-17 (double precision) regardless of prec -- it is a float-style
 * approximate QR that does not refine to arb precision, and an ok=0 return on a
 * diagonal matrix is untrustworthy (CLAUDE.md rule 5, fail fast).  (b) also
 * forces a complex pack/unpack and discards imaginary parts.  Jacobi is pure
 * real arb, point-mode, and refines with prec -- it wins on accuracy and avoids
 * the complex round-trip.  This empirically CONFIRMS the FLINT_NOTES.md verdict.
 * acb_mat_eig_* is kept only in reserve as an optional cross-check oracle.
 *
 * Algorithmic source: PsdCone.ts eighJacobi (lines 63-130), psdSqrt (133-149),
 * psdInvSqrt (152-167).  Math: any real symmetric A = Q diag(lambda) Q^T with Q
 * orthogonal; Jacobi rotations zero off-diagonals one pair at a time.
 *
 * Memory discipline (CLAUDE.md rule 7): callers own init/clear of every
 * argument (arb_ptr eigvals of length n, arb_mat outputs of size n x n); these
 * functions allocate only scoped temporaries that they clear themselves.
 * Square-matrix preconditions are asserted (CLAUDE.md rule 5).
 */

#ifndef ARBSDP_LINALG_H
#define ARBSDP_LINALG_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_eigh -- real-symmetric eigendecomposition (POINT MODE, cyclic Jacobi).
 *
 * Computes eigenvalues (ASCENDING) and an orthogonal Q with
 *     A = Q * diag(eigvals) * Q^T,
 * i.e. column j of Q is the unit eigenvector for eigvals[j].  Only the symmetric
 * part of A is used (A is symmetrized internally); A is not modified.
 *
 * `eigvals` must point to at least n initialized arb elements (n = side of A).
 * `Q` must be an initialized n x n arb_mat.  A must be square (asserted).
 *
 * POINT MODE: results are midpoints; radii are not trusted (see header banner).
 */
void arbsdp_eigh(arb_ptr eigvals, arb_mat_t Q, const arb_mat_t A, slong prec);

/*
 * arbsdp_psd_sqrt -- out <- A^{1/2} for symmetric PSD A (POINT MODE).
 *
 * Via eigendecomposition: A^{1/2} = Q diag(sqrt(max(0, lambda_i))) Q^T.
 * Tiny/negative eigenvalues are clamped to 0 before the sqrt (matching
 * PsdCone.ts psdSqrt, which uses Math.max(0, lambda)) so a slightly-indefinite
 * input still yields a real PSD result.
 *
 * `out` must be an initialized n x n arb_mat; A must be square (asserted).
 * out and A may NOT alias.
 */
void arbsdp_psd_sqrt(arb_mat_t out, const arb_mat_t A, slong prec);

/*
 * arbsdp_psd_invsqrt -- out <- A^{-1/2} for symmetric PD A (POINT MODE).
 *
 * Via eigendecomposition: A^{-1/2} = Q diag(1/sqrt(max(eps, lambda_i))) Q^T.
 * Eigenvalues are floored at a tiny positive eps before the reciprocal sqrt
 * (matching PsdCone.ts psdInvSqrt, which uses Math.max(1e-300, lambda)) to keep
 * the result finite for a near-singular input.
 *
 * `out` must be an initialized n x n arb_mat; A must be square (asserted).
 * out and A may NOT alias.
 */
void arbsdp_psd_invsqrt(arb_mat_t out, const arb_mat_t A, slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_LINALG_H */
