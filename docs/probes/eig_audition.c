/*
 * docs/probes/eig_audition.c -- bead arb-prec-IPM-b05 AUDITION probe.
 *
 * Compares two realistic routes for real-symmetric eigendecomposition usable in
 * POINT mode at arb precision (CLAUDE.md rule 3 -- audition, do not default to
 * the first/most familiar option):
 *
 *   (a) Ported cyclic-Jacobi (pure real arb), port of PsdCone.ts eighJacobi.
 *   (b) acb_mat_approx_eig_qr: pack the real-symmetric A into a complex acb_mat,
 *       run Arb's approximate QR eig, take real parts of eigenvalues and the
 *       right eigenvectors.
 *
 * Metric: max error in eigenvalues vs KNOWN closed-form spectra, and the
 * reconstruction residual ||A - Q diag(lambda) Q^T||_max, at prec = 256.
 *
 * Test matrices (known spectra):
 *   1. diag(1,2,3,4,5,6)              -> {1,2,3,4,5,6}
 *   2. [[2,1],[1,2]]                  -> {1,3}
 *   3. 3x3 [[2,-1,0],[-1,2,-1],[0,-1,2]] (1D Laplacian) -> {2-sqrt2, 2, 2+sqrt2}
 *
 * Build (from repo root):
 *   gcc -std=c11 -O2 -g -I/usr/include docs/probes/eig_audition.c \
 *       -lflint -lmpfr -lgmp -o /tmp/eig_audition && /tmp/eig_audition
 *
 * This file is a one-off measurement harness; it is NOT compiled into libarbsdp.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>
#include <flint/acb.h>
#include <flint/acb_mat.h>

/* ---- option (a): ported cyclic Jacobi (real arb) ------------------------- */
/* eigvals ascending; A = Q diag(lambda) Q^T.  Operates on arb midpoints. */
static void
jacobi_eigh(arb_ptr eigvals, arb_mat_t Q, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t M;
    arb_t apq, app, aqq, theta, t, c, s, tmp, tmp2, off, tol2;
    arb_mat_init(M, n, n);
    arb_mat_set(M, A);
    /* symmetrize */
    for (slong i = 0; i < n; i++)
        for (slong j = i + 1; j < n; j++) {
            arb_add(arb_mat_entry(M, i, j), arb_mat_entry(M, i, j),
                    arb_mat_entry(M, j, i), prec);
            arb_mul_2exp_si(arb_mat_entry(M, i, j), arb_mat_entry(M, i, j), -1);
            arb_set(arb_mat_entry(M, j, i), arb_mat_entry(M, i, j));
        }
    arb_mat_one(Q);
    arb_init(apq); arb_init(app); arb_init(aqq); arb_init(theta);
    arb_init(t); arb_init(c); arb_init(s); arb_init(tmp); arb_init(tmp2);
    arb_init(off); arb_init(tol2);
    arb_set_d(tol2, 1e-30); /* (tol)^2 with tol ~ 1e-15 scaled by n */
    for (slong sweep = 0; sweep < 100; sweep++) {
        arb_zero(off);
        for (slong i = 0; i < n; i++)
            for (slong j = i + 1; j < n; j++) {
                arb_mul(tmp, arb_mat_entry(M, i, j), arb_mat_entry(M, i, j), prec);
                arb_add(off, off, tmp, prec);
            }
        if (arf_cmp(arb_midref(off), arb_midref(tol2)) < 0)
            break;
        for (slong p = 0; p < n - 1; p++)
            for (slong q = p + 1; q < n; q++) {
                arb_set(apq, arb_mat_entry(M, p, q));
                if (arf_is_zero(arb_midref(apq)))
                    continue;
                arb_set(app, arb_mat_entry(M, p, p));
                arb_set(aqq, arb_mat_entry(M, q, q));
                /* theta = (aqq-app)/(2 apq) */
                arb_sub(theta, aqq, app, prec);
                arb_mul_2exp_si(tmp, apq, 1);
                arb_div(theta, theta, tmp, prec);
                /* t = sign(theta)/(|theta|+sqrt(1+theta^2)) */
                arb_mul(tmp, theta, theta, prec);
                arb_add_ui(tmp, tmp, 1, prec);
                arb_sqrt(tmp, tmp, prec);
                arb_abs(tmp2, theta);
                arb_add(tmp, tmp, tmp2, prec);
                arb_inv(t, tmp, prec);
                if (arf_sgn(arb_midref(theta)) < 0)
                    arb_neg(t, t);
                /* c = 1/sqrt(1+t^2), s = t c */
                arb_mul(tmp, t, t, prec);
                arb_add_ui(tmp, tmp, 1, prec);
                arb_rsqrt(c, tmp, prec);
                arb_mul(s, t, c, prec);
                /* update diagonal */
                arb_mul(tmp, t, apq, prec);
                arb_sub(arb_mat_entry(M, p, p), app, tmp, prec);
                arb_add(arb_mat_entry(M, q, q), aqq, tmp, prec);
                arb_zero(arb_mat_entry(M, p, q));
                arb_zero(arb_mat_entry(M, q, p));
                for (slong r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    arb_set(tmp, arb_mat_entry(M, r, p));
                    arb_set(tmp2, arb_mat_entry(M, r, q));
                    /* nrp = c*tmp - s*tmp2 */
                    arb_t nrp, nrq;
                    arb_init(nrp); arb_init(nrq);
                    arb_mul(nrp, c, tmp, prec);
                    arb_mul(aqq, s, tmp2, prec); /* reuse aqq as scratch */
                    arb_sub(nrp, nrp, aqq, prec);
                    arb_mul(nrq, s, tmp, prec);
                    arb_mul(aqq, c, tmp2, prec);
                    arb_add(nrq, nrq, aqq, prec);
                    arb_set(arb_mat_entry(M, r, p), nrp);
                    arb_set(arb_mat_entry(M, p, r), nrp);
                    arb_set(arb_mat_entry(M, r, q), nrq);
                    arb_set(arb_mat_entry(M, q, r), nrq);
                    arb_clear(nrp); arb_clear(nrq);
                }
                for (slong r = 0; r < n; r++) {
                    arb_set(tmp, arb_mat_entry(Q, r, p));
                    arb_set(tmp2, arb_mat_entry(Q, r, q));
                    arb_mul(arb_mat_entry(Q, r, p), c, tmp, prec);
                    arb_mul(aqq, s, tmp2, prec);
                    arb_sub(arb_mat_entry(Q, r, p), arb_mat_entry(Q, r, p), aqq, prec);
                    arb_mul(arb_mat_entry(Q, r, q), s, tmp, prec);
                    arb_mul(aqq, c, tmp2, prec);
                    arb_add(arb_mat_entry(Q, r, q), arb_mat_entry(Q, r, q), aqq, prec);
                }
            }
    }
    for (slong i = 0; i < n; i++)
        arb_set(eigvals + i, arb_mat_entry(M, i, i));
    /* selection sort ascending, swap Q columns */
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
    arb_clear(t); arb_clear(c); arb_clear(s); arb_clear(tmp); arb_clear(tmp2);
    arb_clear(off); arb_clear(tol2);
    arb_mat_clear(M);
}

/* ---- option (b): acb_mat_approx_eig_qr ----------------------------------- */
static int
acb_eigh(arb_ptr eigvals, arb_mat_t Q, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    acb_mat_t Ac, R;
    acb_ptr E;
    mag_t tol;
    int ok;
    acb_mat_init(Ac, n, n);
    acb_mat_init(R, n, n);
    E = _acb_vec_init(n);
    mag_init(tol);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++)
            arb_set(acb_realref(acb_mat_entry(Ac, i, j)), arb_mat_entry(A, i, j));
    ok = acb_mat_approx_eig_qr(E, NULL, R, Ac, tol, 0, prec);
    /* take real parts */
    for (slong i = 0; i < n; i++)
        arb_set(eigvals + i, acb_realref(E + i));
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++)
            arb_set(arb_mat_entry(Q, i, j), acb_realref(acb_mat_entry(R, i, j)));
    /* sort ascending */
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
    mag_clear(tol);
    _acb_vec_clear(E, n);
    acb_mat_clear(R);
    acb_mat_clear(Ac);
    return ok;
}

/* reconstruction residual max|A - Q diag(lambda) Q^T| */
static double
recon_err(arb_ptr eigvals, arb_mat_t Q, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t QD, R, D;
    arb_mat_init(QD, n, n); arb_mat_init(R, n, n); arb_mat_init(D, n, n);
    for (slong i = 0; i < n; i++)
        arb_set(arb_mat_entry(D, i, i), eigvals + i);
    arb_mat_mul(QD, Q, D, prec);
    arb_mat_transpose(R, Q);
    arb_mat_mul(D, QD, R, prec); /* D = Q diag Q^T */
    double mx = 0;
    arb_t diff;
    arb_init(diff);
    for (slong i = 0; i < n; i++)
        for (slong j = 0; j < n; j++) {
            arb_sub(diff, arb_mat_entry(D, i, j), arb_mat_entry(A, i, j), prec);
            double v = fabs(arf_get_d(arb_midref(diff), ARF_RND_NEAR));
            if (v > mx) mx = v;
        }
    arb_clear(diff);
    arb_mat_clear(QD); arb_mat_clear(R); arb_mat_clear(D);
    return mx;
}

static double
eigval_err(arb_ptr got, const double *expect, slong n)
{
    double mx = 0;
    for (slong i = 0; i < n; i++) {
        double v = fabs(arf_get_d(arb_midref(got + i), ARF_RND_NEAR) - expect[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

static void
run_case(const char *name, arb_mat_t A, const double *expect, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_ptr ea = _arb_vec_init(n), eb = _arb_vec_init(n);
    arb_mat_t Qa, Qb;
    arb_mat_init(Qa, n, n); arb_mat_init(Qb, n, n);

    jacobi_eigh(ea, Qa, A, prec);
    int okb = acb_eigh(eb, Qb, A, prec);

    double erra = eigval_err(ea, expect, n);
    double reca = recon_err(ea, Qa, A, prec);
    double errb = eigval_err(eb, expect, n);
    double recb = recon_err(eb, Qb, A, prec);

    printf("%-22s  (a) Jacobi: eig_err=%.3e recon=%.3e | "
           "(b) acb_qr[ok=%d]: eig_err=%.3e recon=%.3e\n",
           name, erra, reca, okb, errb, recb);

    _arb_vec_clear(ea, n); _arb_vec_clear(eb, n);
    arb_mat_clear(Qa); arb_mat_clear(Qb);
}

int
main(void)
{
    slong prec = 256;
    printf("eig audition @ prec=%ld (errors vs known closed-form spectra)\n", prec);

    /* case 1: diag(1..6) */
    {
        arb_mat_t A; arb_mat_init(A, 6, 6);
        double exp1[6] = {1,2,3,4,5,6};
        for (int i = 0; i < 6; i++) arb_set_si(arb_mat_entry(A, i, i), i + 1);
        run_case("diag(1..6)", A, exp1, prec);
        arb_mat_clear(A);
    }
    /* case 2: [[2,1],[1,2]] -> {1,3} */
    {
        arb_mat_t A; arb_mat_init(A, 2, 2);
        double exp2[2] = {1,3};
        arb_set_si(arb_mat_entry(A,0,0),2); arb_set_si(arb_mat_entry(A,1,1),2);
        arb_set_si(arb_mat_entry(A,0,1),1); arb_set_si(arb_mat_entry(A,1,0),1);
        run_case("[[2,1],[1,2]]", A, exp2, prec);
        arb_mat_clear(A);
    }
    /* case 3: 1D Laplacian 3x3 -> {2-sqrt2, 2, 2+sqrt2} */
    {
        arb_mat_t A; arb_mat_init(A, 3, 3);
        double r2 = sqrt(2.0);
        double exp3[3] = {2 - r2, 2, 2 + r2};
        arb_set_si(arb_mat_entry(A,0,0),2); arb_set_si(arb_mat_entry(A,1,1),2);
        arb_set_si(arb_mat_entry(A,2,2),2);
        arb_set_si(arb_mat_entry(A,0,1),-1); arb_set_si(arb_mat_entry(A,1,0),-1);
        arb_set_si(arb_mat_entry(A,1,2),-1); arb_set_si(arb_mat_entry(A,2,1),-1);
        run_case("laplacian3", A, exp3, prec);
        arb_mat_clear(A);
    }

    /* a measurement of how accuracy scales with precision for Jacobi (point) */
    printf("\nJacobi eigenvalue accuracy vs prec on laplacian3:\n");
    for (slong p = 64; p <= 1024; p *= 4) {
        arb_mat_t A; arb_mat_init(A, 3, 3);
        double r2 = sqrt(2.0);
        double exp3[3] = {2 - r2, 2, 2 + r2};
        arb_set_si(arb_mat_entry(A,0,0),2); arb_set_si(arb_mat_entry(A,1,1),2);
        arb_set_si(arb_mat_entry(A,2,2),2);
        arb_set_si(arb_mat_entry(A,0,1),-1); arb_set_si(arb_mat_entry(A,1,0),-1);
        arb_set_si(arb_mat_entry(A,1,2),-1); arb_set_si(arb_mat_entry(A,2,1),-1);
        arb_ptr e = _arb_vec_init(3);
        arb_mat_t Q; arb_mat_init(Q, 3, 3);
        jacobi_eigh(e, Q, A, p);
        /* report rel accuracy bits of the smallest eigenvalue and abs err */
        printf("  prec=%4ld  lam0 rel_acc_bits=%ld  eig_err=%.3e\n",
               p, arb_rel_accuracy_bits(e + 0), eigval_err(e, exp3, 3));
        _arb_vec_clear(e, 3); arb_mat_clear(Q); arb_mat_clear(A);
    }

    flint_cleanup();
    return 0;
}
