/*
 * docs/probes/schur_nt_audition.c -- bead arb-prec-IPM-b12 AUDITION probe.
 *
 * The Schur formula (MATH_SPEC §3.4) is
 *      M_ik = sum_b <A_i^b, W^b A_k^b W^b>,
 * and the question for b12 is WHICH NT representation to feed it:
 *
 *   (1) the symmetric W  (arbsdp_nt_scaling, W = X^{1/2}(X^{1/2} S X^{1/2})^{-1/2}
 *       X^{1/2}, satisfying W S W = X)  -- assemble M_ik = <A_i, W A_k W> directly;
 *   (2) the TS HSDE G-factor (buildNtFactor: L=chol(S), [V,sv2]=eigh(L^T X L),
 *       G = diag(sqrt(sv)) V^T L^{-1}, W = G^T G).  Identity G^T diag(sv) G = X.
 *
 * KEY MATHEMATICAL FACT (this is the audition's whole point): both representations
 * produce the SAME symmetric W (W = G^T G is the unique symmetric PD square root
 * solving W S W = X).  So the Schur matrix M is IDENTICAL either way -- the
 * G-factor's role in the TS solver is the DIRECTION RECOVERY (hdZ = G dS G^T,
 * dX = G^T diag(.) G; HsdeNtSdpSolver.ts:703-758), NOT a differently-conditioned
 * Schur assembly.  The only freedom is the ASSEMBLY ROUTE:
 *
 *   route A (symmetric-W): WAW = W A_k W ; M_ik = <A_i, WAW_k>          (2 mat-muls/k)
 *   route B (G-factor):    GAGt = G A_k G^T ; M_ik = <G A_i G^T, GAGt_k> (Gram form)
 *
 * Route B's Gram form M_ik = <G A_i G^T, G A_k G^T> is manifestly symmetric PSD by
 * construction; route A relies on exact arithmetic for symmetry.  This probe
 * measures, on a fixture, whether route B yields a meaningfully better-conditioned
 * M (smaller kappa(M)) that would justify also threading the G-factor through b12
 * (we only have the symmetric W from b06 today).
 *
 * Metric: M from each route on a constructed (A_i, X, S); report max|M_A - M_B|
 * (should be ~0 -- same matrix) and kappa(M) = lam_max/lam_min via the point-mode
 * Jacobi eig.  Decision recorded in c/src/schur.c banner.
 *
 * Build (from repo root):
 *   gcc -std=c11 -O2 -g docs/probes/schur_nt_audition.c \
 *       c/src/linalg.c c/src/ntscaling.c c/src/svec.c \
 *       -Ic/include -lflint -lmpfr -lgmp -o /tmp/schur_nt_audition \
 *     && /tmp/schur_nt_audition
 *
 * One-off measurement harness; NOT compiled into libarbsdp.
 */

#include <stdio.h>
#include <stdlib.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/linalg.h"
#include "arbsdp/ntscaling.h"
#include "arbsdp/svec.h"

#define PREC 256

/* G-factor a la buildNtFactor: L=chol(S); MLXL = L^T X L; [V,sv2]=eigh(MLXL);
 * sv=sqrt(sv2); G = diag(sqrt(sv)) V^T L^{-1}.  Point mode. */
static void
build_G(arb_mat_t G, const arb_mat_t X, const arb_mat_t S, slong n, slong prec)
{
    arb_mat_t L, invL, Ssym, Xsym, tmp, MLXL, V, VtinvL;
    arb_ptr sv2;
    arb_t s;
    arb_mat_init(L, n, n);
    arb_mat_init(invL, n, n);
    arb_mat_init(Ssym, n, n);
    arb_mat_init(Xsym, n, n);
    arb_mat_init(tmp, n, n);
    arb_mat_init(MLXL, n, n);
    arb_mat_init(V, n, n);
    arb_mat_init(VtinvL, n, n);
    sv2 = _arb_vec_init(n);
    arb_init(s);

    arb_mat_set(Ssym, S);
    arbsdp_symmetrize(Ssym, prec);
    arb_mat_cho(L, Ssym, prec);             /* S = L L^T, lower */
    arb_mat_one(invL);
    arb_mat_solve_tril(invL, L, invL, 0, prec); /* invL = L^{-1} */

    arb_mat_set(Xsym, X);
    arbsdp_symmetrize(Xsym, prec);
    arb_mat_transpose(tmp, L);              /* tmp = L^T */
    arb_mat_mul(MLXL, tmp, Xsym, prec);     /* L^T X */
    arb_mat_mul(MLXL, MLXL, L, prec);       /* L^T X L */
    arbsdp_symmetrize(MLXL, prec);

    arbsdp_eigh(sv2, V, MLXL, prec);        /* MLXL = V diag(sv2) V^T */
    arb_mat_transpose(VtinvL, V);           /* V^T */
    arb_mat_mul(VtinvL, VtinvL, invL, prec);/* V^T L^{-1} */
    for (slong i = 0; i < n; i++) {
        arb_set(s, sv2 + i);
        arb_sqrt(s, s, prec);               /* sv_i */
        arb_sqrt(s, s, prec);               /* sqrt(sv_i) */
        for (slong j = 0; j < n; j++)
            arb_mul(arb_mat_entry(G, i, j), s, arb_mat_entry(VtinvL, i, j), prec);
    }

    arb_mat_clear(L);
    arb_mat_clear(invL);
    arb_mat_clear(Ssym);
    arb_mat_clear(Xsym);
    arb_mat_clear(tmp);
    arb_mat_clear(MLXL);
    arb_mat_clear(V);
    arb_mat_clear(VtinvL);
    _arb_vec_clear(sv2, n);
    arb_clear(s);
}

static double
mat_max_abs_diff(const arb_mat_t A, const arb_mat_t B)
{
    arb_t d;
    double worst = 0.0;
    arb_init(d);
    for (slong i = 0; i < arb_mat_nrows(A); i++)
        for (slong j = 0; j < arb_mat_ncols(A); j++) {
            arb_sub(d, arb_mat_entry(A, i, j), arb_mat_entry(B, i, j), PREC);
            arb_abs(d, d);
            double v = arf_get_d(arb_midref(d), ARF_RND_UP);
            if (v > worst) worst = v;
        }
    arb_clear(d);
    return worst;
}

/* kappa(M) = lam_max/lam_min via point-mode Jacobi eig (M symmetric PD). */
static double
cond_number(const arb_mat_t M, slong n)
{
    arb_mat_t Q;
    arb_ptr ev;
    double lo, hi;
    arb_mat_init(Q, n, n);
    ev = _arb_vec_init(n);
    arbsdp_eigh(ev, Q, M, PREC);  /* ascending eigenvalues */
    lo = arf_get_d(arb_midref(ev + 0), ARF_RND_DOWN);
    hi = arf_get_d(arb_midref(ev + (n - 1)), ARF_RND_UP);
    arb_mat_clear(Q);
    _arb_vec_clear(ev, n);
    return hi / lo;
}

int
main(void)
{
    const slong m = 3;
    const slong n = 3; /* single block */

    arb_mat_t X, S, W, G, Gt;
    arb_mat_t A[3];
    arb_mat_t WAk, GAk, MA, MB, T;
    arb_t v;

    arb_mat_init(X, n, n);
    arb_mat_init(S, n, n);
    arb_mat_init(W, n, n);
    arb_mat_init(G, n, n);
    arb_mat_init(Gt, n, n);
    arb_mat_init(WAk, n, n);
    arb_mat_init(GAk, n, n);
    arb_mat_init(MA, m, m);
    arb_mat_init(MB, m, m);
    arb_mat_init(T, n, n);
    arb_init(v);
    for (slong k = 0; k < m; k++) arb_mat_init(A[k], n, n);

    /* PD X, S (same fixtures as test_ntscaling). */
    arb_set_si(arb_mat_entry(X, 0, 0), 5);
    arb_set_si(arb_mat_entry(X, 1, 1), 4);
    arb_set_si(arb_mat_entry(X, 2, 2), 6);
    arb_set_si(arb_mat_entry(X, 0, 1), 1); arb_set_si(arb_mat_entry(X, 1, 0), 1);
    arb_set_si(arb_mat_entry(X, 0, 2), 2); arb_set_si(arb_mat_entry(X, 2, 0), 2);
    arb_set_si(arb_mat_entry(X, 1, 2), 1); arb_set_si(arb_mat_entry(X, 2, 1), 1);
    arb_set_si(arb_mat_entry(S, 0, 0), 6);
    arb_set_si(arb_mat_entry(S, 1, 1), 7);
    arb_set_si(arb_mat_entry(S, 2, 2), 4);
    arb_set_si(arb_mat_entry(S, 0, 1), 2); arb_set_si(arb_mat_entry(S, 1, 0), 2);
    arb_set_si(arb_mat_entry(S, 0, 2), 1); arb_set_si(arb_mat_entry(S, 2, 0), 1);
    arb_set_si(arb_mat_entry(S, 1, 2), -1); arb_set_si(arb_mat_entry(S, 2, 1), -1);

    /* A_0 = E_00, A_1 = E_11+ off-diag, A_2 = full symmetric (3 independent). */
    arb_set_si(arb_mat_entry(A[0], 0, 0), 1);
    arb_set_si(arb_mat_entry(A[1], 1, 1), 1);
    arb_set_si(arb_mat_entry(A[1], 0, 1), 1); arb_set_si(arb_mat_entry(A[1], 1, 0), 1);
    arb_set_si(arb_mat_entry(A[2], 2, 2), 1);
    arb_set_si(arb_mat_entry(A[2], 0, 2), 1); arb_set_si(arb_mat_entry(A[2], 2, 0), 1);
    arb_set_si(arb_mat_entry(A[2], 1, 2), 1); arb_set_si(arb_mat_entry(A[2], 2, 1), 1);

    arbsdp_nt_scaling(W, X, S, PREC);
    build_G(G, X, S, n, PREC);
    arb_mat_transpose(Gt, G);

    /* route A: M_ik = <A_i, W A_k W> */
    for (slong k = 0; k < m; k++) {
        arb_mat_mul(WAk, W, A[k], PREC);
        arb_mat_mul(WAk, WAk, W, PREC);
        arbsdp_symmetrize(WAk, PREC);
        for (slong i = 0; i < m; i++) {
            arbsdp_frob_inner(v, A[i], WAk, PREC);
            arb_set(arb_mat_entry(MA, i, k), v);
        }
    }
    /* route B: M_ik = <G A_i G^T, G A_k G^T> (Gram form) */
    {
        arb_mat_t GAi[3];
        for (slong k = 0; k < m; k++) arb_mat_init(GAi[k], n, n);
        for (slong k = 0; k < m; k++) {
            arb_mat_mul(GAk, G, A[k], PREC);
            arb_mat_mul(GAi[k], GAk, Gt, PREC);
            arbsdp_symmetrize(GAi[k], PREC);
        }
        for (slong i = 0; i < m; i++)
            for (slong k = 0; k < m; k++) {
                arbsdp_frob_inner(v, GAi[i], GAi[k], PREC);
                arb_set(arb_mat_entry(MB, i, k), v);
            }
        for (slong k = 0; k < m; k++) arb_mat_clear(GAi[k]);
    }

    /* verify W = G^T G (the audition's central claim) */
    arb_mat_mul(T, Gt, G, PREC);
    arbsdp_symmetrize(T, PREC);
    printf("max|W - G^T G|          = %.3e   (expect ~0: SAME symmetric W)\n",
           mat_max_abs_diff(W, T));
    printf("max|M_routeA - M_routeB| = %.3e   (expect ~0: SAME Schur matrix)\n",
           mat_max_abs_diff(MA, MB));
    printf("kappa(M) route A         = %.6e\n", cond_number(MA, m));
    printf("kappa(M) route B         = %.6e\n", cond_number(MB, m));

    arb_mat_clear(X); arb_mat_clear(S); arb_mat_clear(W);
    arb_mat_clear(G); arb_mat_clear(Gt);
    arb_mat_clear(WAk); arb_mat_clear(GAk);
    arb_mat_clear(MA); arb_mat_clear(MB); arb_mat_clear(T);
    arb_clear(v);
    for (slong k = 0; k < m; k++) arb_mat_clear(A[k]);
    flint_cleanup();
    return 0;
}
