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
 *   invert == 0:  f(lam) = sqrt(lam)         (matrix square root)
 *   invert != 0:  f(lam) = 1/sqrt(lam)       (matrix inverse square root)
 *
 * STRICT-PD GATE (CLAUDE.md rule 5; bead arb-prec-IPM-b30).  The prior port
 * mirrored PsdCone.ts's FLOORS (Math.max(0,lam) for sqrt, Math.max(1e-300,lam)
 * for invsqrt).  Those floors are float64 lifeboats that turn a non-PD or
 * radius-degraded input into a HUGE-but-finite garbage result rather than a
 * detectable failure -- a rule-5 violation in arbitrary precision.  Diagnosis
 * (bead b30): at low working precision the HSDE iterate's complementarity
 * measure mu -> 0 drives the Schur condition number ~1/mu (CLAUDE invariant 3);
 * the ball radius on the iterate's eigenvalues degrades ~30-60 bits per IPM
 * step until lambda_min's BALL straddles zero, at which point arb_rsqrt /
 * arb_inv return a NaN midpoint that the old floor masked into garbage.  The
 * matrices were still genuinely strictly PD (lambda_min midpoint > 0) but their
 * inverse-sqrt could no longer be computed to meaningful accuracy.
 *
 * The honest contract: succeed iff A is strictly PD at the working precision,
 * i.e. its smallest eigenvalue is clear of the precision-relative floor
 *     lambda_min  >  lambda_max * 2^(-prec/2).
 * (Justification: the inverse-sqrt loses ~log2(cond) = log2(lambda_max/lambda_min)
 * bits; floor at cond = 2^(prec/2) keeps at least prec/2 meaningful bits in the
 * result, well clear of the zero-straddling regime that produces NaN, yet far
 * below any genuinely-PD iterate the solver meets -- the b30 traces show
 * lambda_min ~ 1e-13 at prec=256 sitting ~25 orders of magnitude above this
 * floor, so a strictly-PD input is never spuriously rejected.)  The 2^(-prec)
 * scale is also defended against by the contains_zero check below.
 *
 * Returns 0 iff A is strictly PD (floor cleared) and `out` holds the requested
 * matrix power; returns 1 (NOT PD / accuracy-exhausted) WITHOUT writing a
 * garbage `out` (out is left as a finite, symmetric best-effort using the
 * floored eigenvalue, so callers that ignore the status still get a finite
 * matrix -- but the status is the source of truth).
 */
static int
psd_pow_half(arb_mat_t out, const arb_mat_t A, int invert, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t Q, QD, Qt;
    arb_ptr lam;
    arb_t d, floor_lam, lam_max;
    int not_pd = 0;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A));     /* square */
    assert(arb_mat_nrows(out) == n && arb_mat_ncols(out) == n);
    assert(out != A);                                  /* no aliasing */

    arb_mat_init(Q, n, n);
    arb_mat_init(QD, n, n);
    arb_mat_init(Qt, n, n);
    lam = _arb_vec_init(n);
    arb_init(d);
    arb_init(floor_lam);
    arb_init(lam_max);

    arbsdp_eigh(lam, Q, A, prec);   /* eigenvalues ASCENDING: lam[0]=min, lam[n-1]=max */

    /* Precision-relative strict-PD floor = max(0, lam_max) * 2^(-prec/2).
     * lam_max is the largest eigenvalue midpoint (lam[n-1]); if it is itself
     * non-positive the matrix is not PD (floor degenerates to 0, and the
     * lam_min <= 0 test below fires). */
    arb_get_mid_arb(lam_max, lam + (n - 1));
    if (arf_sgn(arb_midref(lam_max)) <= 0)
        arb_zero(floor_lam);
    else
        arb_mul_2exp_si(floor_lam, lam_max, -(prec / 2));

    /* not_pd iff lambda_min <= floor (midpoint test, POINT MODE) OR the
     * lambda_min ball straddles zero (arb_rsqrt/arb_inv would NaN -- the b30
     * mechanism, confirmed by /tmp/inv.c).  Eigenvalues are ascending so lam[0]
     * is the smallest.
     *
     * With epic.6 (c/src/solve.c:iterate_clear_radii) the iterate radii are
     * zeroed each IPM step, so in normal Layer-0 operation inputs are clean
     * midpoints and the arb_contains_zero branch below does NOT fire.  It is
     * kept as a defensive backstop for any future caller that passes ball inputs
     * (e.g. a Layer-1 path probing a perturbed matrix). */
    if (arf_cmp(arb_midref(lam + 0), arb_midref(floor_lam)) <= 0)
        not_pd = 1;
    if (arb_contains_zero(lam + 0))
        not_pd = 1;

    /* QD = Q * diag(f(lambda)) : scale column i of Q by f(lambda_i).  On the
     * not_pd path we still write a finite best-effort (floor the eigenvalue at
     * `floor_lam`, or eps if floor is zero) so `out` is never NaN -- but the
     * returned status is the contract callers must honor (rule 5). */
    arb_mat_set(QD, Q);
    for (slong i = 0; i < n; i++) {
        arb_set(d, lam + i);
        if (arf_cmp(arb_midref(d), arb_midref(floor_lam)) < 0)
            arb_set(d, floor_lam);
        if (arf_sgn(arb_midref(d)) <= 0)
            arb_set_d(d, 1e-300);              /* keep finite when floor==0 */
        if (!invert)
            arb_sqrt(d, d, prec);
        else
            arb_rsqrt(d, d, prec);
        for (slong r = 0; r < n; r++)
            arb_mul(arb_mat_entry(QD, r, i), arb_mat_entry(Q, r, i), d, prec);
    }

    /* out = QD * Q^T */
    arb_mat_transpose(Qt, Q);
    arb_mat_mul(out, QD, Qt, prec);
    arbsdp_symmetrize(out, prec); /* enforce exact symmetry (REUSE svec.h) */

    arb_clear(d);
    arb_clear(floor_lam);
    arb_clear(lam_max);
    _arb_vec_clear(lam, n);
    arb_mat_clear(Q);
    arb_mat_clear(QD);
    arb_mat_clear(Qt);
    return not_pd;
}

int
arbsdp_psd_sqrt(arb_mat_t out, const arb_mat_t A, slong prec)
{
    return psd_pow_half(out, A, 0, prec);
}

int
arbsdp_psd_invsqrt(arb_mat_t out, const arb_mat_t A, slong prec)
{
    return psd_pow_half(out, A, 1, prec);
}
