/*
 * src/certify.c -- Layer-1 (RIGOROUS, BALL-arithmetic) certification primitives.
 *
 * See include/arbsdp/certify.h for the API contract, the AUDITION rationale
 * (MATH_SPEC §5.3.1), and the references.  Everything here runs in Arb *ball*
 * arithmetic and produces THEOREMS (CLAUDE.md rule 1, invariant 2): the only
 * rigor source is verified Cholesky (arb_mat_cho), never eigendecomposition.
 *
 * Arb memory discipline (CLAUDE.md rule 7): every arb_init / arb_mat_init / arf_init
 * has a matching clear in the same scope.
 */

#include <assert.h>

#include <flint/arb.h>
#include <flint/arb_mat.h>
#include <flint/arf.h>
#include <flint/flint.h>

#include "arbsdp/certify.h"
#include "arbsdp/problem.h"
#include "arbsdp/iterate.h"
#include "arbsdp/linalg.h"   /* arbsdp_eigh: POINT-MODE top eigenvector (primal dir) */
#include "arbsdp/svec.h"

/* TAU_HEALTHY: the HSDE convergence threshold below which (tau, kappa) carry an
 * infeasibility certificate (b20), NOT a usable approximate optimum.  Kept in
 * lockstep with ARBSDP_TAU_HEALTHY in c/src/solve.c (the 6-flag convergence test,
 * solve.c:298); not exposed in a header, so mirrored here with this note (b19
 * TAU GUARD, certify.h).  If solve.c's value changes, change this too. */
#define ARBSDP_CERTIFY_TAU_HEALTHY 1e-6

/* ---------------------------------------------------------------------------
 * arbsdp_verified_psd
 *
 * Rump, BIT 2006, 46:433-452, Cor 2.4: a successfully completed (ball/interval)
 * Cholesky factorization of A is a PROOF that A is positive definite.  FLINT
 * 3.0.1 arb_mat.h:378: arb_mat_cho returns 1 iff the matrix is CERTAINLY PD over
 * the input ball enclosures.  A return of 0 is NOT a counter-proof -- it means
 * indefinite/semidefinite OR precision-starved (CLAUDE.md rule 5: a 0 is the
 * honest "not provably PD", never a silent floor).
 *
 * arb_mat_cho reads only the lower triangle, so we symmetrize a scratch copy
 * first (a non-symmetric A could otherwise factor on a triangle that does not
 * represent the symmetric matrix the caller means).
 * ------------------------------------------------------------------------- */
int
arbsdp_verified_psd(const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t S, L;
    int ok;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A) && "verified_psd: A must be square");

    arb_mat_init(S, n, n);
    arb_mat_init(L, n, n);

    arb_mat_set(S, A);
    arbsdp_symmetrize(S, prec);

    ok = arb_mat_cho(L, S, prec);

    arb_mat_clear(S);
    arb_mat_clear(L);
    return ok;
}

/* ---------------------------------------------------------------------------
 * arbsdp_gershgorin_lower_bound
 *
 * Gershgorin disk theorem (1931): every eigenvalue of A lies in some disk
 *     |z - A_ii| <= R_i,   R_i = sum_{j!=i} |A_ij|,
 * hence lambda_min(A) >= min_i ( A_ii - R_i ).  For a rigorous LOWER bound on
 * the true matrix enclosed by the input balls we take the diagonal at its LOWER
 * endpoint, each off-diagonal magnitude at its UPPER endpoint, and round the
 * arithmetic conservatively DOWNWARD (arf directed rounding): the result is a
 * proven lower bound regardless of where the truth sits inside the balls.
 *
 * Note A need NOT be symmetric here (we form |A_ij| per off-diagonal entry); for
 * the certify use case the caller passes a symmetrized Z, but Gershgorin is valid
 * for any matrix with real spectrum and we keep it self-contained.
 * ------------------------------------------------------------------------- */
void
arbsdp_gershgorin_lower_bound(arb_t out, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arf_t best;       /* running min over rows */
    arf_t diag_lo;    /* lbound(A_ii) */
    arf_t absu;       /* ubound(|A_ij|) */
    arf_t row_sum;    /* sum_{j!=i} ubound(|A_ij|), rounded UP (overestimate) */
    arf_t row_lb;     /* diag_lo - row_sum, rounded DOWN */
    arb_t tmp;        /* scratch for |A_ij| */
    int first = 1;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A) && "gershgorin: A must be square");

    arf_init(best);
    arf_init(diag_lo);
    arf_init(absu);
    arf_init(row_sum);
    arf_init(row_lb);
    arb_init(tmp);

    for (slong i = 0; i < n; i++) {
        /* sum of off-diagonal magnitudes in row i, each at its UPPER endpoint,
         * accumulated with CEIL rounding so the subtracted radius is never
         * underestimated. */
        arf_zero(row_sum);
        for (slong j = 0; j < n; j++) {
            if (j == i)
                continue;
            arb_abs(tmp, arb_mat_entry(A, i, j));
            arb_get_ubound_arf(absu, tmp, prec);                 /* >= |A_ij| */
            arf_add(row_sum, row_sum, absu, prec, ARF_RND_CEIL); /* overestimate */
        }

        /* diagonal at its LOWER endpoint (underestimate the center). */
        arb_get_lbound_arf(diag_lo, arb_mat_entry(A, i, i), prec);

        /* row_lb = diag_lo - row_sum, rounded DOWN (FLOOR) -> true lower bound. */
        arf_sub(row_lb, diag_lo, row_sum, prec, ARF_RND_FLOOR);

        if (first) {
            arf_set(best, row_lb);
            first = 0;
        } else {
            arf_min(best, best, row_lb);
        }
    }

    /* `best` is a proven lower bound on lambda_min(A); store it as an exact arb
     * point (zero radius) -- its value already satisfies best <= lambda_min. */
    arb_set_arf(out, best);

    arf_clear(best);
    arf_clear(diag_lo);
    arf_clear(absu);
    arf_clear(row_sum);
    arf_clear(row_lb);
    arb_clear(tmp);
}

/* ---------------------------------------------------------------------------
 * shift_mat -- B <- symmetrize(A) - s*I, with s subtracted at its UPPER endpoint.
 *
 * Subtracting upper(s) (not s itself) keeps the rigor one-directional: if
 * arb_mat_cho(B) then B = A - upper(s)*I >= 0, so lambda_min(A) >= upper(s) >= s.
 * A larger-than-s shift makes the success a STRONGER claim, never a weaker one,
 * so the returned bound (a successful s) is always valid for the true s.
 *
 * B must be an initialized n x n arb_mat; A square (asserted by symmetrize).
 * ------------------------------------------------------------------------- */
static void
shift_mat(arb_mat_t B, const arb_mat_t A, const arb_t s, slong prec)
{
    slong n = arb_mat_nrows(A);
    arf_t su;
    arf_init(su);
    arb_mat_set(B, A);
    arbsdp_symmetrize(B, prec);
    arb_get_ubound_arf(su, s, prec);             /* upper endpoint of the shift */
    for (slong i = 0; i < n; i++)
        arb_sub_arf(arb_mat_entry(B, i, i), arb_mat_entry(B, i, i), su, prec);
    arf_clear(su);
}

/* gershgorin_upper -- cheap scale estimate ||A||_approx = max_i (ubound(A_ii) + R_i),
 * R_i = sum_{j!=i} ubound(|A_ij|) (Gershgorin upper radius).  Used only to set the
 * bisection tolerance tol = 2^(-prec/2) * max(1, ||A||_approx); it need not be
 * rigorous (it scales a STOPPING tolerance, not the returned bound).  Stored into
 * the arf `out`. */
static void
gershgorin_upper(arf_t out, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arf_t row, u, diag_u;
    arb_t tmp;
    int first = 1;
    arf_init(row);
    arf_init(u);
    arf_init(diag_u);
    arb_init(tmp);
    arf_zero(out);
    for (slong i = 0; i < n; i++) {
        arf_zero(row);
        for (slong j = 0; j < n; j++) {
            if (j == i)
                continue;
            arb_abs(tmp, arb_mat_entry(A, i, j));
            arb_get_ubound_arf(u, tmp, prec);
            arf_add(row, row, u, prec, ARF_RND_CEIL);
        }
        arb_get_ubound_arf(diag_u, arb_mat_entry(A, i, i), prec);
        arf_abs(diag_u, diag_u);
        arf_add(row, diag_u, row, prec, ARF_RND_CEIL);
        if (first) { arf_set(out, row); first = 0; }
        else       { arf_max(out, out, row); }
    }
    arf_clear(row);
    arf_clear(u);
    arf_clear(diag_u);
    arb_clear(tmp);
}

/* g(s) = "arb_mat_cho(symmetrize(A) - s*I) succeeds" -- the MONOTONE-DECREASING
 * rigorous predicate (Rump 2006).  A 1 PROVES lambda_min(A) >= s. */
static int
cho_passes(const arb_mat_t A, const arb_t s, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t B, L;
    int ok;
    arb_mat_init(B, n, n);
    arb_mat_init(L, n, n);
    shift_mat(B, A, s, prec);
    ok = arb_mat_cho(L, B, prec);
    arb_mat_clear(B);
    arb_mat_clear(L);
    return ok;
}

/* ---------------------------------------------------------------------------
 * arbsdp_lambda_min_lower_bound -- verified-Cholesky shift + bisection.
 * MATH_SPEC §5.3 / §5.3.1; Rump BIT 2006 Cor 2.4; Higham-Strabic-Sego 2016.
 *
 * Seed the Gershgorin floor, probe s=0, establish the bracket invariant
 * [s_lo: chol PASSES, s_hi: chol FAILS], then BISECT the monotone predicate to
 * tol = 2^(-prec/2) * max(1, ||A||_approx).  Return s_lo (the last shift with a
 * successful ball Cholesky), so d = s_lo <= lambda_min(A) is a THEOREM.
 *
 * ITER_CAP: bisection terminates on width <= tol (the real stopping rule); the
 * cap is a fail-loud safety net (CLAUDE.md rule 5).  Reaching tol = 2^(-prec/2)
 * from an O(1) bracket needs ~prec/2 halvings, so a FIXED 60 (MATH_SPEC §5.3.1's
 * sketch) would hard-stop at ~2^-60 BEFORE tol for prec > 120, leaving the bound
 * loose -- a stall, not convergence.  The cap therefore scales with prec
 * (max(60, prec/2 + 32)) so the tol-driven stop is the one that fires; 60 is kept
 * as the floor for very low precisions.  (Deviation from the literal "=60" in the
 * sketch, justified: the sketch's tol formula and a fixed 60 are mutually
 * inconsistent for prec=256; the binding TOL formula wins, and the cap must let
 * the loop reach it.) */
static slong
lmin_iter_cap(slong prec)
{
    slong c = prec / 2 + 32;
    return c < 60 ? 60 : c;
}

void
arbsdp_lambda_min_lower_bound(arb_t d, const arb_mat_t A, slong prec)
{
    arb_t s_lo;     /* chol(A - s_lo*I) PASSES => lambda_min(A) >= s_lo (proof) */
    arb_t s_hi;     /* chol(A - s_hi*I) FAILS  (upper fence on the bracket)     */
    arb_t s_try;    /* growth probe                                             */
    arb_t s_mid;    /* bisection midpoint                                       */
    arb_t width;    /* s_hi - s_lo                                              */
    arb_t tol;      /* stopping tolerance                                       */
    arb_t gersh;    /* Gershgorin floor: always-valid lower bound on lambda_min */
    arb_t abs_g;
    arf_t scale;    /* ||A||_approx (Gershgorin upper)                          */

    assert(arb_mat_nrows(A) == arb_mat_ncols(A) && "lambda_min: A must be square");

    arb_init(s_lo);
    arb_init(s_hi);
    arb_init(s_try);
    arb_init(s_mid);
    arb_init(width);
    arb_init(tol);
    arb_init(gersh);
    arb_init(abs_g);
    arf_init(scale);

    /* SEED: the Gershgorin floor is a guaranteed-valid lower bound on lambda_min
     * and the safe return value of last resort. */
    arbsdp_gershgorin_lower_bound(gersh, A, prec);

    /* PROBE s = 0. */
    arb_zero(s_try);
    if (cho_passes(A, s_try, prec)) {
        /* A is (provably) PD: lambda_min >= 0.  s_lo = 0; GROW s upward until a
         * failure gives s_hi.  Growth start = max(1, |gershgorin|), doubling. */
        arb_zero(s_lo);
        arb_abs(abs_g, gersh);
        arb_set_si(s_try, 1);
        if (arb_gt(abs_g, s_try))            /* s_try = max(1, |gershgorin|) */
            arb_set(s_try, abs_g);

        /* Double s_try until chol(A - s_try*I) fails -> that is s_hi.  Cap the
         * growth (a generous fence; lambda_min <= max diagonal <= |A| which is
         * bounded by the Gershgorin spread, so this terminates quickly). */
        {
            int found_hi = 0;
            for (int k = 0; k < 200; k++) {
                if (!cho_passes(A, s_try, prec)) {
                    arb_set(s_hi, s_try);
                    found_hi = 1;
                    break;
                }
                arb_set(s_lo, s_try);        /* s_try still passes: advance s_lo */
                arb_mul_2exp_si(s_try, s_try, 1); /* double */
            }
            /* Mathematically a failure MUST appear (lambda_min is finite); the
             * cap is only a fail-loud guard (CLAUDE.md rule 5). */
            assert(found_hi && "lambda_min: PD-growth never found a failing shift");
        }
    } else {
        /* s = 0 FAILS: A is indefinite or on the PSD boundary (lambda_min <= 0).
         * s_hi = 0 (fails); s_lo = Gershgorin floor.  The floor MUST pass a ball
         * Cholesky (gershgorin <= lambda_min, so A - gershgorin*I >= 0); if it
         * fails it is precision starvation -> retry once at prec*3/2; if it still
         * fails, RETURN the floor (rigorous, just loose). */
        arb_zero(s_hi);
        arb_set(s_lo, gersh);

        if (!cho_passes(A, s_lo, prec)) {
            if (!cho_passes(A, s_lo, prec * 3 / 2)) {
                /* Precision-starved even at the Gershgorin floor: the floor is
                 * still a rigorous lower bound (Gershgorin is unconditional), so
                 * return it directly. */
                arb_set(d, gersh);
                goto done;
            }
        }
    }

    /* BISECTION on the monotone predicate g(s) = "chol(A - s*I) succeeds".
     * tol = 2^(-prec/2) * max(1, ||A||_approx).  Invariant: chol passes at s_lo,
     * fails at s_hi, s_lo <= s_hi.  Each iteration halves (s_hi - s_lo). */
    gershgorin_upper(scale, A, prec);
    {
        arf_t one;
        arf_init(one);
        arf_one(one);
        arf_max(scale, scale, one);          /* max(1, ||A||_approx) */
        arf_clear(one);
    }
    arb_set_arf(tol, scale);
    arb_mul_2exp_si(tol, tol, -(prec / 2));  /* tol = scale * 2^(-prec/2) */

    {
    slong iter_cap = lmin_iter_cap(prec);
    for (slong iter = 0; iter < iter_cap; iter++) {
        arb_sub(width, s_hi, s_lo, prec);
        if (arb_le(width, tol))
            break;

        arb_add(s_mid, s_lo, s_hi, prec);
        arb_mul_2exp_si(s_mid, s_mid, -1);   /* s_mid = (s_lo + s_hi)/2 */

        if (cho_passes(A, s_mid, prec)) {
            arb_set(s_lo, s_mid);            /* proof lambda_min >= s_mid */
        } else {
            /* precision-starvation guard (CLAUDE.md rule 5): a failure near the
             * boundary may be the prec, not indefiniteness -- retry once higher
             * before trusting it.  Only a CONFIRMED failure moves s_hi (keeping
             * the bracket, hence the returned bound, rigorous). */
            if (cho_passes(A, s_mid, prec * 3 / 2))
                arb_set(s_lo, s_mid);
            else
                arb_set(s_hi, s_mid);
        }
    }
    }

    /* d = s_lo: the last shift with a SUCCESSFUL ball Cholesky -> d <= lambda_min(A). */
    arb_set(d, s_lo);

done:
    arb_clear(s_lo);
    arb_clear(s_hi);
    arb_clear(s_try);
    arb_clear(s_mid);
    arb_clear(width);
    arb_clear(tol);
    arb_clear(gersh);
    arb_clear(abs_g);
    arf_clear(scale);
}

/* ---------------------------------------------------------------------------
 * arbsdp_dual_residual -- per-block dual residual Z[b] = C[b] - sum_i y_i A_i[b]
 * in BALL arithmetic (bead arb-prec-IPM-b18).
 *
 * Jansson-Chaykin-Keil 2007 (SIAM J. Numer. Anal. 46(1):180-200): the Layer-1
 * lower bound rests on the dual slack Z = C - sum_i y_i A_i and its lambda_min
 * floor.  Z must be a rigorous ENCLOSURE (CLAUDE.md rule 2): its ball must
 * contain the exact residual for the supplied y.
 *
 * Materialization REUSE (no re-parse; same helper as iterate.c::apply_At):
 * arbsdp_problem_block_mat(., p, matno, b, prec) builds the symmetric block of
 * matno (0 = C_file, 1..m = A_matno) at `prec`, each entry a correctly-rounded
 * ball of the stored decimal (problem.h banner).
 *
 * The linear combination is accumulated PER ENTRY with arb_dot (one rounding for
 * the whole length-m dot product, the tightest rigorous enclosure FLINT offers):
 *     Z[b][r,c] = C[b][r,c] - sum_{i=0}^{m-1} y[i] * A_{i+1}[b][r,c].
 * arb_dot(res, init=C, subtract=1, x=y, y=avec, len=m) returns
 *     res = C - sum y[i]*avec[i]
 * with the radius bounding all rounding (FLINT arb_dot contract).
 *
 * SIGN-AGNOSTIC (REPORTED to b19): C is taken at matno 0 with the FILE sign (NOT
 * the solver's C_int = -C_file), and the A_i at matno 1..m with the file sign,
 * exactly matching arbsdp_apply_At.  This routine does not choose/purify the sign
 * of y -- that is b19's decision (certify.h SIGN note).
 *
 * Memory (CLAUDE.md rule 7): Cblk + the m A-blocks + the length-m gather vector
 * are scoped here and cleared; the caller owns Z's init/clear.
 * ------------------------------------------------------------------------- */
void
arbsdp_dual_residual(arb_mat_t *Z, const arbsdp_problem *p, arb_srcptr y,
                     slong prec)
{
    int m, nb;

    assert(Z != NULL && "dual_residual: Z must be non-NULL");
    assert(p != NULL && "dual_residual: p must be non-NULL");
    /* y may be NULL only if m == 0 (no linear combination terms). */

    m = p->m;
    nb = p->nblocks;
    assert((m == 0 || y != NULL) && "dual_residual: y must be non-NULL when m>0");

    for (int b = 0; b < nb; b++) {
        slong n = p->block_sizes[b] < 0 ? -p->block_sizes[b]
                                        : p->block_sizes[b];
        arb_mat_t Cblk;
        arb_mat_struct *Ablk = NULL;   /* m materialized A_i[b] blocks */
        arb_ptr avec = NULL;           /* length-m gather of A_i[b][r,c] */

        assert(arb_mat_nrows(Z[b]) == n && arb_mat_ncols(Z[b]) == n
               && "dual_residual: Z[b] must be n x n");

        arb_mat_init(Cblk, n, n);
        arbsdp_problem_block_mat(Cblk, p, 0, b, prec);   /* C_file block (matno 0) */

        if (m > 0) {
            Ablk = flint_malloc((size_t) m * sizeof *Ablk);
            avec = _arb_vec_init(m);
            for (int i = 0; i < m; i++) {
                arb_mat_init(&Ablk[i], n, n);
                arbsdp_problem_block_mat(&Ablk[i], p, i + 1, b, prec); /* A_{i+1} */
            }
        }

        /* Z[b][r,c] = C[b][r,c] - sum_i y[i] * A_{i+1}[b][r,c], one rounding per
         * entry (arb_dot). */
        for (slong r = 0; r < n; r++) {
            for (slong c = 0; c < n; c++) {
                if (m > 0) {
                    for (int i = 0; i < m; i++)
                        arb_set(&avec[i], arb_mat_entry(&Ablk[i], r, c));
                    arb_dot(arb_mat_entry(Z[b], r, c),
                            arb_mat_entry(Cblk, r, c), /* initial = C[b][r,c] */
                            1,                          /* subtract the dot product */
                            y, 1, avec, 1, m, prec);
                } else {
                    arb_set(arb_mat_entry(Z[b], r, c), arb_mat_entry(Cblk, r, c));
                }
            }
        }

        /* Downstream verified-Cholesky / lambda_min read only the lower triangle
         * (certify.c above); make Z[b] exactly symmetric.  C and the A_i are each
         * symmetric (block_mat writes A_{ij}=A_{ji}), so this only re-rounds. */
        arbsdp_symmetrize(Z[b], prec);

        arb_mat_clear(Cblk);
        if (m > 0) {
            for (int i = 0; i < m; i++)
                arb_mat_clear(&Ablk[i]);
            flint_free(Ablk);
            _arb_vec_clear(avec, m);
        }
    }
}

/* ---------------------------------------------------------------------------
 * arbsdp_apriori -- a-priori bound storage (bead arb-prec-IPM-b18).
 *
 * CLAUDE.md invariant 5 (NO silent boundedness): the SET/UNSET distinction is the
 * *_set flag, NOT a sentinel value -- so an explicitly-set huge xbar[b] is a
 * FINITE bound, distinct from "unset" (which b19 reads as +inf -> lb = -inf for
 * that block).  This module only STORES; b19 consumes.
 *
 * Memory (CLAUDE.md rule 7): init allocates xbar (length nblocks), xbar_set
 * (length nblocks), and ybar; clear frees all three and re-zeros.
 * ------------------------------------------------------------------------- */
void
arbsdp_apriori_init(arbsdp_apriori *ab, int nblocks)
{
    assert(ab != NULL && "apriori_init: ab must be non-NULL");
    assert(nblocks >= 0 && "apriori_init: nblocks must be >= 0");

    ab->nblocks = nblocks;
    ab->xbar = (nblocks > 0) ? _arb_vec_init(nblocks) : NULL;
    ab->xbar_set = (nblocks > 0)
                       ? flint_calloc((size_t) nblocks, sizeof *ab->xbar_set)
                       : NULL;
    arb_init(ab->ybar);
    ab->ybar_set = 0;
}

void
arbsdp_apriori_clear(arbsdp_apriori *ab)
{
    assert(ab != NULL && "apriori_clear: ab must be non-NULL");

    if (ab->xbar != NULL)
        _arb_vec_clear(ab->xbar, ab->nblocks);
    flint_free(ab->xbar_set);   /* flint_free(NULL) is a no-op */
    arb_clear(ab->ybar);

    ab->nblocks = 0;
    ab->xbar = NULL;
    ab->xbar_set = NULL;
    ab->ybar_set = 0;
}

void
arbsdp_apriori_set_xbar(arbsdp_apriori *ab, int b, const arb_t value)
{
    assert(ab != NULL && "apriori_set_xbar: ab must be non-NULL");
    assert(b >= 0 && b < ab->nblocks && "apriori_set_xbar: b out of range");

    arb_set(&ab->xbar[b], value);
    ab->xbar_set[b] = 1;
}

void
arbsdp_apriori_set_ybar(arbsdp_apriori *ab, const arb_t value)
{
    assert(ab != NULL && "apriori_set_ybar: ab must be non-NULL");

    arb_set(ab->ybar, value);
    ab->ybar_set = 1;
}

/* ===========================================================================
 * The rigorous two-sided bracket [lb, ub] (bead arb-prec-IPM-b19).
 * MATH_SPEC §5.4 (lower) + §5.5 (upper, UB-A).  THE PROJECT'S PRODUCT.
 *
 * All BALL arithmetic (CLAUDE.md rule 1, invariant 2): a THEOREM about the true
 * optimum p*_ext = max <C_file, X>.  Memory (rule 7): scoped temporaries cleared.
 * ======================================================================== */

/* ---------------------------------------------------------------------------
 * arbsdp_lambda_max_upper_bound -- rigorous upper bound dhi >= lambda_max(A) via
 * lambda_max(A) = -lambda_min(-A) (MATH_SPEC §5.5; certify.h contract).
 *
 * A successful ball Cholesky of (-A - s*I) proves lambda_min(-A) >= s, i.e.
 * lambda_max(A) <= -s; the b17 kernel returns the largest such s as d_lo(-A), so
 * out = -d_lo(-A) >= lambda_max(A) is a THEOREM (Rump 2006 Cor 2.4).  We negate A
 * into a scratch matrix (NOT in place -- A is const), run the shipped b17 kernel,
 * and negate its result.
 * ------------------------------------------------------------------------- */
void
arbsdp_lambda_max_upper_bound(arb_t out, const arb_mat_t A, slong prec)
{
    slong n = arb_mat_nrows(A);
    arb_mat_t negA;
    arb_t dlo;

    assert(arb_mat_nrows(A) == arb_mat_ncols(A)
           && "lambda_max_upper: A must be square");

    arb_mat_init(negA, n, n);
    arb_init(dlo);

    arb_mat_neg(negA, A);                          /* negA = -A (rigorous) */
    arbsdp_lambda_min_lower_bound(dlo, negA, prec);/* dlo <= lambda_min(-A) */
    arb_neg(out, dlo);                             /* out = -dlo >= lambda_max(A) */

    arb_mat_clear(negA);
    arb_clear(dlo);
}

/* ---------------------------------------------------------------------------
 * assemble_bty -- out <- b^T y_ext in BALL arithmetic (rigorous enclosure).
 *
 * b is materialized from the problem (FILE sign; arbsdp_problem_b) and the dot
 * product is accumulated with arb_dot (one rounding for the whole length-m sum,
 * the tightest rigorous enclosure FLINT offers).  m may be 0 (-> out = 0).
 * ------------------------------------------------------------------------- */
static void
assemble_bty(arb_t out, const arbsdp_problem *p, arb_srcptr y_ext, slong prec)
{
    int m = p->m;
    if (m == 0) {
        arb_zero(out);
        return;
    }
    {
        arb_ptr bvec = _arb_vec_init(m);
        for (int i = 0; i < m; i++) {
            int rc = arbsdp_problem_b(&bvec[i], p, i, prec);
            assert(rc == 0 && "assemble_bty: b_i failed to parse");
            (void) rc;
        }
        /* out = sum_i b[i] * y_ext[i], one rounding (arb_dot, subtract = 0). */
        arb_dot(out, NULL, 0, bvec, 1, y_ext, 1, m, prec);
        _arb_vec_clear(bvec, m);
    }
}

/* ---------------------------------------------------------------------------
 * arbsdp_lower_bound -- Jansson lower bound (MATH_SPEC §5.4):
 *     lb = b^T y_ext + sum_b min(0, dlo_b) * xbar_b,   dlo_b <= lambda_min(Z_ext^b).
 *
 * HONEST -inf (CLAUDE.md invariant 5): a block with dlo_b < 0 AND xbar_set[b]==0
 * makes lb = -inf (that block's tr(X_b) is a-priori unbounded so its penalty is
 * unbounded below).  A block with dlo_b >= 0 contributes 0 regardless of xbar
 * (min(0,dlo_b) = 0), so an UNSET xbar there is harmless -- only a NEGATIVE
 * dlo_b needs a finite xbar.  Z_ext is built once for all blocks via
 * arbsdp_dual_residual (SAME y_ext as the upper bound; §5.5 sign contract).
 * ------------------------------------------------------------------------- */
void
arbsdp_lower_bound(arb_t lb, const arbsdp_problem *p, arb_srcptr y_ext,
                   const arbsdp_apriori *ab, slong prec)
{
    int nb;

    assert(p != NULL && ab != NULL && "lower_bound: p, ab non-NULL");
    assert(ab->nblocks == p->nblocks && "lower_bound: ab->nblocks != p->nblocks");

    nb = p->nblocks;

    /* Z_ext^b = C_file^b - sum_i (y_ext)_i A_i^b (ball; arbsdp_dual_residual).
     * Z is an array of nb arb_mat_t -- the established convention (iterate.c R_d,
     * test_certify.c): flint_malloc(nb * sizeof *Z), arb_mat_init(Z[b], ..). */
    {
        arb_mat_t *Z = flint_malloc((size_t) nb * sizeof *Z);
        arb_t acc;     /* running b^T y_ext + sum_b min(0,dlo_b)*xbar_b */
        arb_t dlo;     /* dlo_b <= lambda_min(Z_ext^b) */
        arb_t term;    /* min(0,dlo_b) * xbar_b */
        arb_t zero;
        int is_neg_inf = 0;

        for (int b = 0; b < nb; b++) {
            slong sz = p->block_sizes[b] < 0 ? -p->block_sizes[b]
                                             : p->block_sizes[b];
            arb_mat_init(Z[b], sz, sz);
        }
        arbsdp_dual_residual(Z, p, y_ext, prec);

        arb_init(acc);
        arb_init(dlo);
        arb_init(term);
        arb_init(zero);
        arb_zero(zero);

        assemble_bty(acc, p, y_ext, prec);         /* acc <- b^T y_ext */

        for (int b = 0; b < nb && !is_neg_inf; b++) {
            arbsdp_lambda_min_lower_bound(dlo, Z[b], prec);
            /* term = min(0, dlo_b). */
            arb_min(term, dlo, zero, prec);
            if (!arb_is_zero(term)) {
                /* a strictly-negative penalty: needs a finite a-priori xbar. */
                if (!ab->xbar_set[b]) {
                    is_neg_inf = 1;                 /* HONEST -inf (invariant 5) */
                    break;
                }
                arb_mul(term, term, &ab->xbar[b], prec);
                arb_add(acc, acc, term, prec);
            }
            /* dlo_b >= 0 -> min(0,dlo_b) = 0 -> no contribution, no xbar needed. */
        }

        if (is_neg_inf)
            arb_neg_inf(lb);
        else
            arb_set(lb, acc);

        for (int b = 0; b < nb; b++)
            arb_mat_clear(Z[b]);
        flint_free(Z);
        arb_clear(acc);
        arb_clear(dlo);
        arb_clear(term);
        arb_clear(zero);
    }
}

/* ---------------------------------------------------------------------------
 * arbsdp_upper_bound -- dual-side upper bound (MATH_SPEC §5.5, UB-A):
 *     ub = b^T y_ext + sum_b max(0, dhi_b) * xbar_b,   dhi_b >= lambda_max(Z_ext^b).
 *
 * Exact dual-side MIRROR of arbsdp_lower_bound (SAME y_ext, SAME Z_ext): swap
 * lambda_min -> lambda_max (via arbsdp_lambda_max_upper_bound), min(0,.) ->
 * max(0,.), and -inf -> +inf for the honest unbounded case.
 * ------------------------------------------------------------------------- */
void
arbsdp_upper_bound(arb_t ub, const arbsdp_problem *p, arb_srcptr y_ext,
                   const arbsdp_apriori *ab, slong prec)
{
    int nb;

    assert(p != NULL && ab != NULL && "upper_bound: p, ab non-NULL");
    assert(ab->nblocks == p->nblocks && "upper_bound: ab->nblocks != p->nblocks");

    nb = p->nblocks;

    {
        arb_mat_t *Z = flint_malloc((size_t) nb * sizeof *Z);
        arb_t acc;
        arb_t dhi;     /* dhi_b >= lambda_max(Z_ext^b) */
        arb_t term;    /* max(0,dhi_b) * xbar_b */
        arb_t zero;
        int is_pos_inf = 0;

        for (int b = 0; b < nb; b++) {
            slong sz = p->block_sizes[b] < 0 ? -p->block_sizes[b]
                                             : p->block_sizes[b];
            arb_mat_init(Z[b], sz, sz);
        }
        arbsdp_dual_residual(Z, p, y_ext, prec);

        arb_init(acc);
        arb_init(dhi);
        arb_init(term);
        arb_init(zero);
        arb_zero(zero);

        assemble_bty(acc, p, y_ext, prec);         /* acc <- b^T y_ext */

        for (int b = 0; b < nb && !is_pos_inf; b++) {
            arbsdp_lambda_max_upper_bound(dhi, Z[b], prec);
            /* term = max(0, dhi_b). */
            arb_max(term, dhi, zero, prec);
            if (!arb_is_zero(term)) {
                if (!ab->xbar_set[b]) {
                    is_pos_inf = 1;                 /* HONEST +inf (invariant 5) */
                    break;
                }
                arb_mul(term, term, &ab->xbar[b], prec);
                arb_add(acc, acc, term, prec);
            }
        }

        if (is_pos_inf)
            arb_pos_inf(ub);
        else
            arb_set(ub, acc);

        for (int b = 0; b < nb; b++)
            arb_mat_clear(Z[b]);
        flint_free(Z);
        arb_clear(acc);
        arb_clear(dhi);
        arb_clear(term);
        arb_clear(zero);
    }
}

/* ---------------------------------------------------------------------------
 * block_rayleigh_lb -- the per-block rigorous Rayleigh lower bound
 *     lb_b = beta_b * R(v_b),   R(v_b) = (v_b^T C_b v_b)/(v_b^T v_b)
 * where v_b is the midpoint of the top (largest-eigenvalue) eigenvector of the
 * solver's primal block X_b.  This is exactly the proven vry inner computation,
 * parameterized over (C_b, beta_b, X_b).  Returns 1 and sets lb_b on success; 0
 * (lb_b untouched) iff the den = v_b^T v_b lower-endpoint guard trips (rule 5).
 *
 * Parlett 1998 ("The Symmetric Eigenvalue Problem", Rayleigh quotient / Courant-
 * Fischer): R(v) <= lambda_max(C_b) for ANY v != 0.  RIGOR (rule 2): the quotient
 * is computed in BALL arithmetic, so lb_b is a true enclosure and its lower
 * endpoint <= beta_b*lambda_max(C_b) for the FIXED exact v_b (assuming beta_b>=0,
 * which the caller's detector guarantees).  No eig rigor, no PSD certification of
 * X (the rank-1 block is PSD by construction; invariant 1).  v_b is the POINT-MODE
 * top eigenvector FROZEN to an exact midpoint vector -- a direction only, never a
 * rigor dependency (a poor v_b merely loosens lb_b).
 *
 * Memory (rule 7): every init/_arb_vec_init/arb_mat_init has a matching clear.
 * ------------------------------------------------------------------------- */
static int
block_rayleigh_lb(arb_t lb_b, const arb_mat_t C_b, const arb_t beta_b,
                  const arb_mat_t X_b, slong prec)
{
    slong     n = arb_mat_nrows(C_b);
    arb_mat_t Q;        /* eigenvectors (POINT MODE); top vector = last col      */
    arb_ptr   eigvals;  /* length n (ASCENDING; unused except via the col)       */
    arb_ptr   v;        /* exact (zero-radius) top eigenvector                   */
    arb_ptr   Cv;       /* C_b v (ball)                                          */
    arb_t     num;      /* v^T C_b v (ball)                                      */
    arb_t     den;      /* v^T v     (ball)                                      */
    arb_t     ray;      /* num / den                                            */
    arf_t     den_lo;   /* lower endpoint of den (divisibility guard)            */
    int       ok = 1;   /* set 0 if the den guard trips                          */

    assert(arb_mat_ncols(C_b) == n && arb_mat_nrows(X_b) == n
           && arb_mat_ncols(X_b) == n && "block_rayleigh_lb: square, matched");

    arb_mat_init(Q, n, n);
    eigvals = _arb_vec_init(n);
    v  = _arb_vec_init(n);
    Cv = _arb_vec_init(n);
    arb_init(num);
    arb_init(den);
    arb_init(ray);
    arf_init(den_lo);

    /* POINT-MODE top eigenvector of the solver's primal block (direction only;
     * NOT a rigor dependency).  arbsdp_eigh: eigenvalues ASCENDING, column j of Q
     * is the unit eigenvector for eigvals[j] -> TOP vector is column n-1.  Freeze
     * it to an EXACT (zero-radius) midpoint vector: the bound is rigorous for ANY
     * FIXED v, and using the midpoint stops the point-mode eigenvector's radius
     * from widening the result while staying sound. */
    arbsdp_eigh(eigvals, Q, X_b, prec);
    for (slong i = 0; i < n; i++)
        arb_get_mid_arb(v + i, arb_mat_entry(Q, i, n - 1));

    /* BALL arithmetic for the rigorous quotient.  Cv_i = sum_j C_b[i][j] v_j via
     * arb_dot (one rounding per row), then num = sum_i v_i (Cv)_i and den =
     * sum_i v_i^2, each one arb_dot (tightest rigorous enclosure). */
    for (slong i = 0; i < n; i++) {
        arb_dot(Cv + i, NULL, 0,
                C_b->rows[i], 1,   /* row i of C_b: entries C_b[i][0..n-1]       */
                v, 1, n, prec);
    }
    arb_dot(num, NULL, 0, v, 1, Cv, 1, n, prec);   /* num = v^T C_b v            */
    arb_dot(den, NULL, 0, v, 1, v,  1, n, prec);   /* den = v^T v                */

    /* GUARD (rule 5): a rigorous divide needs den strictly positive.  If the LOWER
     * endpoint of den is <= 0 the quotient is not enclosable -> 0 (caller keeps
     * the dual bound).  For an orthonormal Q this never fires (den ~ 1). */
    arb_get_lbound_arf(den_lo, den, prec);
    if (arf_sgn(den_lo) <= 0) {
        ok = 0;
    } else {
        arb_div(ray, num, den, prec);     /* ray = R(v) = (v^T C_b v)/(v^T v)     */
        arb_mul(lb_b, beta_b, ray, prec); /* lb_b = beta_b * R(v) <= beta_b l_max */
    }

    arb_mat_clear(Q);
    _arb_vec_clear(eigvals, n);
    _arb_vec_clear(v, n);
    _arb_vec_clear(Cv, n);
    arb_clear(num);
    arb_clear(den);
    arb_clear(ray);
    arf_clear(den_lo);

    return ok;   /* 1: lb_b set; 0: den guard tripped, lb_b untouched */
}

/* ---------------------------------------------------------------------------
 * arbsdp_primal_lower_bound -- rigorous PRIMAL (Rayleigh) lower bound from a
 * FEASIBLE block-diagonal rank-1 primal point, for the FULLY-SEPARABLE
 * per-block-trace family (bead arb-prec-IPM-oc7; generalizes the single-block vry
 * case to nblocks >= 1).
 *
 * SHAPE (the applicability detector, CONSERVATIVE -- a wrong ACCEPT is a P0, so we
 * prefer rejecting; rule 2/8).  The construction below is valid IFF m == nblocks
 * and there is a BIJECTION constraint i <-> block b_i where constraint A_i is
 * EXACTLY the identity I on block b_i and EXACTLY zero on every other block (each
 * block has exactly one constraint pinning its trace, tr(X_{b_i}) = beta_b_i, and
 * no constraint couples two blocks), with beta_b_i = b_{i-1} >= 0.  Then the
 * block direct-sum of rank-1 trace-beta points
 *     X_hat = (+)_b  beta_b * v_b v_b^T / (v_b^T v_b)
 * is EXACTLY feasible: PSD by construction (rank-1 PSD blocks since beta_b >= 0)
 * and A_i(X_hat) = tr(X_hat_{b_i}) = beta_{b_i} = b_i for every i (no cross-block
 * coupling), so by Courant-Fischer (R(v_b) <= lambda_max(C_b))
 *     <C, X_hat> = sum_b beta_b R(v_b) <= sum_b beta_b lambda_max(C_b) = opt
 * is a rigorous LOWER bound on the external max optimum, for ANY fixed v_b.  The
 * single-block single-trace vry case is exactly nblocks == 1 (one block, beta_0 =
 * b_1, A_1 = I): this routine reproduces it bit-identically.
 *
 * RIGOR (rule 2, invariant 2): each per-block quotient is evaluated in Arb *ball*
 * arithmetic (block_rayleigh_lb) and the blocks are summed in ball arithmetic, so
 * lb_out is a true enclosure with lower endpoint <= opt REGARDLESS of solve
 * quality.  Needs NO eigenvalue rigor and NO PSD certification of X (the rank-1
 * blocks are PSD by construction -- exactly why it works at the PSD boundary where
 * a verified Cholesky of X FAILS; invariant 1).  The exact-identity / exact-zero
 * ball checks (arb_is_one / arb_is_zero) accept ONLY zero-radius exact 1/0 balls
 * (the goldens store A_i from "1.0"/"0.0" strings -> exact), so a constraint that
 * is not LITERALLY a per-block trace is rejected and the feasibility proof holds.
 * The beta_b >= 0 check (lower endpoint sgn >= 0) is REQUIRED: a negative beta_b
 * makes beta_b * v v^T NOT PSD, breaking feasibility.
 *
 * APPLICABILITY: returns 1 and sets lb_out only when ALL the shape conditions
 * hold; otherwise returns 0 (lb_out untouched) and the caller falls back to the
 * rigorous dual bound (safe).  A tripped den guard in ANY block (block_rayleigh_lb
 * returns 0) also returns 0 (the whole primal bound is unavailable; conservative).
 *
 * Coupled / multi-constraint-per-block / higher-rank feasible points are oc7 (b)/(c).
 *
 * Citation: R(v) <= lambda_max(C_b) is standard (Parlett, "The Symmetric
 * Eigenvalue Problem", 1998, Rayleigh-quotient / Courant-Fischer min-max); this is
 * the primal mirror of the dual-residual Jansson lower bound (arbsdp_lower_bound).
 *
 * Memory (rule 7): lb_out is a caller-init'd output; every per-constraint /
 * per-block materialization temporary and the malloc'd covered[] map are scoped
 * and freed on every return path.  p, it non-NULL.  prec is the working precision.
 * ------------------------------------------------------------------------- */
int
arbsdp_primal_lower_bound(arb_t lb_out, const arbsdp_problem *p,
                          const arbsdp_iterate *it, slong prec)
{
    int   nb;
    int  *covered;   /* covered[b] = constraint matno (1..m) that pins block b, 0=none */
    int   applicable = 1;

    assert(lb_out != NULL && p != NULL && it != NULL
           && "primal_lower_bound: lb_out, p, it non-NULL");

    /* APPLICABILITY part 1 (cheap structural gate): one constraint per block. */
    if (p->nblocks != p->m)
        return 0;
    if (it->nblocks != p->nblocks || it->m != p->m)
        return 0;

    nb = p->nblocks;
    if (nb <= 0)
        return 0;

    covered  = flint_calloc((size_t) nb, sizeof *covered);   /* 0 = uncovered    */

    /* APPLICABILITY part 2: each constraint i (matno i, i=1..m) must be EXACTLY I
     * on EXACTLY ONE block b_i and EXACTLY zero on every other block; the map
     * i -> b_i must be a bijection onto {0..nblocks-1}.  Exact-ball checks
     * (arb_is_one / arb_is_zero) accept ONLY zero-radius exact 1/0 (rule 2/8). */
    for (int i = 1; i <= nb && applicable; i++) {
        int found = -1;   /* the unique block on which A_i is the (nonzero) identity */

        for (int b = 0; b < nb && applicable; b++) {
            slong sz = it->block_n[b];
            arb_mat_t A;
            int is_identity = 1;   /* A == I_sz exactly                            */
            int is_zero     = 1;   /* A == 0 exactly                               */

            if (sz <= 0) { applicable = 0; break; }   /* guard (rule 5)            */

            arb_mat_init(A, sz, sz);
            arbsdp_problem_block_mat(A, p, /*matno=*/i, /*block=*/b, prec);
            for (slong r = 0; r < sz; r++) {
                for (slong cc = 0; cc < sz; cc++) {
                    arb_srcptr e = arb_mat_entry(A, r, cc);
                    if (!arb_is_zero(e))                 is_zero = 0;
                    if (r == cc) { if (!arb_is_one(e))   is_identity = 0; }
                    else         { if (!arb_is_zero(e))  is_identity = 0; }
                }
            }
            arb_mat_clear(A);

            if (is_zero) {
                continue;                /* A_i is zero on block b: fine            */
            } else if (is_identity) {
                if (found != -1) { applicable = 0; break; } /* identity on TWO blocks */
                found = b;
            } else {
                applicable = 0; break;   /* nonzero but NOT exactly I -> reject     */
            }
        }

        if (!applicable)
            break;
        if (found == -1) { applicable = 0; break; }   /* A_i is zero on ALL blocks  */
        if (covered[found] != 0) { applicable = 0; break; } /* block pinned twice    */
        covered[found] = i;              /* block `found` pinned by constraint i     */
    }

    /* APPLICABILITY part 3: bijection -- every block covered exactly once.  (Each
     * was covered at most once above; require at least once here.) */
    for (int b = 0; b < nb && applicable; b++)
        if (covered[b] == 0) applicable = 0;

    /* APPLICABILITY part 4: beta_b = b_{i_b - 1} >= 0 for the pinning constraint
     * i_b of each block (a negative trace makes beta_b v v^T NOT PSD -> the rank-1
     * point is infeasible).  Reject on a NEGATIVE lower endpoint (rule 2/5). */
    if (applicable) {
        arb_t  beta;
        arf_t  beta_lo;
        arb_init(beta);
        arf_init(beta_lo);
        for (int b = 0; b < nb; b++) {
            int rc = arbsdp_problem_b(beta, p, covered[b] - 1, prec); /* b is 0-indexed */
            assert(rc == 0 && "primal_lower_bound: beta_b failed to parse");
            (void) rc;
            arb_get_lbound_arf(beta_lo, beta, prec);
            if (arf_sgn(beta_lo) < 0) { applicable = 0; break; }
        }
        arb_clear(beta);
        arf_clear(beta_lo);
    }

    if (!applicable) {
        flint_free(covered);
        return 0;
    }

    /* COMPUTE: lb_out = sum_b beta_b * R(v_b), accumulated in BALL arithmetic.  Any
     * block's den guard tripping (block_rayleigh_lb -> 0) makes the WHOLE primal
     * bound unavailable (conservative; caller keeps the dual bound). */
    {
        arb_mat_t C_b;
        arb_t     beta_b;
        arb_t     lb_b;
        arb_t     acc;
        int       ok = 1;

        arb_init(beta_b);
        arb_init(lb_b);
        arb_init(acc);
        arb_zero(acc);

        for (int b = 0; b < nb && ok; b++) {
            slong sz = it->block_n[b];
            arb_mat_init(C_b, sz, sz);
            arbsdp_problem_block_mat(C_b, p, /*matno=*/0, /*block=*/b, prec); /* C_file^b */

            int rc = arbsdp_problem_b(beta_b, p, covered[b] - 1, prec); /* beta_b   */
            assert(rc == 0 && "primal_lower_bound: beta_b failed to parse");
            (void) rc;

            if (block_rayleigh_lb(lb_b, C_b, beta_b, it->X[b], prec)) {
                arb_add(acc, acc, lb_b, prec);   /* sum_b beta_b R(v_b)  (ball)     */
            } else {
                ok = 0;                          /* den guard: whole bound off      */
            }
            arb_mat_clear(C_b);
        }

        if (ok)
            arb_set(lb_out, acc);

        arb_clear(beta_b);
        arb_clear(lb_b);
        arb_clear(acc);
        flint_free(covered);

        return ok;   /* 1: lb_out set; 0: a den guard tripped, lb_out untouched */
    }
}

/* ---------------------------------------------------------------------------
 * arbsdp_certify_bracket -- THE PRODUCT (bead arb-prec-IPM-b19): the rigorous
 * [lb, ub] with lb <= p*_ext <= ub (MATH_SPEC §5.4 + §5.5).  certify.h contract.
 *
 * THE P0 SIGN LOCUS (ONE place): y_ext[i] = -( it->y[i] / it->tau ).  Tau-purify
 * (divide by tau), THEN negate (the iterate carries the internal-min dual y_int =
 * tau * y_int_unpurified; the external max problem's dual is y_ext = -y_int).
 * This single y_ext feeds BOTH the lower and upper bound (§5.5 sign contract).
 *
 * TAU GUARD (CLAUDE.md rule 5): tau < TAU_HEALTHY -> infeasibility path (b20),
 * not a usable optimum -> INCONCLUSIVE with [-inf, +inf], no purify/certify.
 *
 * lb is taken at the ball's LOWER endpoint and ub at its UPPER endpoint, so the
 * SCALAR interval [lb, ub] rigorously contains p*_ext even after ball rounding.
 * ------------------------------------------------------------------------- */
arbsdp_certify_status
arbsdp_certify_bracket(arb_t lb, arb_t ub, const arbsdp_problem *p,
                       const arbsdp_iterate *it, const arbsdp_apriori *ab,
                       slong prec_c)
{
    int m;
    arb_ptr y_ext;
    arb_t lb_ball, ub_ball;
    arf_t lo, hi;
    arbsdp_certify_status status;

    assert(p != NULL && it != NULL && ab != NULL
           && "certify_bracket: p, it, ab non-NULL");
    assert(ab->nblocks == p->nblocks
           && "certify_bracket: ab->nblocks != p->nblocks");
    assert(it->nblocks == p->nblocks
           && "certify_bracket: it->nblocks != p->nblocks");
    assert(it->m == p->m && "certify_bracket: it->m != p->m");

    /* TAU GUARD (rule 5): an unhealthy tau means the HSDE iterate is on the
     * infeasibility path (b20), not an approximate optimum.  Do NOT purify or
     * certify -- return INCONCLUSIVE with the honest [-inf, +inf].
     *
     * Take tau's MIDPOINT into a local arb first (the convergence.c arb_to_d
     * idiom) rather than arb_midref(it->tau) directly: the latter aliases the
     * iterate field as an arf_struct view and trips a GCC -O2 -Wstringop-overread
     * false positive when arb_div later reads the full arb_struct from it->tau. */
    {
        arb_t tau_mid;
        double tau_d;
        arb_init(tau_mid);
        arb_get_mid_arb(tau_mid, it->tau);
        tau_d = arf_get_d(arb_midref(tau_mid), ARF_RND_DOWN);
        arb_clear(tau_mid);
        if (!(tau_d >= ARBSDP_CERTIFY_TAU_HEALTHY)) {  /* NaN-safe (NaN -> guard) */
            arb_neg_inf(lb);
            arb_pos_inf(ub);
            return ARBSDP_CERTIFY_INCONCLUSIVE;
        }
    }

    m = p->m;

    /* --- THE P0 SIGN LOCUS: y_ext = -(it->y / it->tau), formed ONCE. --------- */
    y_ext = NULL;
    if (m > 0) {
        y_ext = _arb_vec_init(m);
        for (int i = 0; i < m; i++) {
            /* element-pointer form (vec + i), as in FLINT's own _arb_vec code,
             * to keep -Wstringop-overread (a known -O2 false positive on the
             * &vec[i] form for arb_ptr) quiet without suppressing the warning. */
            arb_div(y_ext + i, it->y + i, it->tau, prec_c); /* y_int = y / tau */
            arb_neg(y_ext + i, y_ext + i);                  /* y_ext = -y_int  */
        }
    }

    arb_init(lb_ball);
    arb_init(ub_ball);

    arbsdp_lower_bound(lb_ball, p, y_ext, ab, prec_c);
    arbsdp_upper_bound(ub_ball, p, y_ext, ab, prec_c);

    /* Take lb at the LOWER endpoint and ub at the UPPER endpoint of the assembled
     * balls (the rigorous scalar interval).  arf +/-inf flows through verbatim. */
    arf_init(lo);
    arf_init(hi);
    arb_get_lbound_arf(lo, lb_ball, prec_c);
    /* vry: tighten lb with the rigorous PRIMAL Rayleigh bound where applicable.
     * Both the dual-residual lb (lb_ball) and the primal Rayleigh lb are rigorous
     * lower bounds on p*_ext, so their MAX is rigorous and tighter -- it dominates
     * the structurally-loose single-dual bound for max-eigenvalue problems
     * (arf_cmp handles a -inf dual lo correctly: max(-inf, finite) = finite). */
    {
        arb_t lb_primal;
        arb_init(lb_primal);
        if (arbsdp_primal_lower_bound(lb_primal, p, it, prec_c)) {
            arf_t lo_p;
            arf_init(lo_p);
            arb_get_lbound_arf(lo_p, lb_primal, prec_c);
            if (arf_cmp(lo_p, lo) > 0)
                arf_set(lo, lo_p);
            arf_clear(lo_p);
        }
        arb_clear(lb_primal);
    }
    arb_get_ubound_arf(hi, ub_ball, prec_c);
    arb_set_arf(lb, lo);
    arb_set_arf(ub, hi);
    arf_clear(lo);
    arf_clear(hi);

    status = (arb_is_finite(lb) && arb_is_finite(ub))
                 ? ARBSDP_CERTIFY_OPTIMAL_BRACKET
                 : ARBSDP_CERTIFY_INCONCLUSIVE;

    arb_clear(lb_ball);
    arb_clear(ub_ball);
    if (m > 0)
        _arb_vec_clear(y_ext, m);

    return status;
}
