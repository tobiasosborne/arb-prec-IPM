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

#include "arbsdp/certify.h"
#include "arbsdp/svec.h"

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
