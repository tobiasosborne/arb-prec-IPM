/*
 * src/regularize.c -- adaptive 3-way Tikhonov regularization of the Schur
 * Cholesky factorization (POINT MODE, Layer 0).
 *
 * GROUND TRUTH (CLAUDE.md rule 3) -- see include/arbsdp/regularize.h for the
 * full citation list, the invariant-4 boundary, and the failRow porting note.
 *   - docs/MATH_SPEC.md §4 (M_reg = M + (d_p+d_d+d_g)I), §4.2 (diagnose),
 *     §4.3 (retry), §10.4 (carry d -- no reset), §10.6 (maxRefactor = 20).
 *   - solver-ipm/src/solver/Regularization.ts:96-115 / :174-192 / :231-305.
 *   - solver-ipm/src/solver/Defaults.ts:40-46 (caps, bumps, initialJitter).
 *   - docs/FLINT_NOTES.md: arb_mat_cho returns 1 iff PROVEN PD.
 *
 * REUSE (no duplication): arbsdp_problem_block_mat (problem.h) to materialize
 * each A_i^b for the diagnose row norms; arb_mat_cho (FLINT) for the factor;
 * arb_mat_frobenius_norm (FLINT) for the per-block Frobenius norm.  The Schur M
 * is supplied by the caller (arbsdp_schur_assemble, schur.h) -- not rebuilt here.
 */

#include <assert.h>

#include <flint/flint.h>
#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/regularize.h"

/* ---- Fixed tier parameters (Defaults.ts:40-46, Regularization.ts:96-115) ---
 * Caps and multiplicative bump factors.  The asymmetry is load-bearing
 * (Regularization.ts:18-33): the primal tier is bounded tight because a small
 * lift suffices for a primal-degenerate row, while the dual/gap tiers need many
 * orders of headroom for generic Schur conditioning blowups. */
#define CAP_P 1e-2
#define CAP_D 1e2
#define CAP_G 1e2
#define BUMP_P 10.0
#define BUMP_D 100.0
#define BUMP_G 100.0

/* initialJitter (Defaults.ts:40) -- the GAP tier's starting value; also the
 * multiplicative base for the first bump off zero (Regularization.ts:283,291,
 * 299: Math.max(jitter, 1e-12)). */
#define INITIAL_JITTER 1e-12
#define BUMP_BASE 1e-12

/* maxRefactor (MATH_SPEC §10.6, decision 6) -- the SDP call-site value. */
#define MAX_REFACTOR 20

/* diagnose threshold scale (Regularization.ts:187): 1e-10 * max(globalNorm,1e-300). */
#define DIAG_THRESH_REL 1e-10
#define DIAG_THRESH_FLOOR 1e-300

void
arbsdp_regstate_init(arbsdp_regstate *rs, slong prec)
{
    assert(rs != NULL);
    arb_init(rs->delta_p);
    arb_init(rs->delta_d);
    arb_init(rs->delta_g);
    arbsdp_regstate_reset(rs, prec);
}

void
arbsdp_regstate_reset(arbsdp_regstate *rs, slong prec)
{
    assert(rs != NULL);
    /* Defaults.ts:40-46: d_p = d_d = 0, d_g = initialJitter = 1e-12. */
    arb_zero(rs->delta_p);
    arb_zero(rs->delta_d);
    arb_set_d(rs->delta_g, INITIAL_JITTER);
    /* arb_set_d is exact for representable doubles; round to prec for tidiness. */
    arb_set_round(rs->delta_g, rs->delta_g, prec);
    rs->bumps_p = 0;
    rs->bumps_d = 0;
    rs->bumps_g = 0;
    rs->refactors = 0;
}

void
arbsdp_regstate_clear(arbsdp_regstate *rs)
{
    assert(rs != NULL);
    arb_clear(rs->delta_p);
    arb_clear(rs->delta_d);
    arb_clear(rs->delta_g);
    rs->bumps_p = rs->bumps_d = rs->bumps_g = rs->refactors = 0;
}

/* side length of block b (|block_sizes[b]|; negative = diagonal/LP block). */
static slong
block_side(const arbsdp_problem *p, int b)
{
    int s = p->block_sizes[b];
    return (s < 0) ? -(slong) s : (slong) s;
}

/*
 * has_primal_degenerate_row -- the row-agnostic port of makeSdpDiagnose
 * (Regularization.ts:174-192).  Returns 1 iff SOME constraint i has Frobenius
 * row norm  rowNorm_i = sqrt(sum_b ||A_i^b||_F^2)  below the global threshold
 *     1e-10 * max(globalNorm, 1e-300).
 * A `1` => diagnose "primal" (a primal-degenerate row exists); `0` => "gap".
 * See the regularize.h porting note: arb_mat_cho gives no failing-row index, so
 * we test "does a degenerate row exist at all" rather than classifying a
 * specific failed row, preserving the TS primal-vs-gap intent.
 *
 * Computed in POINT mode (midpoints); the row norms only steer which tier to
 * bump, no rigor is claimed.
 */
static int
has_primal_degenerate_row(const arbsdp_problem *p, slong prec)
{
    int m = p->m;
    int nb = p->nblocks;
    arb_ptr row_norm;        /* rowNorm_i for i = 0..m-1 */
    arb_t global, thresh, sq, fn, floor_a;
    int found;

    if (m == 0)
        return 0;

    row_norm = _arb_vec_init(m);
    arb_init(global);
    arb_init(thresh);
    arb_init(sq);
    arb_init(fn);
    arb_init(floor_a);

    for (int i = 0; i < m; i++) {
        /* rowNorm_i^2 = sum_b ||A_{i+1}^b||_F^2  (matno = i+1). */
        arb_zero(sq);
        for (int b = 0; b < nb; b++) {
            slong n = block_side(p, b);
            arb_mat_t Ai;
            arb_mat_init(Ai, n, n);
            arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);
            arb_mat_frobenius_norm(fn, Ai, prec);
            arb_addmul(sq, fn, fn, prec); /* sq += ||A_i^b||_F^2 */
            arb_mat_clear(Ai);
        }
        arb_sqrt(row_norm + i, sq, prec);
        if (arb_gt(row_norm + i, global))
            arb_set(global, row_norm + i);
    }

    /* threshold = 1e-10 * max(globalNorm, 1e-300). */
    arb_set_d(floor_a, DIAG_THRESH_FLOOR);
    if (arb_lt(global, floor_a))
        arb_set(global, floor_a);
    arb_set_d(thresh, DIAG_THRESH_REL);
    arb_mul(thresh, thresh, global, prec);

    found = 0;
    for (int i = 0; i < m; i++) {
        if (arb_lt(row_norm + i, thresh)) {
            found = 1;
            break;
        }
    }

    _arb_vec_clear(row_norm, m);
    arb_clear(global);
    arb_clear(thresh);
    arb_clear(sq);
    arb_clear(fn);
    arb_clear(floor_a);
    return found;
}

/* RegKind, as in Regularization.ts:84. */
typedef enum { REG_PRIMAL, REG_DUAL, REG_GAP } reg_kind;

/*
 * try_bump -- bump the given tier by its factor, clamped at the tier's cap
 * (Regularization.ts:278-305).  Returns 1 on a successful bump, 0 if the tier
 * is already at (or above) its cap (caller escalates to the next tier).
 * Uses max(delta, 1e-12) as the multiplicative base so the first bump off zero
 * is non-zero.
 */
static int
try_bump(arbsdp_regstate *rs, reg_kind kind, slong prec)
{
    arb_t *delta;
    double cap, bump;
    ulong *bumps;
    arb_t base, capa, val;
    int ok;

    switch (kind) {
        case REG_PRIMAL: delta = &rs->delta_p; cap = CAP_P; bump = BUMP_P; bumps = &rs->bumps_p; break;
        case REG_DUAL:   delta = &rs->delta_d; cap = CAP_D; bump = BUMP_D; bumps = &rs->bumps_d; break;
        case REG_GAP:    delta = &rs->delta_g; cap = CAP_G; bump = BUMP_G; bumps = &rs->bumps_g; break;
        default:         return 0;
    }

    arb_init(capa);
    arb_set_d(capa, cap);
    /* If the tier is already at/above its cap, it has no headroom. */
    if (arb_ge(*delta, capa)) {
        arb_clear(capa);
        return 0;
    }

    arb_init(base);
    arb_init(val);
    /* base = max(delta, 1e-12). */
    arb_set(base, *delta);
    arb_set_d(val, BUMP_BASE);
    if (arb_lt(base, val))
        arb_set(base, val);
    /* val = base * bump, clamped at cap. */
    arb_set_d(val, bump);
    arb_mul(val, base, val, prec);
    if (arb_gt(val, capa))
        arb_set(val, capa);
    arb_set(*delta, val);

    (*bumps)++;
    ok = 1;
    arb_clear(base);
    arb_clear(val);
    arb_clear(capa);
    return ok;
}

int
arbsdp_factor_with_reg(arb_mat_t L, const arb_mat_t M, arbsdp_regstate *rs,
                       const arbsdp_problem *p, slong prec, int *attempts_out)
{
    slong m;
    int primal_degen;
    arb_mat_t Mreg;
    arb_t lift;
    int attempts = 0;
    int success = 0;

    assert(L != NULL);
    assert(M != NULL);
    assert(rs != NULL);
    assert(p != NULL);
    assert(arb_mat_nrows(M) == arb_mat_ncols(M));
    assert(arb_mat_nrows(L) == arb_mat_nrows(M));
    assert(arb_mat_ncols(L) == arb_mat_ncols(M));
    assert(p->m == arb_mat_nrows(M)); /* diagnose row count must match M (rule 5) */

    m = arb_mat_nrows(M);

    /* Diagnose input precomputed ONCE (matches makeSdpDiagnose closing over its
     * rowNorms; Regularization.ts:174-186): is any constraint row degenerate? */
    primal_degen = has_primal_degenerate_row(p, prec);

    arb_mat_init(Mreg, m, m);
    arb_init(lift);

    /* Retry loop (Regularization.ts:231-269): up to maxRefactor attempts.  Each
     * attempt factors M + (carried lift)*I; the lift starts from the CARRIED
     * deltas (MATH_SPEC §10.4 -- NOT reset). */
    for (int attempt = 0; attempt < MAX_REFACTOR; attempt++) {
        reg_kind kind;
        int bumped;

        attempts = attempt + 1;

        /* lift = delta_p + delta_d + delta_g  (MATH_SPEC §4.1). */
        arb_add(lift, rs->delta_p, rs->delta_d, prec);
        arb_add(lift, lift, rs->delta_g, prec);

        /* Mreg = M + lift*I.  M is never mutated (invariant 4 auditability). */
        arb_mat_set(Mreg, M);
        for (slong i = 0; i < m; i++)
            arb_add(arb_mat_entry(Mreg, i, i), arb_mat_entry(Mreg, i, i), lift, prec);

        if (arb_mat_cho(L, Mreg, prec)) {
            success = 1;
            break;
        }

        /* Failure: diagnose primal-vs-gap, then bump (Regularization.ts:247-266).
         * The dual tier is reachable only via the cap fall-through. */
        kind = primal_degen ? REG_PRIMAL : REG_GAP;
        if (try_bump(rs, kind, prec)) {
            rs->refactors++;
            continue;
        }
        /* Diagnosed tier capped -> fall through gap -> dual -> primal, skipping
         * the already-tried diagnosed kind (Regularization.ts:255-265). */
        bumped = 0;
        {
            const reg_kind order[3] = { REG_GAP, REG_DUAL, REG_PRIMAL };
            for (int t = 0; t < 3; t++) {
                if (order[t] == kind)
                    continue;
                if (try_bump(rs, order[t], prec)) {
                    bumped = 1;
                    break;
                }
            }
        }
        if (!bumped) {
            /* Every tier capped and still not PD -> honest failure (rule 5). */
            success = 0;
            break;
        }
        rs->refactors++;
    }

    if (attempts_out != NULL)
        *attempts_out = attempts;

    arb_mat_clear(Mreg);
    arb_clear(lift);
    return success;
}
