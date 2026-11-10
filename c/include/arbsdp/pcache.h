/*
 * include/arbsdp/pcache.h -- the per-(solve, prec) POINT-MODE materialization
 * cache of the constraint data A_i^b and b (bead arb-prec-IPM-6b6).
 *
 * ==========================================================================
 * WHY (measured, bead 6b6)
 * ==========================================================================
 * problem.h stores the data as decimal strings and materializes on demand via
 * arb_set_str.  Before this cache the Layer-0 loop re-materialized A_i^b inside
 * its inner loops: O(m^2 * nb) parses per iteration in the Schur assembly
 * (schur.c), O(m * nb) per apply_A / apply_At (residuals, twice per iteration),
 * O(m * nb) per RHS / A^T dy loop in direction.c (6 such loops plus 4 b loops
 * per direction), and O(m * nb) per
 * factorization in the regularization diagnose (regularize.c) -- all on data
 * that is CONSTANT for the whole solve.  The cache parses every stored entry
 * exactly once per (solve, prec).
 *
 * ==========================================================================
 * SEMANTICS -- BIT-IDENTICAL TO ON-DEMAND MATERIALIZATION
 * ==========================================================================
 * A[i * nblocks + b] is arb_mat_set-equal (midpoint AND radius) to what
 * arbsdp_problem_block_mat(., p, i + 1, b, prec) produces, and b[i] to
 * arbsdp_problem_b(., p, i, prec).  arb_set_str of an inexact decimal yields a
 * ball with a nonzero radius; the cache KEEPS that radius verbatim, because the
 * pre-cache consumers computed with it and the cache is required to be
 * numerically neutral (bit-identical results; test_pcache).  The radius is not
 * trusted (POINT MODE, CLAUDE.md rule 1).
 *
 * The binary rounding of a decimal depends on prec, so a cache is valid for ONE
 * prec only: the adaptive controller (precision.c) runs one arbsdp_solve per
 * precision level and each arbsdp_solve builds its own cache.  Consumers assert
 * pc->prec == prec.
 *
 * ==========================================================================
 * LAYER BOUNDARY (CLAUDE.md rule 1/2, invariant 4)
 * ==========================================================================
 * Layer 0 only.  Layer 1 (certify.c) materializes the data itself, as balls at
 * the certification precision; it never sees this cache.
 *
 * ==========================================================================
 * LIFETIME / OWNERSHIP (CLAUDE.md rule 7)
 * ==========================================================================
 * arbsdp_pcache_init allocates every matrix and the b vector; arbsdp_pcache_clear
 * frees them.  arbsdp_solve owns one cache for the duration of the solve and
 * lends it to its working iterate (arbsdp_iterate.pc, borrowed); it detaches the
 * pointer before clearing the cache, so the iterate handed back in the result
 * never holds a dangling pointer.  An iterate with pc == NULL (tests, external
 * callers) still works: the consumers build a scoped temporary cache per call.
 */

#ifndef ARBSDP_PCACHE_H
#define ARBSDP_PCACHE_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#include "arbsdp/problem.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct arbsdp_pcache {
    const arbsdp_problem *prob;  /* borrowed; not owned                         */
    slong      prec;             /* the ONE precision the data was parsed at    */
    int        m;                /* number of constraints                       */
    int        nblocks;          /* number of blocks                            */
    arb_mat_t *A;                /* length m*nblocks: A[i*nblocks+b] = A_{i+1}^b */
    arb_ptr    b;                /* length m: b[i] = b_i                        */
    int        primal_degen;     /* regularize.c diagnose memo: -1 = not yet
                                  * computed, else 0/1 (constant per (p, prec)) */
} arbsdp_pcache;

/*
 * arbsdp_pcache_init -- materialize every A_i^b (via arbsdp_problem_block_mat)
 * and every b_i (via arbsdp_problem_b) of `p` at `prec`.  Pair with
 * arbsdp_pcache_clear.  `p` must outlive the cache.
 */
void arbsdp_pcache_init(arbsdp_pcache *pc, const arbsdp_problem *p, slong prec);

/* arbsdp_pcache_clear -- free everything init allocated; re-zero the struct. */
void arbsdp_pcache_clear(arbsdp_pcache *pc);

/* arbsdp_pcache_A -- A_{i+1}^b for 0-indexed constraint i (matno = i + 1). */
static inline const arb_mat_struct *
arbsdp_pcache_A(const arbsdp_pcache *pc, int i, int b)
{
    return pc->A[(slong) i * pc->nblocks + b];
}

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_PCACHE_H */
