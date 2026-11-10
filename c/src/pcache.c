/*
 * src/pcache.c -- the per-(solve, prec) POINT-MODE materialization cache of the
 * constraint data (include/arbsdp/pcache.h, bead arb-prec-IPM-6b6).
 *
 * GROUND TRUTH: problem.h (the decimal-string data model and its materializers
 * arbsdp_problem_block_mat / arbsdp_problem_b, which this module calls -- no
 * second parser).  The cached values are exactly what those materializers return
 * at `prec`, so every consumer is bit-identical to on-demand materialization
 * (test_pcache).
 *
 * Arb memory discipline (CLAUDE.md rule 7): init allocates m*nblocks arb_mat and
 * an m-vector; clear frees exactly those.
 */

#include <assert.h>

#include <flint/flint.h>

#include "arbsdp/pcache.h"

void
arbsdp_pcache_init(arbsdp_pcache *pc, const arbsdp_problem *p, slong prec)
{
    slong nmat;

    assert(pc != NULL);
    assert(p != NULL);
    assert(p->m >= 0 && p->nblocks >= 0);

    pc->prob = p;
    pc->prec = prec;
    pc->m = p->m;
    pc->nblocks = p->nblocks;
    pc->primal_degen = -1;

    nmat = (slong) p->m * p->nblocks;
    pc->A = flint_malloc((size_t) (nmat > 0 ? nmat : 1) * sizeof *pc->A);
    for (int i = 0; i < p->m; i++)
        for (int b = 0; b < p->nblocks; b++) {
            int s = p->block_sizes[b];
            slong n = (s < 0) ? -(slong) s : (slong) s;
            arb_mat_struct *Ai = pc->A[(slong) i * p->nblocks + b];
            arb_mat_init(Ai, n, n);
            arbsdp_problem_block_mat(Ai, p, i + 1, b, prec);   /* matno i+1 */
        }

    pc->b = _arb_vec_init(p->m);
    for (int i = 0; i < p->m; i++) {
        int bad = arbsdp_problem_b(pc->b + i, p, i, prec);
        assert(bad == 0);   /* b validated at read time (problem.h) */
        (void) bad;
    }
}

void
arbsdp_pcache_clear(arbsdp_pcache *pc)
{
    if (pc == NULL)
        return;
    if (pc->A != NULL) {
        slong nmat = (slong) pc->m * pc->nblocks;
        for (slong k = 0; k < nmat; k++)
            arb_mat_clear(pc->A[k]);
        flint_free(pc->A);
    }
    if (pc->b != NULL)
        _arb_vec_clear(pc->b, pc->m);
    pc->prob = NULL;
    pc->prec = 0;
    pc->m = 0;
    pc->nblocks = 0;
    pc->A = NULL;
    pc->b = NULL;
    pc->primal_degen = -1;
}
