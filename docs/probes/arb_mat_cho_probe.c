/*
 * arb_mat_cho_probe.c
 *
 * Smoke-test for arb_mat_cho (FLINT 3.0.1).
 *
 * Tests:
 *   1. 3x3 positive-definite matrix  -> arb_mat_cho must return 1 (success)
 *      and L*L^T must enclose A.
 *   2. 3x3 indefinite matrix         -> arb_mat_cho must return 0 (failure /
 *      could not prove PD).
 *
 * Build:
 *   gcc -fsanitize=address,undefined -g arb_mat_cho_probe.c \
 *       -lflint -lmpfr -lgmp -o /tmp/arb_mat_cho_probe
 * Run:
 *   /tmp/arb_mat_cho_probe
 */

#include <stdio.h>
#include <assert.h>
#include <flint/arb.h>
#include <flint/arb_mat.h>

static void set_entry_d(arb_mat_t M, slong r, slong c, double v)
{
    arb_set_d(arb_mat_entry(M, r, c), v);
}

int main(void)
{
    slong prec = 256;  /* bits of working precision */

    /* ------------------------------------------------------------------ */
    /* Test 1: PD matrix  A = [[4, 2, 1], [2, 3, 0], [1, 0, 2]]           */
    /*   eigenvalues ~ 5.56, 2.63, 0.81  -- all positive                  */
    /* ------------------------------------------------------------------ */
    arb_mat_t A_pd, L_pd;
    arb_mat_init(A_pd, 3, 3);
    arb_mat_init(L_pd, 3, 3);

    set_entry_d(A_pd, 0, 0, 4.0);
    set_entry_d(A_pd, 0, 1, 2.0);
    set_entry_d(A_pd, 0, 2, 1.0);
    set_entry_d(A_pd, 1, 0, 2.0);
    set_entry_d(A_pd, 1, 1, 3.0);
    set_entry_d(A_pd, 1, 2, 0.0);
    set_entry_d(A_pd, 2, 0, 1.0);
    set_entry_d(A_pd, 2, 1, 0.0);
    set_entry_d(A_pd, 2, 2, 2.0);

    int ok_pd = arb_mat_cho(L_pd, A_pd, prec);
    printf("[Test 1] arb_mat_cho on PD matrix: return = %d (expected 1)\n", ok_pd);
    assert(ok_pd == 1 && "PD matrix must succeed");

    /* Verify L*L^T encloses A  (spot-check entry [0][0]: L[0][0]^2 = A[0][0]) */
    arb_mat_t LLT;
    arb_mat_init(LLT, 3, 3);
    arb_mat_t L_pd_T;
    arb_mat_init(L_pd_T, 3, 3);
    arb_mat_transpose(L_pd_T, L_pd);
    arb_mat_mul(LLT, L_pd, L_pd_T, prec);

    /* Check that each entry of LLT contains the corresponding entry of A_pd */
    int enclosure_ok = 1;
    for (slong i = 0; i < 3; i++) {
        for (slong j = 0; j < 3; j++) {
            if (!arb_contains(arb_mat_entry(LLT, i, j),
                              arb_mat_entry(A_pd, i, j))) {
                enclosure_ok = 0;
            }
        }
    }
    printf("[Test 1] L*L^T encloses A: %s\n", enclosure_ok ? "YES" : "NO");
    assert(enclosure_ok && "L*L^T must enclose A");

    arb_mat_clear(LLT);
    arb_mat_clear(L_pd_T);
    arb_mat_clear(L_pd);
    arb_mat_clear(A_pd);

    /* ------------------------------------------------------------------ */
    /* Test 2: indefinite matrix  A = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]  */
    /* ------------------------------------------------------------------ */
    arb_mat_t A_indef, L_indef;
    arb_mat_init(A_indef, 3, 3);
    arb_mat_init(L_indef, 3, 3);

    set_entry_d(A_indef, 0, 0,  1.0);
    set_entry_d(A_indef, 0, 1,  0.0);
    set_entry_d(A_indef, 0, 2,  0.0);
    set_entry_d(A_indef, 1, 0,  0.0);
    set_entry_d(A_indef, 1, 1, -1.0);
    set_entry_d(A_indef, 1, 2,  0.0);
    set_entry_d(A_indef, 2, 0,  0.0);
    set_entry_d(A_indef, 2, 1,  0.0);
    set_entry_d(A_indef, 2, 2,  1.0);

    int ok_indef = arb_mat_cho(L_indef, A_indef, prec);
    printf("[Test 2] arb_mat_cho on indefinite matrix: return = %d (expected 0)\n", ok_indef);
    assert(ok_indef == 0 && "indefinite matrix must fail");

    arb_mat_clear(L_indef);
    arb_mat_clear(A_indef);

    printf("\nAll assertions passed. No leaks (ASan will confirm).\n");
    flint_cleanup();
    return 0;
}
