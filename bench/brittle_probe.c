/*
 * bench/brittle_probe.c -- Layer-0 brittleness/performance probe (see HANDOFF.md).
 * NOT part of the cmake build. Run from the repo root so golden/ paths resolve:
 *   cmake --build c/build
 *   gcc bench/brittle_probe.c -o /tmp/bp -I c/include -L c/build -larbsdp \
 *       -lflint -lmpfr -lgmp -Wl,-rpath,c/build
 *   /tmp/bp <target_digits=15> <prec_max=8192>
 * Reproduces: the read_sdpa heap-corruption (remove the arbsdp_problem_init line),
 * the per-precision NUM trajectory (iterate leaving the cone), and the wasted-solve cost.
 */
/* Brittleness/performance probe for the Layer-0 solver. Not part of the build. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <flint/arb.h>
#include "arbsdp/problem.h"
#include "arbsdp/solve.h"
#include "arbsdp/precision.h"

static const char *st_name(arbsdp_solve_status s){
    switch(s){case ARBSDP_SOLVE_OPTIMAL:return"OPT";case ARBSDP_SOLVE_ITER_LIMIT:return"ITER";
    case ARBSDP_SOLVE_STALL:return"STALL";case ARBSDP_SOLVE_NUMERICAL:return"NUM";
    case ARBSDP_SOLVE_PRIMAL_INFEASIBLE:return"PINF";case ARBSDP_SOLVE_DUAL_INFEASIBLE:return"DINF";default:return"?";}
}

int main(int argc, char **argv){
    const char *names[] = {"trivial_2x2","max_eigenvalue_2x2","sdp_sqrt2","lp_diagonal_block",
                           "ill_conditioned_3x3","mixed_blocks","max_eig_tridiag_3x3"};
    setvbuf(stdout, NULL, _IONBF, 0);
    int target = (argc>1)? atoi(argv[1]) : 15;
    slong prec_max = (argc>2)? atol(argv[2]) : 8192;
    printf("target_digits=%d prec_max=%ld\n", target, (long)prec_max);
    printf("%-20s %-6s %8s %7s %8s %9s  trajectory(prec:status:digits)\n",
           "problem","status","finalp","solves","iters","wall_ms");
    for (unsigned k=0;k<sizeof(names)/sizeof(names[0]);k++){
        char path[512];
        snprintf(path,sizeof path,"golden/%s/%s.dat-s",names[k],names[k]);
        arbsdp_problem p; arbsdp_problem_init(&p);   /* REQUIRED: read_sdpa clears at entry */
        if (arbsdp_read_sdpa(&p, path)!=0){ printf("%-20s READ-FAIL\n",names[k]); arbsdp_problem_clear(&p); continue; }
        arbsdp_precision_params pp; arbsdp_precision_default_params(&pp);
        pp.target_digits=target; pp.prec_max=prec_max;
        arbsdp_adaptive_result r;
        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        arbsdp_solve_adaptive(&r,&p,&pp);
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;
        int sumit=0; for(int i=0;i<r.prec_history_len;i++) sumit+=r.res.iters; /* approx */
        printf("%-20s %-6s %8ld %7d %8d %9.1f  ",
               names[k], r.status==ARBSDP_ADAPTIVE_OPTIMAL?"AOPT":"ALIM",
               (long)r.final_prec, r.prec_history_len, r.res.iters, ms);
        for(int i=0;i<r.prec_history_len;i++)
            printf("%ld:%s:%.1f ",(long)r.prec_history[i].prec, st_name(r.prec_history[i].status), r.prec_history[i].digits);
        printf("\n");
        /* recovered value */
        char *vs=arb_get_str(r.res.value, 32, 0);
        printf("    recovered = %s\n", vs); flint_free(vs);
        (void)sumit;
        arbsdp_adaptive_result_clear(&r);
        arbsdp_problem_clear(&p);
    }
    flint_cleanup();
    return 0;
}
