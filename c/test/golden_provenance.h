#ifndef ARBSDP_GOLDEN_PROVENANCE_H
#define ARBSDP_GOLDEN_PROVENANCE_H
typedef struct {
    char   name[128];
    char   dats_path[512];               /* "<golden_dir>/<name>/<name>.dat-s" */
    char   optimal_value[256];           /* decimal string */
    int    optimal_value_digits;
    int    m;
    int    nblocks;
    int    maximize;                     /* 1=maximize (default), 0=minimize (bead n1x) */
    char   expected_status[32];          /* default "optimal" */
    int    target_digits;                /* default 15 */
    long   prec_cap;                     /* default 8192 */
    int    correctness_tol_digits;       /* default 0 */
    double expected_max_bits_per_digit;  /* default 1e9 (== no budget) */
    int    expected_max_iters;           /* default INT_MAX */
    int    gap_status;                   /* 1 if a known_gaps entry has kind "status" */
    int    gap_correctness;              /* 1 if kind "correctness" */
    int    gap_efficiency;               /* 1 if kind "efficiency" */
    char   gap_reason[256];              /* first known_gaps entry's reason ("" if none) */
    char   gap_bead[64];                 /* first known_gaps entry's bead ("" if none) */
} golden_case;
/* Discover golden cases under golden_dir: each subdirectory that contains both
 * provenance.json and <subdir>/<subdir>.dat-s yields one parsed case. Fills
 * cases[0..max-1], returns the count (>=0), or -1 on hard error (dir unreadable
 * or malformed provenance). Cases are sorted by name. Fails loud on malformed. */
int golden_discover(const char *golden_dir, golden_case *cases, int max);
#endif
