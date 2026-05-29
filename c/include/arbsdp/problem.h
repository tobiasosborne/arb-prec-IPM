/*
 * include/arbsdp/problem.h -- the SDP problem data model consumed by the solver
 * (Layer 0, b11+) and the certifier (Layer 1).  This is the in-memory result of
 * parsing an SDPA-sparse (.dat-s) file; it is the single source of truth for the
 * problem data and is deliberately precision-agnostic.
 *
 * Ground truth / sources:
 *   - docs/MATH_SPEC.md §1 (SDPA standard form: max <C,X> s.t. <A_i,X>=b_i,
 *     X = blkdiag >= 0; internal min -<C,X>; sign convention) and §2 (svec /
 *     Frobenius inner product conventions).
 *   - SdpaSparse.ts / SdpProblem.ts (the TS reference parser + data model whose
 *     semantics this matches exactly so golden-master comparison is meaningful).
 *   - golden/README.md (the .dat-s off-diagonal "2v" convention, see below).
 *   - CLAUDE.md rule 5 (fail loud on malformed input), rule 7 (arb memory).
 *
 * ==========================================================================
 * PRECISION-PRESERVATION DECISION (binding for the whole struct)
 * ==========================================================================
 * The golden .dat-s files store every value to 65 decimal digits.  Parsing an
 * entry to `double` would throw away ~50 of those digits before the solver ever
 * sees them -- fatal for a tool whose whole product is an arbitrary-precision
 * certificate.  We therefore store the RAW DECIMAL STRINGS exactly as they
 * appear in the file (b vector entries and matrix entries alike) and materialize
 * arb / arb_mat objects on demand at a caller-given `prec` via `arb_set_str`.
 *
 * Strings (not a high-fixed-prec arb) were chosen because:
 *   - they are truly precision-agnostic: the same struct serves a 128-bit solve
 *     and a 4096-bit certification without re-reading the file or losing digits;
 *   - they are lossless: 65 digits round-trip exactly, no premature rounding;
 *   - `arb_set_str` already yields a correctly-rounded enclosure of the exact
 *     decimal at the requested prec, so rigor (a true ball enclosure) is free.
 * The cost is a re-parse of each materialized entry, which is negligible for the
 * small dense problems in scope (PRD v1) and happens O(few) times per solve.
 *
 * ==========================================================================
 * THE .dat-s OFF-DIAGONAL "2v" CONVENTION (critical; see golden/README.md)
 * ==========================================================================
 * An entry "<matno> <blk> <k> <l> <v>" with k < l (strict upper triangle) names
 * BOTH symmetric positions: A_{kl} = A_{lk} = v.  In the Frobenius inner product
 * <A,X> = sum_{ij} A_{ij} X_{ij} it therefore contributes
 *     A_{kl} X_{kl} + A_{lk} X_{lk} = v X_{kl} + v X_{lk} = 2 v X_{kl}.
 * The factor of 2 is NOT stored in the matrix.  arbsdp_problem_block_mat builds
 * the genuine symmetric matrix with A_{kl} = A_{lk} = v (the materializer below
 * is documented to do exactly this), and the "2v" emerges only when you take the
 * Frobenius inner product over all n*n entries via arbsdp_frob_inner -- i.e. the
 * factor lives in the INNER PRODUCT, not in the stored data.  (This is the exact
 * same mechanism as the svec sqrt(2): see svec.h.)  The golden generator already
 * encodes objective off-diagonals as v = 1/2 to compensate, so e.g. sdp_sqrt2's
 * "0 1 1 2 0.5" gives <C,X> = x_12, as intended.
 */

#ifndef ARBSDP_PROBLEM_H
#define ARBSDP_PROBLEM_H

#include <flint/arb.h>
#include <flint/arb_mat.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A single sparse entry, stored precision-preserving as the raw decimal string
 * `value` (owned, heap-allocated, NUL-terminated).  Indices i, j are 0-indexed
 * with i <= j (upper triangle), as in the file (which is 1-indexed; we subtract
 * one at parse time).  The entry names A_{ij} = A_{ji} = value (see the "2v"
 * note above): off-diagonal entries (i < j) are symmetric, not doubled.
 */
typedef struct {
    int   i;       /* 0-indexed row, i <= j                          */
    int   j;       /* 0-indexed col                                  */
    char *value;   /* owned raw decimal string from the .dat-s file  */
} arbsdp_entry;

/*
 * The matrix data for one (matno, block) pair: a growable array of sparse
 * entries.  matno 0 is C; matno 1..m is A_matno.  Entries are appended in file
 * order; duplicates accumulate (matching SdpProblem.ts which does `M += v`).
 */
typedef struct {
    arbsdp_entry *entries;
    int           count;     /* number of entries in use                  */
    int           capacity;  /* allocated slots                           */
} arbsdp_matblock;

/*
 * arbsdp_problem -- the parsed SDPA problem.
 *
 * Layout of the (m+1) x nblocks matrix-block grid `mats`:
 *   mats[matno * nblocks + block]  holds the entries of matrix `matno`
 *   (0 = C, 1..m = A_matno) restricted to block `block` (0-indexed).
 *
 * block_sizes[b] is SIGNED exactly as in the file: a positive value is a dense
 * PSD block of that side length; a negative value is a diagonal (LP) block of
 * |size| (only i==j entries are meaningful).  The materialized matrix side is
 * always |block_sizes[b]|.
 *
 * b is the constraint right-hand side, stored precision-preserving as decimal
 * strings (length m, each owned).  `maximize` records the SDPA convention (the
 * golden files are all `maximize = true`); the sign flip to the internal
 * min -<C,X> form is the SOLVER's job, NOT this reader's -- this struct stores
 * the problem verbatim (MATH_SPEC §1.2).
 */
typedef struct {
    int               m;            /* number of constraints                 */
    int               nblocks;      /* number of blocks                      */
    int              *block_sizes;  /* length nblocks, signed (neg=diagonal) */
    char            **b;            /* length m, owned decimal strings        */
    arbsdp_matblock  *mats;         /* length (m+1)*nblocks                   */
    int               maximize;     /* SDPA convention flag (1 = maximize)   */
} arbsdp_problem;

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_problem_init -- zero-initialize an empty problem (no allocation).
 * Pair with arbsdp_problem_clear.  Safe to clear an init'd-but-never-read
 * problem.
 */
void arbsdp_problem_init(arbsdp_problem *p);

/*
 * arbsdp_problem_clear -- free all heap owned by the problem (block_sizes, the
 * b strings, every entry string, the mats grid) and re-zero it.  Idempotent on
 * an init'd problem (CLAUDE.md rule 7: matching init/clear, no leaks).
 */
void arbsdp_problem_clear(arbsdp_problem *p);

/* -------------------------------------------------------------------------
 * Parsing
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_read_sdpa -- parse a .dat-s file into `p` (which must be init'd; any
 * prior contents are cleared first).  Matches SdpaSparse.ts semantics: comment
 * lines starting with '*' or '"' are skipped; a trailing "= ..." on a line is
 * treated as a comment (stripped); the first four non-comment lines are
 * m, nblocks, the block sizes, the b vector; remaining lines are
 * "<matno> <blk> <k> <l> <v>" entries with 1-indexed k <= l and 1-indexed blk.
 *
 * Returns 0 on success, nonzero on malformed input (CLAUDE.md rule 5, fail
 * loud): unreadable file, bad/missing integer or decimal token, block-size
 * count != nblocks, b count != m, out-of-range matno/block/index, or k > l.
 * On failure `p` is left cleared (init'd, empty).
 */
int arbsdp_read_sdpa(arbsdp_problem *p, const char *path);

/* -------------------------------------------------------------------------
 * Materialization (precision-parameterized; rigorous ball enclosures)
 * ---------------------------------------------------------------------- */

/*
 * arbsdp_problem_block_mat -- build the symmetric block matrix for matrix
 * `matno` (0 = C, 1..m = A_matno) restricted to block `block`, at precision
 * `prec`, into the pre-initialized arb_mat `out`.
 *
 * `out` MUST already be an initialized arb_mat of side |block_sizes[block]| x
 * |block_sizes[block]| (asserted).  It is zeroed, then each stored entry value
 * (parsed from its decimal string via arb_set_str at `prec`) is written to
 * A_{ij}; for i < j the same value is also written to A_{ji}, producing the
 * genuine symmetric matrix A_{ij} = A_{ji} = v.
 *
 * NO factor of 2 is applied here.  The "2v" doubling for off-diagonals lives in
 * the Frobenius inner product (arbsdp_frob_inner over all n*n entries), where
 * the symmetric pair (i,j),(j,i) is naturally counted twice.  See the header
 * banner and golden/README.md.  For diagonal (LP) blocks (block_sizes<0), only
 * i==j entries occur; the result is a diagonal matrix.
 *
 * matno must be in 0..m and block in 0..nblocks-1 (asserted).
 */
void arbsdp_problem_block_mat(arb_mat_t out, const arbsdp_problem *p,
                              int matno, int block, slong prec);

/*
 * arbsdp_problem_b -- materialize the i-th right-hand side b_i into `out` at
 * precision `prec` (parsed from the stored decimal string via arb_set_str).
 * `out` must be initialized.  i must be in 0..m-1 (asserted).  Returns 0 on
 * success, nonzero if the stored string fails to parse (should not happen for a
 * problem produced by arbsdp_read_sdpa, which validates b at parse time).
 */
int arbsdp_problem_b(arb_t out, const arbsdp_problem *p, int i, slong prec);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_PROBLEM_H */
