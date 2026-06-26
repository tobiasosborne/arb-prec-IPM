/*
 * include/arbsdp/io.h -- I/O entry points for libarbsdp.
 *
 * Scope:
 *   - b07: the SDPA-sparse (.dat-s) reader (declared in problem.h; re-exported
 *     here so a caller gets the full read-and-materialize API in one #include).
 *   - b21: the JSON result serializer (below) -- the testable core of the CLI
 *     (the CLI driver itself is a later step).  It turns a full solve+certify
 *     result into a single JSON object.  The directed-decimal helper it uses
 *     (arbsdp_arb_decimal) is rigour-critical: the printed [obj_lb, obj_ub] must
 *     still ENCLOSE the optimum after decimal rounding (CLAUDE.md rule 2).
 *
 * The .dat-s parsing semantics match SdpaSparse.ts exactly (so golden-master
 * comparison against the TS solver is meaningful) -- see problem.h for the data
 * model and the off-diagonal "2v" convention.
 */

#ifndef ARBSDP_IO_H
#define ARBSDP_IO_H

#include <flint/arb.h>

#include "arbsdp/problem.h"
#include "arbsdp/precision.h"
#include "arbsdp/certify.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_read_sdpa is declared in problem.h (it produces an arbsdp_problem).
 * This header re-exports problem.h so callers can `#include "arbsdp/io.h"` for
 * the full read-and-materialize API in one place.
 */

/*
 * arbsdp_arb_decimal -- a RIGOROUS directed decimal string for x.  round_up=0
 * rounds toward -inf (a guaranteed LOWER bound on x); round_up=1 rounds toward
 * +inf (a guaranteed UPPER bound).  `digits` = significant decimal digits.
 * Special: +/-inf -> "+inf"/"-inf"; NaN -> "nan".  Returns a malloc'd string the
 * caller frees.  Used so the printed [obj_lb, obj_ub] still ENCLOSES the optimum
 * after decimal rounding (CLAUDE.md rule 2).
 *
 * GUARANTEE (THEOREM, CLAUDE.md rule 2): the returned decimal satisfies
 *   round_up=0 -> value <= true x ;  round_up=1 -> value >= true x .
 * Mechanism: take x's directed endpoint (round_up ? ubound : lbound) into an
 * arf_t, convert to mpfr with the matching directed rounding (RNDU / RNDD), then
 * format with the matching directed rounding ("%.*RUe" / "%.*RDe").  Every
 * rounding step goes the SAME direction, so the printed decimal never crosses x.
 */
char *arbsdp_arb_decimal(const arb_t x, int round_up, slong digits);

/*
 * arbsdp_result_to_json -- serialize a full solve+certify result to a JSON
 * string (malloc'd; caller frees).  Hand-rolled (no JSON library); keys are
 * emitted in a fixed order.  The schema (see the .c banner) is a single object:
 *   { "arbsdp_version", "problem"{m,nblocks,block_sizes,maximize},
 *     "solve"{adaptive_status,value,final_prec,iters,prec_history},
 *     "certificate"{status,prec_c,obj_lb,obj_ub,trace_bounds} }.
 *
 * obj_lb uses arbsdp_arb_decimal(lb, round_up=0) and obj_ub uses (ub,
 * round_up=1) so the printed interval is a rigorous ENCLOSURE (CLAUDE.md rule 2);
 * lb/ub are the certify bracket endpoints (may be +/-inf).  `ar` is asserted
 * non-NULL (NOT NULL-safe).  prec_c is the certification precision; `ab` supplies
 * the per-block a-priori trace bounds printed in "trace_bounds".
 */
char *arbsdp_result_to_json(const arbsdp_problem *p,
                            const arbsdp_adaptive_result *ar,
                            arbsdp_certify_status cstatus,
                            const arb_t lb, const arb_t ub,
                            slong prec_c, const arbsdp_apriori *ab);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_IO_H */
