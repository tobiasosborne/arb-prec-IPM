/*
 * include/arbsdp/io.h -- I/O entry points for libarbsdp.
 *
 * v1 scope (this bead, b07): the SDPA-sparse (.dat-s) reader.  The reader builds
 * the precision-preserving arbsdp_problem data model defined in problem.h.  A
 * JSON result serializer is planned (see PRD §4.7 / api.h asdp_result) and will
 * be added here in a later bead.
 *
 * The .dat-s parsing semantics match SdpaSparse.ts exactly (so golden-master
 * comparison against the TS solver is meaningful) -- see problem.h for the data
 * model and the off-diagonal "2v" convention.
 */

#ifndef ARBSDP_IO_H
#define ARBSDP_IO_H

#include "arbsdp/problem.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * arbsdp_read_sdpa is declared in problem.h (it produces an arbsdp_problem).
 * This header re-exports problem.h so callers can `#include "arbsdp/io.h"` for
 * the full read-and-materialize API in one place.
 */

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_IO_H */
