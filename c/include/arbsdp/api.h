/*
 * include/arbsdp/api.h -- public ABI for libarbsdp
 *
 * This is the single versioned header that external callers (and ArbSDP.jl via
 * ccall) include.  Keep it minimal; implementation headers live in src/.
 *
 * Dependency: FLINT 3.x (Arb is folded into FLINT 3).
 * Link: -lflint -lmpfr -lgmp
 *
 * PRD §5.1, §5.3: the ABI is tiny, flat, and versioned.  Data crosses as
 * strings (option a) or status ints; never live Arb structs.
 */

#ifndef ARBSDP_API_H
#define ARBSDP_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Version
 * ---------------------------------------------------------------------- */

#define ARBSDP_VERSION "0.1.0"

/*
 * arbsdp_version -- return the library version string.
 * Returns a pointer to a static string equal to ARBSDP_VERSION.
 * Never returns NULL.
 */
const char *arbsdp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ARBSDP_API_H */
