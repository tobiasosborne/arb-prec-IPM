# FLINT_NOTES.md — confirmed Arb/FLINT API for `libarbsdp`

Bead arb-prec-IPM-b01. Every symbol below was verified against the **installed**
headers in `/usr/include/flint/` (CLAUDE.md rule 12: names drift across releases).

## Environment
- FLINT **3.0.1** (`pkg-config --modversion flint` = 3.0.1; `libflint.so.18.0.1`).
- Arb is folded into FLINT 3; headers `flint/arb.h`, `flint/arb_mat.h`, `flint/acb_mat.h`.
- Link line: **`-lflint -lmpfr -lgmp`** (GMP 10, MPFR 6 present as runtime+dev).
- Compile flags (project standard): `-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -O2 -g -std=c11`.

## THE load-bearing fact (Layer-1 PSD certification kernel)
**verified-Cholesky-success ⇒ rigorous positive-definiteness proof: YES, via `arb_mat_cho` (returns 1).**
`int arb_mat_cho(arb_mat_t L, const arb_mat_t A, slong prec)` returns **1** iff `A`
is provably positive-definite over the input balls (and fills `L` with a lower
Cholesky enclosure), **0** if it cannot prove PD. Empirically confirmed by
`docs/probes/arb_mat_cho_probe.c`: returns 1 on a PD matrix (with `L·Lᵀ ⊇ A`),
0 on an indefinite matrix. This is the primitive behind the `λ_min`-via-shift
routine (b17) and all PSD certification (CLAUDE invariant 2 — no eig needed).

## Confirmed symbols

### Ball scalars — `flint/arb.h`
| Symbol | Signature | Notes |
|---|---|---|
| arb_init / arb_clear | `void arb_init(arb_t)` (inline) / `void arb_clear(arb_t)` | init/clear discipline (CLAUDE rule 7) |
| arb_set_str | `int arb_set_str(arb_t, const char*, slong prec)` | returns nonzero on parse error |
| arb_get_str | `char* arb_get_str(const arb_t, slong digits, ulong flags)` | caller frees with `flint_free` |
| arb_set_d | `void arb_set_d(arb_t, double)` | exact for representable doubles |
| arb_add/sub/mul/div | `void arb_*(arb_t z, const arb_t x, const arb_t y, slong prec)` | ball arithmetic |
| arb_sqrt | `void arb_sqrt(arb_t z, const arb_t x, slong prec)` | |
| arb_midref / arb_radref | macros → `arf_ptr` / `mag_ptr` | midpoint (arf) and radius (mag) for serialization |

### Ball matrices — `flint/arb_mat.h`
| Symbol | Signature | Semantics |
|---|---|---|
| arb_mat_init / arb_mat_clear | `void arb_mat_init(arb_mat_t, slong r, slong c)` / clear | |
| arb_mat_mul | `void arb_mat_mul(arb_mat_t res, const arb_mat_t, const arb_mat_t, slong prec)` | rigorous (ball) product |
| **arb_mat_cho** | `int arb_mat_cho(arb_mat_t L, const arb_mat_t A, slong prec)` | **1 ⇒ proven PD** (see above) |
| arb_mat_solve_cho_precomp | `void arb_mat_solve_cho_precomp(arb_mat_t X, const arb_mat_t L, const arb_mat_t B, slong prec)` | reuse one `cho` factor for many RHS — **predictor+corrector share this** (b14) |
| arb_mat_inv_cho_precomp | `void arb_mat_inv_cho_precomp(arb_mat_t X, const arb_mat_t L, slong prec)` | |
| arb_mat_spd_solve | `int arb_mat_spd_solve(arb_mat_t X, const arb_mat_t A, const arb_mat_t B, slong prec)` | 1 ⇒ proven SPD; one-shot factor+solve |
| arb_mat_solve | `int arb_mat_solve(arb_mat_t X, const arb_mat_t A, const arb_mat_t B, slong prec)` | **1 ⇒ A provably invertible** (rigorous enclosure); 0 otherwise |
| arb_mat_solve_lu / _precond | `int …(arb_mat_t X, const arb_mat_t A, const arb_mat_t B, slong prec)` | same return contract; `_precond` better-conditioned |
| arb_mat_inv | `int arb_mat_inv(arb_mat_t X, const arb_mat_t A, slong prec)` | 1 ⇒ proven invertible |
| arb_mat_ldl | `int arb_mat_ldl(arb_mat_t L, const arb_mat_t A, slong prec)` + `arb_mat_solve_ldl_precomp` | LDL alternative if needed |
| arb_mat_solve_tril / _triu | `void …(arb_mat_t X, const arb_mat_t L/U, const arb_mat_t B, int unit, slong prec)` | triangular solves |

### Approximate (point-mode, Layer-0) — `flint/arb_mat.h`
`arb_mat_approx_mul`, `arb_mat_approx_solve` (`int`, no rigorous guarantee),
`arb_mat_approx_lu`, `arb_mat_approx_inv`, `arb_mat_approx_solve_tril/_triu`.
Use these inside the IPM loop where rigor is not claimed (CLAUDE rule 1); they are
faster and do not inflate radii.

### Init/cleanup & memory — `flint/flint.h`
`void flint_free(void*)` (free strings from `arb_get_str`), `void flint_cleanup(void)`
(per-thread teardown). String ownership: `arb_get_str` allocates; caller must `flint_free`.

## Gotchas / findings
1. **No eigendecomposition in `arb_mat.h`** (grep count = 0). Real-symmetric eig is
   NOT available as an `arb_mat_*` routine.
   - Available only via **`acb_mat.h`**: approximate `acb_mat_approx_eig_qr`, and
     rigorous `acb_mat_eig_simple` / `acb_mat_eig_multiple` /
     `acb_mat_eig_global_enclosure` / `acb_mat_eig_enclosure_rump` — all **complex**.
   - **VERDICT for b05:** **port the TS cyclic-Jacobi eig from `PsdCone.ts` (option a).**
     Layer-0 needs real-symmetric `sqrt`/`invsqrt`/`eig` in POINT mode; a ported
     Jacobi is pure real-`arb` arithmetic, avoids packing into complex `acb` and the
     associated overhead, and is point-mode anyway (no rigor lost — Layer-1 uses
     verified Cholesky, never eig, per CLAUDE invariant 2). Keep `acb_mat_eig_*` in
     reserve only as an optional cross-check oracle.
2. **`arb_mat_solve` return is a theorem:** nonzero ⇒ `A` provably invertible and `X`
   is a true enclosure of `A⁻¹B`. A `0` return is itself information (not proven
   invertible) — fail loud (CLAUDE rule 5), do not treat as a numeric zero.
3. **Factor-once / solve-many:** `arb_mat_cho` + `arb_mat_solve_cho_precomp` is the
   intended pattern for the Mehrotra predictor+corrector sharing one Schur
   factorization (b12/b14). `arb_mat_spd_solve` is the one-shot convenience form.
4. **String memory:** always `flint_free` the result of `arb_get_str`; ASan/valgrind
   gate (CLAUDE rule 7) will catch omissions.

## Probe
`docs/probes/arb_mat_cho_probe.c` — compiles under `-fsanitize=address,undefined -g`
and prints: PD matrix → `arb_mat_cho` = 1 with `L·Lᵀ ⊇ A`; indefinite matrix → 0.
Leak-clean. This is the empirical anchor for the load-bearing fact above.
