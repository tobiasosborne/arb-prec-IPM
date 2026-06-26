# arbsdp

An arbitrary-precision semidefinite programming (SDP) solver that returns a
**rigorous interval (ball) enclosure of the optimal value** plus verified status —
not just high-precision digits, but a machine-checkable certificate. Built on
FLINT 3.x / Arb.

The architecture: **solve approximately in Arb point arithmetic, then certify
rigorously in Arb ball arithmetic** (Jansson-style verified bounds). The
interior-point iteration is never run in rigorous ball mode — rigor comes from a
separate certification step keyed off the dual residual and verified Cholesky.

This is aimed at the "small matrix, but the answer must be trustworthy to many
digits" regime: quantum-information bounds (NPA, entropy, ground-state energy),
sum-of-squares / Lasserre / moment relaxations, and extremal combinatorics —
where a certified bound is the scientific deliverable.

## Layout

- `c/` — `libarbsdp`, the standalone C core (FLINT/Arb). Builds and tests with
  `cmake` + `ctest`, no other dependencies.
- `ArbSDP.jl/` — the Julia cockpit: modeling, golden-master harness, diagnostics
  (planned).
- `golden/` — golden-master problems with closed-form / high-precision optima.
- `docs/` — `PRD.md` (scope + roadmap), `MATH_SPEC.md` (cited formulas),
  `FLINT_NOTES.md` (confirmed Arb API).
- `CLAUDE.md` — engineering rules and invariants.

## Build & test (C core)

```bash
cmake -S c -B c/build -DCMAKE_BUILD_TYPE=Debug
cmake --build c/build
ctest --test-dir c/build --output-on-failure

# memory-checked build
cmake -S c -B c/build-asan -DENABLE_ASAN=ON && cmake --build c/build-asan
ctest --test-dir c/build-asan
```

Requires FLINT 3.x (bundles Arb), GMP, MPFR, and a C11 compiler.
On Debian/Ubuntu: `sudo apt-get install libflint-dev`.

## Usage (CLI)

`arbsdp solve` reads an SDPA-sparse `.dat-s` problem, solves it (adaptive
precision), certifies a rigorous `[lb, ub]` bracket on the optimum (or a verified
infeasibility status), and prints a JSON result:

```bash
arbsdp solve problem.dat-s --target-digits 30 --trace-bound 1
```

```json
{
  "problem": { "m": 1, "nblocks": 1, "block_sizes": [2], "maximize": true },
  "solve": { "adaptive_status": "optimal", "final_prec": 256, "iters": 9, ... },
  "certificate": {
    "status": "optimal",
    "obj_lb": "2.999...994e+00",
    "obj_ub": "3.000...005e+00"
  }
}
```

`--trace-bound V` (or `B:V` per block) supplies the a-priori `tr X ≤ V` bound that
a finite rigorous bracket requires; without it an unbounded block is reported
honestly as `-inf`/`+inf`. The printed `obj_lb`/`obj_ub` are *directed* decimals
(lb rounded down, ub rounded up) so the printed interval still encloses the
optimum. A `status` of `primal_infeasible` / `dual_infeasible` is a verified
ball-arithmetic Farkas certificate. Run `arbsdp --help` for all options.

## Status

Early development, but the core product runs end to end: the Layer-0 adaptive
solve, the Layer-1 rigorous `[lb, ub]` certificate, verified primal/dual
infeasibility certificates, and the `arbsdp solve` CLI / JSON serializer are in
place. Tightening the lower bound for general coupled problems, and the Julia
cockpit (modeling, golden-master harness), are in progress. See `docs/PRD.md` for
the roadmap.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
