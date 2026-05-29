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

## Status

Early development. Foundations (svec, point-mode spectral/NT scaling, build
infra, golden masters) are in place; the Layer-0 solve and Layer-1 certification
are in progress. See `docs/PRD.md` for the roadmap.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
