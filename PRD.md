# PRD — `arbsdp`: an arbitrary-precision, rigorously-bounded SDP solver

**Status:** draft v0.1 (scoping)
**Author:** Tobias Osborne (with Claude)
**Date:** 2026-05-29
**Repo:** `~/Projects/arb-prec-IPM` (standalone; *not* part of `scientist-workbench`)

---

## 0. One-liner

A self-contained C library (`libarbsdp`) with a Julia cockpit (`ArbSDP.jl`) that
solves small semidefinite programs in arbitrary precision and returns **rigorous
interval (ball) enclosures of the optimal value and of the primal/dual
solution**, built on FLINT 3.x / Arb. High precision *and* a machine-checkable
certificate — not one or the other.

---

## 1. Vision & motivation

The market has high-*precision* SDP (SDPA-GMP/-DD/-QD — and SDPA-GMP looks
effectively unmaintained) and it has rigorous-*verification* tooling (VSDP /
INTLAB, Jansson's bounds) that sits *on top of* an external float solver. Nobody
ships a single, maintained, scriptable tool that does both: solve to N digits
*and* hand back a proof-grade enclosure. For "small matrix, but the answer has to
be *trustworthy* to many digits" problems — exactly the regime where a rigorous
bound is scientifically load-bearing — that gap is real.

**The product is the certificate.** High precision is table stakes (SDPA-GMP
already does it); what makes this worth building is that the output is a
*rigorous* `[lb, ub]` bracket on the optimum plus verified status
(optimal / primal-infeasible / dual-infeasible / inconclusive), with ball
enclosures on the iterates. If we only wanted digits, we would wrap SDPA-GMP and
stop.

### Target use cases (the "small but hard" regime)
- **Quantum information / many-body bounds:** NPA-style nonlocality hierarchies,
  separability/entanglement witnesses, ground-state energy lower bounds,
  entropy/relative-entropy bounds, channel capacities. Matrices are small;
  the *rigor* of the bound is the scientific deliverable. (Cf. the surrounding
  `~/Projects` tree: `qinfo`, FQHE, NPA-flavoured relaxations.)
- **Sum-of-squares / Lasserre / moment relaxations:** moment matrices are
  catastrophically ill-conditioned; float64 silently lies; a certified bound
  is the whole point of the relaxation.
- **Extremal / combinatorial bounds:** kissing numbers, code bounds, Delsarte-LP
  and its SDP strengthenings — fields where reviewers want a certificate.
- **Pedagogy / reference oracle:** a slow, exact, inspectable solver to validate
  fast float solvers (including the author's own TS solver and the
  MOSEK/COPT reverse-engineering work).

### Non-goals (hard "no" for v1)
- Not a general LP/conic workhorse, not fast, not large-scale, no sparsity
  exploitation, no crossover, no GPU. Performance target is "tractable for
  n up to ~tens per block, m up to ~low hundreds," not competitive wall-clock.
- Not a dependency of, nor dependent on, `scientist-workbench`. The TS solver is
  the *algorithmic source* (a cleanroom, IP-clean spec) and the *golden master*,
  nothing more.

---

## 2. The central architectural thesis

> **Solve approximately in high-precision point arithmetic; certify rigorously in
> ball arithmetic. Never run the interior-point iteration itself in rigorous ball
> mode.**

This is the single most important decision in the document, and the code we are
porting forces it:

1. **The NT path is built on matrix square roots, hence on symmetric
   eigendecomposition.** In `solver-ipm/src/cone/PsdCone.ts`, `ntScaling` =
   `X^{1/2}(X^{1/2} S X^{1/2})^{-1/2} X^{1/2}`, implemented via `psdSqrt` /
   `psdInvSqrt`, implemented via `eighJacobi`. Step lengths (`psdMaxStep`) need
   `λ_min` of a scaled increment. **Rigorous symmetric eigendecomposition is
   exactly what Arb does *not* do well** (its rigorous eig is complex/general via
   `acb_mat_eig_*`); its *strength* is verified Cholesky and verified linear
   solves. Trying to run NT scaling in rigorous balls means fighting the library
   on its weakest axis on every iteration.

2. **Ball radii blow up exactly at convergence.** The Schur/KKT condition number
   grows like `1/μ` by design as the path approaches the optimum. In ball
   arithmetic the enclosure width of each Cholesky grows like `κ·2^{-prec}`, so
   the radii balloon precisely when you want them tight. Rigorous-ball-IPM
   therefore needs precision escalating like `−log μ` *and* loses the eig kernel.

3. **You'd be enclosing the iterate, not the optimum.** An enclosure of an
   approximate point is not a bound on the optimal value. The rigorous bound has
   to come from a separate certification step *regardless* — so the ball
   arithmetic inside the loop is wasted.

4. **The optimum lives on the boundary of the PSD cone.** SDP solutions are
   typically rank-deficient (`X` singular at optimality), so a verified Cholesky
   of `X` *will fail* — you cannot certify `X ≻ 0`. The Jansson/VSDP construction
   sidesteps this: it certifies a bound from the **dual residual** `Z = C − Σ yᵢAᵢ`,
   which we keep strictly interior, using an a-priori bound on `tr X`. We never
   need to prove the primal `X` is PSD. A naive "enclose everything" approach
   dies on exactly this point.

**Consequence for the layers:**

| Layer | What | Numeric regime | Rigor |
|------|------|----------------|-------|
| 0 | Approximate HSDE-NT IPM solve (port of TS) | Arb **point** arithmetic (midpoints; radii not trusted) at adaptive working precision `prec` | none |
| 1 | Certification → `[lb, ub]`, verified status | Arb **ball** arithmetic + verified Cholesky + Jansson bounds | **rigorous** |
| 2 | Refinement (only if the L1 gap is too wide) | interval-Newton/Krawczyk on KKT, or precision-bump-and-resolve | rigorous |

The author's instinct — "implement a C version of my IPM, then add the layers" —
is correct, *and the reasoning matters*: you must port the solver (rather than
shell out to an external float solver as the Layer-0 oracle) **because you need
precision control inside the solve** that no external float64 solver can give. So
Layer 0 is genuinely the IPM port; Layers 1–2 are new code with no analogue in
MOSEK/COPT (which are not rigorous solvers).

---

## 3. Scope

### In scope (v1)
- **Primal-dual SDP in SDPA standard form**, multi-block:
  `max ⟨C,X⟩  s.t.  ⟨Aᵢ,X⟩ = bᵢ,  X = blkdiag(X₁,…,X_{nb}) ⪰ 0`
  (internally `min -⟨C,X⟩`, per `SdpProblem.ts`).
- **LP is subsumed** as `1×1` diagonal blocks; no separate LP path needed for
  correctness, though we keep the option of a fast diagonal specialization.
- **Dense** blocks only. Block sizes `n` up to ~tens; number of constraints `m`
  up to ~low hundreds. (These are soft; the limit is wall-clock at the precision
  requested.)
- HSDE-NT direction with Mehrotra predictor-corrector and 3-way Tikhonov
  regularization (port of `HsdeNtSdpSolver.ts` + `Regularization.ts`).
- Rigorous certification (Jansson lower/upper bound) + verified infeasibility
  certificates from the HSDE `(τ,κ)`.
- Adaptive precision controller.
- Input: SDPA sparse (`.dat-s`) reader; direct C struct / Julia builder.

### Out of scope (v1, revisit later)
- SOCP / exponential / power cones (NT theory is general, but the verified
  cone-membership kernels differ; defer).
- Sparse / chordal Schur exploitation, low-rank, presolve beyond diagonal scaling.
- Crossover to a vertex / exact rational solution.
- Distributed / parallel / GPU.
- An AHO or HKM direction (NT only; AHO retained in TS as a cross-check oracle).

---

## 4. Mathematical specification

### 4.1 Standard form & vectorization
Match the TS solver to make golden-master comparison exact:
- Blocks stored **dense row-major**, symmetrized (`symmetrize`).
- Internal `svec` with the **strict-Mosek √2 off-diagonal scaling** (as used by
  `sdp-solve`/`PSDCone`) so that `⟨A,X⟩ = svec(A)ᵀ svec(X)` and the Schur
  assembly is a plain Gram matrix. Document the convention once, in one header,
  and never deviate.

### 4.2 Layer 0 — the HSDE-NT solve (port target)
Port `HsdeNtSdpSolver.ts` (1175 LOC) — *not* the plain `NtSdpSolver` — because
the homogeneous self-dual embedding gives clean `(τ,κ)` infeasibility
certificates, which are part of the product (and match MOSEK's approach per
`MOSEK-decomp`). Per iteration:

1. Residuals (primal `Ax−bτ`, dual `Aᵀy+s−cτ`, gap `−cᵀx+bᵀy−κ`) — `Residuals.ts`.
2. NT scaling `W` per block — `ntScaling` (via approx eig/sqrt at `prec` bits).
3. Schur assembly `M_{ik} = Σ_b ⟨Aᵢᵇ, Wᵇ Aₖᵇ Wᵇ⟩` — `SchurAssembler.ts`.
4. 3-way Tikhonov regularization `(δ_p, δ_d, δ_g)` with bump-on-failure —
   `Regularization.ts`. Caps and escalation as reverse-engineered from COPT.
5. Cholesky factor `M` once; reuse for predictor + corrector.
6. Mehrotra affine (`σ=0`) then corrector (`σ=(μ_aff/μ)³`, clipped `[1e-?, 0.9]`).
7. Step length keeping `(X,S,τ,κ)` in their cones — `HsdeStepLength.ts` +
   `psdMaxStep`; Mehrotra safeguard `clip(max(0.95α, 2α−1), 1−ε)`.
8. Update; convergence test (6-flag, `Convergence.ts`); precision-controller hook.

**Spectral kernels run in approximate (point) mode.** `psdSqrt`/`psdInvSqrt`/
`eighJacobi` are ported to operate on arb midpoints at `prec` bits (or use Arb's
`arb_mat_approx_eig_qr`). Their inexactness is irrelevant: Layer 1 re-derives all
rigor from the *result*.

### 4.3 Precision controller (new; core design element)
Working precision `prec` (bits) is **adaptive**, not fixed:
- Start at `prec₀` (e.g. 128 bits) or `prec₀ ≈ requested_output_digits·log₂10 + margin`.
- **Escalate** (`prec ← 2·prec`, or model-based `prec ≈ c·(−log₂ μ) + margin`) when:
  (a) Schur Cholesky needs regularization beyond a threshold / fails;
  (b) the duality gap stalls (μ fails to decrease >X% for K iters — mirrors COPT's
      stall counter);
  (c) the relative residual floor stops improving.
- **Acceptance gate (cross-precision stability, bead arb-prec-IPM-p68):** an
  OPTIMAL solve at precision P is a CANDIDATE; the next-higher precision (2P)
  CONFIRMS it iff that solve is also OPTIMAL and its recovered objective agrees
  with the candidate's to ≥ `target_digits` relative decimal digits.  On
  confirmation, the controller returns the CANDIDATE result (lowest
  confirmed-sufficient precision, not the confirming solve).
  `ARBSDP_ADAPTIVE_OPTIMAL` is a POINT-MODE heuristic stop, not a rigorous
  guarantee; rigor is Layer 1's ball-arithmetic bracket (certify.c, b18/b19).
- **Caps:** `prec_max`, `iter_max`; on hitting them without a confirmed solve,
  return best iterate with status `inconclusive` / `ARBSDP_ADAPTIVE_LIMIT` and
  let Layer 1 still try to certify whatever bound it can.
- Optional **float64 warm-start**: run the existing solver (or a float64 inner
  loop) to get a starting point, then lift to arb — cheap and often halves iters.

### 4.4 Layer 1 — rigorous certification (the product)
Given an approximate primal `X̃ ⪰ 0`-ish and dual `(ỹ, S̃)` from Layer 0, compute
in **ball arithmetic** at certification precision `prec_c`:

**Rigorous lower bound on the optimum** (primal `max ⟨C,X⟩`):
1. Form the dual residual matrix per block `Zᵇ = Cᵇ − Σᵢ ỹᵢ Aᵢᵇ` as an
   `arb_mat` (enclosure includes rounding of `ỹ`).
2. Get a **rigorous lower bound** `d̲ ≤ λ_min(Z)` per block (negative allowed) —
   *not* via eig but via the verified-Cholesky shift trick: binary-search the
   largest `s` for which `arb_mat_cho(Z − sI)` succeeds (proves `Z ⪰ sI`), giving
   `d̲ = s`; if even `s` slightly negative fails, fall back to a verified
   Gershgorin/`‖·‖`-based bound.
3. With an **a-priori bound** `tr Xᵇ ≤ x̄ᵇ` (supplied; see §4.6),
   `optimum ≥ bᵀỹ + Σᵇ min(0, d̲ᵇ)·x̄ᵇ`  (Jansson–Chaykin–Keil).
4. The dual side is symmetric → **rigorous upper bound** from an approximate
   primal-feasible `X̃` and a bound on `‖y‖`.

Output `[lb, ub]` is a *theorem*: the true optimum is provably in the interval,
regardless of how badly Layer 0 behaved. Status:
- `optimal` if `ub − lb ≤ tol` (and both finite);
- `primal_infeasible` / `dual_infeasible` if the HSDE `(τ,κ)` certificate verifies
  in ball arithmetic (rigorous Farkas: e.g. `‖Aᵀy+s‖` enclosure ≤ `rel·(bᵀy)` with
  `τ→0, κ>0`);
- `inconclusive` otherwise (gap too wide / bounds infinite) → escalate to Layer 2.

### 4.5 Layer 2 — refinement (later phase)
Only invoked when L1 returns `inconclusive` with a finite-but-wide gap:
- **Cheapest:** bump `prec`, re-run Layer 0 from the warm iterate, re-certify.
- **Self-validating:** apply an interval-Newton / Krawczyk operator to the KKT
  system around the iterate box; if it contracts, an exact KKT point is *proved*
  to lie in the box, tightening both bounds. This is the only place an
  interval-Newton appears — and it is optional.

### 4.6 A-priori bounds (the catch — make it a first-class input)
Jansson bounds need a finite `tr Xᵇ ≤ x̄ᵇ` (and dually `‖y‖ ≤ ȳ`). For the target
use cases these are *free and natural*: density matrices have `tr = 1`, NPA moment
matrices have a normalized `[1,1]` entry, etc. The API must:
- accept explicit per-block trace bounds;
- offer a heuristic/auto mode (e.g. from a verified bounded-feasibility probe, or
  Jansson's perturbed-problem trick) that **clearly reports** when it had to guess;
- never silently assume boundedness — an unbounded `tr X` ⇒ `lb = −∞`, reported.

### 4.7 The output object (define the deliverable precisely)
```
struct asdp_result {
  status            // optimal | primal_infeasible | dual_infeasible | inconclusive | limit
  arb  obj_lb, obj_ub        // rigorous bracket on the optimal value
  arb_mat X_blocks[]         // primal enclosure (balls)
  arb     y[]; arb_mat S_blocks[]   // dual enclosure (balls)
  arb     tau, kappa         // HSDE homogenizers (enclosures)
  certificate{ kind, data }  // optimality bound provenance OR Farkas ray (verified)
  diagnostics{ iters, prec_history[], reg_history[], gap_history[], wall_ns }
}
```

---

## 5. System architecture

### 5.1 C core — `libarbsdp` (standalone, no Julia required)
Pure C against FLINT 3.x (Arb is folded into FLINT 3). Depends only on
**FLINT, GMP, MPFR** (and a C compiler). Self-testable via its own C test suite
and a tiny CLI driver (`arbsdp solve problem.dat-s --prec 256`).

Modules:
- `arbsdp/svec.{c,h}` — vectorization, √2 convention, block bookkeeping.
- `arbsdp/linalg.{c,h}` — `arb_mat` helpers: symmetrize, Frobenius inner, NT
  scaling (approx-eig path), Cholesky wrappers, triangular solves, **verified**
  Cholesky/`λ_min`-bound (for L1). Thin layer over Arb (`arb_mat_cho`,
  `arb_mat_solve`, `arb_mat_approx_eig_qr` — *exact spellings confirmed against
  the installed FLINT version at build time*).
- `arbsdp/solve.{c,h}` — HSDE-NT Mehrotra loop (Layer 0).
- `arbsdp/regularize.{c,h}` — 3-way Tikhonov.
- `arbsdp/precision.{c,h}` — the precision controller / escalation policy.
- `arbsdp/certify.{c,h}` — Jansson bounds, verified PSD test, Farkas verification (Layer 1).
- `arbsdp/refine.{c,h}` — interval-Newton/Krawczyk (Layer 2; later).
- `arbsdp/io.{c,h}` — SDPA `.dat-s` reader, result serializer (JSON).
- `arbsdp/api.h` — the **public ABI** (see §5.3).

Memory discipline: every `arb_*`/`arb_mat_*` has init/clear; adopt a strict
`SCOPED` macro convention + run the whole test suite under ASan/valgrind in CI to
catch leaks early (the #1 source of C-against-Arb bugs).

### 5.2 Julia cockpit — `ArbSDP.jl`
Depends on `libarbsdp` (via JLL or local `dlopen`) + standard Julia pkgs. Owns
everything that is *not* a hot numeric loop:
- **FFI bindings** (`ccall`) and marshaling (Julia `BigFloat`/`Arblib.Arb` ↔ C arb).
- **Modeling & I/O:** build problems, read/write SDPA format, pretty-print.
- **Public Julia API:** `solve(problem; prec, tol, trace_bounds, ...) ::Result`.
- **MathOptInterface backend** (recommended, later phase): plugs into JuMP, and —
  crucially — gives one-line access to *dozens of golden-master solvers*
  (Clarabel, Hypatia, SCS, SDPA, ProxSDP, and via wrappers Mosek/COPT).
- **Golden-master harness** (§6): generate/store/compare references, the
  "beat-or-bracket" reports, regression suite.
- **Diagnostics & viz:** central-path plots, ball-radius-vs-iteration,
  precision-escalation traces, `[lb,ub]` gap convergence.

### 5.3 The C/Julia boundary (small, flat, versioned ABI)
Keep the ABI tiny and stable:
- Opaque `asdp_ctx*` handle; create/configure/solve/query/destroy.
- Data crosses as **flat arrays of strings or limbs**, never live Arb structs, to
  avoid ABI coupling to FLINT's internal layout. Two options, pick one and pin it:
  (a) decimal/hex strings for arb midpoint+radius (simplest, version-proof);
  (b) shared `arb_struct` arrays (faster, but couples Julia's `Arblib.jl` FLINT
  version to the C lib's — workable since both can target the same JLL).
  **Recommendation:** start with (a) for correctness/stability; offer (b) behind a
  flag once the JLL pins a single FLINT.
- Status as `int` codes; rich diagnostics as a JSON string the Julia side parses.

### 5.4 "Completely independent" guarantee
```
libarbsdp  ──depends on──>  FLINT 3.x (Arb), GMP, MPFR        [no Julia, no sci-wb]
ArbSDP.jl  ──depends on──>  libarbsdp (JLL) + MOI/JuMP + Arblib.jl (refs only)
```
The TS `solver-ipm` appears nowhere in the dependency graph — it is consulted as
documentation + golden master only. The C lib builds and tests with `cmake && ctest`
on a machine that has never heard of Julia.

---

## 6. Golden-master strategy (the author's preferred workflow)

The author works best by beating reference implementations. Make this a
first-class subsystem, not an afterthought. **Two relations matter:**
- *beat*: our high-precision value agrees with a trusted reference and exposes
  more correct digits than a float solver could;
- *bracket*: our rigorous `[lb,ub]` **contains** the reference value — this is
  self-certifying and is the stronger claim.

Reference sources, in rough order of trust:
1. **Analytic / closed-form optima** — hand-built SDPs with known exact values
   (the ground truth; resolves "who's right when we disagree with a float solver").
2. **Independent arb computation** — for tiny problems, compute the optimum a
   second way in Julia via Nemo/Arblib (e.g. KKT root-find at high precision) to
   cross-check our own arb path.
3. **On-disk commercial solvers** — Mosek (`~/Projects/MOSEK-decomp/mosek`) and
   COPT (`~/Projects/COPT-decomp/copt`) binaries: float64 references + their
   iteration logs (the existing `probe_*.log` oracles).
4. **Julia conic solvers via MOI/JuMP** — Clarabel, Hypatia (both pure-Julia,
   clean references), SCS; one interface, many oracles.
5. **The author's TS solver** (`scientist-workbench/tools/sdp-solve`) — the
   algorithmic sibling; agreement here validates the *port* specifically.
6. **SDPA-GMP/-DD** if a working binary can be coaxed up — high-precision (not
   rigorous) cross-check; treat as "dead but useful," not a dependency.

Standard test sets: **SDPLIB** (small instances), **DIMACS**, plus a curated
"pathological" set (rank-deficient optima, no strict feasibility, ill-conditioned
moment matrices) — the cases where float solvers fail and rigor earns its keep.

Harness (in `ArbSDP.jl`): for each problem store reference value(s) + provenance;
run our solver; emit a report with `(our [lb,ub], reference, digits_agreed,
brackets?, wall, prec_used, iters)`; fail CI on regressions. **Log every silent
cap** (iter/prec/time limit hit) so "inconclusive" is never mistaken for "solved."

---

## 7. Numerical design — things that will bite (push from all angles)

- **Don't run the IPM in rigorous balls** (§2). Layer 0 is point-mode arb.
- **Certify from the dual side** — `X` is singular at the optimum, so verified
  Cholesky of `X` fails; the Jansson bound never needs it. Get this right or the
  whole thing collapses on its first real (boundary) problem.
- **`λ_min(Z)` lower bound via verified Cholesky-shift**, not eig — plays to Arb's
  strength, avoids its weakness.
- **Precision must escalate**; a fixed `prec` either wastes time (too high early)
  or stalls (too low late). The controller (§4.3) is core, not polish.
- **Regularization interacts with rigor:** the Tikhonov `(δ_p,δ_d,δ_g)` perturbs
  the *solved* system; that's fine for Layer 0 (approximate) but Layer 1 must
  certify the *original* problem, using the regularized solution only as a
  starting guess. Keep the books separate.
- **Warm-starting from float64** is high-value and cheap; design Layer 0 to accept
  an external starting point.
- **`tr X` bound provenance** is the usability cliff (§4.6) — make it explicit.
- **Reproducibility:** arb is deterministic given `prec`; record `prec_history`
  and the FLINT version in provenance so a bracket is reproducible bit-for-bit.

---

## 8. Testing & validation
- **C unit tests** (`ctest`): svec round-trips, NT-scaling identity `WSW=X`
  (to `prec` tolerance), Schur symmetry, verified-Cholesky correctness
  (must succeed iff PD), Jansson bound formula on hand-checked tiny cases.
- **Property tests:** randomly generated feasible SDPs with known constructed
  optima; the bracket must always *contain* the constructed value (rigor is
  falsifiable — a bracket that ever excludes the truth is a P0 bug).
- **Analytic oracle suite** (§6.1).
- **Golden regressions** (§6) wired into CI.
- **Memory:** full suite under ASan + valgrind; zero leaks gate merges.
- **Cross-impl:** agreement with the TS solver on shared instances (validates the
  port); bracketing of Mosek/COPT/Clarabel values (validates rigor).

---

## 9. Roadmap / phasing

Each phase has a crisp exit criterion. Phases 1–3 deliver the core product;
4–5 are hardening/ergonomics.

**Phase 0 — Foundations.**
Install/build FLINT 3.x (+GMP/MPFR dev). C skeleton, build system (`cmake`),
`arb_mat` helper layer, svec/cone structs, SDPA `.dat-s` reader, the minimal ABI,
Julia FFI binding + smoke test (round-trip a matrix C↔Julia), CI with ASan.
*Exit:* `arbsdp` CLI reads a problem and echoes it; Julia can `ccall` a stub and
get an arb back; ASan-clean.

**Phase 1 — Approximate solve (port, no rigor yet).**
Port HSDE-NT Mehrotra + regularization + convergence at *fixed* `prec`. Spectral
kernels in approx mode.
*Exit:* matches the TS solver and an analytic-optima set to working precision on
SDPLIB-small; "we ported it and it agrees."

**Phase 2 — Adaptive precision + iterate enclosures.**
Precision controller (escalation, caps, stall detection); float64 warm-start;
emit ball enclosures of the final iterate.
*Exit:* solves the pathological/ill-conditioned set that float solvers fail on, to
requested digits, with auto precision escalation; produces iterate balls.

**Phase 3 — Certification (the product).**
Layer 1: dual-residual Jansson `[lb,ub]`, verified Cholesky `λ_min` bound,
a-priori `tr X` interface, verified `(τ,κ)` infeasibility certificates, the
`asdp_result` object + status semantics.
*Exit:* every problem returns a *rigorous* bracket (or honest `inconclusive`);
brackets provably contain all reference values on the golden set.

**Phase 4 — Refinement + robustness.**
Layer 2 (interval-Newton/Krawczyk and/or precision-bump-resolve) to close wide
gaps; harden infeasible/unbounded handling; broaden the golden suite.
*Exit:* `inconclusive` rate near zero on the target use-case problems.

**Phase 5 — Ergonomics & packaging.**
MathOptInterface/JuMP backend; BinaryBuilder/Yggdrasil `ArbSDP_jll`; docs,
examples (a quantum-info bound end-to-end), diagnostics/viz; tagged release.
*Exit:* `using ArbSDP, JuMP; ...` works; `]add ArbSDP` installs a prebuilt binary.

*(Rough relative sizing: P0 small, P1 the biggest single chunk (the port),
P2 medium, P3 medium-large (new math), P4 medium, P5 medium. Single-developer
multi-month effort; P1+P3 are the long poles.)*

---

## 10. Risks & mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Rigorous symmetric eig is weak in Arb | High | Architecture avoids it: point-mode eig in L0, verified-Cholesky-shift for `λ_min` in L1. |
| Optimum on PSD boundary → can't certify `X≻0` | High | Certify from the dual side (Jansson); never require primal verified Cholesky. |
| Ball radii / precision blow-up near optimum | High | Adaptive precision controller with caps + honest `inconclusive`. |
| A-priori `tr X` bound unavailable | Medium | First-class API input; auto-mode that *reports* guessing; natural for target uses. |
| C-against-Arb memory bugs | Medium | Strict init/clear discipline, SCOPED macros, ASan/valgrind gate. |
| FLINT API spelling/version drift | Medium | Confirm symbol names at build; pin FLINT via JLL; thin wrapper isolates churn. |
| "Who's right?" when we beat a float ref | Medium | Analytic oracles + independent arb cross-check + the bracket is self-certifying. |
| C/Julia ABI fragility | Low-Med | Tiny flat string-based ABI v1; opaque handle; versioned. |
| Scope creep (SOCP/sparse/large) | Medium | Explicit non-goals; SDP-dense-small only in v1. |

---

## 11. Open decisions (need a call; defaults proposed)

1. **MOI/JuMP backend** — recommend *yes, Phase 5* (huge golden-master payoff).
   Default unless you want to avoid the JuMP dep in the core.
2. **ABI data format** — recommend *strings (option a)* for v1, shared-arb later.
3. **arb-midpoint vs `arf`/`mpfr` for L0 arithmetic** — **RESOLVED (bead
   arb-prec-IPM-epic.6, 2026-06-02):** Layer 0 operates on pure MIDPOINTS via
   **radius-clearing**: after each IPM iterate update, `iterate_clear_radii()`
   in `c/src/solve.c` zeros the Arb radii of `X`, `S`, `y`, `tau`, `kappa`
   in-place (`mag_zero(arb_radref(...))`, no allocation). This is what CLAUDE
   rule 1 ("solve in points, radii not trusted") requires; the tested
   `arb_mat_*` kernels are reused as-is; rigor remains exclusively Layer-1's
   job (ball arithmetic, separate code path).

   **Why `arf_t` was rejected:** FLINT 3.0.1 ships no `arf_mat` module — no
   `arf_mat_cho`, no arf eigendecomposition, no arf triangular solve; only
   scalar `arf.h`/`arf_types.h`. Switching would require reimplementing
   Cholesky, the Jacobi eigensolver, triangular solve, NT scaling, and Schur
   assembly from scratch, discarding the tested `arb_mat_*` kernels, for zero
   measured benefit over radius-clearing (CLAUDE rule 12). Radius-clearing
   IS the `arf` semantics (midpoint-only) obtained by reusing those kernels.
   `arf_mat` remains a future option only if a later FLINT ships it AND a
   measured improvement materialises.

   **Mechanism eliminated:** near the PSD boundary the Schur condition number
   grows ~1/μ, so the untrusted ball radius on each iterate eigenvalue was
   growing 30–60 bits per IPM step until λ_min's ball straddled zero, at which
   point `arb_rsqrt`/`arb_inv` in NT scaling returned a NaN midpoint →
   `iterate_is_valid` failed → `ARBSDP_SOLVE_NUMERICAL` → precision escalation.
   Clearing the radii each step prevents this without touching the arithmetic.

   **Measured outcome (test_golden, 2026-06-02):** bits/digit before → after:
   sdp_sqrt2 158→18, mixed_blocks 153→20, disparate_scale_sqrt2 153→38,
   max_eig_path_16 138→32, two_block_corr_coupled 131→33, lp_diagonal_block
   81→10, max_eig_tridiag_3x3 75→19, max_eigenvalue_2x2 74→19, max_eig_path_8
   68→17, max_eig_path_4 37→18, separable_12block 33→16, diag_weight_kappa6/10/12
   18→9, ill_conditioned_3x3 ~58→9, trivial_2x2 now 15 digits at prec=256 (was
   512 bits for 15 digits). Suite: 16/16 corr=PASS eff=PASS; wall time
   22.85 s → 6.37 s; recovered objectives unchanged. Residual efficiency gaps
   (disparate_scale_sqrt2 38, two_block_corr_coupled 33, max_eig_path_16 32
   b/d) are driven by Schur conditioning and precision-controller granularity;
   tracked by bead arb-prec-IPM-wdz (relates to b31 warm-start, p68 controller).

4. **Cone scope** — recommend *SDP + LP-via-diagonal-blocks only* in v1; SOCP later.
5. **Distribution** — recommend *local build first*, `ArbSDP_jll` via Yggdrasil at
   Phase 5.
6. **Repo layout** — single repo with `c/` (libarbsdp) + `ArbSDP.jl/` (Julia pkg),
   or two repos? Recommend *monorepo* for lockstep dev, split later if needed.

---

## 12. Appendix A — port map (`solver-ipm` → `libarbsdp`)

| TS source | LOC | → C module | Notes |
|-----------|-----|-----------|-------|
| `solver/HsdeNtSdpSolver.ts` | 1175 | `solve.c` | **the** port target (HSDE-NT) |
| `solver/NtSdpSolver.ts` | 931 | — | reference only (non-HSDE variant) |
| `solver/AhoSdpSolver.ts` | 640 | — | golden-master cross-check only |
| `solver/Regularization.ts` | 331 | `regularize.c` | 3-way Tikhonov |
| `cone/PsdCone.ts` | 202 | `linalg.c` | NT scaling, eig→approx; +verified kernels |
| `linalg/SchurAssembler.ts` | 50 | `linalg.c` | Schur Gram assembly |
| `linalg/Cholesky.ts` | 52 | `linalg.c` | → Arb `arb_mat_cho` |
| `linalg/IterativeRefinement.ts` | 172 | `linalg.c` | optional |
| `solver/Convergence.ts` | 88 | `solve.c` | 6-flag test, precision-relative tols |
| `solver/HsdeStepLength.ts` | 128 | `solve.c` | step-to-boundary |
| `solver/Residuals.ts` | 48 | `solve.c` | HSDE 3-row residuals |
| `format/SdpaSparse.ts` | 67 | `io.c` | `.dat-s` reader |
| `problem/SdpProblem.ts` | 46 | `svec.h` | data model, √2 convention |

## 13. Appendix B — environment / prerequisites (this machine)
- ✅ Julia 1.12.5, gcc 13.3, clang, cmake, make.
- ✅ GMP 10 + MPFR 6 runtime libs present.
- ❌ **FLINT 3.x not installed** — *prerequisite*: build/install FLINT 3.x with
  dev headers (bundles Arb). Confirm `arb_mat_*` symbol spellings against the
  installed version before coding `linalg.c`.
- ✅ Golden-master binaries on disk: Mosek (`MOSEK-decomp/mosek`),
  COPT (`COPT-decomp/copt`).
- Julia ref pkgs to add: `Arblib.jl`/`Nemo.jl` (independent arb), `JuMP` +
  `Clarabel.jl`/`Hypatia.jl`/`SCS.jl` (MOI golden masters).
```
```
