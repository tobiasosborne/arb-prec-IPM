# HANDOFF — arbsdp (2026-06-01)

Five-minute orientation for the next agent. **Read this, then `CLAUDE.md`, then
`docs/PRD.md` §2, then `docs/MATH_SPEC.md`.**

## Your mandate (from the owner)

> The approximate solver is **very brittle**. Focus on getting it **correct and
> performant**.

This **reprioritizes** the roadmap. The Layer-1 certification chain (the eventual
"product") is **deferred** (`bd` status: b27, b17–b20 are `deferred`). Do **not**
start Layer-1 work until the Layer-0 approximate solver is correct and performant.
Your work is beads **b29 (P0 crash), b30 (P0 cone-safety), b31 (P1 perf), b32 (P1
tests)** — in roughly that order.

## What exists and works

`libarbsdp` (C, FLINT/Arb) builds with `cmake -S c -B c/build && cmake --build
c/build && ctest --test-dir c/build`. **11 ctests, green** in both the normal and
ASan (`-DENABLE_ASAN=ON`) configs. Layer 0 is feature-complete:
svec → linalg/spectral → NT scaling → SDPA reader/problem model → HSDE iterate /
residuals / convergence → NT-scaled Schur + Cholesky → 3-way Tikhonov → Mehrotra
predictor-corrector + step → main solve loop (`arbsdp_solve`) → adaptive precision
(`arbsdp_solve_adaptive`). Golden masters: 7 problems with 65-digit analytic
optima under `golden/` (incl. rank-deficient boundary cases).

**The solver LOGIC is correct.** Under ASan, all 7 golden problems reach the right
optimum (`bench/brittle_probe.c`). The brittleness is *not* wrong math — it is the
specific, fixable issues below.

## The brittleness — concrete evidence

Reproduce everything with `bench/brittle_probe.c` (build/run instructions in its
header; run from the repo root).

### 1. P0 CRASH — `read_sdpa` heap-corrupts on a non-init'd struct → **b29**
`arbsdp_read_sdpa` (io.c:273) calls `arbsdp_problem_clear(p)` at entry, which frees
`p->b` / `p->mats`. If the caller did **not** `arbsdp_problem_init(p)` first, those
are garbage pointers → heap corruption → SIGSEGV:
```
_int_free_merge_chunk  <-  arbsdp_problem_clear (io.c:63)  <-  arbsdp_read_sdpa (io.c:273)
```
**Masked under ASan; missed by every test** (tests init first). This is the kind of
latent UB that makes the binary "brittle" in real use. The b14 agent hit it in a
test and worked around it instead of fixing the library — the footgun is still
there.

### 2. P0 CORRECTNESS — the point-mode iterate leaves the PSD cone → **b30**
This is the **core brittleness**. Probe trajectory at `target=15` digits (each entry
is `prec:status:achieved_digits`):
```
trivial_2x2     128:NUM:10.8 256:NUM:12.4 512:NUM:13.0 1024:OPT:14.2
sdp_sqrt2       128:NUM:1.6 256:NUM:2.6 512:NUM:5.8 1024:NUM:11.5 2048:OPT:13.0
```
Every problem **returns NUM (the iterate left the cone) at low precision** and only
OPT at the final precision. Even `trivial_2x2` (optimum = 1, trivial) needs **1024
bits for 15 digits** — that is **~70 bits per digit**, versus the ~3.3 bits/digit
theoretically required. Most of the precision is being spent fighting the cone
boundary, not representing the answer.

Root-cause hypotheses (ranked; the fix should audition these — CLAUDE rule 3):
- **Step length is far too aggressive.** The Mehrotra safeguard clips the step to
  `0.999999` of the way to the boundary (direction.c / HsdeStepLength). Standard
  IPMs use a fraction-to-boundary `γ ∈ [0.9, 0.99]` to stay *strictly interior*.
  Stepping to 0.999999 puts the iterate essentially **on** the boundary, where the
  next NT scaling (which needs `X,S ≻ 0`) breaks down at finite precision.
- **No central-path neighborhood control / no recentering.** Nothing keeps the
  iterate near the central path; once it skews toward the boundary it cannot recover
  in point mode.
- **Silent garbage instead of fail-loud.** `psd_invsqrt` floors eigenvalues at
  `1e-300` (linalg.c) — near a singular `X` it returns huge finite numbers rather
  than signaling "left the cone" (violates CLAUDE rule 5). The NT scaling then
  produces garbage that surfaces as a later NUM, far from the root cause.

### 3. P0/perf — irrational rank-deficient ceiling + wasted work → **b30/b31**
`sdp_sqrt2` at `target=50`, `prec_max=20000`:
```
ALIM  20000 bits  9 solves  1336.9 ms
  128:NUM 256:NUM 512:NUM 1024:NUM 2048:NUM 4096:NUM:22.7 8192:NUM:28.1 16384:NUM:29.8 20000:NUM:31.8
```
**1.34 s, 9 full solves, 8 wasted, and it still cannot pass ~32 digits.** The ~28–32
digit ceiling is the point-mode iterate being driven out of the cone near `μ≈1e-28`;
b30 should raise/remove it. The wasted solves are b31.

### 4. P1 PERFORMANCE — cold-restart escalation → **b31**
`arbsdp_solve_adaptive` re-solves from the *initial point* at every precision,
doubling from `prec0=128`. The lower-precision solves almost always NUM (wasted),
yet are run anyway. No warm-start (lift the prior iterate). Fix: warm-start + skip
doomed low precisions (after b30 fixes bits/digit, `prec0` can track
`target_digits`).

### 5. P1 TEST GAP — green suite, broken binary → **b32**
The ctest suite is green yet the normal-build binary heap-corrupts on a realistic
usage pattern (#1) and wastes ~20× precision (#2). The tests give **false
confidence**: they only use the safe init pattern, only assert correctness (not
efficiency), and some memory bugs only appear in the **normal** build (not ASan).
Need: a bits-per-digit efficiency regression, `read_sdpa` fuzz/edge tests under the
normal build, and a CTest that *would have caught* the heap corruption.

## Prioritized work plan (the beads)

1. **b29 (P0):** fix the `read_sdpa` footgun. Make it foolproof. Add a normal-build
   CTest that passes a garbage struct and must not corrupt the heap.
2. **b30 (P0):** keep the iterate strictly inside the cone — fraction-to-boundary
   `γ`, neighborhood control, fail-loud cone ops. **Target: ~3–6 bits/digit**
   (trivial_2x2 → 15 digits at ≤ ~256 bits, not 1024). This is the heart of the
   mandate.
3. **b31 (P1):** warm-start escalation + skip doomed precisions. Measure wall-ms /
   sum-iters before vs after with `bench/brittle_probe.c`.
4. **b32 (P1):** the brittleness/robustness/efficiency test harness so this can't
   silently regress again.

`b28` (warm-start) is absorbed into b31. The Layer-1 chain (b27, b17–b20) stays
`deferred` until Layer 0 is correct + performant — then un-defer and resume there.

## How to work here (non-negotiable — see CLAUDE.md)

- **Ground truth + audition (rule 3):** research best-in-class (what SDPA/SDPT3/
  SeDuMi/Clarabel do to stay interior at high precision) and audition before
  implementing. Don't default to the TS port — the TS solver is float64 and never
  faced this regime.
- **Red-green TDD (rule 9)** with a **mutation check** that proves the test bites.
- **Test under BOTH build configs.** The crash here only showed in the normal
  build. ASan-green is necessary, not sufficient. Run `bench/brittle_probe.c`
  (normal build) as part of your loop.
- **Fail loud (rule 5):** a cone exit / singular scaling must return an honest
  status or assert — never produce silent garbage.
- **Arb memory discipline (rule 7):** ASan + valgrind clean, zero leaks.
- **One bead, one commit; commit and push after each** (`git push origin main`;
  remote is GitHub `tobiasosborne/arb-prec-IPM`, public, AGPL-3.0).
- **Serial only; no parallel Julia.** (Julia work — b08 etc. — is untouched and
  not on the critical path.)
- Track work in `bd` (beads); `bd ready` shows the unblocked set. `TaskCreate` is
  allowed for in-session scratch (owner override).

## Reproduce in 30 seconds
```bash
cmake --build c/build
gcc bench/brittle_probe.c -o /tmp/bp -I c/include -L c/build -larbsdp -lflint -lmpfr -lgmp -Wl,-rpath,c/build
/tmp/bp 15 8192      # the NUM-trajectory / bits-per-digit story
/tmp/bp 50 20000     # the sdp_sqrt2 ceiling + wasted-solve cost
# crash repro: delete the `arbsdp_problem_init(&p)` line in bench/brittle_probe.c, rebuild, run -> SIGSEGV
```

## Definition of done for this phase
- b29: no heap corruption on misuse; normal-build CTest guards it.
- b30: trivial/easy goldens converge at ~3–6 bits/digit; cone ops fail loud; goldens
  still match to working precision.
- b31: measurably fewer solves/iters/wall-ms to the same accuracy.
- b32: an efficiency + robustness suite that fails on today's code and passes after.
Then the solver is correct + performant, and Layer 1 (un-defer b27, b17–b20) begins.
