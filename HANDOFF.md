# HANDOFF — arbsdp (2026-06-02)

Five-minute orientation for the next agent. **Read this, then `CLAUDE.md`, then
`PRD.md` §2, then `docs/MATH_SPEC.md`.**

## Your mandate (from the owner)

> The approximate solver is **very brittle**. Focus on getting it **correct and
> performant**.

This **reprioritizes** the roadmap. The Layer-1 certification chain (the eventual
"product") is **deferred** (`bd` status: b27, b17–b20 are `deferred`). Do **not**
start Layer-1 work until the Layer-0 approximate solver is correct and performant.

## Completed beads (all DONE)

- **b29 DONE** (commit fb57325): `read_sdpa` is foolproof on a non-init'd struct
  (init-not-clear at entry); normal-build regression test guards it.
- **epic.1 DONE** (commit cd8dc3b): `test_golden` — the GATED golden-master runner.
  Provenance schema v2 (`expected_status`, `target_digits`, `prec_cap`,
  `expected_max_bits_per_digit`, `expected_max_iters`,
  `known_gaps[{kind,bead,reason}]`) + auto-discovery (`golden_provenance.{h,c}`) +
  correctness/efficiency assertions with a `known_gap` XFAIL/XPASS **ratchet** (a
  stale gap fails the gate, forcing removal). Runs both build configs (12/12).
- **b30 DONE** (commit 1763bde): step rule auditioned empirically on the 7 baseline
  goldens (γ=0.9999 Mehrotra cap beats both pure fraction-to-boundary and the old
  0.999999) + **fail-loud cone ops** (strict-PD gate `λ_min > λ_max·2^(-prec/2)`
  replaces the silent 1e-300 floor; honest NUMERICAL on degradation, rule 5). All 7
  goldens reach target; the p68 false-optimals (trivial/ill_cond) resolved; 12/12
  both build configs.
- **epic.3 DONE** (commit effa9e9): 9 stress golden problems added (size family
  max_eig_path_4/8/16; conditioning families diag_weight_kappa6/10/12,
  disparate_scale_sqrt2; multi-block two_block_corr_coupled, separable_12block).
  All 9 correct; all 9 carry `efficiency` known_gaps pointing to epic.6. Analysis
  in `golden/STRESS_FINDINGS.md`.
- **epic.6 DONE** (2026-06-02): true-midpoint Layer-0 arithmetic via
  **radius-clearing** (`iterate_clear_radii` in `c/src/solve.c`). See the diagnosis
  correction below.
- **LAYER 1 — THE PRODUCT — FIRST RIGOROUS BRACKET SHIPPED (2026-06-02):** the
  certification chain was un-deferred (epic.6 met the correct+performant trigger) and
  built: **b27** (λ_min-bound audition → MATH_SPEC §5.3.1: verified-Cholesky shift
  beats eig/Weyl/Newton), **b17** (`certify.c`: `arbsdp_verified_psd`,
  `arbsdp_gershgorin_lower_bound`, `arbsdp_lambda_min_lower_bound` — `d <= λ_min` as a
  theorem via bisection on `arb_mat_cho(A-sI)`), **b18** (`arbsdp_dual_residual` ball
  enclosure + `arbsdp_apriori` xbar/ybar API + MATH_SPEC §5.5 upper-bound derivation
  = the dual-side MIRROR, xbar-only), **b19** (`arbsdp_certify_bracket` → rigorous
  `[lb,ub]`). RIGOR GATE PASS: `lb <= opt <= ub` for 7 tr-bounded goldens; ub tight to
  56–191 digits, lb loose (single-dual). 14/14 ctest normal + ASan. All pushed.

- **p68 DONE (2026-06-19):** the adaptive accuracy-contract bug. The controller's
  gate was status-only (accept the first `OPTIMAL` at inner tol `10^-target`),
  which gave NO guarantee the recovered objective met `target_digits`. Replaced
  with a CROSS-PRECISION STABILITY gate (`arbsdp_adaptive_confirmed`,
  `c/src/precision.c`): an `OPTIMAL` solve at precision P is a CANDIDATE; the next
  doubling (2P) CONFIRMS it iff that solve is also `OPTIMAL` and its recovered
  objective agrees with the candidate to `>= target_digits`. On confirmation the
  controller RETURNS THE CANDIDATE (final_prec stays at P; the confirming solve is
  verification overhead) — so `bits/digit` and provenance are UNCHANGED (zero
  golden churn). De-risk measurement first (rule 3): post-b30 all 16 goldens
  already deliver `digits >= target` with favorable mu->objective transfer
  (+0.6..+3.9 digits), so p68 was latent/b30-masked; the obvious `digits_from_mu >=
  target` gate was measured and REJECTED (it under-estimates accuracy, would
  over-reject 13/16). TDD: unit `test_confirm_gate` (the p68 discriminator; a
  status-only revert fails it) + integration `test_p68_victims_confirmed`
  (trivial_2x2 13.756d, ill_conditioned_3x3 13.744d at td=12, both
  ADAPTIVE_OPTIMAL, final_prec=128, `prec_history_len>=2` proving confirmation).
  14/14 ctest normal+ASan, valgrind definitely/indirectly-lost=0. HONEST framing
  (rule 1): ADAPTIVE_OPTIMAL is a POINT-MODE heuristic, NOT rigorous; the residual
  tolerance-limited case (both solves agree yet are short of the optimum) is
  Layer-1's certified-bracket job — follow-up **arb-prec-IPM-wf5** (blocked on
  oc7). Cost: one confirming solve per problem (warm-start b31 would amortize it).

- **oc7 part (a) DONE (2026-06-19):** tighten the rigorous `lb` for the
  FULLY-SEPARABLE per-block-trace family (vry generalization). `vry` made the
  primal Rayleigh `lb` tight only for single-block single-trace max-eig problems;
  `arbsdp_primal_lower_bound` (`certify.c`) is now generalized via a static
  `block_rayleigh_lb` helper summed over blocks, gated by a CONSERVATIVE
  applicability detector (m==nblocks, bijective decoupled exact-`I`-per-block
  constraints, `beta_b>=0` — a wrong accept is a P0, so anything else falls back
  to the dual bound). For `X_hat = (+)_b beta_b v_b v_b^T/(v_b^T v_b)` (PSD by
  construction, exactly feasible since each constraint is `I` on one block),
  `lb = sum_b beta_b R(v_b) <= opt`. Measured: `separable_12block` lb moved from
  `-2.5708` (gap_lo 83.14, the loose single-dual bound) to `opt-lb = 3.6e-113`
  (prec_c=384); the 8 single-block goldens are bit-identical (no regression);
  `two_block_corr_coupled`/`sdp_sqrt2`/`trivial_2x2` correctly REJECTED (detector
  returns 0, lb stays the valid dual bound). TDD: `test_separable_primal_tight`
  (tightness gate `opt-lb<1e-40` vs the closed-form `sum_c c/2+sqrt(c^2/4+1)`) +
  `test_detector_rejects_coupled`; detector + rigor mutations both bit. 14/14
  ctest normal+ASan, -Wall clean. REMAINING oc7: (b) general coupled/higher-rank
  via rigorous primal-PSD-projection (`two_block_corr_coupled` gap 6.36,
  `mixed_blocks`) -> **arb-prec-IPM-8jr** (blocks oc7, shares 9tm/UB-B machinery);
  (c) optional Rayleigh-Ritz subspace -> **arb-prec-IPM-801**. `wf5` stays blocked
  on oc7 until (b).

## Current state (2026-06-02)

`libarbsdp` (C, FLINT/Arb) builds with `cmake -S c -B c/build && cmake --build
c/build && ctest --test-dir c/build`. **12 ctests, green** in both the normal and
ASan (`-DENABLE_ASAN=ON`) configs. Layer 0 is feature-complete:
svec → linalg/spectral → NT scaling → SDPA reader/problem model → HSDE iterate /
residuals / convergence → NT-scaled Schur + Cholesky → 3-way Tikhonov → Mehrotra
predictor-corrector + step → main solve loop (`arbsdp_solve`) → adaptive precision
(`arbsdp_solve_adaptive`). Golden masters: **16 problems** with 65-digit analytic
optima under `golden/` (7 baseline + 9 stress; all correct; all reach target).

`test_golden`: **16/16 corr=PASS eff=PASS**. Suite wall time 6.37 s (was 22.85 s
before epic.6). Recovered objectives unchanged (point-mode answer identical →
no correctness regression). 12/12 ctest in both normal and ASan builds.

**The solver LOGIC is correct.** Under ASan, all 16 golden problems reach the right
optimum. The remaining gaps are efficiency-only (residual bits/digit; tracked by
bead arb-prec-IPM-wdz).

## Brittleness diagnosis + resolution

Reproduce with `bench/brittle_probe.c` (build/run instructions in its header; run
from the repo root).

### Diagnosis correction (important — b30 era finding)

The brittleness was **NOT** "the iterate leaves the cone" — at the low-prec NUMs,
X and S stayed *strictly PD* (e.g. λ_min(X)=9.52, λ_min(S)=1.06e-13 for trivial
at prec 256) and the cone kernels were accurate (~1e-54). The root cause was
point-mode **radius corruption**: the *untrusted* ball RADIUS grew like cond~1/μ
(invariant 3) until λ_min's ball straddled zero, so `arb_rsqrt`/`arb_inv` returned
a NaN midpoint → NUMERICAL → escalate. This explained the bits/digit floor of
18–158 across boundary-optimum problems.

### epic.6 RESOLUTION (2026-06-02)

`iterate_clear_radii()` in `c/src/solve.c` zeros the Arb radii of `X`, `S`, `y`,
`tau`, `kappa` after each IPM iterate update. In-place `mag_zero(arb_radref(...))`,
no allocation cost. This makes Layer 0 operate on pure midpoints as CLAUDE rule 1
requires. Rigor remains exclusively Layer-1's job.

Measured outcome (`test_golden`, 2026-06-02, bits/digit before → after):

| problem | before | after |
|---------|--------|-------|
| sdp_sqrt2 | 158 | 18 |
| mixed_blocks | 153 | 20 |
| disparate_scale_sqrt2 | 153 | 38 |
| max_eig_path_16 | 138 | 32 |
| two_block_corr_coupled | 131 | 33 |
| lp_diagonal_block | 81 | 10 |
| max_eig_tridiag_3x3 | 75 | 19 |
| max_eigenvalue_2x2 | 74 | 19 |
| max_eig_path_8 | 68 | 17 |
| max_eig_path_4 | 37 | 18 |
| separable_12block | 33 | 16 |
| diag_weight_kappa6/10/12 | 18 | 9 |
| ill_conditioned_3x3 | ~58 | 9 |
| trivial_2x2 | 512 bits for 15 digits | 15 digits at prec=256 |

Suite wall time: 22.85 s → 6.37 s (fewer cold-restart solves). 16/16
corr=PASS eff=PASS. Recovered objectives unchanged.

### Remaining inefficiency

Residual bits/digit: disparate_scale_sqrt2=38, two_block_corr_coupled=33,
max_eig_path_16=32. These are driven by Schur conditioning + precision-controller
granularity (geometric doubling overshoots). Tracked by bead **arb-prec-IPM-wdz**
(relates to b31 warm-start, p68 controller). This is a P1 performance issue, not
a correctness issue.

## Prioritized work plan (current)

Layer 1 (the product) is un-deferred and its **first rigorous bracket shipped**
(b27→b17→b18→b19 done). `arbsdp_certify_bracket` returns a rigorous `[lb,ub]`
containing the optimum for tr-bounded goldens. Remaining, roughly by value:

1. **Tighten `lb` (highest-value Layer-1 step) -- DONE 2026-06-02 (bead arb-prec-IPM-vry).**
   The dual-residual `lb` is structurally stuck at `lambda_min(C)` for max-eigenvalue
   problems (the [1,3] bracket; NB the old "project y_ext to make Z_ext PSD" idea does NOT
   work -- for a max problem that only lowers `b^T y_ext`). The fix is a rigorous PRIMAL
   Rayleigh lower bound: a feasible rank-1 point `X_hat = b_1 v v^T/(v^T v)` gives
   `lb = b_1 (v^T C v)/(v^T v) <= opt` for any `v` (PSD by construction; no Cholesky;
   MATH_SPEC §5.4.1). `arbsdp_certify_bracket` now returns `lb = max(dual, primal)`.
   Measured: max_eigenvalue_2x2/tridiag/path_4 lb moves from `lambda_min(C)` (gap ~2) to
   within `~1e-192` of opt; 14/14 ctest normal+ASan; mutation-checked (disable primal ->
   tightness fails; inflate lb -> rigor gate fails). Single-block single-trace only;
   per-block + general higher-rank/coupled follow-up: bead **arb-prec-IPM-oc7**.
2. **arb-prec-IPM-om9 (P1, rigor) — DONE 2026-06-02.** Trace-bound convention (no JCK
   `s_b` factor) re-verified against JCK 2007 directly (MATH_SPEC §5.3.1 "om9
   reconfirmation": JCK's `s_b` is Loewner-only; our trace form is `tr(DX) >=
   min(0,lambda_min(D))*xbar`, no factor) and CLAUDE.md citation corrected
   (`SIAM J. Numer. Anal.` 46(1):180-200). All 7 rigor-gate goldens audited: each `xbar`
   is an EXACT trace bound (trace pinned by diagonal equality constraints). New
   `test_trace_vs_loewner` discriminator (full-rank `X*=I`, n_b=2, opt=2): TRACE xbar=2
   brackets opt exactly; LOEWNER xbar=1 EXCLUDES (lb=2+2^-10, ub=2-2^-10). Mutation check
   (rule 9): a spurious `s_b` factor fails 4/6 assertions while rank-1 goldens stay green.
   14/14 ctest normal + ASan (zero leaks).
3. **arb-prec-IPM-9kg (P2):** a sign-discriminating golden so the bracket gate itself
   catches a `y_ext` sign regression (current max-eig goldens give symmetric
   `[-opt,+opt]` brackets that still contain opt under a wrong sign).
4. **b20 (P1, verified Farkas):** primal/dual-infeasibility certificates from
   `(tau,kappa)`. BLOCKED on **epic.2** (infeasible/unbounded goldens) — build epic.2
   first. (`tau < TAU_HEALTHY` already routes to INCONCLUSIVE in `certify_bracket`.)
5. **b21 (P2): CLI `arbsdp solve` + JSON serializer** — expose the bracket through the
   public ABI/CLI so the product is usable end to end.
6. **Performance track (P1, parallel):** **wdz** (residual bits/digit 32–38 from Schur
   conditioning + controller granularity), **b31** (warm-start / skip-doomed solves),
   **b32** (efficiency regression harness).
7. **arb-prec-IPM-9tm (P3):** UB-B residual upper bound (ybar + primal PSD projection)
   for problems with a known dual bound but no finite trace bound.

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
Layer 1 has since BEGUN and shipped its first rigorous bracket (b27→b17→b18→b19,
2026-06-02; see the Layer-1 entry above). Next: tighten `lb` (b24 / dual
correction), b20 (verified Farkas, needs epic.2), b21 (CLI/public ABI).
