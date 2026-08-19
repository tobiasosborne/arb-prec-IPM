# HANDOFF — arbsdp (2026-08-19)

Five-minute orientation for the next agent. **Read this, then `CLAUDE.md`, then
`PRD.md` §2, then `docs/MATH_SPEC.md`.**

## ACTIVE MANDATE (2026-08-19): remediate the full-source review findings

A slow serial review of the entire C source (owner-requested, 2026-08-19) found
no P0 rigor bug on any reachable path, but a cluster of **float64 idioms ported
verbatim from the TS solver** that together cap the product's headline promise
(many certified digits), one latent rigor trap, and substantial avoidable cost.
Every finding is filed as a bead. **The remediation plan and its ordering are in
"Review remediation plan (2026-08-19)" below — orchestrate from there.**

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

- **epic.2 DONE (2026-06-19):** infeasible/unbounded corpus + the POINT-MODE
  (tau,kappa) status-class detection it needs. `convergence.c:166` previously
  STUBBED the infeasibility branch citing invariant 8; epic.2 (owner-created)
  authorizes the Layer-0 point-mode classification (it STEERS the loop / yields the
  status CLASS, exactly like the point-mode `OPTIMAL` status — the RIGOROUS
  ball-verified Farkas cert is still b20). Wired (port of TS `checkHsdeTermination`,
  ART 2003): primal-infeasible (`prstatus<-0.5 && dObj>1e-6 && dualInf small`) and
  dual-infeasible/unbounded (`prstatus<-0.5 && pObj<-1e-6 && primalInf small`),
  primal tested FIRST to disambiguate. **A `tau<1` collapse guard was REQUIRED
  beyond the TS port** (a P0 found in verification): on `disparate_scale_sqrt2`
  (cond~4e32) a precision-starved transient blows kappa to 3.6e16 with tau~10,
  prstatus=-1.0, faking a Farkas witness — `tau<1` rejects it (the cert is the
  tau->0 limit; genuine firings have tau in [7e-11, 4e-4]). `solve.c` breaks at the
  FIRING iterate (skips the best-by-gap snapshot restore — else the cert iterate is
  lost). `precision.c` is terminal on infeasibility (`ARBSDP_ADAPTIVE_INFEASIBLE`,
  no escalation). Corpus (6, all with analytic witnesses, auto-discovered):
  `primal_infeasible_psd`, `primal_infeasible_minor` (PSD 2x2-minor, Farkas
  y=(-1,-1,1)), `unbounded_eig`, `unbounded_lp` (LP/diag block), `dual_infeasible_diag`,
  `dual_infeasible_3x3`. test_golden now 22 problems, GATE PASS (38 pairs, 0 FAIL);
  the 16 optimal goldens UNCHANGED (no misfire). new `c/test/test_infeasible.c`
  (TDD: P2/P3/P5 incl. the firing-iterate case + feasible-no-misfire + mutation).
  15/15 ctest normal+ASan, -Wall clean, valgrind definitely/indirectly-lost=0.
  **b20** (rigorous ball-verified Farkas cert, certify side) now has its corpus and
  is the natural next bead.

- **b20 DONE (2026-06-26):** rigorous ball-verified Farkas infeasibility
  certificates — the certify-side, user-facing status (epic.2 supplied the corpus
  + the point-mode steering; b20 is the RIGOROUS verification, invariant 8).
  MATH_SPEC §5.7. Two theorem-grade verifiers in `certify.c` (ball arithmetic):
  `arbsdp_verify_primal_infeasible` — `D^b = sum_i y_i A_i^b` (RAW iterate y,
  file-sign A_i, NO purify/negate; sign pinned from iterate.c `R_d`) is NSD via a
  rigorous lambda_max upper bound `min(gershgorin_upper, verified-Cholesky-shift)
  <= 0` (INCLUSIVE — the singular witness `primal_infeasible_psd` has lambda_max
  EXACTLY 0, caught by Gershgorin's structural-zero row; the Cholesky shift CANNOT
  certify a singular NSD matrix, so both are kept) AND `b^T y > 0` (lbound, strict);
  `arbsdp_verify_dual_infeasible` — exact rank-1 COORDINATE recession ray `X_hat =
  E_kk` (PSD by construction, no Cholesky, works at the boundary, invariant 1):
  scans all (b,k) for `(A_i^b)_{kk} == 0` EXACTLY for all i (`arb_is_zero`,
  zero-radius, the oc7 exact-data idiom) AND `(C_file^b)_{kk} > 0`. New top-level
  `arbsdp_certify` dispatch is INFEASIBILITY-FIRST (primal, then dual, then
  bracket), NOT tau-gated — REQUIRED because `dual_infeasible_diag` fires at
  tau=4.1e-4 > TAU_HEALTHY=1e-6, and SAFE because the rigorous verifiers cannot
  misfire on a feasible problem (Farkas exclusivity). New
  `arbsdp_gershgorin_upper_bound` (rigorous lambda_max upper, mirror of the lower).
  DE-RISK FIRST (rule 3): a measurement probe confirmed all 6 goldens verify (no
  known_gap). GROUND TRUTH auditioned (rule 3): JCK 2007 §5 eq (5.1)/(5.3),
  Jansson 2009 Prop 3.1/3.2, Rump 2006 (verifying PSD of a SINGULAR matrix is
  ill-posed → use lambda_max-upper for the primal cert, rank-1 construction for the
  dual), Andersen-Roos-Terlaky 2003. TDD: new `c/test/test_farkas.c` — the 6
  goldens get the correct rigorous certify status (`primal_infeasible_psd` is
  GENUINELY both-infeasible → reports PRIMAL, tried first); discriminators (b^T y
  sign on the 4 dual goldens, coordinate detector on `minor`) + a feasible
  no-misfire battery including the FILE-SIGN-C dual guard (a regression to the
  internal `C_int` would spuriously fire dual-infeasible). ADVERSARIAL rigor review
  (rule 2/8): 8 feasible/boundary problems + 6 positive controls, 0 false
  positives. 16/16 ctest normal+ASan; valgrind definitely/indirectly-lost=0.
  Unblocks b23, b24. REMAINING: general non-coordinate / higher-rank dual-
  infeasibility rays (verified-linear-solve / Krawczyk, needs Layer-2 b24) → new
  bead **arb-prec-IPM-psv**.

- **b21 DONE (2026-06-26):** the `arbsdp solve` CLI + JSON result serializer — the
  product is now usable end to end. `c/src/cli.c` (`arbsdp solve file.dat-s
  [--prec N] [--prec-max M] [--target-digits D] [--trace-bound V | B:V]`) wires
  read_sdpa → `arbsdp_solve_adaptive` → build `arbsdp_apriori` from `--trace-bound`
  → `arbsdp_certify` (prec_c = final_prec+128) → `arbsdp_result_to_json` → stdout.
  Serializer in `io.c` (`arbsdp_result_to_json`, `arbsdp_arb_decimal`): emits a
  single JSON object (problem dims, solve {adaptive_status, value, final_prec,
  iters, prec_history}, certificate {status, prec_c, obj_lb, obj_ub,
  trace_bounds}). RIGOR (rule 2): `obj_lb`/`obj_ub` are DIRECTED decimals (lb via
  MPFR RNDD, ub via RNDU) so the printed interval still encloses the optimum after
  decimal rounding — unit-tested (lb-string <= x <= ub-string on 2/3, -2/3, exact,
  ±inf). Fail-loud CLI (rule 5): usage/IO/parse errors → stderr + nonzero exit;
  a completed run exits 0 regardless of optimal/infeasible/inconclusive (those are
  RESULTS in the JSON, not errors). Build note (rule 12): `<mpfr.h>` must precede
  FLINT headers for `arf_get_mpfr`; `cli.c` excluded from the library glob so no
  stray `main` in `libarbsdp.so`. TDD: `test_serialize.c` (directed-decimal rigor
  + serialize-after-solve, python3 `json.load` round-trip) and `test_cli.c` (drives
  the real binary: optimal/dual_infeasible JSON, error-path exit codes, JSON
  round-trip through python3). 18/18 ctest normal+ASan; valgrind 0 leaks on the
  solve path AND the error-exit paths. Demo: `arbsdp solve max_eigenvalue_2x2
  --target-digits 12 --trace-bound 1` → status optimal, [2.9999…994, 3.0000…005].
  Unblocks **b22** (Julia FFI). v1 JSON omits the full X/y/S enclosure dumps (just
  the bracket + status + diagnostics) — a richer payload is a later option.

- **n1x DONE (2026-06-26):** fixed a LATENT SIGN BUG in the minimize (maximize=0)
  objective path + wired minimize into the gated golden suite. ROOT CAUSE (rule 8,
  found by a de-risk probe): `iterate.c` negated `C_int = -C_file` UNCONDITIONALLY,
  but per MATH_SPEC §1.2 / invariant 7 the negate is correct ONLY for maximize —
  the internal loop minimizes `<C_int,X>`, so max needs `C_int=-C_file` and min
  needs `C_int=+C_file`; `recover_value` already applied the matching output sign
  conditionally. So a minimize solve returned `-(max<C,X>)` (measured: −3 for
  `min<diag(1,3),X> s.t. tr=1`, true min = 1). Every golden was maximize=1, so the
  whole green suite missed it. FIX: negate `C_int` iff `p->maximize` (one line +
  banner). Then minimize recovers `+pObj/tau = min<C_file,X>` (measured: 1; maximize
  unchanged at 3). GOLDEN INTEGRATION: added a `maximize` provenance key
  (`golden_provenance.{h,c}`, default 1 → 22 existing goldens unchanged), a
  `test_golden` override (`p.maximize = c->maximize` after read_sdpa; read_sdpa
  hardcodes 1), and the minimize golden `min_eig_path_4` (min `<tridiag(2,1)_4,X>`
  s.t. tr X=1; opt = lambda_min = (3−√5)/2 = 0.38196601125… to 65 digits;
  maximize=false). TDD: RED-GREEN unit test `test_minimize_sign` in test_solve.c
  (embedded min<diag(1,3)> → 1; sign-flip guard: maximize → 3) + the golden gate
  (min_eig_path_4 recovers 0.381966 to 31 digits, corr=PASS). Cross-check: the SAME
  C gives 3.618 maximizing / 0.382 minimizing (CLI confirms lambda_max=3.618…).
  23-golden gate PASS (40 pairs, 0 FAIL), 18/18 ctest normal+ASan, valgrind 0/0/0
  (also fixed a pre-existing test_solve.c flint_cleanup leak baseline). CAVEAT
  (flagged, follow-up **arb-prec-IPM-fjw**): Layer-1 certify (`arbsdp_certify` /
  certify_bracket) is still hardwired for max `<C_file,X>` — minimize CERTIFICATION
  is unsound until fjw (the minimize SOLVE is correct; the CLI has no minimize flag
  so certify-on-minimize is not yet reachable).

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
3. **arb-prec-IPM-9kg (P2) — CLOSED 2026-06-26 (premise disproven).** Goal was a
   golden whose wrong-sign bracket EXCLUDES opt so the containment gate bites. PROVEN
   IMPOSSIBLE: `lb <= opt <= ub` holds for ANY dual `y` given `xbar >= tr(X*)` (the
   Jansson bound is valid at every dual point), so bracket-containment is sign-robust
   and no discriminating golden exists. A wrong sign is just a non-optimal dual point
   — it loosens the bracket / mirror-swaps the tight side, never excludes opt. The
   sign IS defended by the lb-VALUE assertion already in `test_certify_bracket`'s sign
   self-test (wrong-sign lb strictly below correct lb). Corrected the overstated
   "wrong sign EXCLUDES opt / Z_int = -Z_ext" claims in `certify.h` + MATH_SPEC §5.5
   (that identity holds only in the both-flipped internal frame). De-risk probe +
   proof: rule 8 caught a false claim in the spec.
4. **b20 (P1, verified Farkas) — DONE 2026-06-26.** Rigorous ball-verified
   primal/dual-infeasibility certificates (`arbsdp_verify_primal_infeasible`,
   `arbsdp_verify_dual_infeasible`, `arbsdp_certify`; MATH_SPEC §5.7; test_farkas).
   All 6 epic.2 goldens certify rigorously; adversarial review found 0 false
   positives. Follow-up for general non-coordinate dual rays: **arb-prec-IPM-psv**.
5. **b21 (P2): CLI `arbsdp solve` + JSON serializer — DONE 2026-06-26.** The
   product runs end to end (`arbsdp solve file.dat-s --trace-bound V`); rigorous
   directed-decimal `[obj_lb,obj_ub]` + verified status as JSON. Unblocks b22 (Julia).
   (original note:) expose the bracket through the
   public ABI/CLI so the product is usable end to end.
6. **Performance track (P1, parallel):** **wdz** (residual bits/digit 32–38 from Schur
   conditioning + controller granularity), **b31** (warm-start / skip-doomed solves),
   **b32** (efficiency regression harness).
7. **arb-prec-IPM-9tm (P3):** UB-B residual upper bound (ybar + primal PSD projection)
   for problems with a known dual bound but no finite trace bound.

## Review remediation plan (2026-08-19) — the active work queue

Full-source serial review (headers + all 14 `c/src/*.c`, cross-checked against
MATH_SPEC; two suspicions verified by probe, one confirmed, one refuted and
dropped). Verdict: architecture right, Layer-1 ball arithmetic genuinely careful
(directed endpoints correct everywhere; no P0 on any reachable path). The golden
targets (13–15 digits) sit just below the threshold where the worst findings
bite — a textbook rule-6 situation. Findings, each filed as a bead:

### The anchor finding

- **qx4 (P1) — the convergence test runs in float64** (`convergence.c:107-135`,
  `solve.c:275-289,303`; `precision.c:171` silently clamps target_digits at 300).
  The purified gap is a difference of two doubles, quantized at ~1 ulp
  (~2.2e-16 rel): above ~16 digits the stop decision is rounding-noise-driven —
  the loop either overshoots μ far past target (how `min_eig_path_4` td=30
  passes; wasted iterations, a hidden wdz driver) or fires early at true gap
  ~1e-16 (the p68 class, papered over by the confirm gate, not fixed at source).
  td>300 is a hard silent ceiling. Fix: all six flags + prstatus + soft-optimal
  + achieved metric in arb midpoint arithmetic; tolerances from target_digits
  with no double round-trip; MATH_SPEC §3.8 update. Acceptance: new td=50 and
  td=120 goldens pass; iteration counts DROP at td=13–15; p68 discriminator
  still bites. **Land FIRST — it changes stopping behavior, so every
  performance re-baseline waits on it (wdz now depends on qx4).**

### Phase 0 — close the traps (days; independent, ready now)

- **agc (P1)** — certify.c never consults `p->maximize`: a minimize problem gets
  a wrong-objective "rigorous" bracket silently (unreachable via CLI today, one
  API caller away). One-line fail-loud guard now; real minimize certify stays fjw.
- **0v1 (P1)** — `nt_gfactor_build` ignores `arb_mat_cho`'s return
  (`direction.c:315`): silent garbage L poisons the corrector. Propagate as the
  honest cone-exit → NUMERICAL.
- **37w (P2)** — reader accepts `inf`/`nan` value tokens (verified by probe:
  `arb_set_str` accepts them); reject non-finite in `decimal_is_valid`.
- **zuf (P2, pre-existing)** — generate.wl FormatNum sign bug; quick, guards
  golden trust.

### Phase 2 — float64-idiom auditions (rule 3: measure, record the loser)

- **pw1 (P2, dep 0v1)** — fixed absolute 1e-14 jitter on S before Cholesky
  (`direction.c:307-315`) is comparable to/dominates λ_min(S)~μ in the endgame;
  audition none-vs-relative (`||S||·2^(-prec/2)`) vs status quo.
- **pp8 (P2)** — regularize.c constants (1e-12 lift, 1e-2/1e2 caps) are absolute
  float64 defaults; the Schur system permanently carries a ≥1e-12 absolute lift.
  Audition scale/precision-relative tiers.

### Phase 3 — performance (after qx4 re-baseline)

- **6b6 (P2, blocks ggj)** — constraint matrices re-parsed from decimal strings
  O(m²·nb) per iteration (schur.c inner loops, apply_A/At ×2 per iter,
  direction.c RHS loops, regularize.c diagnose on constant data). One-time
  per-(solve,prec) materialization cache; fold in the redundant stall-probe
  residual recompute (`solve.c:474` vs `:366`).
- **czx (P2)** — ~11 Jacobi eighs per block per iteration (NT 2 + gfactor 1 +
  4×psd_max_step×2); consolidate around one G-factor (W = GᵀG; TS computes
  max-step from the Cholesky factor). Cuts ~8/11. Fold-in: eigh's silent
  100-sweep cap → status. (Probe: eigh converges scale-invariantly, ~10 sweeps,
  1.5 ms n=8/prec=256 — the suspected norm-dependent tolerance stall was
  REFUTED by measurement.)
- **nux (P2)** — audition iterative refinement on the Schur solve (reuse L);
  standard in SDPA-GMP/SDPT3; likely the strongest wdz lever after qx4.
- **1mm (P3)** — audition `arb_mat_approx_*` for the point-mode KKT solve vs
  ball cho (ball cho fails-to-prove before approx stops being useful → premature
  escalation).

### Phase 4 — certificate tightness

- **ml6 (P2)** — λ_min bisection tol `2^(-prec/2)·||A||` (`certify.c:414`): only
  half of prec_c becomes certificate digits, and ~prec/2 O(n³) cho probes per
  bound. Tighten toward `2^-(prec-O(log n))`; seed the bracket from a point
  eigenvalue estimate (rigor unchanged).
- **t0u (P3, dep qx4)** — dual refinement of y at prec_c before certification
  (bounds hold for ANY y; tightens b^Ty cheaply; precursor to b24).
- **ehz (P2, deps agc + qx4)** — extend test_golden to CERTIFY tr-bounded
  goldens (the bracket half of rule 4 is currently not enforced per golden;
  min_eig_path_4 excluded until fjw; add the td=50/120 goldens to this gate).

### Hygiene (P3, mechanical)

- **5ly** — ball comparisons where point-mode midpoint compares are meant
  (iterate.c inf-norms, direction.c step min, regularize.c bumps).
- **jhw** — load-bearing `assert()`s become silent garbage under NDEBUG
  (`certify.c:380` found_hi, `io.c:483`).
- Pre-existing, confirmed still open: **7o8** (memcheck gate unwired), **axm**,
  **epic.1.1**.

### Suggested execution order

1. Phase 0 in any order (agc, 0v1, 37w, zuf) — one bead one commit each.
2. **qx4** (the anchor), including the td=50/120 goldens.
3. pw1 + pp8 auditions (direction quality), then 6b6 → czx → nux (perf, ratchet
   provenance budgets down), ggj once 6b6 lands.
4. ml6 → t0u → ehz (certificate tightness + gating).
5. Hygiene beads opportunistically, each its own bead/commit (rule 13 — no
   drive-bys).

Everything above is beads-tracked; `bd ready` reflects the dependency wiring
(wdz blocked on qx4, ggj on 6b6, ehz on agc+qx4, t0u on qx4, pw1 on 0v1).

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
2026-06-02; see the Layer-1 entry above). b20 (verified Farkas) DONE 2026-06-26.
Next: b21 (CLI/public ABI), 8jr (tighten `lb`, general coupled primal bound).
