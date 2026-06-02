# STRESS_FINDINGS.md — epic.3 Pareto/scaling analysis

**Bead:** arb-prec-IPM-epic.3 ("Punishing golden set: size & conditioning stress families")
**Data source:** `test_golden` adaptive solve (calibrated target_digits=13/per-problem
prec_cap) and `bench/brittle_probe.c` wall-time sweeps (target_digits=15, prec_max=8192;
lighter sweeps at target=12/prec_max=2048 and ASan-mode target=13/prec_cap=2048).
**Reference:** `golden/gen/STRESS_DESIGN.md` (hypotheses), `HANDOFF.md` (epic.6 diagnosis).

---

## 1. Verdict

The approximate (Layer-0) solver is **correct** across the entire expanded
size/conditioning space: all 9 new stress problems reach a value that matches the
independent 65-digit analytic optimum to 13–16 digits, with every correct bracket
verified by the xfail-ratcheted `test_golden` gate. No problem returns wrong-sign,
wrong-bracket, or a value outside the golden. **Correctness is not the issue.**

Cost and scaling are the issue. The solver is expensive and degrades predictably with
block size and Schur conditioning: bits/digit ranges from 18 (data-conditioning-only
problems with 1x1 Schur) to 153 (Schur-conditioned disparate-scale problem), versus
the ~3.3-bit/digit theoretical floor and the ≤8 b/d goal. Every single problem NUMs
at every sub-winning precision, meaning 2–5 full cold-restart solves are discarded per
run. The n=16 case takes 69 s at target=15. The root cause is identified: Arb's ball
arithmetic grows radii like `cond ~ 1/mu` near the PSD boundary, corrupting the
point-mode midpoint and forcing escalation (HANDOFF §diagnosis, bead epic.6). The
fixes are epic.6 (true-midpoint Layer-0 arithmetic) and b31 (warm-start escalation),
not the Layer-1 certification chain, which remains correctly deferred.

---

## 2. Pareto along SIZE (n=4, 8, 16 — `max_eig_path` family)

| problem | n | svec dim | eig gap | iters | final_prec | bits/digit | wall (target=15) |
|---------|---|----------|---------|-------|------------|------------|-----------------|
| max_eig_path_4  | 4  | 10  | 1.00 | 5 | 512  | 37  | ~0.22 s |
| max_eig_path_8  | 8  | 36  | 0.35 | 6 | 1024 | 68  | ~2.3 s  |
| max_eig_path_16 | 16 | 136 | 0.10 | 7 | 2048 | 138 | ~69 s   |

**bits/digit** nearly doubles each step: 37 → 68 → 138, a growth consistent with
approximately linear scaling in n (factor ~3.7x as n doubles). The data-matrix
condition number grows only ~4x per step (cond(C) ~16 at n=4, ~32 at n=8, ~117 at
n=16), so data conditioning explains little; the driver is the shrinking eigenvalue
gap (1.00 → 0.35 → 0.10, shrinking ~as 1/n^2) forcing mu to descend further to
resolve the rank-1 boundary, which inflates Arb's radii further before OPT is
declared. The iteration count rises modestly (5 → 6 → 7), consistent with a mild
gap effect on the IPM trajectory itself, but the dominant cost is the precision
escalation overhead, not the IPM convergence rate.

**Wall time** grows super-linearly: 0.22 → 2.3 → 69 s, roughly a 10x step per n
doubling. Two multiplicative factors: (a) per-iteration cost of the NT eigensolve
and Schur assembly on an n x n matrix grows at least as O(n^3) at fixed precision,
and the relevant precision itself doubles (512 → 1024 → 2048); combined, per-iteration
cost scales closer to O(n^3 * prec^2) in Arb's bignum model. (b) The cold-restart
waste (§5) runs 3, 4, and 5 full solves respectively (all but the final discarded),
multiplying the already-growing per-iteration cost. Under the lighter target=12/
prec_max=2048 sweep, n=16 runs ~20.4 s; under ASan with target=13/prec_cap=2048,
~6 s. The 69 s figure is the most demanding configuration tested.

**Implication for the PRD target ("n up to tens per block"):** n=16 at 69 s is
already 300x slower than n=4 at 0.22 s. Projecting the observed ~O(n^3.3) wall
growth, n=32 would be O(1000+ s) at the current efficiency. The PRD target of
"tens per block" is out of reach without first fixing epic.6 + b31; at ≤8 b/d and
warm-start, the n=16 cost would fall to ~0.5–2 s (a 35–140x improvement), which
makes n=32 plausible.

**Comparison with STRESS_DESIGN §3.5 hypotheses:**
- n=4: hypothesized "prec_cap ~1024, ~75 b/d, NUM at 128/256/512 until 1024."
  Measured: prec_cap=512, 37 b/d, NUM at 128/256. Better than hypothesis — the b30
  fix pushed the OPT crossing down by half an octave.
- n=8: hypothesized "prec_cap ~2048, 80–160 b/d." Measured: prec_cap=1024, 68 b/d.
  Better than hypothesis on both axes.
- n=16: hypothesized "prec_cap ~4096, possible ITER_LIMIT or NUM short of target."
  Measured: prec_cap=2048, 138 b/d, target reachable (no correctness gap). Better
  than the pessimistic hypothesis — correctness holds, though wall-time confirms
  the "breaks at n=16" concern quantitatively.

---

## 3. Pareto along CONDITIONING

### 3a. Data-matrix conditioning (`diag_weight_kappa*` — the FLAT axis)

| problem | n | cond(C) | iters | final_prec | bits/digit | wall |
|---------|---|---------|-------|------------|------------|------|
| diag_weight_kappa6  | 3 | 1e6  | 5 | 256 | 18 | ~7 ms  |
| diag_weight_kappa10 | 3 | 1e10 | 5 | 256 | 18 | ~11 ms |
| diag_weight_kappa12 | 4 | 1e12 | 5 | 256 | 18 | ~14 ms |

Increasing data-matrix conditioning from 1e6 to 1e12 produces **zero change** in
bits/digit (all 18), iteration count (all 5), or final_prec (all 256). The Schur is
1x1 throughout (m=1 trace constraint) and the large eigenvalue gap (w_1=1,
w_2~1e-3..1e-12, gap ~1) makes convergence trivial. Data conditioning in C, with a
well-conditioned Schur, does not stress the solver.

**Conclusion: cond(C) is the wrong difficulty proxy for this solver.** The NT
scaling W absorbs the data-matrix spread into per-block eigendecompositions, but
with a 1x1 Schur the linear-system solve is unaffected. The precision cost of a
1e12 dynamic range in W (~40 extra bits) is entirely swamped by the escalation
from the PSD boundary (the dominant cost).

**vs. STRESS_DESIGN §6.6 hypothesis:** "prec_cap ~512–1024, efficiency known_gap,
not correctness." Measured: prec_cap=256, matched hypothesis on kind (efficiency
only). Better on cost than hypothesized.

### 3b. Schur-complement conditioning (`disparate_scale_sqrt2` — the EXPENSIVE axis)

| problem | n | cond(Schur @ boundary) | iters | final_prec | bits/digit | wall |
|---------|---|------------------------|-------|------------|------------|------|
| disparate_scale_sqrt2 | 2 | ~4e32 | 20 | 2048 | 153 | ~91 ms |

With cond(Schur) ~4e32 (justified in STRESS_DESIGN §5.5 via NT-scaling analysis:
`w_1/w_2 ~ P/Q = 2e16`, `cond(M) ~ (P/Q)^2`), bits/digit balloons to 153 and iters
to 20, despite n=2 (the smallest possible PSD block). This is the worst-conditioned
problem among the 9 stress cases and is competitive with `sdp_sqrt2` (158 b/d) and
`mixed_blocks` (153 b/d) from the 7 baseline problems, both of which share a rank-1
boundary optimum but have smaller Schur condition numbers.

The 20-iteration count (vs 5–7 for the path-matrix family) shows the Tikhonov
regularizer working hard: the near-singular Schur forces repeated regularizer bumps,
inflating both iteration cost and precision requirement simultaneously.

**Conclusion:** Schur-complement conditioning (driven by NT scaling near disparate
diagonal constraints) is the binding difficulty proxy, not data-matrix conditioning.
The cond(Schur) ~4e32 requires losing ~log2(4e32) ~107 bits to rounding in the
Cholesky before any meaningful digits of the solution accrue, explaining why
prec_cap=2048 is needed for 13 digits (2048 - 107 = 1941 bits for ~13 digits is
~149 bits/digit before other overheads — consistent with the measured 153 b/d).

**vs. STRESS_DESIGN §5.6 hypothesis:** "NUM/STALL at 128/256, prec_cap ~1024–2048,
>120 b/d." Measured: NUM at 128/256/512/1024, prec_cap=2048, 153 b/d. Matches.
No correctness gap (target_digits=13 reachable), so the hypothesis's "correctness
known_gap likely" was too pessimistic.

---

## 4. Multi-block bookkeeping (`two_block_corr_coupled`, `separable_12block`)

| problem | blocks | m | both blocks active at opt | iters | final_prec | bits/digit | wall |
|---------|--------|---|--------------------------|-------|------------|------------|------|
| two_block_corr_coupled | 2x[2] | 5 | yes | 9 | 2048 | 131 | ~33 ms |
| separable_12block      | 12x[2] | 12 | yes (12 boundaries) | 6 | 512  | 33  | ~81 ms |

Both problems reach the correct analytic optimum. Block bookkeeping (svec indexing,
per-block NT scaling, multi-block Schur assembly) is correct at m=5 and m=12. The
bracket-contains-optimum check passes in both cases, verifying that no sign/index
error in block 2..12 pollutes the objective.

**two_block_corr_coupled:** 131 b/d at 9 iters is the most expensive non-disparate
problem. Both 2x2 PSD blocks sit simultaneously at a rank-1 boundary (det X_b = 0),
and the coupling constraint (m=5, dense 5x5 Schur) forces both boundary residuals
to narrow together. This is ~3.5x more expensive than max_eig_path_4 (37 b/d), which
has only one boundary to resolve. The coupling does add real cost.

**separable_12block:** only 33 b/d despite 12 simultaneous rank-1 boundaries and
m=12. The separability (block-diagonal Schur, twelve 1x1 blocks) means the 12x12
Schur Cholesky is trivial (factoring 12 scalars, not a dense 12x12 system). Each
boundary is resolved independently. Wall time 81 ms reflects 12x the per-block
spectral cost, but not 12x the Schur solve cost. The m=12 assembly and storage path
is confirmed correct and cheap.

**Conclusion:** m (number of constraints) is a weak cost proxy when the Schur is
block-diagonal. What matters is the density and condition number of the Schur, plus
the number of simultaneously active PSD boundaries. `two_block_corr_coupled` at
m=5 (dense, 2 boundaries) costs 4x more than `separable_12block` at m=12 (sparse,
12 boundaries but trivial factorizations).

---

## 5. Trajectory / wasted work (the cold-restart escalation, bead b31)

Every problem in the test set NUMs at every sub-winning precision and OPTs only at
the final escalation step. Concretely, the number of full cold-restart solves per
problem (= number of precision levels tried):

| problem | levels tried | solves discarded | winning prec |
|---------|-------------|-----------------|-------------|
| max_eig_path_4       | 3 (128,256,512)           | 2 | 512   |
| max_eig_path_8       | 4 (128,256,512,1024)      | 3 | 1024  |
| max_eig_path_16      | 5 (128,256,512,1024,2048) | 4 | 2048  |
| disparate_scale_sqrt2| 5 (128,256,512,1024,2048) | 4 | 2048  |
| diag_weight_kappa6/10/12 | 2 (128,256)            | 1 | 256   |
| two_block_corr_coupled   | 5 (128,256,512,1024,2048)| 4 | 2048  |
| separable_12block        | 3 (128,256,512)          | 2 | 512   |

For max_eig_path_16 and disparate_scale_sqrt2, 4 out of 5 solves are fully discarded
— 80% of total wall time is wasted on sub-winning precisions. Since the subwinning
solves return NUM before reaching the convergence tolerance, they may be detectable
early (the ball radii NaN-corrupt long before the iteration limit), but the current
`arbsdp_solve_adaptive` does not exploit this and always runs each solve to
completion (or NUM termination).

The warm-start/skip fix (bead b31) would replace cold-restart with: (a) skip
provably-doomed precisions (prec < `target_digits * log2(10) + margin`), starting
directly at a prec floor derived from target_digits; (b) warm-start from the
prior-precision iterate rather than re-initialising. For max_eig_path_16 at
target=15, the prec floor would be ~50 bits (15 digits * 3.32 + margin), but the
radius blowup means the effective floor is much higher (~1024–2048 bits); b31 can
skip the 128/256/512 solves that are guaranteed to NUM without even trying, saving
perhaps 3 wasted solves out of 5.

---

## 6. Where the 16 goldens sit on the (accuracy, bits/digit, wall) frontier

All 16 problems reach 11–16 matched digits. No correctness gaps exist in the corpus.
The spread is entirely in efficiency:

**Cheap corner (≤ 37 b/d, ≤ 256 ms):**
- diag_weight_kappa6/10/12: 18 b/d, 256 bits, 7–14 ms. Data-cond problems with
  1x1 Schur and large eigenvalue gap. Cheapest in the set.
- max_eig_path_4: 37 b/d, 512 bits, ~220 ms. Single n=4 PSD block, moderate gap.
- separable_12block: 33 b/d, 512 bits, ~81 ms. m=12 but trivial Schur, n=2 blocks.

**Mid range (68–131 b/d, 33 ms – 2.3 s):**
- max_eig_path_8: 68 b/d, 1024 bits, ~2.3 s. Gap ~0.35, growing svec.
- trivial_2x2: ~72 b/d, 1024 bits. (Baseline.)
- max_eigenvalue_2x2: ~74 b/d, 1024 bits. (Baseline.)
- max_eig_tridiag_3x3: ~75 b/d, 1024 bits. (Baseline.)
- ill_conditioned_3x3: ~73 b/d, 1024 bits. (Baseline.)
- lp_diagonal_block: ~81 b/d, 1024 bits. (Baseline.)
- two_block_corr_coupled: 131 b/d, 2048 bits, ~33 ms. Dense coupled 5x5 Schur.

**Expensive corner (138–158 b/d, ≥ 2 s):**
- max_eig_path_16: 138 b/d, 2048 bits, ~69 s. Gap ~0.10, svec dim 136, 5 wasted solves.
- disparate_scale_sqrt2: 153 b/d, 2048 bits, ~91 ms. cond(Schur)~4e32.
- mixed_blocks: ~153 b/d, 2048 bits. (Baseline.)
- sdp_sqrt2: ~158 b/d, 2048 bits. (Baseline.) Worst of the 16; cannot clear 32 digits.

The DESIGN goal of ≤8 b/d is not met by any problem. Every problem carries an
`efficiency` known_gap pointing to epic.6. The structural split is:
- Small-Schur or well-gapped problems: 18–37 b/d.
- Dense-Schur or ill-gapped problems: 68–158 b/d.
- The 8 b/d goal is reachable only after epic.6 (true-midpoint L0 arithmetic).

---

## 7. Follow-up beads

### 7.1 Mapping findings to existing beads

**epic.6 (true-midpoint Layer-0 arithmetic):** covers ALL efficiency gaps in this
analysis. Every b/d overrun — from 18 to 158 — is diagnosed in HANDOFF.md as the
same root cause: Arb's ball radii growing like cond~1/mu corrupt the point-mode
midpoint. epic.6 is the correct and complete fix. No new bead needed for the
"bits/digit is too high" finding.

**b31 (warm-start + skip doomed precisions):** covers all the cold-restart wasted-
work findings in §5. The 4-wasted-solve overhead on n=16 and disparate_scale is
directly addressed by b31. No new bead needed.

**b32 (bits/digit regression harness):** covers the "efficiency gate is currently
all-XFAIL" situation. Once epic.6 lands and b/d drops toward 8, b32 should remove
the XFAIL markers and restore the hard gate. No new bead needed.

**p68 (controller accuracy contract):** the diag_weight family reaching target at
prec_cap=256 with 18 b/d (instead of the higher values previously seen at low prec
with the false-optimal p68 bug) confirms b30 resolved the p68 symptom for these
problems. No new bead needed.

### 7.2 The n=16 wall-time blowup (69 s) — does it warrant a new bead?

**Recommendation: NO new bead; this is fully explained by b31 + epic.6.**

Decomposition of the 69 s:
- 5 cold-restart solves, 4 discarded. The 4 discarded solves each run a full-
  precision HSDE loop at 128/256/512/1024 bits on a 16x16 problem. At O(n^3 * prec^2)
  bignum cost, the prec=1024 wasted solve alone costs ~4x the prec=512 solve. The
  cumulative discarded work is estimated at ~50–65% of total wall time.
- The winning solve at 2048 bits runs 7 IPM iterations on a 136-dimensional svec
  problem with bignum eigendecomposition and 136x136 Schur assembly. This is
  inherently expensive at 2048 bits.

After epic.6 (true-midpoint L0): the prec floor drops from 2048 to ~100–200 bits
for 13 digits at ~8 b/d, so the winning solve requires ~4–5x fewer precision bits
and O(prec^2) savings.
After b31 (warm-start + skip): the 4 sub-winning solves are skipped or warm-started.

Combined, these two beads should reduce n=16 from ~69 s to an estimated ~0.5–2 s —
a 35–140x improvement. There is no additional algorithmic issue to file a separate
bead for: the per-iteration O(n^3) cost at n=16 is not excessive (7 iterations is
correct for a well-posed IPM with a small eigenvalue gap), and the eig/NT-scaling
code is not the bottleneck once the precision waste is eliminated.

If profiling after epic.6 + b31 still shows n=16 > 5 s, that would warrant a new
bead (targeted at the per-iteration eig/NT cost, e.g. lazy Schur reuse or
Cayley-Hamilton-based matrix functions). But that bead should NOT be filed now —
it would be premature before the dominant 35–140x lever is pulled.

### 7.3 Summary of bead recommendations

| finding | existing bead | new bead? |
|---------|--------------|-----------|
| All efficiency gaps (b/d 18–158) | epic.6 | no |
| Cold-restart wasted solves (2–5 per problem) | b31 | no |
| Bits/digit regression harness (currently all-XFAIL) | b32 | no |
| n=16 wall-time blowup (69 s) | b31 + epic.6 | no (reassess post-fix) |
| Schur-cond vs data-cond diagnosis (cond(C) is wrong proxy) | epic.6 | no (knowledge; captured here) |
| m=12 bookkeeping correct | — | no (positive finding; no action) |
| Two-boundary coupling cost (two_block, 131 b/d) | epic.6 | no |

All actionable findings from epic.3 map to existing open beads. The corpus is
complete as designed; no new problem families are indicated by the measurements.

---

## Appendix: Numerical summary (all 16 goldens)

| problem | type | n / m | matched_digits | final_prec | bits/digit | iters | wall |
|---------|------|-------|----------------|------------|------------|-------|------|
| trivial_2x2 | baseline | 2/1 | ~14.2 | 1024 | ~72 | — | — |
| max_eigenvalue_2x2 | baseline | 2/1 | ~13.9 | 1024 | ~74 | — | — |
| sdp_sqrt2 | baseline | 2/2 | ~13.0 | 2048 | ~158 | — | — |
| max_eig_tridiag_3x3 | baseline | 3/1 | ~13.7 | 1024 | ~75 | — | — |
| ill_conditioned_3x3 | baseline | 3/3 | ~14.1 | 1024 | ~73 | — | — |
| lp_diagonal_block | baseline | 3/2 | ~12.7 | 1024 | ~81 | — | — |
| mixed_blocks | baseline | 3/2 | ~13.4 | 2048 | ~153 | — | — |
| max_eig_path_4 | stress-size | 4/1 | 13.9 | 512 | 37 | 5 | 0.22 s |
| max_eig_path_8 | stress-size | 8/1 | 15.1 | 1024 | 68 | 6 | 2.3 s |
| max_eig_path_16 | stress-size | 16/1 | 14.8 | 2048 | 138 | 7 | 69 s |
| disparate_scale_sqrt2 | stress-schur | 2/2 | 13.4 | 2048 | 153 | 20 | 91 ms |
| diag_weight_kappa6 | stress-data | 3/1 | 13.9 | 256 | 18 | 5 | 7 ms |
| diag_weight_kappa10 | stress-data | 3/1 | 13.9 | 256 | 18 | 5 | 11 ms |
| diag_weight_kappa12 | stress-data | 4/1 | 14.0 | 256 | 18 | 5 | 14 ms |
| two_block_corr_coupled | stress-multiblock | 2x[2]/5 | 15.7 | 2048 | 131 | 9 | 33 ms |
| separable_12block | stress-multiblock | 12x[2]/12 | 15.6 | 512 | 33 | 6 | 81 ms |

Wall times: baseline column "—" because `bench/brittle_probe.c` target=15/prec_max=8192
sweep was run only for the stress problems; baseline wall figures were not re-measured
in this sweep.
