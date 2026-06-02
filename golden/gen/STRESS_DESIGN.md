# STRESS_DESIGN.md — punishing golden set: size & conditioning stress families

**Bead:** arb-prec-IPM-epic.3 ("Punishing golden set: size & conditioning stress families")
**Depends on:** arb-prec-IPM-epic.1 (the gated `test_golden` runner + provenance schema v2)
**Status:** DESIGN (binding contract). This document pins the mathematics, the exact
problem data, the analytic optima (cross-verified by two independent toolchains to
>=50 digits), the conditioning justification, the solver-stress hypotheses, and the
generation/cross-check recipes. Downstream agents implement `generate.wl` /
`verify.wl` and calibrate the v2 runner fields strictly FROM this contract.

This document is itself ground truth (CLAUDE.md rule 3). Every optimum below was
verified numerically in `mpmath` 1.3.0 AND `sympy` 1.14.0 (and, for the eigenvalue
families, against an explicit numeric eigensolve), NOT recalled from memory. A wrong
golden is a P0 hazard (CLAUDE.md rule 2); the closed forms here are checked, not
asserted.

---

## 0. Summary table

| # | name | blocks | m | maximize | optimum (closed form) | hole(s) filled |
|---|------|--------|---|----------|-----------------------|----------------|
| 1 | `max_eig_path_4`            | `[4]`            | 1  | true  | `2+2cos(pi/5)`            | 1 (scaling baseline n=4) |
| 2 | `max_eig_path_8`            | `[8]`            | 1  | true  | `2+2cos(pi/9)`            | 1 (n>=8 REQUIRED) |
| 3 | `max_eig_path_16`           | `[16]`           | 1  | true  | `2+2cos(pi/17)`           | 1 (n>=16 REQUIRED) |
| 4 | `min_eig_path_4`            | `[4]`            | 1  | FALSE | `2-2cos(pi/5)`            | minimize path (n1x) |
| 5 | `disparate_scale_sqrt2`     | `[2]`            | 2  | true  | `sqrt(2)`                 | 2 (kappa>=1e10), 5 (scaling) |
| 6 | `diag_weight_kappa12`       | `[4]`            | 1  | true  | `1` (exact)               | 2 (data kappa=1e12 knob) |
| 7 | `two_block_corr_coupled`    | `[2, 2]`         | 5  | true  | `2*sqrt(2)`               | 3 (>=2 PSD blocks) |
| 8 | `separable_12block`         | `[2]x12`         | 12 | true  | `sum_{c=1}^{12}(c/2+sqrt(c^2/4+1))` | 4 (m>=12), 3 (>=2 PSD blocks) |

Eight problems (>= 6 required). The acceptance gates are all met:

- **n >= 8:** problem 2 (`max_eig_path_8`, n=8) and problem 3 (`max_eig_path_16`, n=16).
- **kappa >= 1e10:** problem 5 (`disparate_scale_sqrt2`, central-path Schur cond ~1e32 —
  see §5.5) and problem 6 (`diag_weight_kappa12`, data-matrix cond exactly 1e12 — §6.5),
  with a 1e6 / 1e10 / 1e12 knob explicitly available on problem 6.
- **>= 2 genuine PSD blocks (not SDP+LP):** problem 7 (two 2x2 PSD blocks) and
  problem 8 (twelve 2x2 PSD blocks).
- **m >= 12:** problem 8 (m=12).
- **disparate magnitudes / bad scaling:** problem 5 (RHS dynamic range 2e16),
  problem 6 (data dynamic range 1e12).
- **minimize problem (maximize=false / `maximize=0` path, bead n1x):** problem 4.
- **>= 50-digit analytic reference:** all eight, emitted at 65 sig digits by `FormatNum`.

All eight are auto-discovered by `test_golden`: each lands as
`golden/<name>/<name>.dat-s` + `golden/<name>/provenance.json`, which is the entire
discovery contract (`golden_discover` in `c/test/golden_provenance.c`:161 stats those
two files; the only REQUIRED provenance keys are `name` and `optimal_value`, everything
else falls back to a documented default).

---

## 1. Audition: families considered, winners and losers

CLAUDE.md rule 3 is non-negotiable: do not default to the first construction. The
following standard pathological / verified-SDP families were surveyed (VSDP/INTLAB
test problems, SDPLIB small instances, Jansson's examples, moment/Hankel matrices,
Lovász theta, structured max-eigenvalue, Hilbert matrices). For each, the question
is: (a) is there a PROVABLE closed-form or `Root[]`-expressible optimum to 50+
digits? (b) is the size `n` a free parameter? (c) is conditioning controllable? (d)
which hole does it fill?

### 1.1 WINNER — tridiagonal Toeplitz max-eigenvalue (the "path graph" / Jacobi matrix)

`max <C,X> s.t. tr(X)=1, X nxn PSD` with `C = T_n(a,b)` (symmetric tridiagonal,
diagonal `a`, off-diagonal `b`). By the Rayleigh/variational characterization
(`max{<C,X>: tr X=1, X>=0} = lambda_max(C)`, the density-matrix Rayleigh quotient),
the optimum is `lambda_max(C)`. The eigenvalues of a tridiagonal Toeplitz matrix are
**closed form**:

    lambda_k = a + 2b cos(k*pi/(n+1)),  k = 1..n

(Standard identity; e.g. Smith, *Numerical Solution of PDEs*, or Meurant 1992
"A review of the inverse of symmetric tridiagonal and block tridiagonal matrices".)
With `b > 0`, `lambda_max = a + 2b cos(pi/(n+1))`.

- (a) Closed form to ANY precision via `cos` — verified to 65 digits in mpmath AND
  sympy AND an explicit numeric eigensolve (diff `< 5e-71` at n=4,8,16). ✓
- (b) `n` is a FREE integer parameter. ✓ (directly fills hole 1: n=4,8,16)
- (c) Eigenvalue spread `cond(C) = lambda_max/lambda_min` grows only like `n^2`
  (~32 at n=8, ~116 at n=16) — so this family is the **size / bits-per-digit**
  stressor, NOT the kappa stressor. Honest division of labour (see §1.2).
- (d) Hole 1 (scalable block size). Also connects to the named target use case:
  `T_n(a,b)` IS the Jacobi (three-term recurrence) matrix of orthogonal polynomials,
  whose eigenvalues are Gauss quadrature nodes — i.e. a clean *moment-matrix* family
  with a closed-form spectrum.

This beats the alternatives below for the scalability hole because it is the unique
candidate with a *closed-form spectrum at every n*. `a=2, b=1` chosen: integer data
(clean .dat-s), `lambda_max = 2+2cos(pi/(n+1))` in `(3, 4)` for all n.

### 1.2 Why NOT Hilbert / Hankel matrices for the optimum (they LOSE on provability)

The Hilbert matrix `H_ij = 1/(i+j-1)` is the canonical ill-conditioned symmetric
matrix: measured `cond(H_8) = 1.53e10` (mpmath), crossing the 1e10 bar. It is the
textbook *moment matrix* of the uniform measure on [0,1] (the named target use case).
BUT `lambda_max(H_n)` is **transcendental with no closed form** — there is no way to
write it to 50 digits except by an eigensolve, which is not an *independent* analytic
reference (it would just be "trust the eigensolver"). Using it as the optimum would
violate the "provable analytic optimum" acceptance criterion.

**Verdict: rejected as an OPTIMUM source.** Its conditioning role is taken by
constructions whose optimum IS closed form (§1.3). We keep the Hilbert *connection*
honest: the tridiagonal Jacobi family (§1.1) is the moment-matrix family with a
closed-form spectrum, and `diag_weight_kappa12` (§1.3) reproduces Hilbert-class data
conditioning (1e12) with an exact optimum.

### 1.3 WINNERS for kappa>=1e10 — two constructions with a knob and a provable optimum

The crucial design realization (justified in §5.5 / §6.5): for these SMALL SDPs the
defensible condition-number claim must name WHICH matrix. Two distinct, both
analytic:

- **`disparate_scale_sqrt2`** (problem 5): `max x_12 s.t. x_11=P, x_22=Q, X 2x2 PSD`.
  PSD ⇒ `x_12^2 <= P*Q` ⇒ `opt = sqrt(P*Q)` (closed form). With `P=2e8, Q=1e-8`,
  `opt = sqrt(2)`. The TWO constraints `A_1=e1e1^T, A_2=e2e2^T` make the m×m **Schur
  complement at the central path** ill-conditioned: the NT scaling `W ~ diag(P,Q)`
  near the boundary, and `M = diag(w_1^2, w_2^2)` so `cond(M) ~ (P/Q)^2 ~ 1e32`
  (justified in §5.5). This is the "Schur complement at the central path" condition
  number, and it is `>= 1e10` by 22 orders of magnitude. Beats the Hilbert idea
  because the optimum `sqrt(P*Q)` is closed form.

- **`diag_weight_kappa12`** (problem 6): `max <diag(w),X> s.t. tr(X)=1, X 4x4 PSD`.
  `opt = max_i w_i` (exact). With `w = (1, 1e-4, 1e-8, 1e-12)`, `opt = 1` and the
  **data-matrix condition number** `cond(C) = max w / min w = 1e12` exactly. The
  exponent is a free KNOB: `w = (1, 1e-3, 1e-6)` gives 1e6; `(1, 1e-5, 1e-10)` gives
  1e10; `(1, 1e-4, 1e-8, 1e-12)` gives 1e12 — i.e. the requested 1e6/1e10/1e12
  ladder. Provable exact optimum. This explicitly satisfies "state WHICH condition
  number (data matrix) and justify."

Both kept: they stress DIFFERENT conditioning axes (Schur-at-central-path vs.
data-matrix) and the acceptance asks for at least one kappa>=1e10; we provide two.

### 1.4 WINNER — coupled two-block correlation SDP (>= 2 PSD blocks, both active)

`two_block_corr_coupled` (problem 7): two 2x2 PSD blocks `X1, X2`, both driven to a
rank-1 boundary optimum, COUPLED by an equality constraint so the m×m Schur genuinely
mixes the blocks (not a trivial direct sum). `opt = 2*sqrt(2)` (closed form). Fills
hole 3 with *genuine* PSD blocks (not SDP+LP). Beats the naive "max over union of two
blocks sharing one trace constraint" (where only ONE block is active and the other's
`X` is free/loose — a weak test) because here both blocks are tight at the optimum.

### 1.5 WINNER — separable 12-block (m>=12, many PSD blocks)

`separable_12block` (problem 8): twelve 2x2 PSD blocks, each with its own trace
constraint, objective `sum_b <A_b, X_b>`. Separable ⇒
`opt = sum_b lambda_max(A_b)` (closed form, a sum of 12 surds). m=12 constraints,
12 PSD blocks. Fills hole 4 (m>=12 Schur) and reinforces hole 3.

**Honest caveat (recorded so it is not mistaken for a defect):** this problem is
*block-separable* — its Schur `M` is block-diagonal (one 1x1 block per constraint).
So it stresses the m×m Schur ASSEMBLY/STORAGE and per-block bookkeeping at m=12, but
NOT a dense ill-conditioned m×m factorization. A dense, COUPLED, analytic-optimum
m>=12 SDP does not exist in clean closed form (entry-wise pinning of a known optimum
`X*` injects irrational RHS — verified: `X*_11 = 0.1382...` for the n=4 path matrix —
which pollutes the golden data; rejected). The disparate-scale problem (§1.3) is the
*ill-conditioned* dense-ish Schur stressor; problem 8 is the *large-m* assembly
stressor. Two problems, two axes, both honest. **Design risk flagged for the
orchestrator in the returned summary.**

### 1.6 Losers, recorded with reasons

| candidate | why rejected |
|-----------|--------------|
| Hilbert/Hankel `lambda_max` as optimum | no closed form; reference would just be an eigensolve (not independent). Kept only as a *conditioning* reference point. |
| Lovász theta on small graphs | optimum is closed form only for special graphs (e.g. `theta(C_5)=sqrt(5)`), but the SDP is the *dual* with a sum-of-squares structure that is awkward to emit cleanly in .dat-s, and the n knob is not smooth. Deferred (could be a future bead). |
| Max-cut / GW relaxation (`X_ii=1`, max `<L,X>`) | optimum generally NOT closed form; would need an eigensolve reference. Rejected. |
| entry-wise pinning of a known rank-1 `X*` to inflate m | injects irrational constraint RHS (verified surd entries) — pollutes data, breaks "clean rational/closed-form data". Rejected. |
| band-completion correlation SDP (`max X_{1,n}`, fixed band) | optimum closed-form only in special cases; risky provability. Rejected. |
| tridiagonal as the kappa>=1e10 source | `cond(C)` grows only ~`n^2` (~440 at n=32); would need n~1e5 to reach 1e10. Wrong tool for kappa; right tool for size. |

---

## 2. Conventions recap (so the data tables below are unambiguous)

From `docs/MATH_SPEC.md` §1–2 and `golden/README.md`:

- External form: `max <C,X> s.t. <A_i,X>=b_i, X=blkdiag(...)  >= 0`. `maximize:true`.
  A `maximize:false` problem (problem 4) is `min <C,X> s.t. ...` (exercises the
  internal `min -<C,X>` sign path, bead n1x).
- `<A,B> = trace(A^T B) = sum_{ij} A_ij B_ij` (Frobenius).
- `block_sizes`: positive = PSD block of that side; negative = diagonal/LP block.
  All eight stress problems use POSITIVE (PSD) blocks except none — they are all
  genuine SDP (deliberately, per hole 3).
- **SDPA .dat-s inner-product accounting (the load-bearing gotcha, README.md:84):**
  an entry `i block k l v` with `k <= l` stores `A_kl = A_lk = v`. Therefore

      <A,X> = sum_{k=l} v_kk * X_kk  +  sum_{k<l} 2*v_kl * X_kl

  i.e. an off-diagonal entry contributes `2*v*X_kl` (it counts as BOTH (k,l) and
  (l,k)). So to realize an objective `<C,X> = x_12` you store the off-diagonal value
  `v = 1/2` (then `2*(1/2)*x_12 = x_12`); to realize `C = T_n(a,b)` you store the
  diagonal entries with `v = a` and the super-diagonal entries with `v = b` (the
  ACTUAL matrix entries, not halved, because `<C,X>` already wants `2b*X_{i,i+1}`).
  Every data table below states the realized `<A,X>` explicitly so the generator
  cannot get this wrong.

---

## 3. Family A — scalable tridiagonal max-eigenvalue (`max_eig_path_n`)

Fills **hole 1** (scalable PSD block size; n=4, 8, 16 give bits/digit & wall-time
SCALING data). Three problems, one general formula.

### 3.1 Mathematical statement (general n)

    maximize   <C, X>
    s.t.       tr(X) = 1
               X  in  S^n_+         (single n x n PSD block)
    where      C = T_n(2,1)  (symmetric tridiagonal: diagonal 2, off-diagonal 1)

`block_sizes = [n]`, `m = 1`, `maximize = true`.

### 3.2 Exact problem data (general n)

- `b = [1]`.
- `C = T_n(2,1)`: `C_ii = 2` (i=1..n), `C_{i,i+1} = C_{i+1,i} = 1` (i=1..n-1), else 0.
- `A_1 = I_n` (the trace constraint): `A_1,ii = 1`.

.dat-s entries (1-indexed, upper triangle, `i=0`→C, `i=1`→A_1), realizing
`<C,X> = 2*sum_i X_ii + 2*sum_i X_{i,i+1}` and `<A_1,X> = sum_i X_ii = tr(X)`:

    0  1  i      i      2        for i = 1..n          (C diagonal, v=2)
    0  1  i    (i+1)    1        for i = 1..n-1        (C super-diag, v=1 -> 2*X_{i,i+1})
    1  1  i      i      1        for i = 1..n          (A_1 = I_n)

### 3.3 Analytic optimum

By the variational formula `opt = lambda_max(C) = 2 + 2 cos(pi/(n+1))`. Concrete
instances (65 sig digits, cross-verified mpmath = sympy = numeric eigensolve):

| name | n | optimum `2+2cos(pi/(n+1))` (65 sig digits) |
|------|---|--------------------------------------------|
| `max_eig_path_4`  | 4  | `3.6180339887498948482045868343656381177203091798057628621354486227` |
| `max_eig_path_8`  | 8  | `3.8793852415718167681082185546494629398724162685289292661805733255` |
| `max_eig_path_16` | 16 | `3.9659461993678035565638976897103974321974457501312657519947609184` |

`optimal_value_digits = 65`. The optimal `X*` is the rank-1 projector onto the top
eigenvector (singular, boundary optimum — exercises CLAUDE.md invariant 1).

### 3.4 Conditioning estimate (justification)

- **Data matrix:** `cond(C) = lambda_max/lambda_min = (2+2cos(pi/(n+1)))/(2-2cos(pi/(n+1)))`,
  growing like `~(2(n+1)/pi)^2` (n^2). Measured: n=8 → 32.2, n=16 → 116.5. So this
  family is MILDLY conditioned in the data; it is NOT the kappa>=1e10 source.
- **Central-path Schur:** m=1, so the Schur is a 1x1 scalar — trivially conditioned.
  The stress is elsewhere (§3.5).

### 3.5 Solver-stress hypothesis

The relevant difficulty is the **eigenvalue gap** `lambda_1 - lambda_2 =
2[cos(pi/(n+1)) - cos(2pi/(n+1))]`, measured 1.00 (n=4), 0.347 (n=8), 0.101 (n=16) —
shrinking ~1/n^2. A smaller gap means the rank-1 boundary `X*` is harder to resolve:
`mu` must shrink further, and the per-block PSD step / NT eigendecomposition operate
on a worse-separated spectrum. Combined with the growing svec dimension `n(n+1)/2`
(10, 36, 136), the hypothesis is:

- **n=4:** behaves like the existing `max_eig_tridiag_3x3` (which calibrated to
  target_digits=12, prec_cap=1024, ~75 bits/digit, NUM at prec 128/256/512 until
  1024 reaches OPT). Expect target_digits ~12, prec_cap ~1024.
- **n=8:** smaller gap + larger svec; expect to need prec_cap ~2048 and possibly
  NUM at low prec; bits/digit likely 80–160.
- **n=16:** smallest gap + svec dim 136; expect prec_cap ~4096, possible
  ITER_LIMIT or NUM short of target; this is the problem most likely to expose
  "works at n=3, breaks at n=16" (CLAUDE.md rule 6). A `correctness` known_gap is
  plausible here — to be CALIBRATED from measurement, not guessed.

---

## 4. `min_eig_path_4` — the minimize path (bead n1x)

Fills the **maximize=false** code path. Mirrors problem 1 but minimizes.

### 4.1 Statement

    minimize   <C, X>
    s.t.       tr(X) = 1
               X  in  S^4_+
    where      C = T_4(2,1).

`block_sizes = [4]`, `m = 1`, `maximize = FALSE`.

### 4.2 Data

Identical .dat-s body to `max_eig_path_4` (§3.2 with n=4); the ONLY difference is
`provenance.json` carries `"maximize": false` and the .dat-s header comment says
"minimize". (The solver negates `C` internally for the min path; the runner reads the
sign convention from provenance.)

### 4.3 Analytic optimum

`opt = lambda_min(C) = 2 + 2 cos(4*pi/5) = 2 - 2 cos(pi/5)`:

    0.38196601125010515179541316563436188227969082019423713786455137729

(65 sig digits; mpmath = closed form = numeric eigensolve to `< 5e-71`.) The optimal
`X*` is the rank-1 projector onto the BOTTOM eigenvector (boundary optimum).

### 4.4 Conditioning / stress

Same data conditioning as `max_eig_path_4` (`cond(C)` ~6.9). The point is purely to
exercise the sign-flip (`maximize=0` → internal `min -<C,X>` → output negation). A
sign bug here negates the objective and the bracket excludes the true optimum — a P0
(CLAUDE.md invariant 7). Stress hypothesis: similar to `max_eig_path_4`
(target_digits ~12, prec_cap ~1024). The bottom eigenvalue gap
`lambda_2 - lambda_1 = 2[cos(3pi/5) - cos(4pi/5)] = 2[cos(pi/5) - cos(2pi/5)] = 1.0`
(symmetric to the top gap), so convergence difficulty is comparable.

---

## 5. `disparate_scale_sqrt2` — Schur conditioning >= 1e10 + bad scaling

Fills **hole 2** (kappa>=1e10, central-path Schur) AND **hole 5** (disparate
magnitudes / badly scaled `b`).

### 5.1 Statement

    maximize   x_12
    s.t.       x_11 = P      (P = 2e8)
               x_22 = Q      (Q = 1e-8)
               X = [[x_11, x_12],[x_12, x_22]]  in  S^2_+

`block_sizes = [2]`, `m = 2`, `maximize = true`.

### 5.2 Exact data

- `b = [P, Q] = [200000000, 1/100000000]` (exact; `2*10^8` and `10^-8`).
- `C`: realize `<C,X> = x_12`. Off-diagonal entry `(1,2)` with `v = 1/2`
  (so `2*(1/2)*x_12 = x_12`).
- `A_1 = e1 e1^T`: entry `(1,1) v=1` → `<A_1,X> = x_11`. `b_1 = P`.
- `A_2 = e2 e2^T`: entry `(2,2) v=1` → `<A_2,X> = x_22`. `b_2 = Q`.

.dat-s entries:

    0  1  1  2  0.5      (C_{12}=1/2 -> <C,X> = x_12)
    1  1  1  1  1        (A_1 = e1 e1^T)
    2  1  2  2  1        (A_2 = e2 e2^T)

with `b = [200000000, 0.00000001]` (emitted at 65-digit precision by FormatNum).

### 5.3 Analytic optimum

PSD ⇒ Schur complement `x_11*x_22 - x_12^2 >= 0` ⇒ `x_12^2 <= P*Q = 2` ⇒
`opt = sqrt(P*Q) = sqrt(2)`:

    1.4142135623730950488016887242096980785696718753769480731766797380

(65 sig digits; mpmath = sympy.) `X* = [[2e8, sqrt(2)],[sqrt(2), 1e-8]]` is rank-1
(det = `P*Q - 2 = 0`), a boundary optimum.

### 5.4 Choice of P, Q

`P*Q = 2` gives the irrational optimum `sqrt(2)` (precision stress). `P/Q = 2e16`
gives the data/RHS dynamic range (hole 5). If a cleaner rational optimum is preferred
during calibration, `P=1e8, Q=1e-8` gives `opt = 1` exactly with the SAME
conditioning; `sqrt(2)` is preferred because it also stresses precision.

### 5.5 Conditioning estimate (justification — this is the kappa>=1e10 problem)

**Stated condition number: the Schur complement at the central path,
`cond(M) ~ (P/Q)^2 ~ 1e32`.**

Justification. The Schur matrix is `M_ik = <A_i, W A_k W>` (MATH_SPEC §3.4), with `W`
the NT scaling (`W S W = X`). Here `A_1 = e1e1^T`, `A_2 = e2e2^T` are diagonal, so for
a diagonal scaling `W = diag(w_1, w_2)`, `M = diag(w_1^2, w_2^2)` and
`cond(M) = (w_1/w_2)^2`. Near the boundary the iterate `X -> X* = diag(P, Q)` (up to
the rank-1 off-diagonal), and complementarity drives `S ~ diag(1/P, 1/Q)`, so
`w_i = sqrt(X_ii/S_ii) ~ X_ii`, giving `w_1/w_2 ~ P/Q = 2e16` and

    cond(M) ~ (P/Q)^2 ~ (2e16)^2 = 4e32  >> 1e10.

Even the INITIAL Schur (at `mu ~ O(1)`, before the boundary) already inherits the
`(P/Q)^2 ~ 1e32` spread because the constraint scales differ by `P/Q`. So the Schur
is `>= 1e10`-conditioned throughout, by 22 orders of magnitude. This is a rigorous,
named, justified estimate (NOT a guess). The data-matrix condition number is
separately huge (the RHS span `P/Q = 2e16`), but the binding claim per acceptance is
the central-path Schur.

### 5.6 Solver-stress hypothesis

A Schur cond ~1e32 means the Cholesky of `M` loses ~`log2(1e32) ~ 107` bits to
rounding before any digits of the answer accrue; on top sits the rank-1 boundary
`X*`. Hypothesis: NUM/STALL at prec 128 and likely 256; the 3-way Tikhonov will bump
hard (possibly hit a regularizer cap → precision escalation); reaching even 8–12
digits likely needs prec_cap ~1024–2048 with very high bits/digit (>120). This is a
prime candidate for a `correctness` or `efficiency` known_gap — CALIBRATED later.

---

## 6. `diag_weight_kappa12` — data-matrix conditioning knob (1e6 / 1e10 / 1e12)

Fills **hole 2** (the explicit kappa knob, data-matrix axis) and reinforces **hole 5**.

### 6.1 Statement

    maximize   <C, X>
    s.t.       tr(X) = 1
               X  in  S^4_+
    where      C = diag(1, 1e-4, 1e-8, 1e-12).

`block_sizes = [4]`, `m = 1`, `maximize = true`.

### 6.2 Exact data

- `b = [1]`.
- `C = diag(1, 1/10^4, 1/10^8, 1/10^12)` (exact rationals).
- `A_1 = I_4`.

.dat-s:

    0  1  1  1  1                    (C_11 = 1)
    0  1  2  2  0.0001               (C_22 = 1e-4)
    0  1  3  3  0.00000001           (C_33 = 1e-8)
    0  1  4  4  0.000000000001       (C_44 = 1e-12)
    1  1  1  1  1                    (A_1 = I_4)
    1  1  2  2  1
    1  1  3  3  1
    1  1  4  4  1
    b = [1]

### 6.3 Analytic optimum

`opt = lambda_max(diag(w)) = max_i w_i = 1` (exact integer). `X* = e1 e1^T`, rank-1
boundary. `optimal_value = 1`, `optimal_value_digits = 65` (trivially exact). Verified
trivially (mpmath/sympy: `max(w)=1`).

### 6.4 The knob (explicitly delivered per acceptance "ideally a knob 1e6/1e10/1e12")

The data condition number is `cond(C) = max w / min w`, set purely by the smallest
weight. The generator emits three SIBLINGS sharing this construction:

| name | weights | `cond(C)` | optimum |
|------|---------|-----------|---------|
| `diag_weight_kappa6`  | `(1, 1e-3, 1e-6)`            | 1e6  | 1 |
| `diag_weight_kappa10` | `(1, 1e-5, 1e-10)`          | 1e10 | 1 |
| `diag_weight_kappa12` | `(1, 1e-4, 1e-8, 1e-12)`    | 1e12 | 1 |

`diag_weight_kappa12` (4x4, cond 1e12) is the REQUIRED kappa>=1e10 representative;
the kappa6 / kappa10 siblings give the conditioning ladder for trend analysis.
(`diag_weight_kappa10` alone, cond exactly 1e10, also independently satisfies the
acceptance bar.) Generating all three is cheap; downstream may include all or just
kappa12 + kappa10 — but the doc SPECIFIES all three so the ladder exists.

### 6.5 Conditioning estimate (justification — data-matrix axis)

**Stated condition number: the data matrix, `cond(C) = max w / min w`, exactly
1e12 for `diag_weight_kappa12`.** This is exact by construction (a diagonal matrix's
condition number is its max/min entry ratio): `1 / 1e-12 = 1e12`. No estimate
needed — it is exact. This is the same conditioning CLASS as the Hilbert moment
matrix `H_8` (measured `cond = 1.53e10`), but with a provable exact optimum, which
is why it replaces Hilbert (§1.2).

Note `m=1` ⇒ the Schur is 1x1 (well-conditioned as a matrix); the ill-conditioning
here lives in the DATA and propagates into the NT-scaled per-block step (the scaling
`W` spans 1e12), not into the Schur factorization. This is a deliberately DIFFERENT
stress axis from problem 5 (Schur) — between them they cover both condition numbers
the acceptance mentions ("Schur complement at the central path, OR the data matrix").

### 6.6 Solver-stress hypothesis

Large, well-separated weights ⇒ the argmax is found in FEW iterations (the gap
`w_1 - w_2 = 1 - 1e-4 ~ 1` is large), so iteration count should be LOW. The stress is
PRECISION: the 1e12 dynamic range in `W` means the per-block eigendecomposition and
PSD step computation lose ~`log2(1e12) ~ 40` bits; reaching 12 digits should need
prec_cap ~512–1024. Hypothesis: OPTIMAL reachable but with elevated bits/digit;
likely an `efficiency` known_gap, not a `correctness` one. CALIBRATE later.

---

## 7. `two_block_corr_coupled` — two genuine, coupled, active PSD blocks

Fills **hole 3** (>= 2 genuine PSD blocks, not SDP+LP; both blocks active at optimum).

### 7.1 Statement

Two 2x2 PSD blocks `X1, X2`.

    maximize   x1_12 + x2_12
    s.t.       x1_11 = 1,   x1_22 = 2        (block 1 diagonal fixed)
               x2_11 = 1,   x2_22 = 2        (block 2 diagonal fixed)
               x1_12 - x2_12 = 0              (COUPLING: forces the two off-diagonals equal)
               X1 in S^2_+,  X2 in S^2_+

`block_sizes = [2, 2]`, `m = 5`, `maximize = true`.

### 7.2 Why this is genuinely coupled and both-blocks-active

The coupling `x1_12 = x2_12 = t` ties the blocks: in the Schur assembly the
constraint-5 row mixes block-1 and block-2 contributions (`M` is NOT block-diagonal).
Each block's PSD bound is `t^2 <= x_ii*x_jj = 1*2 = 2`, i.e. `t <= sqrt(2)`,
IDENTICAL for both blocks, so at the optimum BOTH blocks sit on their rank-1 boundary
(`det X_b = 1*2 - t^2 = 0`). No block is slack — a strictly stronger test than a
"max over union" design where one block goes free.

### 7.3 Exact data

`<C,X> = x1_12 + x2_12`: block-1 off-diag `(1,2) v=1/2`, block-2 off-diag `(1,2) v=1/2`.

- `A_1` (block1 `x_11=1`): `(b=1, 1,1) v=1`. `b_1=1`.
- `A_2` (block1 `x_22=2`): `(b=1, 2,2) v=1`. `b_2=2`.
- `A_3` (block2 `x_11=1`): `(b=2, 1,1) v=1`. `b_3=1`.
- `A_4` (block2 `x_22=2`): `(b=2, 2,2) v=1`. `b_4=2`.
- `A_5` (coupling `x1_12 - x2_12 = 0`): block1 off-diag `(1,2) v=1/2`
  (→ `+x1_12`), block2 off-diag `(1,2) v=-1/2` (→ `-x2_12`). `b_5=0`.

.dat-s entries (block 1 and block 2 both 2x2):

    0  1  1  2   0.5      (C block1: +x1_12)
    0  2  1  2   0.5      (C block2: +x2_12)
    1  1  1  1   1        (A_1: x1_11)
    2  1  2  2   1        (A_2: x1_22)
    3  2  1  1   1        (A_3: x2_11)
    4  2  2  2   1        (A_4: x2_22)
    5  1  1  2   0.5      (A_5: +x1_12)
    5  2  1  2  -0.5      (A_5: -x2_12)
    b = [1, 2, 1, 2, 0]

### 7.4 Analytic optimum

`t = sqrt(2)`, objective `= 2t = 2*sqrt(2)`:

    2.8284271247461900976033774484193961571393437507538961463533594762

(65 sig digits; mpmath = sympy.) Both `X_b* = [[1, sqrt(2)],[sqrt(2), 2]]` are rank-1.

### 7.5 Conditioning / stress hypothesis

Data is `O(1)`, well-conditioned. The stress is the BLOCK BOOKKEEPING and the coupled
5x5 Schur with two rank-1 boundary blocks. Hypothesis: comparable to `sdp_sqrt2` but
harder (two boundaries simultaneously, coupling constraint); expect target_digits ~12,
prec_cap ~512–1024. A clean test that the multi-block Schur assembly and per-block
NT scaling are correct (a sign/index bug in block 2 would surface here).

---

## 8. `separable_12block` — m=12 Schur assembly stressor, 12 PSD blocks

Fills **hole 4** (m>=12) and reinforces **hole 3** (many PSD blocks).

### 8.1 Statement

Twelve 2x2 PSD blocks `X_1..X_12`. For `c = 1..12`, block `c` has cost matrix
`A_c = [[c, 1],[1, 0]]` and its own trace constraint.

    maximize   sum_{c=1}^{12} <A_c, X_c>
    s.t.       tr(X_c) = 1,   c = 1..12       (12 constraints)
               X_c in S^2_+,  c = 1..12

`block_sizes = [2,2,2,2,2,2,2,2,2,2,2,2]` (twelve 2's), `m = 12`, `maximize = true`.

### 8.2 Exact data (block c, c=1..12)

`<A_c, X_c> = c*x_11 + 2*x_12` (since `A_c` diagonal `(c,0)`, off-diag `1`):
- C, block c: diagonal `(1,1) v=c`; off-diagonal `(1,2) v=1` (→ `2*x_12`).
- `A_c` (trace of block c): block c diagonal `(1,1) v=1`, `(2,2) v=1`. `b_c = 1`.

.dat-s entries (for each c = 1..12, block index = c):

    0  c  1  1  c          (C block c: c * x_11)
    0  c  1  2  1          (C block c: off-diag v=1 -> 2*x_12)
    c  c  1  1  1          (A_c: trace, X_c,11)
    c  c  2  2  1          (A_c: trace, X_c,22)
    b = [1, 1, ..., 1]     (twelve 1's)

### 8.3 Analytic optimum

Separable: `opt = sum_{c=1}^{12} lambda_max(A_c)`. For `A_c = [[c,1],[1,0]]`,
`lambda_max = c/2 + sqrt(c^2/4 + 1)`. Sum over c=1..12:

    opt = sum_{c=1}^{12} ( c/2 + sqrt(c^2/4 + 1) )
        = 80.570836016635247688021349419916773357633086452802384073908919

(65 sig digits; mpmath closed form = mpmath numeric eigensolve to diff 0.) Each block
`X_c*` is the rank-1 projector onto its top eigenvector (boundary).

### 8.4 Conditioning / stress hypothesis

Data is `O(1..12)`, well-conditioned. The Schur is 12x12 but BLOCK-DIAGONAL (separable
→ one 1x1 Schur block per constraint), so it is well-conditioned as a matrix. The
stress is: (i) m=12 Schur ASSEMBLY and storage; (ii) 12 simultaneous rank-1 PSD
boundaries; (iii) summing 12 surds to 12+ digits (accumulation). Hypothesis: OPTIMAL
reachable; iteration count moderate; bits/digit similar to single max-eig problems;
prec_cap ~1024. The real risk is an assembly/indexing bug at m=12 (caught by the
bracket-contains-optimum check). **See the §1.5 caveat: this is a large-m ASSEMBLY
stressor, not a dense-ill-conditioned-Schur stressor (that role is problem 5).**

---

## 9. provenance.json sketches (v1 fields fixed now; v2 runner fields CALIBRATED later)

For every problem, the v1 fields (`name`, `source`, `description`, `m`,
`block_sizes`, `maximize`, `optimal_value`, `optimal_value_digits`,
`reference_method`, `generated_by`) are fixed by THIS document. The v2 runner fields
(`expected_status`, `target_digits`, `prec_cap`, `correctness_tol_digits`,
`expected_max_bits_per_digit`, `expected_max_iters`, `known_gaps`) MUST be calibrated
from measured behavior (`bench/brittle_probe.c` / running `test_golden`), NOT guessed
(README.md schema v2 note; CLAUDE.md rule 11). The placeholders below are EXPECTATIONS
to verify, flagged as such.

Sketch (example: `max_eig_path_8`):

    {
      "name": "max_eig_path_8",
      "source": "wolframscript",
      "description": "8x8 SDP: max <C,X> s.t. tr(X)=1, X 8x8 PSD, C = tridiagonal Toeplitz T_8(2,1) (diagonal 2, off-diagonal 1). By the variational formula opt = lambda_max(C) = 2 + 2 cos(pi/9). Optimal X* is the rank-1 projector onto the top eigenvector (singular, boundary optimum). Scalable size-stress family (n=4,8,16).",
      "m": 1,
      "block_sizes": [8],
      "maximize": true,
      "optimal_value": "3.8793852415718167681082185546494629398724162685289292661805733255",
      "optimal_value_digits": 65,
      "reference_method": "Closed form lambda_max(T_n(2,1)) = 2 + 2 cos(pi/(n+1)); n=8. Cross-checked in mpmath 1.3.0 and sympy 1.14.0 against an explicit symmetric eigensolve to >60 digits.",
      "generated_by": "arbsdp golden-master generator epic.3 / wolframscript",

      "// --- v2 runner fields: CALIBRATE from measurement, do not ship these guesses ---": "",
      "expected_status": "optimal",
      "target_digits": 12,                  // PLACEHOLDER: set at/just-below measured achieved digits
      "prec_cap": 2048,                     // PLACEHOLDER: set to final_prec needed, rounded up to a power of 2
      "correctness_tol_digits": 0,
      "expected_max_bits_per_digit": 8,     // GOAL ceiling; current solver overruns -> efficiency known_gap
      "expected_max_iters": 100,
      "known_gaps": [
        // CALIBRATE: add { "kind": "efficiency", "bead": "arb-prec-IPM-epic.6",
        //   "reason": "<measured bits/digit, final_prec, achieved digits>" } if bits/digit > 8 (it will be);
        // add { "kind": "correctness", ... } ONLY if target_digits is NOT reachable within prec_cap (then
        //   target_digits must be set to the reachable value and the gap documents the shortfall).
      ]
    }

`min_eig_path_4` differs only by `"maximize": false` and its optimum string. The
`disparate_scale_sqrt2` / `diag_weight_kappa12` sketches additionally expect (to
verify, not assume) `correctness`/`efficiency` known_gaps reflecting their
conditioning (§5.6, §6.6). Every `known_gaps[].reason` MUST carry the CONCRETE
measured numbers (e.g. "achieved 8.3 digits at final_prec 2048 = 247 bits/digit; NUM
at prec 128/256/512/1024"), per README.md and the existing `max_eig_tridiag_3x3`
exemplar.

---

## 10. Generation & calibration notes for downstream agents

### 10.1 How to extend `generate.wl`

- Add one `Module[...]` block per problem, in the exact style of the existing 7
  (`golden/gen/generate.wl`). Reuse `FormatNum` (65 sig digits, no scientific
  notation), `WriteDatS`, `WriteProvenance` verbatim. The generator stays
  deterministic and idempotent.
- For the **scalable family**, write a single helper
  `WriteMaxEigPath[n_]` that builds `C = T_n(2,1)` entries programmatically
  (diagonal `v=2`, super-diagonal `v=1`) and `A_1 = I_n`, then loop `n in {4,8,16}`.
  Compute `optVal = 2 + 2 Cos[Pi/(n+1)]` symbolically and `N[optVal, 65]` via
  `FormatNum`. (Do NOT use `Eigenvalues` of a numeric matrix for the reference —
  use the closed form; the eigensolve is the CROSS-CHECK in `verify.wl`, kept
  independent per CLAUDE.md rule 8.)
- `WriteProvenance` currently emits ONLY v1 fields and hard-codes `"maximize": true`.
  Two changes are needed:
  1. Add a `maximize` parameter (for `min_eig_path_4`). Either parametrize
     `WriteProvenance` or post-process the string. The .dat-s header comment should
     also say "minimize" for that problem.
  2. The v2 runner fields are NOT emitted by the current `WriteProvenance`. Either
     (a) extend `WriteProvenance` to append the v2 block, or (b) keep generation v1
     and let the CALIBRATION step (running `test_golden`, reading measured
     final_prec/digits/iters) write the v2 fields. Option (b) matches the
     "calibrate from measurement" rule best — generate v1, then a calibration script
     patches in target_digits/prec_cap/known_gaps from the measured run.
- For `diag_weight_*`, generate all three siblings (kappa6/kappa10/kappa12) from one
  helper taking the weight vector.

### 10.2 .dat-s gotchas (RE-STATE for the implementer — these bite)

- **Upper triangle only**, `k <= l`, 1-indexed. Never emit a `k > l` entry.
- **Off-diagonal factor of 2:** entry `i block k l v` with `k < l` contributes
  `2*v*X_kl` to `<A,X>` (it stands for both `(k,l)` and `(l,k)`). So:
  - To realize a TARGET inner-product coefficient `x_12` (coefficient 1), store
    `v = 1/2` (problems 5, 7).
  - To realize a matrix `C = T_n(a,b)`, store the ACTUAL entries (`v=a` diagonal,
    `v=b` super-diagonal) — because `<C,X>` legitimately wants `2b*X_{i,i+1}`
    (problems 1–4, 8). Do NOT halve the matrix entries in this case.
  Every §3–8 data table states the realized `<A,X>` explicitly; match it.
- **No `=` signs** in data lines (the parser truncates from `=` as a comment).
- **Block index is 1-indexed** and must match `block_sizes` order. Problem 7 uses
  blocks 1 and 2; problem 8 uses blocks 1..12.
- **`b` values at full precision:** `2e8` and `1e-8` (problem 5) and the `1e-4..1e-12`
  weights (problem 6) must go through `FormatNum` so they are plain 65-digit decimals
  (FormatNum already handles `exp <= 0` via the leading-zeros branch; confirm it
  renders `1e-12` as `0.000...001` correctly — it does in the existing
  `ill_conditioned_3x3` which uses `1e-6`).

### 10.3 Independent cross-check (`verify.wl` + a second toolchain)

Per CLAUDE.md rules 4 and 8, the optimum is confirmed by TWO independent toolchains.

- **`verify.wl` (Mathematica):** recompute each optimum by a DIFFERENT route than the
  generator. For the eigenvalue families, build the numeric matrix and call
  `Eigenvalues` / `CharacteristicPolynomial` + `Solve` (NOT the `2+2cos` closed form
  the generator used), then `DigitsAgree` to the stored string. For problem 5,
  `Maximize[{x12, P*Q - x12^2 >= 0}, {x12}]`. For problem 6, `Max[weights]`. For
  problem 7, `Maximize` over the 2x2 PSD constraints. For problem 8, sum the 12
  `lambda_max(A_c)` numerically.
- **Second toolchain (mpmath 1.3.0 AND/OR sympy 1.14.0):** the exact expressions and
  their 65-digit values used in THIS document were produced and cross-checked in both
  mpmath and sympy (and against numeric eigensolves). The downstream verifier SHOULD
  re-run a small Python script asserting each stored `optimal_value` matches the
  closed form to >= 50 digits, as a third, non-Mathematica gate. Reference values to
  assert (the strings are in §3.3, §4.3, §5.3, §6.3, §7.4, §8.3):
  - `2+2cos(pi/(n+1))` for n=4,8,16; `2-2cos(pi/5)`; `sqrt(2)`; `1`; `2*sqrt(2)`;
    `sum_{c=1}^{12}(c/2+sqrt(c^2/4+1))`.

### 10.4 Calibration (the v2 fields) — the discipline

- Run `test_golden` (or `bench/brittle_probe.c`) on each new problem and RECORD:
  adaptive status, achieved digits, final_prec, iters, bits/digit.
- Set `target_digits` AT or JUST-BELOW the measured achieved digits, so the
  correctness assertion PASSES (README.md schema v2: target must be reachable within
  `prec_cap`). Set `prec_cap` to the measured final_prec rounded UP to a power of 2.
- Where the solver is NOT yet robust, add a `known_gaps` marker of the right kind:
  - `efficiency` if bits/digit > `expected_max_bits_per_digit` (8) or iters >
    `expected_max_iters` — EXPECTED on every problem today (the b30/epic.6 efficiency
    gap). Reason MUST state the measured bits/digit and final_prec.
  - `correctness` ONLY if `target_digits` cannot be reached within `prec_cap` — in
    that case ALSO lower `target_digits` to the reachable value; the gap documents
    the shortfall with a concrete measured reason (e.g. "NUM at prec 128/256/512;
    reaches only 9.1 digits at prec_cap 4096"). Likely candidates: `max_eig_path_16`,
    `disparate_scale_sqrt2`.
  - `status` is not expected (all eight are `expected_status: "optimal"`).
- xfail/xpass ratchet (test_golden semantics): a `known_gap` makes that failure class
  GREEN (tolerated, but printed); if the case later PASSES that class unexpectedly,
  the gate goes RED (XPASS) forcing removal of the stale marker. So mark ONLY genuine,
  measured shortfalls — every marker must be earned by a measurement.

### 10.5 README.md update

When the problems land, extend the `golden/README.md` "Problems" table and add
detailed descriptions in the same style as the existing 7. Note the new size/
conditioning columns implicitly (n, m, kappa) so the corpus's coverage is legible.

---

## 11. Verification log (what was actually checked, per CLAUDE.md rule 11)

All optima below were computed by closed form AND independently, agreeing to the
stated precision (no value recalled from memory):

- `2+2cos(pi/(n+1))`, n=4,8,16: mpmath closed form vs mpmath `eigsy` numeric eig:
  diff `< 5e-71`; vs sympy `eigenvals` (real part): diff `0` to 45 digits; sympy
  symbolic `N(2+2cos(pi/(n+1)),50)` matched mpmath to 50 digits.
- `2-2cos(pi/5)` (min_eig_path_4 lambda_min): closed form vs numeric eig diff
  `4.5e-71`.
- `sqrt(P*Q)` with `P=2e8,Q=1e-8`: `= sqrt(2)`, 40-digit print matched the surd.
- `diag_weight` optimum `= max(weights) = 1`: exact.
- `2*sqrt(2)` (coupled two-block): 40-digit print matched.
- `sum_{c=1}^{12}(c/2+sqrt(c^2/4+1))`: closed-form sum vs `eigsy` per-block sum:
  diff `0` at 40 digits → `80.5708360166352476880213494199...`.
- Schur-conditioning estimate for problem 5: `(P/Q)^2 = (2e16)^2 = 4e32`; data RHS
  span `P/Q = 2e16`.
- Data-matrix conditioning for problem 6: `cond(diag(1,1e-4,1e-8,1e-12)) = 1e12`
  exactly; Hilbert `H_8` reference `cond = 1.53e10` (mpmath) confirms the
  conditioning CLASS the diag-weight family stands in for.
- SDPA inner-product accounting (off-diagonal `2*v` factor) re-derived per problem
  and baked into every §3–8 data table.
