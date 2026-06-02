# MATH_SPEC.md — libarbsdp mathematical specification

**Status:** draft v0.1  
**Bead:** arb-prec-IPM-b03  
**Date:** 2026-05-29  
**Sources cited:** PRD.md, CLAUDE.md, HsdeNtSdpSolver.ts, HsdeIterate.ts,
Residuals.ts, HsdeStepLength.ts, PsdCone.ts, SchurAssembler.ts,
Regularization.ts, Convergence.ts, Defaults.ts, SdpProblem.ts, NtSdpSolver.ts
(reference only), Jansson-Chaykin-Keil 2007, Todd-Toh-Tutuncu 1998,
Andersen-Roos-Terlaky 2003.

---

## Purpose

This document is the "ground truth before coding" artifact (CLAUDE.md rule 3).
Every formula that `libarbsdp` implements is stated here with its source
citation. Later coding beads have a single, unambiguous contract. If the PRD and
a stray note disagree, the PRD wins; if the PRD and a paper disagree, the paper
wins (CLAUDE.md rule 3).

---

## 1. Standard form and sign convention

### 1.1 External (SDPA) form

The user-facing problem is the SDPA standard multi-block SDP (PRD.md §3):

    max  <C, X>
    s.t. <A_i, X> = b_i,  i = 1..m
         X = blkdiag(X_1, ..., X_{nb})  >=  0

where each block X_b is a symmetric n_b x n_b positive semidefinite matrix,
inner product `<A, B> = trace(A^T B) = sum_{ij} A_{ij} B_{ij}` (Frobenius
inner product, not the svec dot product -- see §2 for the distinction).

Parameters: m linear constraints, nb blocks, block sizes n_1, ..., n_{nb}.
LP is subsumed as 1x1 diagonal blocks; no separate LP path is needed.

### 1.2 Internal (min) form

Internally `libarbsdp` works with the negated minimization problem
(SdpProblem.ts:7-8, SdpProblem.ts:39-41):

    min  -<C, X>    (equivalently: minimize over (X, y, S))
    s.t. <A_i, X> = b_i,  i = 1..m
         X = blkdiag(X_1, ..., X_{nb})  >=  0

This matches the TS solver's convention exactly.  When the input problem has
`maximize=true`, the C matrices stored in `SdpBlock.C` are negated at parse
time; the final objective is negated again on output.

**CRITICAL sign invariant (CLAUDE.md invariant 7):** The internal objective is
`-<C, X>`, not `<C, X>`.  Get this backwards and every objective is wrong.
Primal objective in the solver loop is `pObj = sum_b <C^b, X^b>` (the negated
C is already stored, so `pObj` corresponds to the internal minimization
objective, while `pObj/tau` purified and negated is the user-facing value).
Source: HsdeNtSdpSolver.ts:392-414.

### 1.3 Block-diagonal structure

Constraint matrices A_i and cost matrices C are block-diagonal:

    A_i = blkdiag(A_i^1, ..., A_i^{nb})
    C   = blkdiag(C^1,   ..., C^{nb})

Each block A_i^b is a dense symmetric n_b x n_b matrix stored row-major
in a flat Float64Array of length n_b^2 (SdpProblem.ts:8-10, HsdeIterate.ts:145-158).
Symmetry is enforced by explicit `symmetrize` calls (PsdCone.ts:8-16) before
each factorization and after each block update.

The dual variable per block is the slack S_b (symmetric, PSD).  The dual
multiplier y is a single m-vector shared across blocks.

---

## 2. Vectorization convention

### 2.1 Two conventions in play -- choose once, never deviate

There are two conventions in the codebase, and a coder MUST understand both
and know which one `libarbsdp` uses internally.

**Convention A (Dense Frobenius -- used by the TS solver internally):**

    <A, B>  =  trace(A^T B)  =  sum_{ij} A_{ij} B_{ij}

The TS solver stores all matrices as dense n x n row-major arrays and computes
inner products via `frobInner` (PsdCone.ts:18-23), which is just a dot product
over all n^2 entries.  This is not an svec convention; it is the plain Frobenius
inner product.  PsdCone.ts comment line 4: "We keep it dense (not svec)."

**Convention B (strict-Mosek sqrt(2) svec -- required by libarbsdp and the
sdp-solve tool layer):**

The svec operator maps a symmetric n x n matrix A to a vector of length
d = n(n+1)/2 with sqrt(2) scaling on the off-diagonal entries:

    svec(A)_k  =  A_{ii}          if i = j  (diagonal entry)
    svec(A)_k  =  sqrt(2) A_{ij}  if i < j  (upper-triangular entry, column-major order)

The index k runs over pairs (i,j) with i <= j in some fixed order (e.g.,
column-major upper triangle, matching Mosek's convention).

The inverse smat operation:

    smat(v)_{ij} = v_k              if i = j
    smat(v)_{ij} = v_k / sqrt(2)    if i != j  (set both A_{ij} and A_{ji})

The key property:

    <A, B>  =  svec(A) . svec(B)     (standard Euclidean dot product in R^d)

This holds because the factor of sqrt(2) on off-diagonal entries compensates
for their double-counting (A_{ij} B_{ij} appears as both (i,j) and (j,i) in
the Frobenius sum, but only once in svec).

**CLAUDE.md invariant 6 (non-negotiable):** `libarbsdp` uses Convention B
(strict-Mosek sqrt(2) svec) as the CANONICAL internal representation for
the Schur assembly and for all operations where inner products are computed via
vector dot products.  Document it once in `svec.h`; never deviate.

### 2.2 Why the TS dense path is NOT directly portable to the Schur assembly

The TS `frobInner` function computes sum_{ij} A_{ij} B_{ij}, which equals
`<A, B>` as a trace inner product.  In the Schur assembly (Section 3.3 below),
the matrix

    M_{ik}^b  =  <A_i^b, W^b A_k^b W^b>

is a Gram matrix in the trace inner product, which is mathematically identical
to the svec Gram matrix if and only if the trace inner product is used
consistently throughout.  The TS solver IS consistent: it uses `frobInner`
everywhere, so its Schur M is correct.

For `libarbsdp`, which operates on arb_mat objects, the svec Convention B is
preferred because:
1. It makes M a standard Gram matrix: M_{ik} = svec(W A_i W)^T svec(W A_k W),
   which is manifestly symmetric and PSD in exact arithmetic.
2. The PRD.md §4.1 explicitly prescribes svec with strict-Mosek sqrt(2) scaling.
3. It enables the `svec.h` module to be the single locus of the convention.

**Coders must NOT blindly copy the TS `frobInner` call pattern for the Schur
assembly without ensuring consistent sqrt(2) treatment.**  If the
constraint matrices A_i^b are stored in svec form and inner products are plain
Euclidean dot products, the result is the same as the trace inner product and
everything is consistent.  If the matrices are stored DENSE and the frobInner
formula is used, that is also consistent -- but mixing the two is wrong.

The recommended implementation for `libarbsdp`:
- Store constraint matrices in svec form (length d_b = n_b(n_b+1)/2 per block).
- Compute all inner products as Euclidean dot products over svec vectors.
- The Schur assembly becomes: M_{ik} = sum_b dot(svec(W^b A_i^b W^b), svec(W^b A_k^b W^b)).
- The module `svec.c` implements `arb_svec` / `arb_smat` with the sqrt(2) convention.

### 2.3 Exact svec/smat formulas (the contract)

For a symmetric n x n real matrix A, let the index set I = {(i,j) : 1<=i<=j<=n}
ordered (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), ... (column-major lower
triangle, equivalently row-major upper triangle).

    svec(A)_k :=  A_{ii}           if i = j  (diagonal)
                  sqrt(2) * A_{ij} if i > j  (strictly lower triangle, column-major)

with k = j*(j-1)/2 + i  (1-indexed) or equivalently j*(j-1)/2 + i - 1 (0-indexed).

The factor of sqrt(2) appears on the OFF-DIAGONAL entries.

For the inverse:

    smat(v)_{ij} := v_k           if i = j
                    v_k / sqrt(2) if i != j  (set both (i,j) and (j,i))

Verification: <A, B> = trace(A^T B) = sum_{i=j} A_{ii} B_{ii}
              + sum_{i<j} (A_{ij} B_{ij} + A_{ji} B_{ji})
              = sum_{i=j} A_{ii} B_{ii} + 2 sum_{i<j} A_{ij} B_{ij}
              = svec(A) . svec(B).

This matches PRD.md §4.1 and is confirmed to be the Mosek convention
by the sdp-solve tool layer design.

---

## 3. HSDE-NT iteration (Layer 0)

### 3.1 Homogeneous self-dual embedding

The HSDE reformulation adds homogenization scalars tau > 0 and kappa > 0
to the iterate (HsdeIterate.ts:9-14, Andersen-Roos-Terlaky 2003):

    sum_b <A_i^b, X^b>  -  b_i * tau       =  0    (primal feasibility, i=1..m)
    sum_i y_i * A_i^b + S^b - C^b * tau    =  0    (dual feasibility, per block b)
    -sum_b <C^b, X^b> + b^T y - kappa      =  0    (gap feasibility)
    sum_b <X^b, S^b> + tau * kappa          =  0    (homogenized complementarity)

Cone: X^b in S^{n_b}_+, S^b in S^{n_b}_+, tau > 0, kappa > 0.

The iterate is (X_1,...,X_{nb}, y, S_1,...,S_{nb}, tau, kappa).

**mu-denominator convention (HsdeIterate.ts:22):** The HSDE mu averages over
(n+1) slots, not n:

    mu  =  (sum_b <X^b, S^b>  +  tau * kappa) / (totalConeDim + 1)

where totalConeDim = sum_b n_b.  This differs from the non-HSDE variant which
divides by sum_b n_b.

### 3.2 Residuals

At each iteration, compute the three HSDE residuals
(HsdeNtSdpSolver.ts:371-407, HsdeIterate.ts:43-56):

    r_p_i  =  sum_b <A_i^b, X^b>  -  b_i * tau       (primal, m-vector)
    r_d^b  =  sum_i y_i A_i^b + S^b - C^b * tau       (dual, per-block n_b x n_b)
    r_g    =  -sum_b <C^b, X^b> + b^T y - kappa        (gap, scalar)

Note: in the internal minimization convention C^b is the negated cost matrix,
so `pObj = sum_b <C^b, X^b>` (using the stored negated C) equals
`-<C_orig, X>`. The gap residual r_g uses this same internal C.

Source: HsdeNtSdpSolver.ts:371-407 implements r_p (lines 375-379), r_d
(lines 380-390), r_g (line 397), mu (lines 398-400).

### 3.3 NT scaling

The Nesterov-Todd (NT) scaling W^b per block satisfies
(PsdCone.ts:169-183, Todd-Toh-Tutuncu 1998):

    W^b S^b W^b  =  X^b

Constructed via (NtSdpSolver.ts:12-20, HsdeNtSdpSolver.ts:126-163):

    L          = chol(S^b)            (S = L L^T, lower triangular)
    tmp        = L^{-T} X^b L^{-1}   (= L^{-1} X L^{-T}^T, equivalently L^T X L)
    [V, sv^2]  = eigh(L^T X L)        (symmetric eigedecomposition)
    sv         = sqrt(sv^2)
    G          = diag(sqrt(sv)) * V^T * L^{-1}
    W^b        = G^T G                (symmetric, >=  0)

Useful identities verified by construction:
    G^T diag(sv)   G  =  X^b
    G^T diag(1/sv) G  =  (S^b)^{-1}
    G^T diag(-sv)  G  =  -X^b

The TS implementation in `buildNtFactor` (HsdeNtSdpSolver.ts:126-163) adds a
diagonal jitter of 1e-14 before the Cholesky to handle near-singular S.

**Layer 0 NOTE:** This computation uses approximate (point-mode) eigendecomposition
via Jacobi iteration (PsdCone.ts:63-130, `eighJacobi`).  Inexactness is irrelevant
for Layer 0; Layer 1 derives all rigor from the result independently.  In `libarbsdp`
use `arb_mat_approx_eig_qr` (Arb's approximate eig) or port the Jacobi routine to
operate on arb midpoints at `prec` bits.

### 3.4 Schur complement assembly

The Schur complement matrix M is the m x m symmetric PSD matrix
(HsdeNtSdpSolver.ts:541-566):

    M_{ik}  =  sum_b  <A_i^b, W^b A_k^b W^b>

where the inner product is the Frobenius/trace inner product (or equivalently
the svec dot product -- see §2).

Implementation: for each block b, precompute the cache
`WAW[b][k] = W^b * A_k^b * W^b` (HsdeNtSdpSolver.ts:511-522), then assemble
M by:

    M_{ik}  +=  frobInner(A_i^b, WAW[b][k])    for each b

M is then symmetrized (average of M and M^T) as it is symmetric in exact
arithmetic (HsdeNtSdpSolver.ts:551-558).

In the svec convention: M_{ik} = sum_b svec(A_i^b)^T (I kron W^b otimes W^b) svec(A_k^b),
which is the Gram matrix structure making M manifestly symmetric PSD.

### 3.5 Mehrotra predictor-corrector with one factorization

The Cholesky factor of M is computed ONCE per iteration (after 3-way Tikhonov
regularization, §3.6) and reused for three right-hand sides (HsdeNtSdpSolver.ts
comment lines 20-27):

**RHS 1 (data direction):**

    M dy1  =  b_i + sum_b <A_i^b, W^b C^b W^b>

**RHS 2a (affine predictor, sigma=0, E_x_aff = -X):**

    M dy2  =  -<A_i, E_x_aff> - r_p_i - <A_i, W r_d W>
           =  <A_i, X^b> - r_p_i - <A_i, W r_d W>

**RHS 2b (combined corrector, sigma from Mehrotra heuristic):**

    M dy2  =  -<A_i, E_x_comb> - eta * r_p_i - eta * <A_i, W r_d W>

where `eta = 1 - sigma`.

The corrector right-hand side uses

    E_x_comb^b  =  -X^b + sigma*mu * (S^b)^{-1} - Rq^b

where Rq^b is the Mehrotra cross-term (HsdeNtSdpSolver.ts:703-758):

    hdZ^b        = G^b * dS_aff^b * (G^b)^T
    hdX^b        = diag(-sv^b) - hdZ^b      (algebraic identity for affine direction)
    symCross^b   = sym(hdX^b * hdZ^b)        (symmetrize)
    RqSv^b_{ij}  = 2 * symCross^b_{ij} / (sv^b_i + sv^b_j)   (element-wise)
    Rq^b         = (G^b)^T * RqSv^b * G^b

Reference: SDPT3 v4 source files NTrhsfun.m and NTdirfun.m.

The gap (tau, kappa) updates use:

    E_tau_aff   = -tau * kappa
    E_tau_comb  = -tau * kappa + sigma*mu - dtau_aff * dkappa_aff

Direction recovery (HsdeNtSdpSolver.ts:573-611 for data direction, 629-659 for affine):

    dS1^b   = C^b - sum_k dy1_k A_k^b
    dX1^b   = -W^b * dS1^b * W^b

    dS2^b   = -sum_k dy2_k A_k^b - eta * r_d^b
    dX2^b   = E_x - W^b * dS2^b * W^b

Scalar tau update (HsdeNtSdpSolver.ts:606-611, 799-806):

    dtau_denom  = kappa/tau - sum_b <C^b, dX1^b> + b^T dy1
    dtau_num    = -eta * r_g + sum_b <C^b, dX2^b> - b^T dy2 + E_tau/tau
    dtau        = dtau_num / dtau_denom
    dkappa      = (E_tau - kappa * dtau) / tau

Full combined direction:

    dy  = dy2 + dtau * dy1
    dS^b = dS2^b + dtau * dS1^b
    dX^b = dX2^b + dtau * dX1^b

### 3.6 Mehrotra sigma and mu_aff

After computing the affine (predictor) direction, compute the affine step
length `alpha_aff` (capped at 1.0), then
(HsdeNtSdpSolver.ts:681-695):

    mu_aff  =  (sum_b <X+alpha_aff*dX_aff, S+alpha_aff*dS_aff>
                + (tau + alpha_aff*dtau_aff)(kappa + alpha_aff*dkappa_aff))
               / (totalConeDim + 1)

    sigma_raw  =  (mu_aff / max(mu, 1e-300))^3
    sigma      =  clip(sigma_raw, 1e-8, 0.9)

Source: HsdeNtSdpSolver.ts:694-695.  The clip bounds are `[1e-8, 0.9]`.

### 3.7 Step-to-boundary and Mehrotra safeguard

**Per-block PSD step (psdMaxStep, PsdCone.ts:186-202):**

To find the maximum `alpha` keeping `X + alpha * dX >= 0`:

    L L^T = X   (Cholesky with 1e-14 diagonal jitter)
    M     = L^{-1} dX L^{-T}   (symmetrize)
    lambda_min(M) from eighJacobi
    alpha_cone = 1 / max(0, -lambda_min(M));  Infinity if lambda_min >= 0

**tau-kappa step (HsdeStepLength.ts:48-64):**

    alpha_tau   = -tau / dtau    if dtau < 0;  else Infinity
    alpha_kappa = -kappa/dkappa  if dkappa < 0; else Infinity
    alpha_tauk  = min(alpha_tau, alpha_kappa)

**Full HSDE step length:**

    alpha_raw  =  min(alpha_cone_primal, alpha_cone_dual, alpha_tauk)

**Mehrotra safeguard (HsdeNtSdpSolver.ts:827, clipStep function lines 1168-1172):**

    alpha  =  clip(max(0.95 * alpha_raw, 2*alpha_raw - 1), 0, 0.999999)

Note: the same safeguard formula appears for both the affine step
(alpha_aff is capped at 1.0 by `Math.min(1, alphaAffRaw)`, line 678) and
the combined step (safeguard applied, line 827).

### 3.8 Convergence test (6-flag HSDE)

Source: HsdeNtSdpSolver.ts:1043-1132 (`checkHsdeTermination`).

**Scaling norms:**

    bInfNorm  =  max(1, ||b||_inf)
    cFrob     =  max(1, ||C||_F)          (Frobenius norm over all blocks)

**Purified residuals (tau > 0):**

    rho_p  =  primalInf / (tau * feasTol * (1 + bInfNorm))
    rho_d  =  dualInf   / (tau * feasTol * (1 + cFrob))

where:
    primalInf  =  ||r_p||_inf
    dualInf    =  max_b ||r_d^b||_inf

**Purified objectives and gap:**

    pObjPure  =  pObj / tau
    dObjPure  =  dObj / tau
    gapAbs    =  |pObjPure - dObjPure|
    gapScale  =  max(1, min(|pObjPure|, |dObjPure|))
    rho_g     =  gapAbs / (optTol * gapScale)

**prstatus (tau-kappa indicator):**

    prstatus  =  (tau - kappa) / max(tau + kappa, 1e-300)

converges to +1 on optimal branch, -1 on infeasibility-certificate branch.

**Anti-collapse floor:** if tau + kappa < 1e-8, return "running" (degenerate
iterate, no classification possible).

**Termination conditions:**

    optimal:             rho_p <= 1 AND rho_d <= 1 AND rho_g <= 1 AND prstatus > 0.5
    primal-infeasible:   prstatus < -0.5 AND dObj > 1e-6 AND dualInf <= eps_inf*(1+|dObj|)
    dual-infeasible:     prstatus < -0.5 AND pObj < -1e-6 AND primalInf <= eps_inf*(1+|pObj|)

where `eps_inf = max(feasTol, 1e-8)`.

Resource limits: iter >= iterLimit => "iter-limit"; wall > timeLimitSec => "time-limit";
stallCount >= stallIterCap => "numerical-difficulty".

**Default tolerances (Defaults.ts:34-49):**

    feasTol      = 1e-8    (primal and dual feasibility)
    optTol       = 1e-8    (optimality gap)
    iterLimit    = 500
    timeLimitSec = 600
    stallIterCap = 10

### 3.9 Initial point

Source: HsdeNtSdpSolver.ts:1134-1143 (`initialScale`):

    bNorm  =  max_i |b_i|       (infinity norm of b)
    cFrob  =  ||C||_F           (Frobenius norm, all blocks)

    xiP  =  max(1, 10 * bNorm)
    xiD  =  max(1, 10 * cFrob + bNorm)

    X^b  =  xiP * I_{n_b}    for each block b
    S^b  =  xiD * I_{n_b}    for each block b
    y    =  0
    tau  =  1
    kappa = 1

### 3.10 Radius zeroing (point-mode guarantee, epic.6)

Source: `c/src/solve.c`, `iterate_clear_radii()`; bead arb-prec-IPM-epic.6;
CLAUDE.md rule 1.

After the iterate update at the end of each IPM step — and before the next NT
scaling — the untrusted Arb radii of every iterate variable are zeroed in-place:

    mag_zero(arb_radref(arb_mat_entry(X[b], i, j)))   for every b, i, j
    mag_zero(arb_radref(arb_mat_entry(S[b], i, j)))   for every b, i, j
    mag_zero(arb_radref(&y[i]))                        for every i
    mag_zero(arb_radref(tau))
    mag_zero(arb_radref(kappa))

This zeroing (`iterate_clear_radii`) has no arb_init/arb_clear cost (rule 7)
and makes Layer 0 operate on pure midpoints, which is what CLAUDE rule 1
("solve in points, radii not trusted inside the loop") requires.

**Why this is necessary:** the Schur/KKT condition number grows like 1/μ as
the iterate approaches the optimum (CLAUDE.md invariant 3). In Arb ball
arithmetic every matrix operation widens radii by a factor proportional to the
condition number. Left unchecked, the accumulated (untrusted) ball radius on
the iterate's eigenvalues grows 30–60 bits per IPM step until λ_min's ball
straddles zero, at which point `arb_rsqrt`/`arb_inv` in NT scaling returns a
NaN midpoint — triggering `ARBSDP_SOLVE_NUMERICAL` and a wasted precision
escalation. Zeroing the radii each step eliminates this without perturbing the
arithmetic content of the midpoints or the correctness of Layer-1 certification.

**No rigor is lost:** Layer-0 radii are *untrusted by design* (the architecture
table in PRD.md §2 lists "none" under Rigor for Layer 0). All rigor comes from
Layer 1, which operates in ball arithmetic on the Layer-0 midpoint output.

### 3.11 Best-iterate tracking

Source: HsdeNtSdpSolver.ts:436-484.

At each iteration, compute the "achieved precision" as

    achieved  =  max(rho_p, rho_d, rho_g, gapAbs / objScale)

where objScale = max(1, |pObjPure|).  Snapshot the iterate whenever
`achieved < bestAchieved`.  On non-convergence exits (iter-limit, stall, etc.),
return the best-achieved snapshot rather than the current iterate.

A "soft optimal" (status "dual-feasible") is set when mu <= feasTol AND
prstatus > 0.5 AND tau >= 1e-6 (the TAU_HEALTHY floor).

---

## 4. Three-way Tikhonov regularization

Source: Regularization.ts.

### 4.1 Three-tier diagonal lift

Before Cholesky factorization, the Schur matrix M is lifted to

    M_reg  =  M + (delta_p + delta_d + delta_g) * I

where the three regularizers are (Regularization.ts:96-115, Defaults.ts:34-49):

    delta_p  in [0, cap_p = 1e-2],  bump factor ×10
    delta_d  in [0, cap_d = 1e+2],  bump factor ×100
    delta_g  in [0, cap_g = 1e+2],  bump factor ×100

Initial (per-iteration) values: delta_p = 0, delta_d = 0, delta_g = initialJitter = 1e-12.
The delta values carry over ACROSS iterations (accumulating).

### 4.2 Failure-diagnosis heuristic

When Cholesky fails at row `r`, diagnose the tier to bump
(Regularization.ts:173-192, SDP path `makeSdpDiagnose`):

    rowNorm_i  =  sqrt(sum_b ||A_i^b||_F^2)     (constraint i norm across all blocks)
    globalNorm =  max_i rowNorm_i
    threshold  =  1e-10 * max(globalNorm, 1e-300)

    diagnose(r) = "primal"  if rowNorm_r < threshold
                  "gap"     otherwise

### 4.3 Retry loop

`factorWith3Way` (Regularization.ts:231-269):
- Try up to `maxRefactor = 20` (HsdeNtSdpSolver.ts:348) factorizations.
- Each failure: diagnose kind, bump that tier.
- If diagnosed tier is capped: fall through to "gap" -> "dual" -> "primal" in order.
- If all tiers capped and still failing: return `success=false`.

### 4.4 Critical separation: regularization is NOT part of the certified problem

**CLAUDE.md invariant 4 (non-negotiable):** The 3-way Tikhonov regularization
perturbs the SOLVED system.  `M_reg x = h` is solved instead of `M x = h`.
This is acceptable for Layer 0 (approximate) because the solution is only used
as a starting point for Layer 1.  Layer 1 MUST certify the ORIGINAL unperturbed
problem using the regularized solution only as a starting guess.

Concretely: when computing the dual residual Z^b = C^b - sum_i y_i A_i^b in
Layer 1, use the actual problem matrices C^b and A_i^b, NOT any regularized
versions.  The iterative solution y from Layer 0 is taken as-is; its error
relative to the original problem is absorbed into the lambda_min bound.

---

## 5. Layer-1 certification (Jansson bounds)

Source: PRD.md §4.4, CLAUDE.md invariants 1-2, Jansson-Chaykin-Keil 2007.

### 5.1 Architecture principle

**CLAUDE.md invariant 1:** The optimal X lives on the BOUNDARY of the PSD cone
(typically rank-deficient, hence singular).  A verified Cholesky of X will fail.
The Jansson lower bound is derived entirely from the DUAL side and never requires
certifying X as PSD.

All certification computations run in Arb BALL arithmetic at certification
precision prec_c.  All intermediate arb_mat operations carry rigorous enclosures.

### 5.2 Dual residual matrix

Given the approximate dual iterate y_tilde from Layer 0, form the dual residual
matrix per block in ball arithmetic (PRD.md §4.4, Jansson-Chaykin-Keil 2007,
Theorem 3.2):

    Z^b  =  C^b - sum_i y_tilde_i * A_i^b

The arb computation: y_tilde is lifted to arb balls (midpoints with small radii
reflecting rounding errors), and Z^b is computed as an arb_mat whose ball radius
includes all rounding in the linear combination.

### 5.3 Rigorous lambda_min lower bound via verified Cholesky shift

**CLAUDE.md invariant 2 (non-negotiable):** lambda_min lower bounds come from
verified Cholesky shift, NOT from eigendecomposition.  Rigorous symmetric
eigendecomposition is Arb's weak spot; verified Cholesky is its strength.

Procedure to find `d_b <= lambda_min(Z^b)`:

    Binary search for the largest s such that  arb_mat_cho(Z^b - s*I)  SUCCEEDS.

A successful ball-arithmetic Cholesky of `(Z^b - s*I)` is a PROOF that
`Z^b - s*I >= 0` (positive semidefinite), hence `lambda_min(Z^b) >= s`.

Implementation sketch (`certify.c`):

    s = 0
    if arb_mat_cho(Z - s*I) FAILS:
        try s = -Gershgorin_lower_bound(Z)   // definitely valid starting point
    else:
        binary search upward for largest s with successful Cholesky
    d_b = s   // rigorous lower bound on lambda_min(Z^b)

Gershgorin fallback: for each row i, `lambda_min(Z) >= min_i (Z_{ii} - sum_{j!=i} |Z_{ij}|)`.
This is always valid (not ball-arithmetic required) and provides a safe starting
point when Z is far from PSD.

If Z is verified PD (all lambda_min bounds > 0), there is no contribution from
this block to the lower bound correction.

### 5.3.1 Audition: verified lambda_min lower bound (CLAUDE.md rule 3; bead b27)

The certified `d_b <= lambda_min(Z^b)` is the single rigor source for the Jansson
lower bound (§5.4).  Candidates auditioned for the arb / FLINT 3.0.1 BALL setting
(small dense Z, high prec, Z typically near-singular at the SDP boundary):

| technique | rigor | tightness | cost (dense n) | verdict |
|-----------|-------|-----------|----------------|---------|
| Verified-Cholesky shift (`arb_mat_cho(Z - s*I)`, bisect s) | rigorous: a 1-return is a PD theorem (FLINT contract; Rump 2006 Cor 2.4) | -> lambda_min within bisection tol (~2^(-prec/2)*scale) | O(n^3 M(prec)) per shift, ~prec/2 shifts | **CHOSEN** |
| Rigorous symmetric eig (`acb_mat_eig`) | weak: eigenvector balls blow up ~1/eigengap at degeneracy | vacuous near boundary | O(n^3 M(prec)) + complex round-trip | rejected: arb's weak axis; `acb_mat_approx_eig_qr` returned ok=0 even on a diagonal matrix in `eig_audition.c` |
| Weyl / Rump perturbation (`verify_eigsym`, VSDP's choice) | rigorous BUT needs an approximate eig for the V*D*V' residual | vacuous near degeneracy | O(n^3) eig + interval matmul (~10-30x a Cholesky) | rejected for arb: VSDP runs it at double prec where V is cheap; in arb the eig dependency + matmul dominate, and Cholesky-shift already handles negative lambda_min |
| Single-shot Gershgorin | rigorous, always valid | loose for non-diagonally-dominant Z | O(n^2) | kept as the bisection SEED only, not the returned bound |
| Newton/secant on the shift | n/a | n/a | n/a | rejected: d(lambda_min)/ds = -1 makes Newton degenerate; Higham-Strabic-Sego 2016 ship pure bisection |

**Decision.** Verified-Cholesky shift with a Gershgorin seed and bisection on the
monotone predicate `g(s) = "arb_mat_cho(Z - s*I) succeeds"`.  Rationale: (1) it uses
Arb's strongest axis (ball linear algebra) and satisfies CLAUDE invariant 2; (2) a
successful ball Cholesky IS the proof -- no separate error analysis (Rump 2006: a
completed factorization of the shifted matrix certifies PD); (3) it handles negative
and ~0 lambda_min uniformly (bisect the shift through 0), so Weyl's only advantage
over Cholesky-shift evaporates while its eig dependency and matmul cost remain;
(4) the eig route is unavailable (no real-symmetric arb eigensolver) and untrustworthy
at arb precision (`eig_audition.c`).  The returned `d` satisfies `d <= lambda_min(Z^b)`
as a theorem; `d` may be `<= 0` for indefinite / rank-deficient Z (the boundary case),
which is exactly what the `min(0, d_b)` penalty in §5.4 consumes.

**Search parameters (binding).** Seed `s_lo` = Gershgorin lower bound
`min_i( lbound(Z_ii) - sum_{j!=i} ubound(|Z_ij|) )`; maintain the bracket invariant
`[s_lo: chol passes, s_hi: chol fails]`; bisect to `tol = 2^(-prec/2) * max(1, ||Z||_Gershgorin)`;
`ITER_CAP = 60`; on a chol failure near the boundary, retry once at `prec*3/2` before
trusting the failure (precision-starvation guard).  The shift is subtracted at its
UPPER endpoint so a success proves the bound for the true (point) shift.  Z is
`arbsdp_symmetrize`d first (arb_mat_cho reads only the lower triangle).

**A-priori-bound convention (no block-dimension factor in our trace form).**  Our
§5.4 uses `x_bar_b >= tr(X_b)` (a TRACE bound).  For symmetric D and `X >= 0`,
`tr(DX) >= lambda_min(D) * tr(X)`; when `lambda_min(D) < 0` and `tr(X) <= x_bar`,
this gives `tr(DX) >= min(0, lambda_min(D)) * x_bar` -- with NO dimension factor.
Jansson-Chaykin-Keil 2007 (Thm 3.2 / Lemma 3.1) carry a block-dimension factor
`s_b` because their a-priori bound is a Loewner / eigenvalue bound `X <= x_bar*I`
(for which `tr(X) <= s_b*x_bar`); the two forms are equivalent under
`x_bar_trace = s_b * x_bar_Loewner`.  **BINDING for b18/b19:** the a-priori-bound
API and every golden MUST supply `x_bar_b` as a TRACE bound -- supplying a Loewner /
lambda_max bound without the `s_b` factor would over-state lb and could EXCLUDE the
optimum (P0, CLAUDE rule 2).  Re-verified against JCK 2007 directly and guarded by a
convention-discriminating bracket test (full-rank optimum, `n_b>=2`): bead
arb-prec-IPM-om9, DONE 2026-06-02; see the om9 reconfirmation below.

**om9 reconfirmation (primary source, 2026-06-02).** Re-verified directly against
Jansson-Chaykin-Keil 2007 (SIAM J. Numer. Anal. 46(1):180-200): their Lemma 3.1 and
Thm 3.2 (eq. 3.7) use a LOEWNER a-priori bound lambda_max(X_j) <= x_j and carry the
block-dimension factor s_j: lb = b^T y + sum_j s_j * min(0,d_j) * x_j.  The s_j
arises in the Lemma 3.1 proof because <D,X> = sum_{k=1..s_j} lambda_k(D) * q_k^T X q_k
and each of the s_j terms is bounded below by min(0,lambda_min(D)) * lambda_max(X) <=
min(0,lambda_min(D)) * x_j, summing to s_j copies.  VSDP 2020 confirms the lambda_max
convention for its primal a-priori input.  Our §5.4 uses a TRACE bound xbar_j >= tr(X_j);
since tr(DX) >= lambda_min(D)*tr(X) directly (no per-eigenvalue split), the factor does
NOT appear.  The two are equivalent under xbar_trace = s_j * x_Loewner.  CONCLUSION:
§5.4 with NO s_b factor is correct iff the supplied xbar is a TRACE bound; supplying a
Loewner/lambda_max bound as xbar (without multiplying by s_b) would over-state lb and
could EXCLUDE the optimum (P0, CLAUDE rule 2).  The certify.h apriori API documents xbar
as a trace bound; a convention-discriminating bracket test (full-rank optimum, n_b>=2)
guards it.  IMPLEMENTED: `test_trace_vs_loewner` in c/test/test_certify_bracket.c builds
max <I,X> s.t. X11=X22=1, X12=0 (unique X*=I: tr=2, lambda_max=1, opt=2) and feeds
synthetic y_ext with eps=2^-10 to arbsdp_lower_bound / arbsdp_upper_bound: with the TRACE
bound xbar=2 the bracket endpoints land at exactly opt=2 (rigorous, tight), while the
LOEWNER value xbar=1 yields lb=2+2^-10 > opt and ub=2-2^-10 < opt (EXCLUDES) -- so the
trace convention is necessary and there is no s_b factor.  Mutation check (CLAUDE rule 9):
injecting a spurious *s_b factor into arbsdp_lower_bound/upper_bound fails 4 of the 6
assertions while the rank-1 rigor-gate goldens (tr == lambda_max) stay green, confirming
the discriminator catches what the existing goldens cannot.

**Scope.** b17 implements only `arbsdp_verified_psd`, `arbsdp_gershgorin_lower_bound`,
and `arbsdp_lambda_min_lower_bound` (returns `d_b`).  The lb/ub ASSEMBLY and the
upper-bound `y_bar` derivation (§5.5) are b18/b19.

**References.** Jansson-Chaykin-Keil 2007, SIAM J. Numer. Anal. 46(1):180-200
(Thm 3.2, Lemma 3.1).  Rump, BIT 2006, 46:433-452 (Thm 2.3 / Cor 2.4: shifted
Cholesky proves PD, incl. interval matrices).  Higham-Strabic-Sego, SIAM Review
2016, 58(2):245-263 (bisection definiteness oracle).  Gershgorin 1931.  FLINT 3.0.1
`arb_mat.h:378` (`arb_mat_cho` returns 1 = certainly PD over input balls).

### 5.4 A-priori trace bound (required for finite lower bound)

**CLAUDE.md invariant 5 (non-negotiable):** A finite lower bound requires a
finite bound `x_bar^b >= tr X^b` for each block b.

The Jansson lower bound is:

    optimum (of max <C,X>)  >=  b^T y_tilde + sum_b min(0, d_b) * x_bar^b

where d_b is the lambda_min lower bound from §5.3.

If lambda_min(Z^b) >= 0 for all b (dual feasible), then min(0, d_b) = 0 and
the lower bound is just b^T y_tilde (the dual objective).

If lambda_min(Z^b) < 0 for some b, the penalty term `min(0, d_b) * x_bar^b`
is negative (making the bound weaker) but remains rigorous.

**No bound => lb = -infinity.** If `x_bar^b` is not supplied for some block b
with `d_b < 0`, the lower bound for that block is `-infinity`.  Report honestly;
never silently assume boundedness.

The a-priori bound is a FIRST-CLASS API input (PRD.md §4.6).  For the primary
use cases it is free: density matrices have tr = 1, normalized moment matrices
have a bounded [1,1] entry.

### 5.4.1 Tight primal lower bound (Rayleigh quotient of a feasible rank-1 point)

Bead arb-prec-IPM-vry.  The dual-residual lower bound (§5.4) is STRUCTURALLY LOOSE
for max-eigenvalue problems.  For  max <C,X> s.t. tr(X)=1, X>=0  (opt = lambda_max(C)),
at any external dual y_ext the residual is Z_ext = C - y_ext I, and the §5.4 bound
lb = b^T y_ext + min(0, lambda_min(Z_ext)) is STUCK at lambda_min(C) for every y_ext:
at the converged dual y_ext = lambda_max(C), Z_ext = C - lambda_max(C) I has
lambda_min = lambda_min(C) - lambda_max(C), giving lb = lambda_max(C) + (lambda_min(C)
- lambda_max(C)) = lambda_min(C).  E.g. the bracket [1,3] for max_eigenvalue_2x2
(opt=3=lambda_max, lb stuck at lambda_min=1).  No choice of y_ext tightens it.

A TIGHT rigorous lower bound comes from the PRIMAL side (a feasible primal point lower-
bounds a max).  For a single trace constraint <I,X>=b_1 and ANY v != 0,
    X_hat = b_1 * v v^T / (v^T v)
is EXACTLY feasible -- PSD by construction (rank-1 outer product; NO Cholesky, works at
the PSD boundary, invariant 1) and tr(X_hat)=b_1 -- so
    lb_primal = b_1 * (v^T C v)/(v^T v) = b_1 * R(v) <= b_1 * lambda_max(C) = opt
where R(v) is the Rayleigh quotient and R(v) <= lambda_max(C) for every v (Courant-
Fischer; Parlett, The Symmetric Eigenvalue Problem, 1998).  Evaluated in BALL arithmetic
(lower endpoint = lower(v^T C v)/upper(v^T v)) it is rigorous for ANY v REGARDLESS of
solve quality, and TIGHT when v ~ top eigenvector (R is quadratically accurate near an
eigenvector).  v is the POINT-MODE top eigenvector of the solver's primal block X~ (a
direction only, never a rigor dependency).

arbsdp_certify_bracket sets lb = max(dual-residual lb §5.4, primal Rayleigh lb): both
are rigorous lower bounds so their max is rigorous and tighter.  Implemented:
arbsdp_primal_lower_bound (certify.c).  Measured (max_eigenvalue_2x2 / max_eig_tridiag_3x3
/ max_eig_path_4 at prec_c = final_prec+128): bracket lb moves from lambda_min(C) (gap ~2)
to within ~2^-192 of opt (true_opt - lb ~ 1e-192).  APPLICABILITY: single block, single
trace constraint A_1 = I (conservative exact check; returns 0 -> caller keeps the dual
bound otherwise).  Per-block trace constraints (separable_12block) and general higher-rank
/ coupled-constraint feasible primal points (two_block_corr_coupled; shares the primal-PSD-
projection machinery of UB-B/§5.5, bead 9tm) are follow-ups: bead arb-prec-IPM-oc7.

GOLDEN-REFERENCE PRECISION (vry finding, CLAUDE rule 2/8).  The golden optimal_value
strings are correctly-rounded to ~65 digits, so for an IRRATIONAL optimum the string sits
up to half a ulp BELOW the true optimum.  A now-tight lb (resolving the optimum to ~190
digits) can land ABOVE the 65-digit point while remaining <= the TRUE optimum; comparing a
bound against the bare rounded point would FALSELY flag a P0.  The rigor tests therefore
compare against (a) an ENCLOSURE ball of the true optimum (reference +/- one ulp, a
conservative over-cover of the half-ulp rounding) and, for the analytic goldens, (b) the
CLOSED-FORM true optimum (3, 2+sqrt3, (5+sqrt5)/2) computed in-test at 4096 bits.  Verified:
lb <= true closed-form opt with margin ~2^-192, 0 violations.  This WIDENS the reference
(rule 2: widen the interval, never the bound); the bound itself is unchanged and rigorous.

### 5.5 Rigorous upper bound (dual-side symmetric mirror of §5.4)

The upper bound on `p*_ext = max <C_file, X>` is the exact dual-side MIRROR of the
§5.4 lower bound. It consumes the SAME a-priori TRACE bound `xbar_b >= tr(X*_b)`
(NOT a dual-norm bound) and reuses the shipped verified-Cholesky-shift kernel
(`arbsdp_lambda_min_lower_bound`, b17). No primal-PSD certification is required,
honoring CLAUDE invariant 1 (certify from the dual side).

**Master identity (for EVERY feasible X and ANY y in R^m):**

    <C_file, X> = <C_file - sum_i y_i A_i, X> + sum_i y_i <A_i, X>
                = <Z, X> + b^T y ,      where  Z := C_file - sum_i y_i A_i.

(uses <A_i,X> = b_i). Splitting per block and bounding the Rayleigh-trace product
both ways gives, for X_b >= 0 with tr(X_b) <= xbar_b:

    sum_b min(0, lambda_min(Z_b)) xbar_b  <=  <Z,X>  <=  sum_b max(0, lambda_max(Z_b)) xbar_b.

Adding `b^T y` and maximizing over feasible X yields the two-sided bracket; the
lower half is §5.4, the upper half is:

**Theorem 5.5 (rigorous upper bound, UB-A).** Let `y_ext = -y_int`,
`y_int = it->y / it->tau` (tau-purified internal dual; §5.4 / sign banner), and
`Z_ext^b = C_file^b - sum_i (y_ext)_i A_i^b` in ball arithmetic. Let
`dhi_b >= lambda_max(Z_ext^b)` be the rigorous upper bound computed via
`dhi_b = -arbsdp_lambda_min_lower_bound(-Z_ext^b, prec_c)` (the verified Cholesky
shift on the NEGATED block; `lambda_max(Z) = -lambda_min(-Z)`). Then:

    p*_ext  <=  b^T y_ext  +  sum_{b=1}^{nb}  max(0, dhi_b) * xbar_b  =:  ub_ext.

`ub_ext` is taken at the Arb UPPER endpoint of the assembled ball. If for some
block `dhi_b > 0` and `xbar_set[b] = 0`, then `ub_ext = +inf` (reported honestly,
mirroring `lb = -inf`; CLAUDE invariant 5, rule 2). The clamp `max(0, dhi_b)`
(vs. `min(0, dlo_b)` for the lower bound) is the correct safe over-estimate of
`<Z_b,X_b>` when `Z_b` is negative definite (the over-estimate -> 0 as tr X_b -> 0).

**Proof.** Master identity + Rayleigh-trace inequality + maximize over feasible X.
A successful ball Cholesky of `(-Z_ext^b - s I)` proves `-Z_ext^b - s I >= 0`, i.e.
`lambda_min(-Z_ext^b) >= s`, i.e. `lambda_max(Z_ext^b) <= -s = dhi_b` (Rump 2006
Cor 2.4; FLINT `arb_mat_cho` contract). QED.

**Sign contract (P0).** Both `lb_ext` (§5.4) and `ub_ext` consume the SAME
external dual `y_ext = -y_int` and the SAME file-sign data (C_file, A_i) via
`arbsdp_dual_residual`. Passing `y_int` instead forms `Z_int = -Z_ext`, swaps
lambda_min <-> lambda_max, and yields a bracket that EXCLUDES the optimum.
Independent cross-check (defends against a double flip): the §5.4 lower bound run
on the INTERNAL min-problem `p*_int = min <C_int,X>` gives
`p*_int >= b^T y_int + sum_b min(0, dlo(Z_int^b)) xbar_b`; negating
(`p*_ext = -p*_int`, with `Z_int = -Z_ext`) reproduces `ub_ext` exactly.

**Tightness.** A single `y` cannot make BOTH bounds tight for a max problem: at the
solver's converged dual (`y_ext -> y*`), `Z_ext` sits on the PSD boundary so
`lambda_max(Z_ext) -> 0+` and `ub_ext -> p*` (TIGHT), while `lambda_min(Z_ext) << 0`
makes `lb_ext` loose but still valid (contains p*). The b19 deliverable is the
rigorous BRACKET (rigor); tightening `lb_ext` is a Layer-2 / dual-correction concern.

**Secondary upper bound (UB-B, ybar fallback; NOT implemented in b19).** When a
tight primal iterate and a dual-norm bound `ybar_i >= |y*_i|` are available, the
Jansson 2009 Thm 4.2 / VSDP `vsdpup` residual form applies:

    p*_ext  <=  <C_file, Xt+>  +  ybar^T r,
        Xt+ = block-wise PSD projection of the approximate primal
              (X_b shifted by |min(0, lambda_min(X_b))| I),
        r_i >= |<A_i, Xt+> - b_i|   (rigorous primal residual, ball upper endpoint).

UB-B needs primal-PSD-projection machinery arbsdp does not yet have; it is the
right tool only when `xbar` is unavailable but `ybar` + a near-feasible primal
are. RE-SCOPE: the FINITE upper bound needs `xbar` (via UB-A), not `ybar`; the
`arbsdp_apriori.ybar` field and §10 decision 3 are retained for UB-B only.
Tracked: bead arb-prec-IPM-9tm (UB-B fallback).

**References.** Jansson-Chaykin-Keil 2007 (Thm 3.2 lower; UB-A is its dual-side
trace-mirror). Jansson 2009, Japan J. Indust. Appl. Math. 26:337-363, Thm 4.2
(UB-B residual form). Rump, BIT 2006, Cor 2.4 (shifted Cholesky proves PD). VSDP
2020 `@vsdp/rigorous_upper_bound.m`.

### 5.6 Status classification from Layer-1 output

    "optimal"            if ub - lb <= tol and both finite
    "primal_infeasible"  if HSDE (tau, kappa) certificate verifies in ball arithmetic
                         (verified Farkas: ||r_d|| enclosure <= rel * b^T y, tau->0, kappa>0)
    "dual_infeasible"    if HSDE (tau, kappa) certificate verifies dually
                         (<C, X> enclosure negative, ||r_p|| enclosure <= rel * |<C, X>|)
    "inconclusive"       gap too wide, or bounds infinite -> escalate to Layer 2 or
                         bump precision and re-solve

The infeasibility certificates are checked in BALL arithmetic -- NOT float64
heuristics (CLAUDE.md invariant 8).

---

## 6. Precision controller

Source: PRD.md §4.3, CLAUDE.md invariant 3.

### 6.1 Adaptive working precision policy

Working precision `prec` (bits) is adaptive, not fixed.

**Initial precision:**

    prec_0  =  max(128, ceil(requested_output_digits * log2(10)) + margin)

where `margin` is a fixed buffer (e.g., 64 bits) to cover rounding accumulation
in the solve.

**Escalation triggers (any one suffices):**

(a) Schur Cholesky requires regularization beyond a threshold, i.e.,
    `delta_p + delta_d + delta_g > escalation_threshold` after a refactor.
(b) Duality gap stalls: mu fails to decrease by more than X% for K consecutive
    iterations (mirrors COPT's stall counter; default K = stallIterCap = 10,
    X ~ 1%).  Source: HsdeNtSdpSolver.ts:850-854 (stall detection: `if (muNew > 0.99 * mu) stallCount++`).
(c) Relative residual floor stops improving.

**Escalation strategy:**

    prec <- 2 * prec          (doubling, simple and robust)
    -- or --
    prec <- c * (-log2(mu)) + margin   (model-based: match Schur condition number growth)

The model-based formula arises from the fact that the Schur condition number
grows like 1/mu (CLAUDE.md invariant 3), so the required precision to maintain
accuracy in the Cholesky grows like log2(1/mu) = -log2(mu) bits.

**Caps and graceful degradation:**

    prec_max    -- hard cap on working precision (e.g., 4096 bits)
    iter_max    -- hard cap on iterations (DEFAULT_PARAMS: iterLimit = 500)

On hitting either cap, return the best iterate with status "limit" (mapped to
"inconclusive" in the output object) and let Layer 1 attempt to certify
whatever bound it can from the current approximate solution.

**Float64 warm-start (optional):** Run the existing TS solver (or a float64
inner loop) to generate a starting point, then lift to arb.  This typically
halves the number of arb iterations.

---

## 7. Output object and status semantics

Source: PRD.md §4.7.

### 7.1 asdp_result struct (C ABI)

    struct asdp_result {
      int    status;              // see §7.2
      arb_t  obj_lb, obj_ub;     // rigorous bracket [lb, ub] on the optimal value
      arb_mat_t *X_blocks;       // primal enclosures (nb arb_mat blocks)
      arb_ptr  y;                // dual multipliers enclosure (m-vector of arb)
      arb_mat_t *S_blocks;       // dual slack enclosures (nb arb_mat blocks)
      arb_t  tau, kappa;         // HSDE homogenization scalars (enclosures)
      // Certificate provenance
      int    cert_kind;          // 0=optimality_bound, 1=primal_farkas, 2=dual_farkas
      char  *cert_data;          // JSON-encoded certificate details
      // Diagnostics
      int    iters;
      int   *prec_history;       // precision at each iteration
      double *reg_history;       // regularization level at each iteration
      double *gap_history;       // mu at each iteration
      double  wall_ns;           // total wall time in nanoseconds
    };

### 7.2 Status codes

    ASDP_OPTIMAL           = 0   // rigorous [lb,ub] with ub-lb <= tol
    ASDP_PRIMAL_INFEASIBLE = 1   // verified Farkas-y certificate
    ASDP_DUAL_INFEASIBLE   = 2   // verified primal recession-ray certificate
    ASDP_INCONCLUSIVE      = 3   // gap too wide; Layer 2 needed or problem hard
    ASDP_LIMIT             = 4   // iter_max or prec_max hit; best-effort bracket

**Semantic contracts:**

- `ASDP_OPTIMAL`: The true optimal value of the original (unregularized, SDPA
  convention `max <C, X>`) problem is proved to lie in `[obj_lb, obj_ub]`.
  This is a theorem.

- `ASDP_PRIMAL_INFEASIBLE`: The original problem has no feasible X.  The
  certificate (y_tilde, S_tilde) satisfies:
      sum_i y_i A_i + S = 0 (S >= 0)  and  b^T y > 0
  verified in ball arithmetic.

- `ASDP_DUAL_INFEASIBLE`: The original problem is unbounded above (the dual
  is infeasible).  Certificate X_tilde satisfies:
      <A_i, X_tilde> = 0 for all i  and  <C, X_tilde> > 0
  verified in ball arithmetic.

- `ASDP_INCONCLUSIVE`: Layer 1 ran successfully but could not produce a finite
  gap or verified certificate.  obj_lb and obj_ub may be finite but wide, or
  one may be infinite.  Layer 2 (interval-Newton/Krawczyk or precision bump)
  should be invoked.

- `ASDP_LIMIT`: A resource cap was reached.  obj_lb/obj_ub are the best
  available Layer-1 bounds from the best-iterate snapshot.  They may not be
  tight.  The `prec_history` and `gap_history` arrays document exactly what
  happened.

**The "rigor is falsifiable" contract (CLAUDE.md rule 2):** Any bracket that
ever excludes a known/constructed optimal value is a P0 bug.  Never widen a
claim to fit a result; widen the interval or report `inconclusive`.

---

## 8. Summary of formula sources

| Formula | Source |
|---------|--------|
| SDPA standard form, internal min convention | SdpProblem.ts:7-8; PRD.md §3 |
| Block-diagonal structure, frobInner | PsdCone.ts:18-23; SdpProblem.ts:6-11 |
| svec convention | PRD.md §4.1; CLAUDE.md invariant 6 |
| HSDE system of equations | HsdeIterate.ts:9-14; Andersen-Roos-Terlaky 2003 |
| mu denominator n+1 | HsdeIterate.ts:22 |
| Residuals r_p, r_d, r_g | HsdeNtSdpSolver.ts:371-407 |
| NT scaling W: WSW=X | PsdCone.ts:169-183; NtSdpSolver.ts:12-20; Todd-Toh-Tutuncu 1998 |
| buildNtFactor (G, W) | HsdeNtSdpSolver.ts:126-163 |
| Schur M_{ik} | HsdeNtSdpSolver.ts:541-566 |
| Mehrotra affine (sigma=0) | HsdeNtSdpSolver.ts:613-659 |
| E_x_comb, Rq cross-term | HsdeNtSdpSolver.ts:697-758 |
| sigma = (mu_aff/mu)^3, clip [1e-8, 0.9] | HsdeNtSdpSolver.ts:694-695 |
| Mehrotra safeguard alpha | HsdeNtSdpSolver.ts:827; PsdCone.ts:186-202 |
| tau-kappa step | HsdeStepLength.ts:48-64 |
| Convergence rho_p, rho_d, rho_g | HsdeNtSdpSolver.ts:1043-1132 |
| prstatus | HsdeNtSdpSolver.ts:1078 |
| Optimal/infeasible classification | HsdeNtSdpSolver.ts:1089-1131 |
| Default tolerances | Defaults.ts:34-49 |
| 3-way Tikhonov caps/bumps | Regularization.ts:96-115; Defaults.ts:34-49 |
| SDP diagnose heuristic | Regularization.ts:173-192 |
| factorWith3Way retry | Regularization.ts:231-269 |
| Jansson lower bound | Jansson-Chaykin-Keil 2007 Thm 3.2; PRD.md §4.4 |
| lambda_min via Cholesky shift | CLAUDE.md invariant 2; PRD.md §7 |
| Gershgorin fallback | PRD.md §4.4 |
| Precision escalation heuristic | PRD.md §4.3; CLAUDE.md invariant 3 |
| Initial point | HsdeNtSdpSolver.ts:1134-1143 |
| Output struct | PRD.md §4.7 |

---

## 9. Open questions for the orchestrator

1. **svec index order.** The svec convention is specified as "strict-Mosek sqrt(2)
   off-diagonal", but the exact index ordering (column-major lower triangle vs.
   row-major upper triangle vs. another) is not pinned in PRD.md §4.1 or CLAUDE.md.
   Both orderings produce `svec(A).svec(B) = <A,B>` but differ in memory layout.
   Mosek uses column-major upper triangle.  A concrete decision must be made before
   coding `svec.c` and locked in the header.  Recommend: document in `svec.h` as
   a `#define SVEC_UPPER_COL_MAJOR` comment with the exact formula
   `k = j*(j+1)/2 + i  (0-indexed, i<=j)`.

2. **Certification precision prec_c vs. working precision prec.** PRD.md §4.4
   mentions "certification precision prec_c" as distinct from the Layer-0 working
   precision prec.  The relationship (prec_c = prec? prec_c = prec + 64? a fixed
   value?) is not specified.  A coder implementing `certify.c` needs this pinned.
   Recommended default: prec_c = prec + 128 (an extra 128 bits for certification
   arithmetic overhead).

3. **Verified upper bound: y_bar requirement.** The rigorous upper bound construction
   in §5.5 requires a bound `y_bar >= ||y||`.  The PRD mentions "a bound on ||y||"
   (§4.4) but does not specify how it is computed or supplied.  For the primary use
   cases (bounded feasible sets), a primal trace bound x_bar implies a dual norm bound
   via the KKT slackness, but the formula is not written down.  This should be derived
   and added to MATH_SPEC before coding `certify.c`.

4. **Regularization carry-over policy.** The TS implementation carries delta values
   ACROSS iterations (Regularization.ts comment: "Per-iteration adaptive init: NOT
   implemented in v1").  The PRD §4.2 step 4 says "bump-on-failure scheme and caps"
   but does not specify carry-over vs. reset-per-iter.  For `libarbsdp`, the
   carry-over behavior should be an explicit design decision (not inherited silently).
   Carrying over helps ill-conditioned problems but may over-regularize well-conditioned
   ones.

5. **Stall detection: exact mu decrease threshold.** The TS solver uses `muNew > 0.99 * mu`
   as the stall criterion (HsdeNtSdpSolver.ts:851), i.e., less than 1% decrease
   triggers a stall count increment.  The precision-controller escalation trigger
   in PRD.md §4.3 says "fails to decrease >X% for K iters" with X unspecified.
   Pinning X = 1% and K = stallIterCap = 10 (matching the TS default) is the
   natural choice; confirm with the orchestrator.

6. **factorWith3Way: maxRefactor discrepancy.** The TS `HsdeNtSdpSolver.ts:348`
   passes `maxRefactor=20` to `regParamsFromIpm`, but `Defaults.ts` comment line 14
   says "MAX_REFACTOR = 10 per spec §6.4 (was 20 in the LP path; aligned here)".
   The SDP solver still uses 20 (hardcoded at the call site).  For `libarbsdp`,
   pick one value and document it.  Recommend 20 for robustness (the extra retries
   cost nothing on well-conditioned problems).

## 10. Decisions (orchestrator resolutions — these are now binding)

1. **svec index order — DECIDED: upper-triangle column-major, √2 off-diagonal.**
   `k = j*(j+1)/2 + i` (0-indexed, `i <= j`); diagonal entries unscaled, strictly
   upper entries scaled by √2, so `svec(A)·svec(B) = <A,B>`. Pin in `svec.h` with a
   header comment + the formula. (Resolves Q1; addressed in b04.)

2. **prec_c — DECIDED: prec_c = prec + 128 initially, escalates independently.**
   Certification starts 128 bits above the Layer-0 working precision; if the bracket
   `ub-lb` is wider than the requested tolerance, `prec_c` escalates on its own
   (geometric) before Layer 2 is invoked. (Resolves Q2; addressed in b18/b19.)

3. **Dual norm bound ȳ — DECIDED: first-class input, symmetric to x̄.** The verified
   upper bound takes a supplied per-block / global `ȳ >= ||y||` exactly as the lower
   bound takes `x̄ >= tr X`. The explicit KKT-slackness derivation of ȳ from problem
   data (when not supplied) is IN SCOPE of b18 and must be written into this spec
   there. No ȳ supplied ⇒ `ub = +∞`, reported honestly (never silently assumed).
   (Resolves Q3; addressed in b18.)

4. **Regularization carry-over — DECIDED: carry δ across iterations (match TS).**
   Carry-over (not reset-per-iter), as the TS path does. (Resolves Q4; addressed in b13.)

5. **Stall criterion — DECIDED: X = 1% (muNew > 0.99·mu), K = 10.** Matches the TS
   default `stallIterCap`. (Resolves Q5; addressed in b16.)

6. **maxRefactor — DECIDED: 20.** Matches the SDP call site; extra retries are free
   on well-conditioned problems. (Resolves Q6; addressed in b13.)

7. **arb-midpoint vs arf for Layer-0 iterate arithmetic — DECIDED: radius-clearing
   on arb_mat (CLAUDE rule 1; see §3.10 and PRD §11.3).** `arf_t` rejected because
   FLINT 3.0.1 ships no `arf_mat` module; radius-clearing gives identical point-mode
   semantics while reusing the tested `arb_mat_*` kernels. (Resolves PRD §11.3;
   implemented in epic.6, `c/src/solve.c:iterate_clear_radii`.)
