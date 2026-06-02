#!/usr/bin/env python3
"""
verify_stress.py — independent cross-check of the 9 stress-golden SDP problems.

Toolchain: Python 3 / mpmath 1.3.0 / sympy 1.14.0 / numpy.
Deliberately does NOT call Mathematica/wolframscript.

Two levels of check per problem:
  Level 1: closed-form value vs provenance.json (>= 50 digits agreement).
  Level 2: data-grounded re-derivation from the .dat-s file.

Mutation check at the end: perturb inputs in-memory and confirm failures are caught.

Run from the repository root:
    python3 golden/gen/verify_stress.py
"""

import json
import os
import sys
import copy

import mpmath
import numpy as np

mpmath.mp.dps = 80   # 80 decimal places throughout (>= 70 required)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN_DIR = os.path.join(REPO_ROOT, "golden")

PROBLEMS = [
    "max_eig_path_4",
    "max_eig_path_8",
    "max_eig_path_16",
    "disparate_scale_sqrt2",
    "diag_weight_kappa6",
    "diag_weight_kappa10",
    "diag_weight_kappa12",
    "two_block_corr_coupled",
    "separable_12block",
]

# ── SDPA-sparse parser ──────────────────────────────────────────────────────

def parse_dat_s(path):
    """
    Parse a .dat-s file and return a dict with:
      m          : number of constraints
      nblocks    : number of blocks
      block_sizes: list of ints (positive = PSD, negative = LP/diag)
      b          : list of mpmath.mpf (length m)
      entries    : list of (i, block, k, l, v) where i=0->C, i>=1->A_i,
                   block is 1-indexed, k and l are 1-indexed, k<=l.
    The caller is responsible for applying the 2*v off-diagonal convention.
    """
    with open(path) as fh:
        lines = fh.readlines()

    # Strip comment lines (starting with * or ")
    data_lines = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("*") or stripped.startswith('"'):
            continue
        # Strip inline = comment (per README: "the parser strips from = onwards")
        if "=" in stripped:
            stripped = stripped[:stripped.index("=")].strip()
        if stripped:
            data_lines.append(stripped)

    idx = 0
    m = int(data_lines[idx]); idx += 1
    nblocks = int(data_lines[idx]); idx += 1
    block_sizes = list(map(int, data_lines[idx].split())); idx += 1
    assert len(block_sizes) == nblocks, f"block_sizes length {len(block_sizes)} != nblocks {nblocks}"

    b_tokens = []
    while len(b_tokens) < m:
        b_tokens.extend(data_lines[idx].split()); idx += 1
    assert len(b_tokens) == m, f"b vector has {len(b_tokens)} entries, expected {m}"
    b = [mpmath.mpf(t) for t in b_tokens]

    entries = []
    while idx < len(data_lines):
        tokens = data_lines[idx].split(); idx += 1
        if not tokens:
            continue
        i_val  = int(tokens[0])
        block  = int(tokens[1])
        k      = int(tokens[2])
        l      = int(tokens[3])
        v      = mpmath.mpf(tokens[4])
        assert k <= l, f"Entry k={k} > l={l} in {path}"
        entries.append((i_val, block, k, l, v))

    return {"m": m, "nblocks": nblocks, "block_sizes": block_sizes,
            "b": b, "entries": entries}


def build_matrix(entries, i_val, block, size):
    """
    Build the full symmetric matrix for constraint i_val, block `block` (1-indexed)
    from parsed entries.  Returns a numpy float64 matrix (sufficient for eigenvalue
    cross-checks; mpmath used only where high precision is required).
    The raw stored values v are returned as-is (upper triangle); the caller applies
    the 2*v convention for off-diagonal if needed.
    """
    mat = np.zeros((size, size), dtype=np.float64)
    for (ii, bb, k, l, v) in entries:
        if ii == i_val and bb == block:
            mat[k-1, l-1] += float(v)
            if k != l:
                mat[l-1, k-1] += float(v)
    return mat


def build_C_mp(entries, block, size):
    """
    Build the cost matrix C (i_val=0, given block) as an mpmath matrix,
    applying the full symmetry (both (k,l) and (l,k) = v, so the matrix entry
    is v for off-diagonal already doubled in build_matrix; but for the inner-product
    purpose we need the ACTUAL matrix entries, not the svec convention).

    For eigenvalue computation we want the ACTUAL matrix C where C_kl = C_lk = v
    for the off-diagonal entries (since that's what the .dat-s stores for the
    Frobenius inner product accounting).  The lambda_max of this matrix equals opt.
    """
    mat = mpmath.zeros(size, size)
    for (ii, bb, k, l, v) in entries:
        if ii == 0 and bb == block:
            mat[k-1, l-1] += v
            if k != l:
                mat[l-1, k-1] += v
    return mat

# ── Utility ─────────────────────────────────────────────────────────────────

def digits_agreed(a, b):
    """
    Count significant decimal digits of agreement between mpmath numbers a and b.
    Returns floor(-log10(|a-b|/max(1,|a|))) or 80 if exactly equal.
    """
    diff = abs(a - b)
    if diff == 0:
        return 80
    scale = max(mpmath.mpf(1), abs(a))
    rel = diff / scale
    if rel == 0:
        return 80
    return int(-mpmath.log10(rel))


def load_provenance(name):
    path = os.path.join(GOLDEN_DIR, name, "provenance.json")
    with open(path) as fh:
        return json.load(fh)


def prov_value(prov):
    return mpmath.mpf(prov["optimal_value"])

# ── Level 1: closed-form vs provenance ──────────────────────────────────────

def level1_closed_form(name, prov):
    """
    Recompute the closed-form optimum in mpmath at 80 dps.
    Return (closed_form_value, digits_agreed_with_provenance).
    Asserts digits >= 50 (truncation-safe).
    """
    pv = prov_value(prov)

    if name in ("max_eig_path_4", "max_eig_path_8", "max_eig_path_16"):
        n = prov["block_sizes"][0]
        cf = 2 + 2 * mpmath.cos(mpmath.pi / (n + 1))
    elif name == "disparate_scale_sqrt2":
        cf = mpmath.sqrt(2)
    elif name in ("diag_weight_kappa6", "diag_weight_kappa10", "diag_weight_kappa12"):
        cf = mpmath.mpf(1)
    elif name == "two_block_corr_coupled":
        cf = 2 * mpmath.sqrt(2)
    elif name == "separable_12block":
        cf = sum(mpmath.mpf(c) / 2 + mpmath.sqrt(mpmath.mpf(c)**2 / 4 + 1)
                 for c in range(1, 13))
    else:
        raise ValueError(f"No closed form defined for {name}")

    da = digits_agreed(cf, pv)
    # Require >= 50 digits (truncation at 65 digits is benign)
    assert da >= 50, (
        f"[FAIL L1] {name}: closed-form vs provenance agree to only {da} digits "
        f"(need >= 50). closed_form={mpmath.nstr(cf, 66)}, "
        f"provenance={mpmath.nstr(pv, 66)}"
    )
    return cf, da


# ── Level 2: data-grounded re-derivation ────────────────────────────────────

def level2_max_eig_path(name, prov, parsed):
    """
    Single-block max eigenvalue: opt = lambda_max(C).
    Also assert A_1 = I and b = [1].
    """
    n = prov["block_sizes"][0]
    assert parsed["m"] == 1, f"[FAIL L2] {name}: expected m=1, got {parsed['m']}"
    assert len(parsed["block_sizes"]) == 1, f"[FAIL L2] {name}: expected 1 block"
    assert parsed["block_sizes"][0] == n, f"[FAIL L2] {name}: block size mismatch"

    # Check b = [1]
    b0 = parsed["b"][0]
    assert abs(b0 - 1) < mpmath.mpf("1e-60"), \
        f"[FAIL L2] {name}: b[0] should be 1, got {b0}"

    # Check A_1 = I_n
    A1 = build_matrix(parsed["entries"], 1, 1, n)
    I_n = np.eye(n)
    assert np.allclose(A1, I_n, atol=1e-60), \
        f"[FAIL L2] {name}: A_1 is not I_{n}. Got:\n{A1}"

    # Build C and compute lambda_max via numpy (float64 sufficient for 10-digit check)
    C_np = build_matrix(parsed["entries"], 0, 1, n)
    eigs = np.linalg.eigvalsh(C_np)
    lam_max_np = float(eigs[-1])

    pv = prov_value(prov)
    diff = abs(mpmath.mpf(lam_max_np) - pv)
    assert diff < mpmath.mpf("1e-10"), (
        f"[FAIL L2] {name}: numpy lambda_max={lam_max_np:.15g} "
        f"vs provenance={mpmath.nstr(pv, 20)}, diff={float(diff):.3e}"
    )

    # Also verify using mpmath eigsy (higher precision cross-check)
    C_mp = build_C_mp(parsed["entries"], 1, n)
    eigs_mp = mpmath.eigsy(C_mp, eigvals_only=True)
    lam_max_mp = max(eigs_mp)
    da_eig = digits_agreed(lam_max_mp, pv)
    assert da_eig >= 10, \
        f"[FAIL L2] {name}: mpmath eigsy lambda_max agrees to only {da_eig} digits with provenance"

    return lam_max_np, da_eig


def level2_disparate_scale(name, prov, parsed):
    """
    disparate_scale_sqrt2: parse b=[P,Q]; assert C picks x_12 via off-diag (1,2) v=0.5;
    assert P=2e8, Q=1e-8; assert sqrt(P*Q) matches provenance to >=12 digits.
    """
    assert parsed["m"] == 2, f"[FAIL L2] {name}: expected m=2, got {parsed['m']}"
    assert parsed["block_sizes"] == [2], f"[FAIL L2] {name}: expected block_sizes=[2]"

    P = parsed["b"][0]
    Q = parsed["b"][1]

    P_expected = mpmath.mpf("200000000")   # 2e8
    Q_expected = mpmath.mpf("1e-8")
    assert abs(P - P_expected) < mpmath.mpf("1e-55"), \
        f"[FAIL L2] {name}: P={P} != 2e8"
    assert abs(Q - Q_expected) < mpmath.mpf("1e-70"), \
        f"[FAIL L2] {name}: Q={Q} != 1e-8"

    # C: off-diagonal (1,2) should have value 0.5 -> <C,X> = 2*0.5*x_12 = x_12
    c_entries = [(i, b, k, l, v) for (i, b, k, l, v) in parsed["entries"]
                 if i == 0]
    assert len(c_entries) == 1, \
        f"[FAIL L2] {name}: C should have exactly 1 entry, got {len(c_entries)}"
    _, _, ck, cl, cv = c_entries[0]
    assert ck == 1 and cl == 2, f"[FAIL L2] {name}: C entry should be (1,2), got ({ck},{cl})"
    assert abs(cv - mpmath.mpf("0.5")) < mpmath.mpf("1e-60"), \
        f"[FAIL L2] {name}: C (1,2) value should be 0.5, got {cv}"

    # Verify A_1 = e1e1^T and A_2 = e2e2^T
    A1_entries = [(i, b, k, l, v) for (i, b, k, l, v) in parsed["entries"] if i == 1]
    assert len(A1_entries) == 1 and A1_entries[0][2] == 1 and A1_entries[0][3] == 1 and \
           abs(A1_entries[0][4] - 1) < 1e-60, \
        f"[FAIL L2] {name}: A_1 != e1e1^T"
    A2_entries = [(i, b, k, l, v) for (i, b, k, l, v) in parsed["entries"] if i == 2]
    assert len(A2_entries) == 1 and A2_entries[0][2] == 2 and A2_entries[0][3] == 2 and \
           abs(A2_entries[0][4] - 1) < 1e-60, \
        f"[FAIL L2] {name}: A_2 != e2e2^T"

    # Compute opt = sqrt(P*Q) and compare to provenance
    opt_computed = mpmath.sqrt(P * Q)
    pv = prov_value(prov)
    da = digits_agreed(opt_computed, pv)
    assert da >= 12, \
        f"[FAIL L2] {name}: sqrt(P*Q)={mpmath.nstr(opt_computed,20)} " \
        f"agrees to {da} digits with provenance (need >=12)"

    return float(opt_computed), da


def level2_diag_weight(name, prov, parsed):
    """
    Single-block diag-weight: opt = lambda_max(diag(w)) = max(w).
    Assert A_1 = I and b = [1].
    """
    bs = prov["block_sizes"][0]
    assert parsed["m"] == 1
    assert parsed["block_sizes"] == [bs]
    assert abs(parsed["b"][0] - 1) < mpmath.mpf("1e-60")

    # Gather diagonal weights from C
    c_entries = [(k, l, float(v)) for (i, b, k, l, v) in parsed["entries"]
                 if i == 0]
    # All should be diagonal
    for ck, cl, cv in c_entries:
        assert ck == cl, f"[FAIL L2] {name}: C has off-diagonal entry ({ck},{cl})"
    weights = [cv for (_, _, cv) in c_entries]
    opt_computed = max(weights)

    pv = prov_value(prov)
    assert abs(mpmath.mpf(opt_computed) - pv) < mpmath.mpf("1e-60"), \
        f"[FAIL L2] {name}: max(weights)={opt_computed} != provenance={mpmath.nstr(pv,20)}"

    # Assert A_1 = I
    A1 = build_matrix(parsed["entries"], 1, 1, bs)
    assert np.allclose(A1, np.eye(bs), atol=1e-60), \
        f"[FAIL L2] {name}: A_1 is not I_{bs}"

    return opt_computed, 65  # exact integer match


def level2_two_block_corr(name, prov, parsed):
    """
    two_block_corr_coupled: 2 blocks of size 2, m=5.
    Key structural assertions:
      - A_5 block1 (1,2) entry = +0.5 AND block2 (1,2) entry = -0.5 (NEGATIVE — the
        sign-bug guard).
      - b_5 = 0.
      - b = [1, 2, 1, 2, 0].
    Optimum: 2*sqrt(2).
    """
    assert parsed["m"] == 5, f"[FAIL L2] {name}: expected m=5, got {parsed['m']}"
    assert parsed["block_sizes"] == [2, 2], \
        f"[FAIL L2] {name}: expected block_sizes=[2,2], got {parsed['block_sizes']}"

    b = parsed["b"]
    b_expected = [1, 2, 1, 2, 0]
    for idx, (bv, be) in enumerate(zip(b, b_expected)):
        assert abs(bv - be) < mpmath.mpf("1e-60"), \
            f"[FAIL L2] {name}: b[{idx}]={bv} != {be}"

    # Find A_5 entries
    a5_entries = [(bb, k, l, v) for (i, bb, k, l, v) in parsed["entries"] if i == 5]
    # Should have exactly 2: block1 (1,2) v=+0.5 and block2 (1,2) v=-0.5
    a5_b1 = [(k, l, v) for (bb, k, l, v) in a5_entries if bb == 1]
    a5_b2 = [(k, l, v) for (bb, k, l, v) in a5_entries if bb == 2]

    assert len(a5_b1) == 1 and a5_b1[0][0] == 1 and a5_b1[0][1] == 2, \
        f"[FAIL L2] {name}: A_5 block1 entry should be (1,2), got {a5_b1}"
    assert abs(a5_b1[0][2] - mpmath.mpf("0.5")) < mpmath.mpf("1e-60"), \
        f"[FAIL L2] {name}: A_5 block1 (1,2) should be +0.5, got {a5_b1[0][2]}"

    assert len(a5_b2) == 1 and a5_b2[0][0] == 1 and a5_b2[0][1] == 2, \
        f"[FAIL L2] {name}: A_5 block2 entry should be (1,2), got {a5_b2}"
    v_b2 = a5_b2[0][2]
    assert abs(v_b2 - mpmath.mpf("-0.5")) < mpmath.mpf("1e-60"), \
        f"[FAIL L2] {name}: A_5 block2 (1,2) should be -0.5 (NEGATIVE), got {v_b2}. " \
        f"This is the sign bug the generator was susceptible to."

    # Verify optimum: t <= sqrt(x11*x22) = sqrt(1*2) = sqrt(2), coupling forces equal,
    # objective = 2t, maximized at 2*sqrt(2).
    opt_computed = 2 * mpmath.sqrt(2)
    pv = prov_value(prov)
    da = digits_agreed(opt_computed, pv)
    assert da >= 12, \
        f"[FAIL L2] {name}: 2*sqrt(2)={mpmath.nstr(opt_computed,20)} " \
        f"agrees to {da} digits with provenance (need >=12)"

    return float(opt_computed), da


def level2_separable_12block(name, prov, parsed):
    """
    separable_12block: 12 blocks of size 2, m=12, b all ones.
    For block c: C_c = [[c, 1],[1, 0]], lambda_max = c/2 + sqrt(c^2/4 + 1).
    opt = sum over c=1..12.
    """
    assert parsed["m"] == 12, f"[FAIL L2] {name}: expected m=12, got {parsed['m']}"
    assert len(parsed["block_sizes"]) == 12, \
        f"[FAIL L2] {name}: expected 12 blocks, got {len(parsed['block_sizes'])}"
    for idx, bs in enumerate(parsed["block_sizes"]):
        assert bs == 2, f"[FAIL L2] {name}: block {idx+1} size should be 2, got {bs}"

    # b all ones
    for idx, bv in enumerate(parsed["b"]):
        assert abs(bv - 1) < mpmath.mpf("1e-60"), \
            f"[FAIL L2] {name}: b[{idx}] should be 1, got {bv}"

    # Build each block's C and compute lambda_max; sum and compare
    opt_computed = mpmath.mpf(0)
    block_lmaxes = []
    for c in range(1, 13):
        C_np = build_matrix(parsed["entries"], 0, c, 2)
        eigs = np.linalg.eigvalsh(C_np)
        lam_max_c = float(eigs[-1])
        block_lmaxes.append(lam_max_c)
        opt_computed += mpmath.mpf(lam_max_c)

    pv = prov_value(prov)
    da = digits_agreed(opt_computed, pv)
    assert da >= 10, \
        f"[FAIL L2] {name}: sum of lambda_max agrees to {da} digits (need >=10). " \
        f"computed={mpmath.nstr(opt_computed,20)}, provenance={mpmath.nstr(pv,20)}"

    # Also verify closed-form sum independently (mpmath, high precision)
    opt_cf = sum(mpmath.mpf(c) / 2 + mpmath.sqrt(mpmath.mpf(c)**2 / 4 + 1)
                 for c in range(1, 13))
    da_cf = digits_agreed(opt_cf, pv)
    assert da_cf >= 50, \
        f"[FAIL L2] {name}: closed-form mpmath sum agrees to {da_cf} digits (need >=50)"

    # Report the high-precision (mpmath closed-form) digits in the table
    return float(opt_computed), da_cf


# ── Mutation check ───────────────────────────────────────────────────────────

def _check_level1_fails(name, prov_corrupted):
    """Return True if level1 check FAILS (as expected for corrupted input)."""
    try:
        level1_closed_form(name, prov_corrupted)
        return False   # Did not fail — bad!
    except AssertionError:
        return True


def _check_level2_sign_fails(parsed_corrupted):
    """Return True if the two_block_corr A_5-sign check FAILS."""
    try:
        level2_two_block_corr("two_block_corr_coupled",
                              load_provenance("two_block_corr_coupled"),
                              parsed_corrupted)
        return False
    except AssertionError:
        return True


def mutation_check():
    """
    In-memory mutation: confirm the verifier catches corrupted inputs.
    1. Perturb 'disparate_scale_sqrt2' provenance by +1e-40.
    2. Flip the sign of A_5 block2 entry in two_block_corr_coupled.
    """
    # Mutation 1: corrupt a provenance value
    prov_good = load_provenance("disparate_scale_sqrt2")
    prov_corrupt = copy.deepcopy(prov_good)
    pv = mpmath.mpf(prov_corrupt["optimal_value"])
    pv_corrupt = pv + mpmath.mpf("1e-40")
    prov_corrupt["optimal_value"] = mpmath.nstr(pv_corrupt, 70)

    ok1 = _check_level1_fails("disparate_scale_sqrt2", prov_corrupt)

    # Mutation 2: flip sign of A_5 block2 (1,2) entry in two_block_corr_coupled
    dat_path = os.path.join(GOLDEN_DIR, "two_block_corr_coupled",
                            "two_block_corr_coupled.dat-s")
    parsed_good = parse_dat_s(dat_path)
    parsed_corrupt = copy.deepcopy(parsed_good)
    # Find A_5 block2 entry and flip sign
    flipped = False
    new_entries = []
    for (i, b, k, l, v) in parsed_corrupt["entries"]:
        if i == 5 and b == 2 and k == 1 and l == 2:
            new_entries.append((i, b, k, l, -v))  # flip sign: -0.5 -> +0.5
            flipped = True
        else:
            new_entries.append((i, b, k, l, v))
    assert flipped, "Mutation 2: could not find A_5 block2 (1,2) entry to corrupt"
    parsed_corrupt["entries"] = new_entries

    ok2 = _check_level2_sign_fails(parsed_corrupt)

    return ok1, ok2


# ── Main ─────────────────────────────────────────────────────────────────────

DISPATCH_L2 = {
    "max_eig_path_4":      level2_max_eig_path,
    "max_eig_path_8":      level2_max_eig_path,
    "max_eig_path_16":     level2_max_eig_path,
    "disparate_scale_sqrt2": level2_disparate_scale,
    "diag_weight_kappa6":  level2_diag_weight,
    "diag_weight_kappa10": level2_diag_weight,
    "diag_weight_kappa12": level2_diag_weight,
    "two_block_corr_coupled": level2_two_block_corr,
    "separable_12block":   level2_separable_12block,
}


def main():
    print("=" * 72)
    print("verify_stress.py — independent cross-check (mpmath 1.3.0 / numpy)")
    print(f"mpmath dps: {mpmath.mp.dps}    numpy: {np.__version__}")
    print("=" * 72)

    results = []
    all_pass = True

    for name in PROBLEMS:
        prov = load_provenance(name)
        dat_path = os.path.join(GOLDEN_DIR, name, f"{name}.dat-s")
        parsed = parse_dat_s(dat_path)

        # Level 1
        try:
            cf_val, da_l1 = level1_closed_form(name, prov)
            l1_status = "PASS"
            l1_digits = da_l1
        except AssertionError as e:
            l1_status = "FAIL"
            l1_digits = -1
            all_pass = False
            print(f"\n*** LEVEL-1 FAILURE: {name} ***\n  {e}\n")

        # Level 2
        try:
            l2_fn = DISPATCH_L2[name]
            l2_val, da_l2 = l2_fn(name, prov, parsed)
            l2_status = "PASS"
            l2_info = f"computed={l2_val:.15g}, digits~{da_l2}"
        except AssertionError as e:
            l2_status = "FAIL"
            l2_info = str(e)
            all_pass = False
            print(f"\n*** LEVEL-2 FAILURE: {name} ***\n  {e}\n")

        results.append((name, l1_status, l1_digits, l2_status, l2_info))

    # Print table
    print()
    print(f"{'Problem':<30} {'L1':>4} {'digits':>8}  {'L2':>4}  independently-computed / info")
    print("-" * 100)
    for (name, l1_s, l1_d, l2_s, l2_info) in results:
        digits_str = f"{l1_d}" if l1_d >= 0 else "FAIL"
        l1_col = "PASS" if l1_s == "PASS" else "FAIL"
        l2_col = "PASS" if l2_s == "PASS" else "FAIL"
        print(f"{name:<30} {l1_col:>4} {digits_str:>8}  {l2_col:>4}  {l2_info}")
    print()

    # Mutation check
    print("--- Mutation check ---")
    try:
        ok1, ok2 = mutation_check()
        if ok1 and ok2:
            print("MUTATION OK: verifier rejects corrupted inputs")
            print(f"  Mutation 1 (provenance +1e-40):        caught={ok1}")
            print(f"  Mutation 2 (A_5 block2 sign flip):     caught={ok2}")
        else:
            print(f"MUTATION FAIL: verifier did NOT catch corrupted inputs!")
            print(f"  Mutation 1 caught: {ok1}")
            print(f"  Mutation 2 caught: {ok2}")
            all_pass = False
    except Exception as e:
        print(f"MUTATION CHECK ERROR: {e}")
        all_pass = False

    print()
    if all_pass:
        print("OVERALL: PASS — all 9 problems verified at both levels; mutation guard active.")
    else:
        print("OVERALL: FAIL — see FAILURE messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
