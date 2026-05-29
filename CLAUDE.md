# CLAUDE.md — arbsdp

## What this is

A self-contained C library (`libarbsdp`) with a Julia cockpit (`ArbSDP.jl`) that
solves small semidefinite programs in arbitrary precision and returns a
**rigorous interval (ball) enclosure of the optimal value** plus verified status.
High precision *and* a machine-checkable certificate. Built on FLINT 3.x (Arb).
The product is the certificate — if we only wanted digits, we would wrap
SDPA-GMP and stop.

**The architecture in one line:** solve approximately in Arb *point* arithmetic,
then certify rigorously in *ball* arithmetic. Never run the interior-point
iteration itself in rigorous ball mode. See rule 1 — this is the single most
violable decision in the project.

## Canonical documents — read these BEFORE writing code

```
PRD.md            <-- start here. Scope, the layer architecture, the math spec,
                      the roadmap, the open decisions. Canonical.
README.md         <-- public-facing intro (once written)
```

External ground truth (cite these in source, see rule 3):
- **Jansson, Chaykin, Keil**, "Rigorous error bounds for the optimal value in
  semidefinite programming", *SIAM J. Optim.* 2007 — the Layer-1 bound formulas.
- **Todd, Toh, Tütüncü**, "On the Nesterov–Todd direction in SDP", 1998 — the NT
  scaling for Layer 0.
- **Andersen–Roos–Terlaky** 2003 — the homogeneous self-dual embedding.
- `../scientist-workbench/packages/solver-ipm/` — the **algorithmic source** for
  Layer 0 (an IP-clean, cleanroom TS implementation). `HsdeNtSdpSolver.ts` is the
  port target; `PsdCone.ts` is the cone kernel. Also the primary golden master.
- `../MOSEK-decomp/`, `../COPT-decomp/` — reverse-engineering notes confirming the
  HSDE/NT/Mehrotra structure; the binaries are golden-master oracles.

If a math routine in `src/` is not anchored to one of these (paper + equation, or
a TS source file + line), it is undocumented — fix that before changing it.

## Non-negotiable rules

These apply to every agent, every session, every commit.

1. **ARCHITECTURE IS LAW: solve in points, certify in balls.** Layer 0 (the
   HSDE-NT IPM) runs in Arb *point* arithmetic at adaptive precision — radii are
   not trusted inside the loop. Layer 1 (certification) runs in *ball* arithmetic
   and is the *only* source of rigor. **Do NOT "just run the IPM in ball
   arithmetic"** — it is a trap (the NT path needs symmetric eigendecomposition,
   Arb's weakest axis; ball radii blow up like `κ·2^-prec ~ 2^-prec/μ` exactly at
   convergence; and you would be enclosing the iterate, not the optimum). If you
   are tempted to track radii through the solve, re-read PRD §2 first.

2. **RIGOR IS FALSIFIABLE.** The output `[lb, ub]` is a theorem: the true optimum
   is provably inside it. A bracket that ever *excludes* a known/constructed
   optimum is a **P0 bug** — stop and fix it before anything else. Any value
   labelled "verified" must trace to ball arithmetic + verified Cholesky, never to
   point arithmetic. Never widen a claim to fit a result; widen the interval or
   report `inconclusive`.

3. **GROUND TRUTH = PRD + literature, not memory.** Every nontrivial math routine
   cites its source in a comment: the paper equation it implements, or the TS file
   it ports. Example:
   ```c
   /* Jansson-Chaykin-Keil 2007, Thm 3.2: opt >= b'y + sum_b min(0, lam_min(Z_b)) * xbar_b
    * Z_b = C_b - sum_i y_i A_i^b ; lam_min lower bound via verified-Cholesky shift. */
   ```
   When the PRD and a stray note disagree, the PRD wins. When the paper and your
   memory disagree, the paper wins.

4. **CROSS-CHECKS > UNIT TESTS — golden masters are the test strategy.** Two
   relations, both first-class (PRD §6): *beat* (more correct digits than a float
   solver) and *bracket* (our rigorous `[lb,ub]` contains the reference — this is
   self-certifying and is the stronger claim). Unit tests catch typos; golden
   masters catch algorithmic errors. Sources, in trust order: analytic optima >
   independent arb computation > the TS solver > Clarabel/Hypatia/SCS via JuMP >
   Mosek/COPT binaries. Every new path adds its cross-check immediately.

5. **FAIL FAST, FAIL LOUD.** `assert()` invariants; abort with context on
   insufficient precision, a failed/over-regularized factorization, an unbounded
   `tr X` with no supplied bound, a malformed problem. A corrupted bound is far
   worse than a crash — a wrong "certificate" is the worst possible output.

6. **ALL BUGS ARE DEEP.** Ill-conditioning, precision starvation, and arb memory
   bugs are subtle and interlocked. A fix that makes one test pass without
   addressing the root cause is a future incident with a longer fuse. The optimum
   sits on the PSD boundary; "works at n=3" silently breaks at n=12. Investigate
   root causes.

7. **ARB MEMORY DISCIPLINE.** This is the #1 source of C-against-Arb bugs. Every
   `arb_init`/`arb_mat_init` has a matching `clear`. Use the SCOPED-init macro
   convention. The full C test suite runs under ASan and valgrind, and zero leaks
   gates a merge. Never leave a `clear` to "later".

8. **SKEPTICISM.** Be skeptical of subagent output, previous agent work, the TS
   solver, the papers (LaTeX/transcription typos happen), and your own
   assumptions. Verify. Reproduce. When a test passes, ask whether it exercised
   the property you think it did (e.g. did it use a *boundary* optimum, or a cosy
   interior one?).

9. **RED-GREEN TDD.** Write the failing test first; watch it fail; minimum code to
   green; refactor. For a new certification routine, the cross-check (or the
   "bracket-contains-constructed-optimum" property test) is written before the code.

10. **GET FEEDBACK FAST.** Run `ctest` (C) / `Pkg.test()` (Julia) after every
    non-trivial change. Don't code 500 lines blind. For a single C test:
    `cmake --build build --target test_certify && ./build/test_certify`.

11. **NO EMOJIS, NO MARKETING.** The code reads like SQLite / TigerBeetle docs.
    Concrete numbers always: "certified 87 digits at prec=512, gap ub-lb=3.1e-88,
    14 iters" — not "highly accurate" / "robust" / "production-grade". State the
    measurement.

12. **DON'T REPLACE FLINT/Arb.** They were chosen deliberately. Do not propose
    alternatives (MPFR-only, custom bignum, GMP-rationals) unless you have shipped
    a measured improvement first. Confirm exact `arb_mat_*` symbol spellings
    against the *installed* FLINT version before coding `linalg.c` — names drift
    across FLINT releases.

13. **ONE BEAD, ONE COMMIT.** A bead is its own commit line. No drive-by cleanup —
    drive-by improvements get their own bead first. The diff reads in one sitting.

14. **PRD-DRIVEN.** Implement only what the current PRD scopes. v1 is
    **dense, small SDP (LP via diagonal blocks)** — no SOCP/exp/power cones, no
    sparse/chordal exploitation, no crossover, no large-scale. If a feature isn't
    in the PRD, file a bead; don't build it.

15. **LITERATE UPDATES.** Touch the math ⇒ update the PRD math spec. Change the
    layer boundary ⇒ update PRD §2/§5. Finish work that opens new opportunities ⇒
    file beads. The documents are the contract; if stale, future-you can't recover
    state from `git log`.

## Project-specific invariants you will trip over

These are non-obvious and grounded in PRD §2/§4 and the TS kernel. Get them wrong
and the solver produces confident garbage.

1. **Certify from the DUAL side.** The optimal `X` lives on the *boundary* of the
   PSD cone (typically rank-deficient, hence singular), so a verified Cholesky of
   `X` **will fail** — you can never prove `X ≻ 0`. The Jansson lower bound is
   derived from the dual residual `Z = C − Σ yᵢAᵢ` plus an a-priori `tr X` bound,
   and never requires the primal to be certified PD. Any design that needs
   "verify X is PSD" is wrong.

2. **`λ_min` lower bounds come from a verified-Cholesky shift, NOT eig.** To bound
   `λ_min(Z) ≥ s` rigorously, find the largest `s` for which `arb_mat_cho(Z − sI)`
   succeeds (a successful ball-Cholesky is a *proof* of PD). Rigorous symmetric
   eigendecomposition is Arb's weak spot; verified Cholesky is its strength. Eig is
   allowed only in Layer 0's *approximate* (point-mode) spectral kernels.

3. **Ball radii blow up at convergence.** The Schur/KKT condition number grows like
   `1/μ` by design. This is *why* Layer 0 is point-mode and *why* the precision
   controller must escalate `prec` (≈ `c·(−log₂ μ) + margin`) as μ shrinks. A fixed
   precision either wastes time early or stalls late.

4. **Regularization perturbs the SOLVED system; certify the ORIGINAL.** The 3-way
   Tikhonov `(δ_p, δ_d, δ_g)` makes Layer 0 tractable, but Layer 1 must certify the
   *unperturbed* problem, using the regularized solution only as a starting guess.
   Keep these two books strictly separate.

5. **The a-priori `tr Xᵇ ≤ x̄ᵇ` bound is REQUIRED for a finite lower bound.** It is a
   first-class API input. No bound ⇒ `lb = −∞`, reported honestly. Never silently
   assume boundedness. (For the target use cases it's free: density matrices have
   `tr = 1`, normalized moment matrices, etc.)

6. **svec uses the strict-Mosek √2 off-diagonal scaling** so `⟨A,X⟩ =
   svec(A)ᵀsvec(X)`. Document it once in `svec.h`; never deviate. It must match the
   TS solver exactly or golden-master comparison is meaningless.

7. **Internal form is `min −⟨C,X⟩`** (sign-flipped on output), matching
   `SdpProblem.ts`. Get the sign convention wrong and every objective is negated.

8. **HSDE `(τ,κ)` carry the infeasibility certificates.** `primal_infeasible` /
   `dual_infeasible` status must be a *verified* Farkas certificate checked in ball
   arithmetic (`τ→0, κ>0` etc.), not a float64 heuristic.

## Architecture

| Layer | What | Numeric regime | Rigor |
|------|------|----------------|-------|
| 0 | HSDE-NT Mehrotra IPM (port of `HsdeNtSdpSolver.ts`) | Arb **point** (midpoints), adaptive `prec` | none |
| 1 | Certification → `[lb,ub]` + status (Jansson, verified Cholesky) | Arb **ball** | **rigorous** |
| 2 | Refinement (interval-Newton/Krawczyk, or precision-bump-resolve) | ball | rigorous |

```
libarbsdp  ──depends on──>  FLINT 3.x (Arb), GMP, MPFR         [no Julia, no sci-wb]
ArbSDP.jl  ──depends on──>  libarbsdp (JLL) + JuMP/MOI + Arblib.jl (refs only)
```

**C/Julia boundary:** C owns every arb-in-a-loop kernel (solve *and* certify) and
is standalone (`cmake && ctest`, no Julia). Julia owns modeling, the golden-master
harness, diagnostics/viz, and the optional MOI/JuMP backend. The ABI is tiny, flat
(strings or limb arrays — never live Arb structs across the boundary in v1), and
versioned. The TS `solver-ipm` appears nowhere in the dependency graph — it is
documentation and golden master only.

## Planned file structure (target layout; P0 creates it)

The repo is currently just `PRD.md` + `CLAUDE.md`. This is the layout the PRD
roadmap builds toward — keep it accurate as files land.

```
arb-prec-IPM/
  PRD.md                            # canonical scope + math spec (read first)
  CLAUDE.md                         # this file — non-negotiable rules
  README.md                         # public intro
  c/                                # libarbsdp — standalone C library
    CMakeLists.txt
    include/arbsdp/api.h            # the public ABI (opaque ctx, status codes)
    src/
      svec.c                        # vectorization, √2 convention, block bookkeeping
      linalg.c                      # arb_mat helpers; NT scaling (approx eig);
                                    #   verified Cholesky + λ_min-shift (Layer 1 kernel)
      solve.c                       # HSDE-NT Mehrotra loop (Layer 0)
      regularize.c                  # 3-way Tikhonov
      precision.c                   # adaptive precision controller
      certify.c                     # Jansson bounds, verified PSD test, Farkas (Layer 1)
      refine.c                      # interval-Newton/Krawczyk (Layer 2; later)
      io.c                          # SDPA .dat-s reader, JSON result serializer
      cli.c                         # `arbsdp solve problem.dat-s --prec 256`
    test/                           # ctest suite (ASan/valgrind-gated)
  ArbSDP.jl/                        # Julia cockpit
    Project.toml
    src/ArbSDP.jl                   # ccall bindings, public API, (later) MOI backend
    test/runtests.jl                # golden-master cross-checks (beat-or-bracket)
    bench/                          # diagnostics, central-path / ball-radius / prec plots
  golden/                           # reference values + provenance (analytic, SDPLIB, ...)
  docs/
    literature/                     # Jansson, TTT, ART papers (cited from source)
```

## Build & test

**Prerequisite (not yet installed on this machine):** FLINT 3.x with dev headers
(bundles Arb). Only GMP 10 + MPFR 6 runtime libs are present today. Install/build
FLINT 3.x before coding `linalg.c`, and confirm `arb_mat_*` symbol spellings
against that version.

```bash
# C library — configure, build, test
cmake -S c -B c/build -DCMAKE_BUILD_TYPE=Debug
cmake --build c/build
ctest --test-dir c/build --output-on-failure

# Single C test
cmake --build c/build --target test_certify && ./c/build/test_certify

# Memory-clean gate (required before merge)
ctest --test-dir c/build -T memcheck        # valgrind
# or build with -DENABLE_ASAN=ON and run the suite

# CLI smoke test
./c/build/arbsdp solve golden/truss1.dat-s --prec 256

# Julia
julia --project=ArbSDP.jl -e 'using Pkg; Pkg.test()'
julia --project=ArbSDP.jl -e 'using ArbSDP; solve(read_sdpa("golden/truss1.dat-s"); prec=256)'
```

Required C flags: `-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -O2 -g
-std=c11`. Keep `-Wshadow` — it catches real bugs in arb code where temporary
`arb_t`s shadow each other. Link `-lflint -lmpfr -lgmp`.

## "Done" checklist for a math-routine change

- [ ] `ctest` green (and `Pkg.test()` if the Julia side changed).
- [ ] The bracket-contains-constructed-optimum property test still passes (rigor).
- [ ] The relevant golden master still *beats* or *brackets* its reference.
- [ ] ASan/valgrind clean — zero leaks, zero errors.
- [ ] Source comment cites the paper equation or the TS source line.
- [ ] PRD updated if the math/scope/boundary changed.
- [ ] Concrete numbers recorded (prec, iters, gap, digits certified) — not adjectives.
- [ ] Bead closed referencing the commit.

## Issue tracking — beads (`bd`) + in-session tasks

This project uses **bd (beads)** as the persistent issue tracker.

```bash
bd ready                 # find available work
bd show <id>             # view an issue
bd update <id> --claim   # claim work
bd close <id>            # complete work
bd remember <...>        # persistent project knowledge
bd prime                 # full command reference + session-close protocol
```

**TaskCreate / TaskList ARE allowed in this session for ephemeral, in-session
task tracking.** The auto-generated `bd init` boilerplate says "do NOT use
TaskCreate" — that line is **overridden here** (per the owner). Division of
labour: `bd` = durable issues that outlive the session; `TaskCreate` = the
working checklist for the current session. Do not use markdown TODO lists.

## Session completion

When ending a work session:

1. **File beads** for remaining/follow-up work.
2. **Run quality gates** if code changed — `ctest` + ASan/valgrind, `Pkg.test()`.
3. **Update bead status** — close finished, update in-progress.
4. **Commit** with the bead reference; bundle any `.beads/` dolt-cache changes
   into the *same* commit as the source change that closed the bead (no separate
   "sync dolt cache" commits).
5. **Push** if a git remote is configured (`git pull --rebase && bd dolt push &&
   git push`); verify `git status` is clean. (This project is not yet a git repo —
   `git init` is part of P0.)
6. **Hand off** — leave context for the next session in the bead or worklog.
```


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
