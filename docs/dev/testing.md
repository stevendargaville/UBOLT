# Testing in UBOLT

## Pass/fail contract
- Pass/fail is the process exit code — no output diffing, no log grepping (this must stay
  true so CI can just run `make tests`, which it does — see
  `.github/workflows/ci_build.yml`). `make` stops on the first non-zero exit.
- Every solve recipe in `tests/Makefile` pins `-ksp_max_it` to the baseline iteration
  count, so an iteration-count regression fails the run. A pin is the count measured on
  the local opt arch (`arch-linux-c-opt`), exactly, with no per-arch slack: every pin was
  re-set that way 2026-09-25 for the ghost-flux vacuum default (see "Switching the
  default" under the ghost-flux section). The same recipes run on every CI arch
  (opt/debug/64-bit/OpenMP), so CI flags any arch that needs one more, and at that point
  the pin takes the max over the arches. The per-arch notes further down ("pinned 10",
  "64-bit CI 18") are the history of the pre-2026-09-25 max-over-arches policy.
- Multigroup caveat: `-ksp_max_it` is one option for the whole group sweep, so a
  multigroup recipe pins the **max over the groups**. A single group getting slower
  without exceeding that max will not fail the recipe — the multigroup baselines below
  are what pin the per-group counts.
- The drivers return 1 unless `KSPGetConvergedReason() > 0` (since Phase 1a), so a
  non-converged solve fails the recipe on its own — no `-ksp_error_if_not_converged`
  or `-on_error_abort` is needed for pass/fail (CI sets `-on_error_abort` in
  `PETSC_OPTIONS` anyway, belt and braces against a hang inside PETSc). Driver
  diagnostics go to stderr so they never appear in a captured baseline. The four
  non-converging `capture_baselines` lines carry `|| true`.
- Targets: `make check` (fast sanity, 6 runs) < `make tests_short` < `make tests`.
  Parallel variants use `$(MPIEXEC) -n 2` — plus one `-n 4` run, see the 2D section — and
  are skipped under MPIUNI, including `run_check`'s single parallel line.
- The `verify_*` drivers are the exception to "pass/fail is a pinned iteration count":
  they are discretisation and quadrature checks and their exit codes come from tolerances
  on numerical checks (below). They print what they measured against the tolerance, so a
  run that is drifting towards failure is visible before it fails.

## Baselines (tests/baselines/)
Captured `-ksp_monitor -ksp_converged_reason` logs. They are the ground truth for a
refactor: identical numerics means the residual history diffs clean against these files
(iteration counts exactly, residuals to ~1e-14).

24 logs in two families: 16 single-group (`slab1d_*`, Phase 0) and 8 multigroup
(`slab1dmg_*`, Phase 2). `make baselines` captures both. A refactor that claims to be
numerically inert must diff clean against all 24. All of them are 1D — the 2D checks are
sharper and need no baseline, see below.

Since the driver unification (Aug 2026) every capture line runs `transportk` on a
problem file (`tests/problems/`); the log names keep their `slab1d`/`slab1dmg` history.
The unification itself was verified the strong way before the old drivers
(`slab_1dk`, `slab_1d_mgk`, `box_2dk`) were deleted: all 24 logs byte-for-byte from the
new driver, plus the t0 structural identity, the painting identity at np=1 and np=4, and
the 1D/2D infinite-medium checks.

### Single-group baselines
Re-captured 2026-09-25 for the switch to the ghost-flux vacuum treatment as the default
(see below). Before that, 2026-08-04 for the quadrature precision fix, then
2026-08-01 after the Dirichlet-row fix to the matrix-free scatter, 2026-07-31 after the
negative-angle upwind sign fix, and before *that* they
came from the pre-refactor code (the single-file `UBOLTk.kokkos.cxx`); Phase 1a and Phase
1b each reproduced all 16 bitwise, which is what those phases were verified against. The
`-ubolt_coo_two_call` assembly fallback reproduces them bitwise.

- Full matrix: {default pc, `-precon_stream -ksp_pc_side right`} x {`slab_st0.json`,
  `slab_st2.json`} x {np 1, 2} x {`-diag_scale` off, on}. File naming:
  `slab1d_<default|stream>_st<0|2>_np<1|2>[_ds].log`.
- Capture is capped at `-ksp_max_it 200`: all four st=2 diag_scale configs are pathological
  (see table) and a 200-iteration residual history is a strong enough fingerprint.
  Non-converged baselines are still valid fingerprints for a refactor diff. Under
  Dirichlet-cell none of those four converged, so all four capture lines carry
  `|| true`; under ghost-flux the default-pc pair scrapes in at 174/173 and the
  streaming-pmat pair hits the cap, and the `|| true` stays for the latter.
- Re-capture with `make baselines` — only do this deliberately (i.e. when the reference
  behavior itself is being intentionally changed), never to make a failing test pass.
- Environment matters for exact reproduction: these were captured with a debug PETSc main
  build (`arch-linux-c-debug`), gcc 13, OpenMPI, np as named — the same environment every
  previous capture used, so a capture-to-capture diff is like for like. (The 2026-09-25
  capture used a debug build of PFLARE main 9ec892b; the unmodified library on it
  reproduced 20 of the 24 previous logs byte for byte and the other 4 in the 12th-13th
  digit, so the environment is like for like.) Different
  compilers/optimization may shift trailing digits; iteration counts should still match
  (measured, on the pre-change code: debug against opt moves no count in any of the 24
  logs, leaves the initial residuals identical, and drifts early iterations by 1.3e-13).
- np=1 and np=2 logs differ in the trailing digits (parallel reduction order); compare a
  log against the baseline for the *same* np, never across np.

## Single-group baseline iteration counts
Captured 2026-09-25 under the ghost-flux default, capped at `-ksp_max_it 200`. The older
columns are kept because earlier phases were verified against them — "Dirichlet-cell" is
the 2026-08-04 capture, the last one before the switch (and what the opt-in
`"vacuum_treatment": "dirichlet_cell"` still reproduces byte for byte), "Dirichlet fix" is the 2026-08-01 capture,
before the quadrature constants were fixed; "sign fix" is 2026-07-31, before the scatter's
Dirichlet rows were fixed; and "pre-refactor" is what the original single file did:

| config | np=1 | np=2 | Dirichlet-cell (np=1, np=2) | Dirichlet fix (np=1, np=2) | sign fix (np=1, np=2) | pre-refactor |
|---|---|---|---|---|---|---|
| default pc, st=0 | 1 | 1 | 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| default pc, st=2 | 6 | 6 | 6, 6 | 6, 6 | 5, 5 | 10, 10 |
| default pc, st=0, diag_scale | 1 | 1 | 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| default pc, st=2, diag_scale | 174 | 173 | DIVERGED_ITS (200) | DIVERGED_ITS (200) | 175, 173 | DIVERGED_ITS |
| precon_stream, st=0 | 1 | 1 | 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| precon_stream, st=2 | 8 | 8 | 9, 9 | 9, 9 | 10, 10 | 16, 16 |
| precon_stream, st=0, diag_scale | 2 | 2 | 3, 3 | 3, 3 | 3, 3 | 3, 3 |
| precon_stream, st=2, diag_scale | DIVERGED_ITS (200) | DIVERGED_ITS (200) | DIVERGED_BREAKDOWN (60, 180) | DIVERGED_BREAKDOWN (150, 150) | DIVERGED_BREAKDOWN (90) | DIVERGED_ITS |

The st=0 rows were identical across every capture up to the ghost-flux switch, and had to
be: `sigma_s = 0` there, so neither the scatter nor its Dirichlet rows exist. The switch
moves the boundary rows themselves, so it is the first capture that moves them (only the
diag-scaled streaming pmat's count, 3 -> 2).

History of these baselines:
- **Ghost-flux became the default vacuum treatment** (2026-09-25, this re-capture): every
  vacuum inflow row is a physical row now, so every log moves from the first residual.
  Iteration counts: the default pc is unchanged; the streaming-only pmat improves by one
  (9 -> 8, and 3 -> 2 diag-scaled); the default-pc diag-scaled pair, pathological since
  the Dirichlet-row fix, converges again at 174/173, while the diag-scaled streaming pair
  goes from breakdown to the 200 cap - neither is a stable quantity and neither is pinned.
  The right-preconditioned (streaming-pmat) logs' INITIAL residual grows from 63 to 1313:
  with right preconditioning it is `||b||`, and the ghost rows' rhs is `|mu|/dx` times the
  inflow where a Dirichlet row's was the inflow itself, which at dx = 1e-3 makes the
  boundary rows dominate `||b||` and hence the rtol reference. The opt-in
  `"vacuum_treatment": "dirichlet_cell"` twin (`slab_st2_dirichlet_cell.json`) reproduces
  the previous capture's `default_st2` (np 1 and 2), `default_st2_ds` and `stream_st2`
  logs byte for byte on the same arch.
- **The quadrature constants were truncated to float precision** (FIXED 2026-08-04, this
  re-capture): the direction cosines were literals of 7 to 10 significant digits in a
  double code (`0.5773502692` for `1/sqrt(3)`, a relative error of 1.8e-10), and the
  quadratures now generate them — 1D by Newton on the Legendre polynomial at `create`
  time, 2D and 3D from a table generated at 60 decimal digits by `src/sn_lqn_table.py`.
  That perturbs every result in the trailing digits and nothing else, which is what the
  re-capture was checked against, on the same arch so the diff is like for like. Over the
  20 non-pathological logs: **no iteration count moved**, and the **initial** residual —
  the direct fingerprint of the constants, with no solver amplification in it — drifts by
  at most 2.2e-10, the size of the perturbation itself. Later iterations amplify that, as
  they must: through the non-converged iterations the drift stays under 1e-10 everywhere
  except the multigroup streaming-pmat pair, whose 11-iteration histories reach 2.2e-4 by
  the end. A run's FINAL residual is not a useful comparison at all — it sits at the rtol
  floor, so the change in it is 1e-13 to 1e-17 of that run's initial residual (3e-7 for
  the multigroup streaming pair) while its *relative* change can be anything.
  What DID move is the two `precon_stream, st=2, diag_scale` logs, from
  DIVERGED_BREAKDOWN at 150 to 60 (np=1) and 180 (np=2). Those are the pathological
  configs below: they break down rather than converge, so where they break down is not a
  stable quantity, and no recipe pins them. Across the whole recipe suite exactly one
  count moved — 2D 80x40 with a streaming-only pmat, 10 to 9 — see the 2D table.
- **The matrix-free scatter wrote to the Dirichlet rows** (FIXED after Phase 4, this
  re-capture): `ScatteringTerm::apply_add` subtracted the scattering source from every row
  including the boundary ones, so the shell's Dirichlet rows were not the identity the
  assembly had written and the prescribed inflow value was polluted. Present since Phase 1;
  the contract in `operator_term.hpp` only ever covered `assemble_add`. Found by
  `tests/verify_2dk`'s reference matrix. In 1D the effect is a wash — the well-behaved
  configs move by one iteration either way and the pathological diag_scale pair gets worse
  — but in 2D it was the whole of an apparent preconditioner problem (see `TODO.md`).
- **Negative-angle upwind sign** (FIXED after Phase 1b, this re-capture): the streaming
  term wrote the upwind neighbour coefficient as `-mu/dx`, correct for `mu > 0` but the
  wrong sign for `mu < 0`, so those rows were `|mu|/dx (psi_i + psi_i+1)` instead of
  `|mu|/dx (psi_i - psi_i+1)`. Fixing it makes the interior streaming rows sum to zero,
  and every well-behaved config converges in fewer iterations (st=2 went 10 -> 5, the
  streaming-pmat st=2 16 -> 10). Phase 1 deliberately preserved the bug so the refactor
  could be verified bit-for-bit, then fixed it in one commit of its own.
- **Parallel out-of-bounds bug in the matrix-free scatter** (FIXED, Phase 1a-pre): the
  original code reshaped the LOCAL vector with the GLOBAL `N_CELLS` and looped kernels
  over `N_CELLS`, so at np>1 it read past the local `x`/`sigma_s` arrays and wrote past
  the end of local `y`, sending `default pc, st=2, np=2` to NaN.
- **diag_scale + strong removal is still pathological** (175 its, or breakdown): diagonal
  scaling makes the removal shell PC an identity and degrades the composite. The sign fix
  improved it a lot but did not cure it. Known current behavior, not a target — excluded
  from pass/fail recipes; revisit when the diag_scale semantics are looked at again.

## Multigroup baselines (Phase 2)
`slab1dmg_<default|stream>_st<0|2>_g4_t<0|05>_np<1|2>.log` — 4 groups, `t0`/`t05` being
downscatter 0.0/0.5 into the next group down (the `slab_mg4_t0.json` /
`slab_mg4_t05.json` / `slab_mg4_t05_st0.json` problem files; before the driver
unification these were `slab_1d_mgk -sigma_transfer` recipes, whose derived per-group
values are exactly what the files now tabulate). Each log is the whole group sweep: one
KSP history after another, in group order. That is what pins the per-group iteration
counts and the downscatter source arithmetic, neither of which a single `-ksp_max_it`
can reach. All eight were re-captured 2026-09-25 for the ghost-flux default (the t0 logs
are still four byte-for-byte copies of the single-group log, re-checked structurally); the
six t05/stream logs were re-captured before that, 2026-08-02, for the isotropic-source
change (below), when the two t0 logs deliberately did not move.

The `t0` logs are exactly four byte-for-byte copies of the matching single-group
`slab1d_default_st2_np<n>.log`. That is the Phase 2 uncoupled-groups verification, and it
is worth re-checking as a structural property rather than only diffing the file — a change
that broke both files the same way would still diff clean. It was re-checked that way on
the 2026-08-01 re-capture, when both files moved, and again on the driver unification.
`slab_mg4_t0.json` carries `Source: [2, 2, 2, 2]`: the isotropic strength that puts 1.0
on each 1D ordinate, which is exactly the rhs the single-group problems carry — that is
what keeps the identity byte-for-byte.

| multigroup config | per-group iterations (np=1 and np=2) |
|---|---|
| default pc, st=0, transfer 0.5 | 1, 1, 1, 1 |
| default pc, st=2, transfer 0.0 | 6, 6, 6, 6 (= single group, four times; Source 2.0) |
| default pc, st=2, transfer 0.5 | 6, 6, 6, 6 |
| default pc, st=2, transfer 2.0 | 6, 6, 5, 5 (recipe only, not captured) |
| precon_stream, st=2, transfer 0.5 | 9, 9, 9, 8 (Dirichlet-cell: 11, 11, 11, 9) |

Group 0 matches the single-group count because its rhs is just the external source (a
count match at Source 1.0, a history match at Source 2.0, see above); the later groups
carry a downscatter source as well. The last group has no outgoing transfer, so its
`sigma_t` is lower than the others' — that is why the counts are not uniform.

With `-precon_stream` the streaming-only pmat does not depend on the group, so PCAIR is
set up once for the whole sweep; with the default pc the assembled matrix is refilled per
group and PCAIR re-runs its setup each time. Both are exercised by the recipes.

## Quadrature verification
`tests/verify_quadraturek` checks the angular quadratures in isolation, and it exists
because since 2026-08-04 they are **generated rather than tabulated**: 1D is a
Gauss-Legendre rule built by Newton in `SNQuadrature::create`, and 2D/3D read the
level-symmetric table `src/sn_lqn_table.py` emits into `src/sn_lqn_table.hpp`. Neither is
anything a reader can eyeball, and nothing downstream can tell a subtly wrong set from a
right one — a wrong quadrature makes a plausible, converging, wrong answer.

Tolerances, not iteration counts, decide its exit code, and it prints every measurement
against its tolerance. It is in both `make check` and `make tests_short`, serial and at
`-n 2`; the quadrature is replicated on every rank, so the parallel run says only that the
driver is clean under MPI. It costs well under a second.

What it pins:
- **1D**, at S2, S4, S6, S8, S12, S16 and S32: exactness on every polynomial of degree
  up to 2N−1, which is the whole of what an N point Gauss rule promises (measures ~1e-15
  against 1e-13); the weights summing to 2; the nodes strictly ascending inside (−1, 1);
  and the ± pairing, weight symmetry and reflection map, all as **exact** equality — the
  reflection search compares cosines with `==`, so the mirrored node has to be the bitwise
  negation of its partner, which is why the generator mirrors by negating a shared value
  rather than recomputing. S2 and S4 are additionally checked against their published
  nodes and weights to 1e-15, which is the only outside reference in the file.
- **2D and 3D**, at every order the table carries (even 2 to 18): the even axis moments
  through order N — the conditions that *define* a level-symmetric set, so this is the
  real test of the generated table (measures ~1e-15 against 1e-12); the ordinate counts
  N(N+2)/2 and N(N+2); weights summing to 4π; strictly positive weights; unit direction
  vectors in 3D and mu²+eta² < 1 strictly in 2D; and all three reflection maps being
  weight-preserving involutions that flip exactly their own cosine. 3D also checks
  invariance under all six permutations of the axes, by search — that is what "level
  symmetric" means, and it is what a class weight attached to the wrong permutation would
  break.
- **The fold**: the 2D set is exactly the ξ > 0 half of the 3D set with doubled weights,
  matched by (mu, eta) with `==`. The two classes build their ordinates from the same
  table but by separate loops, so nothing else would catch them diverging.
- **The two closed-form cosines**: S2's 1/√3 and S4's √((5−√10)/15), to 1e-15. Everything
  above S4 rests on the generator's own assertions, so these are what say the generator is
  solving the right problem.
- **The error paths**: an odd order in any dimension, and S20 in 2D and 3D, all have to
  fail. Run under `PetscReturnErrorHandler` so an expected error neither aborts the run
  (CI puts `-on_error_abort` in `PETSC_OPTIONS`) nor prints a stack trace.

The driver hard-codes 2 and 18 as the range rather than reading it out of the library, so
a silently shortened table fails it — and the S20 rejection is what says the table has not
silently grown either.

The generator carries its own assertions and is checked separately: `python3
src/sn_lqn_table.py` re-derives everything and asserts the moment residuals, positivity,
the cosine ordering, the point counts, and agreement with the published mu_1 values; the
output is deterministic, so `python3 src/sn_lqn_table.py --check` fails if the checked-in
header is not what the script generates. That is not a `make` target — the table changes
about as often as the definition of a level-symmetric set does.

**Why the table stops at S18.** From S14 up, the axis moments do not determine the weights
uniquely — a family of dimension 1 at S14 and S16, 2 at S18, 3 at S20 — and the generator
picks the member minimising the error in the MIXED even moments, one least-squares problem
and the whole rule. At S20 that member has a weight of −1.8e-5, so the generator stops
there; the cap is a computed result, printed in the generated header with the offending
number, not a constant anyone typed. That is the level-symmetric family running out, which
is the well-known reason SN codes cap LQn around here; a higher order needs a different
family (product, Gauss-Chebyshev), not a different choice within this one.

## 2D verification (Phase 4)
There are **no 2D baselines**, and that is deliberate. A residual-history baseline is an
indirect fingerprint of the numerics; in 2D `tests/verify_2dk` compares the operator
itself, so it is a strictly sharper oracle and there is nothing left for a baseline to
catch. `make baselines` is still 1D + multigroup only.

`verify_2dk` runs these checks, both S2 (4 angles) and S4 (12), and exits non-zero on any.
Checks 1 and 2 run under both vacuum treatments; the ghost-flux runs and checks 3-4
are described under "Ghost-flux vacuum treatment" below:

1. **Pure-streaming closed form.** `psi = x + y` is linear, and first-order upwind
   differencing is exact on a linear function, so the *discrete* operator has the PDE's
   solution: `mu dpsi/dx + eta dpsi/dy = mu + eta`, with `psi = x + y` prescribed on the
   inflow boundaries. Two halves — the residual `||A psi_exact - b||_inf` (no solver
   tolerance in the way; ~4e-15 against a 1e-12 tolerance) and then the solve, which says
   the system really does have that solution and is nonsingular. Run on a 16x12 grid over
   a 1x2 box, so nothing about it is symmetric in x and y and a mix-up cannot hide.
2. **Shell operator vs a reference matrix.** `MatComputeOperator` on the *shell*, so the
   matrix-free scatter is included, against a matrix built entry by entry in the driver on
   a 4x3 grid — for three BC configurations apiece: all-vacuum, all-reflect, and mixed
   (left + bottom reflect), so both reflection maps and the vacuum-wins corner rule are in
   the checked matrix. Currently agrees to 0.0, i.e. bitwise, in all six runs. Serial
   only: the reference is written in the natural ordering, which is what a 2D DMDA uses on
   one rank. The reference deliberately reimplements the boundary-row classification —
   including its own search for the mirrored ordinate — rather than reading the
   discretisation's BC mask, reflect slots or reflection maps: a check that asked the code
   under test which rows it thought were boundary rows would agree with it by
   construction. It says a Dirichlet row is the identity and nothing else, and a
   reflective row is the identity plus the -1 on its mirrored angle and nothing else; the
   former is how it caught the matrix-free scatter writing to those rows, see the Phase 4
   postscript in `TODO.md`.

**Parallel.** A 2D DMDA at `-n 2` splits in one direction only, so `-n 4` is the first
decomposition with a genuinely 2D processor grid and the one where the patch-lexicographic
global numbering is least like the natural one. Both are in `run_tests_short_parallel`.

## 2D iteration counts
The `st` in the table and the problem-file names is `Sigma_t` — the retired drivers took
it as `-sigma_t`, and until Aug 2026 that option was `-max_exponent`, a fossil of the
original driver's random per-cell xsection (`mantissa * 10^exponent` with the exponent
drawn up to it); that history is why anything older (commit messages, the history notes
above at their capture dates) says `me`. A file whose within-group `Sigma_s` equals its
`Sigma_t` is a scattering ratio of exactly 1 (no absorption). Counts below are with the
Dirichlet-row fix; before it, ratio 1 took 18 to 87 iterations on square grids and did not
converge at all on any other shape, which is what led to the fix — see the research note
in `TODO.md`. They were re-measured 2026-08-02 under the isotropic external source (see
the painted-regions section): only the two streaming-only pmat serial references moved,
8 -> 9 and 9 -> 10. Re-measured again 2026-09-25 under the ghost-flux default; where a
count moved, the Dirichlet-cell count follows it in brackets.

| config | np=1 | np=2 |
|---|---|---|
| 50x50, st=2 (ratio 1) | 7 (Dirichlet-cell 6) | 7 (Dirichlet-cell 6) |
| 50x50, st=2, ratio 0.5 | 5 | 5 |
| 40x20 (dx != dy), st=2 | 6 | 6 |
| 80x40 (same shape, 4x finer), st=2 | 7 (Dirichlet-cell 6) | 7 (Dirichlet-cell 6) |
| 80x20 in a 1x0.25 box (dx == dy), st=2 | 5 | 5 |
| 50x50, st=0 (pure streaming) | 3 | 3 |
| 60x60, st=0 | — | 3 |
| 60x60, st=2 | 7 | 7 |
| 50x50, streaming-only pmat, st=2 | 7 (Dirichlet-cell 9) | 7 (Dirichlet-cell 9) |
| 80x40, streaming-only pmat, st=2 | 7 (Dirichlet-cell 9) | — |
| 30x30 S4, st=2 | 6 | 6 |
| 60x60 S4, st=2 | 6 | 6 |
| 20x20 S8, left+bottom reflect, ratio 0.5 | 6 | 6 |

**Under Dirichlet-cell the three streaming-only pmat recipes were the arch-fragile ones,
and all three carried a pin of 10 against a reference count of 9.** They converged by
clearing rtol on the last iteration with only a percent or two to spare, so a
perturbation far too small to call a regression decided whether that last iteration
counted. The 2026-08-04 quadrature precision fix was exactly such a perturbation, and it
demonstrated the point twice: 80x40 serial measured 10 before it and 9 after on both
local arches, while 50x50 serial measured 9 locally both before and after but moved
9 -> 10 on all three CI images. Under ghost-flux the three take 7 and are pinned at 7,
the local count, per the contract — if CI asks for 8 on one arch, this is the history.

Mesh independent on every shape: 40x20 to 80x40 holds at 6 under Dirichlet-cell (6 to 7
under ghost-flux), and 200x50 in the 1x0.25 box (not a recipe) is 5, the same as 80x20
(a Dirichlet-cell measurement, not repeated).

These sizes were cut on 2026-08-02 to bring CI down (see "Recipe cost" below): the
"larger grid" family went 100x100 -> 60x60, the non-square mesh-independence pair
60x30/120x60 -> 40x20/80x40 with its 2x ratio intact, and the S4 reflective run joined
the plain S4 recipe at 30x30. The counts above are the re-measured references.

Until 2026-09-25 a pin was the max over the reference build and the two sensitive CI
arches (64-bit indices and the OpenMP Kokkos backend), re-swept 2026-08-02 in both CI images after the
isotropic-source change. Which configs sit one above the reference count changed with
it: the two streaming-only pmat serial recipes now measure the same everywhere (so
their pins equal the table), and instead the 50x50 ratio-1 SERIAL solve takes 7 under
64-bit — pinned 7 on all three recipes that run it (plain, `-ubolt_coo_two_call`, and
the painting identity, which is the same history by construction) — while under OpenMP
the np2 streaming-only pmat takes 10 (pinned 10) and the serial large-grid S4 takes 7
(pinned 7 — that slack was kept across the 2026-08-02 resize). Every other recipe
measures its reference count in both images; the resized recipes' pins are debug-arch
measurements, and CI runs green with them. The cost is
one iteration of slack on the reference build for those five recipes — the 1D
streaming-pmat baselines still pin their counts exactly. (That was the policy until
2026-09-25; since the ghost-flux re-pin every pin is the local opt count, so the 60x60 S4
serial pin is 6 and the 50x50 ratio-1 serial pins are 7 because the count itself is 7.)

**The 2026-08-04 quadrature precision fix was swept locally** (Dirichlet-cell era; opt,
then re-checked on debug) **and then against CI**, which is where the streaming-only pmat pins above got
their slack — the local sweep alone was not enough, and the note above records why. It has
NOT been swept in the sense of tightening every other pin back onto a per-arch maximum;
those are unchanged from before the fix, which is safe because the change only ever moves
a count by one and every other recipe passed on all four arches.

## Unequal quadrature weights
Up to and including S4 a level-symmetric set shares 4π equally between its ordinates; from
S6 up it does not — the weights are constant on the permutation classes of a direction's
level indices, two distinct values at S6, three at S8, ten at S18. Nothing in the
solver ever assumed otherwise (the discretisation backends read only the sign of a cosine,
`w_d()` is always consumed through `UboltAngularIntegral`'s gemm, and DSA's
`/sum_weights` prolongation is the isotropic projection rather than an equal-weight
assumption), but until 2026-08-04 no order above S4 existed, so nothing exercised it.

`box_s8_reflect_coarse.json` is what does: 2D, S8 (40 ordinates, three distinct weights),
coarse 20x20 because none of this is about resolution, with reflective left and bottom
faces so the reflection maps and the reflect partner columns see the unequal weights too.
It runs serial and at `-n 2`, 6 iterations both ways.

Note that `verify_2dk` and `verify_3dk` build their reference matrices with
`sum_weights / n_angles` as the weight, which is only correct at S2 and S4 — the orders
they run. A future reference-matrix run at a higher order has to read the weights out of
the quadrature instead.

## Reflective boundary conditions
A problem file's `boundary_conditions` object takes `vacuum` (the default for an unset
face) or `reflect` per face. The default is bitwise the pre-reflective behaviour —
verified by re-capturing all 24 baselines when the plumbing landed — so every recipe and
baseline above is untouched, and the reflective recipes below are NEW pins, not
adjustments. Reflective rows couple the angle blocks at the boundary (the streaming-only
pmat carries the coupling too), so these counts have no reason to match their vacuum
counterparts.

The sharp oracles are the verify_2dk reference configs above and the **infinite-medium
check**: with every face reflective (or a vacuum face feeding in exactly the
infinite-medium flux, see the ghost-flux section), a uniform source and uniform xsections, the exact
discrete solution is constant in every cell and angle — no discretisation error, so
`-check_inf_medium` compares against 1e-9 under `-ksp_rtol 1e-12` (lands at ~1e-13).
The constant is per group, by forward substitution down the sweep:
`psi_g = (Source_g / sum_weights + sum_{g'<g} Sigma_s[g'][g] psi_g') /
(Sigma_t[g] - Sigma_s[g][g])` — which is the retired `slab_1dk`'s
`1 / (sigma_t - sigma_s)` for the Source 2.0 slab files (its VecSet source was per-angle
by design; 2.0 shared over sum_weights 2 restores it) and the retired `box_2dk`'s
`1 / (sum_weights (sigma_t - sigma_s))` for the Source 1.0 box files. It is
decomposition independent, which is what makes it the parallel oracle for the reflect
partner columns.

Singularity constraint: all-reflect with a scattering ratio of exactly 1 has the
constants in the operator's kernel. Hence the all-reflect problem files carry absorption
(within-group `Sigma_s` 1.0 under `Sigma_t` 2.0 — `slab_inf_medium.json`,
`box_50_inf_medium.json`), and the multigroup reflective file keeps the right face
vacuum — the last group always has ratio 1 (no downscatter out of it).

| reflective config | np=1 | np=2 |
|---|---|---|
| 1D, left reflect, st=2 (ratio 1) | 8 | — |
| 1D, left reflect, st=2, two-call assembly | 8 | — |
| 1D all-reflect infinite medium (rtol 1e-12) | 10 | 11 |
| 1D multigroup 4 groups t05, left reflect | 8 (max over groups) | — |
| 2D 50x50, left+bottom reflect, st=2 ratio 0.5 | 6 | 6 |
| 2D 30x30 S4, left+bottom reflect, ratio 0.5 | 6 | — |
| 2D 20x20 S8, left+bottom reflect, ratio 0.5 | 6 | 6 |
| 2D all-reflect infinite medium (rtol 1e-12) | 10 | 10 |

No row moved under the 2026-09-25 switch to the ghost-flux default: the all-reflect rows
have no vacuum face to change, and the mixed rows land on their Dirichlet-cell counts.

## Ghost-flux vacuum treatment
`"vacuum_treatment": "ghost_flux"` (see `docs/problem_files.md`) changes only the
boundary rows. It landed opt-in, with its checks being the existing ones run a second
time in that mode plus two that only it needs, and became the DEFAULT on 2026-09-25 -
see "Switching the default" at the end of this section for what that moved.

- **Closed form** (`verify_2dk`/`verify_3dk` check 1, ghost mode). A ghost row is a
  stencil row, so its rhs is the streamed source plus `|cosine| / h` times `psi` at the
  GHOST node one cell further upwind, per outside axis. That cell against the Dirichlet
  row's boundary node is exactly the O(h) the two treatments differ by.
- **Reference matrix** (check 2, ghost mode). A ghost row is the interior row with the
  outside-pointing upwind entries dropped. The mixed configs pin the ghost-mode corner
  rule, which is the opposite of the default: reflect wins a corner or edge where a
  direction enters through both a vacuum and a reflective face.
- **Constant inflow** (check 3, both modes). Inflow `psi_in` on every face and a source
  `sigma_t psi_in` make `psi = psi_in` exact, and the rhs is built by `UboltFillInflow` +
  `UboltFillSource` exactly as `transportk` builds it. That pins the per-face
  `|cosine| / h` weights in `BoundaryInfo::ghost_inflow_d` and that `UboltFillSource`
  adds rather than overwrites. Checks 1 and 2 build their rhs by hand and cannot see
  either.
- **Opposite-ordinate identity** (check 4, ghost mode only, parallel too).
  `||A^T - P A P||_F / ||A||_F` on streaming + removal with heterogeneous sigma_t, where P
  swaps `(cell, Omega)` with `(cell, -Omega)` (found by the driver's own cosine search).
  It must be below 1e-14 and is 0.0. This is why the mode exists: it lets one hierarchy
  built on half the ordinates precondition the other half through its transpose. Under
  the default it fails on the boundary rows (residue ~0.1 relative in 2D), so it is not
  run there.
- **Solves** against the infinite-medium oracle. `*_inf_medium_ghost.json` replace some
  or all reflective faces with ghost-flux vacuum faces whose inflow IS the
  infinite-medium flux, so the constant stays exact. The 2D/3D files keep reflective
  faces on the low sides, which puts the mixed corners in the solve. `-check_inf_medium`
  accepts such a face (whole-face, and inflow / sum_weights equal to the expected
  constant in every group, otherwise it errors).
- `DSAPrecon` refused ghost mode until the switch; it now takes either treatment (see
  below).

Counts, measured 2026-09-25 on the local opt arch and pinned on exactly that. The same
day's sweep in the 64-bit and OpenMP CI images measured one more on two rows (noted in
the table) and those pins first took the max; the re-pin for the default switch moved
them back onto the opt count, so these two are where CI would ask for +1:

| ghost-flux config (rtol 1e-12, `-check_inf_medium`) | np=1 | np=2 |
|---|---|---|
| 1D slab, both faces ghost vacuum | 8 | 8 |
| 1D slab, both faces ghost vacuum, `-matfree_removal` | 17 (64-bit CI 18) | — |
| 2D 50x50, left+bottom reflect, right+top ghost vacuum | 12 | 11 (OpenMP CI 12) |
| 2D same, `-matfree_removal` | 25 | — |
| 3D 10^3, left+front+bottom reflect, others ghost vacuum | 11 | 11 |
| 3D same, `-matfree_removal` | 21 | — |

### Switching the default (2026-09-25)
Ghost-flux replaced Dirichlet-cell as the default vacuum treatment; `"vacuum_treatment":
"dirichlet_cell"` keeps the old path. `*_dirichlet_cell.json` twins of `slab_st2`,
`box_50_st2`, `cube_10_st2` and `plex_box_50_st2` were added as recipes and reproduce the
pre-switch counts exactly (6, 6, 5, 7; `box_50_st2_dirichlet_cell` also at np=2, 6, and
with `-precon_dsa`, 5), and the 1D baselines were re-captured — see "Single-group
baselines". Every pin was re-set onto the local opt count at the same time. The recipes
whose count moved (the max over groups for a multigroup recipe, as pinned; "and
`-matfree_removal`" means the right-preconditioned twin, which moved identically):

| recipe | Dirichlet-cell | ghost-flux |
|---|---|---|
| 1D `slab_st2`, `-precon_stream` and `-matfree_removal`, np=1 and 2 | 9 | 8 |
| 1D `slab_st0`, `-diag_scale -precon_stream`, np=1 | 3 | 2 |
| 1D `slab_mg4_t05`, `-precon_stream` and `-matfree_removal`, np=1 and 2 | 11 | 9 |
| 1D `slab_diffusive`, np=1 and 2 | 20 | 23 |
| 1D `slab_diffusive`, `-precon_dsa`, np=2 | 10 | 11 |
| 1D `slab_decades4`, ref-shift `-precon_ref_k 4`, np=1 and 2 | 20 | 22 |
| 1D `slab_decades4`, ref-shift default k, np=1 and 2 | 42 | 44 |
| 1D `slab_decades4_stream0`, ref-shift default k, np=1 and 2 | 20 | 22 |
| 2D `box_50_st2` (also `-ubolt_coo_two_call`; np=2 also `-check_matfree`), np=1 and 2 | 6 | 7 |
| 2D `box_50_identity`, np=1 | 6 | 7 |
| 2D `box_80x40_st2`, np=1 and 2 | 6 | 7 |
| 2D `box_50_st2`, `-precon_stream` and `-matfree_removal`, np=1 and 2 | 9 | 7 |
| 2D `box_80x40_st2`, `-precon_stream` and `-matfree_removal`, np=1 | 9 | 7 |
| 2D `box_decades4`, ref-shift `-precon_ref_k 4`, np=1 and 2 | 30 | 29 |
| 2D `box_decades4`, ref-shift default k, np=1 and 2 | 48 | 51 |
| 2D `box_diffusive`, `-precon_dsa -pc_composite_type additive`, np=1 | 23 | 14 |
| 2D `box_crooked_pipe`, np=1 | 123 | 113 |
| 2D `box_crooked_pipe`, np=2 | 95 | 114 |
| 2D `box_crooked_pipe`, `-precon_dsa`, np=1 | 72 | 28 |
| 2D `box_crooked_pipe`, `-precon_dsa`, np=2 | 69 | 28 |
| 2D `box_layers`, np=1 and 2 | 25 | 24 |
| 2D `box_layers`, `-precon_dsa`, np=1 | 13 | 12 |
| 2D `box_layers`, `-precon_dsa`, np=2 | 12 | 11 |
| 2D `box_random8`, np=1 and 2 | 27 | 19 |
| 2D `box_random8`, `-precon_dsa`, np=1 and 2 | 11 | 8 |
| 3D `cube_10_st2` (also `-ubolt_coo_two_call`), np=1 and 2 | 5 | 6 |
| 3D `cube_10_identity`, np=1 and 4 | 5 | 6 |
| 3D `cube_10_mg4_t05`, np=2 (its last group; serial stays 5) | 5 | 6 |
| 3D `cube_diffusive`, np=1 and 2 | 21 | 18 |
| 3D `cube_diffusive`, `-precon_dsa`, np=1 and 2 | 10 | 8 |
| 3D `cube_diffusive_yreflect`, `-precon_dsa`, np=1 | 10 | 12 |
| plex `plex_box_50_st2`, `-precon_stream` and `-matfree_removal`, np=1 | 9 | 7 |
| plex `plex_box_50_st2`, `-precon_stream` and `-matfree_removal`, np=2 | 9 | 8 |
| plex `plex_cube_10_st2`, np=1 and 2 | 5 | 6 |
| plex `plex_square_msh`, np=1 | 4 | 5 |
| plex `plex_decades4`, ref-shift `-precon_ref_k 4`, np=1 and 2 | 30 | 29 |
| plex `plex_decades4`, ref-shift default k, np=1 and 2 | 48 | 51 |

Every other recipe measures its Dirichlet-cell count, including every all-reflective one
(nothing to change there) and every `*_inf_medium_ghost` one (ghost-flux already).

- The ratio-1 st=2 workhorses take one more in 2D and 3D (`box_50_st2`, `box_80x40_st2`,
  `cube_10_st2` and their identity/two-call copies, `plex_cube_10_st2`); 1D `slab_st2`,
  the ratio-0.5 files and `box_40x20_st2` / `box_60_st2` do not move. `plex_box_50_st2`
  stays at 7, so the structured and plex 50x50 boxes now agree.
- The streaming-only pmats improve everywhere: 9 -> 8 in 1D, 11 -> 9 on the multigroup
  slab, 9 -> 7 on the 2D boxes, 9 -> 7/8 on the plex box; 3D stays at 6.
- The mismatched default-k ref-shift gets worse by 2 or 3 (1D 42 -> 44, 2D 48 -> 51)
  while exact coverage moves with the full-pmat count it reproduces (1D 20 -> 22, 2D 30
  -> 29).
- DSA is much better on the heterogeneous problems — crooked pipe 72 -> 28 (and the
  unaccelerated serial/parallel gap, 123 against 95, is gone: 113 / 114), random8
  11 -> 8, `cube_diffusive` 10 -> 8, and the additive composite 23 -> 14 — but worse on
  `cube_diffusive_yreflect`, 10 -> 12. Off the recipes, the mismatched 2D ref-shift +
  DSA that took 179 on its thick group now takes 10 (see "DSA on a shifted pmat").
- DSA needed no change to its Marshak face for the switch: scaling that coefficient over
  0.25-1.0 moved no DSA count by more than 1.

## Painted regions (MaterialSpec)
A problem file's `regions.paint` list paints shapes — boxes in 2D, intervals in 1D —
over the background material (see `MaterialSpec` and `docs/problem_files.md`). No paint
is the uniform problem bit-for-bit, which was verified the strong way when the plumbing
landed: everything builds its xsections and rhs through the MaterialSpec path, and all
24 baselines reproduce bitwise.

The sharp oracle is the **painting identity**: regions whose values equal the
background's must land exactly on the uniform recipe's residual history, because
painting and expansion change nothing but which table entry a cell reads. The recipe
pins the count; the bitwise history match was checked at np=1 and np=4 when the pins
were captured, and re-checked on the 2026-08-02 isotropic-source re-measure (where the
np=4 count moved 6 -> 7, uniform and painted together). It runs at `-n 4` in the
parallel suite deliberately — painting maps cell centres through the
patch-lexicographic local ordering, and the 2D processor grid is where that ordering is
least like the natural one. The heterogeneous recipes are ordinary pins (and the
absorbing block was eyeballed via `-flux_vtk`: a clear flux depression over the block,
~1.9 against ~7-10 in the surrounding medium).

The **overlap** recipes sharpen the same oracle onto paint ORDER. An extreme sourceless
material (sigma_t 100, sigma_s 0, q 0) is painted first and a copy of the background
over a box that contains it, so later-paint-wins makes the result uniform and the run
has to land on the uniform ratio-0.5 history. Both were checked bitwise against a
uniform twin of the same mesh when the pins were captured. Reversing the paint list
(letting the earlier box win) moves 2D 30x30 from 5 to 4 iterations, so the pin alone
catches it; in 3D 10^3 the count does not move at the default rtol — the history does,
and that is what the bitwise check covers. The covering box stays strictly inside the
domain on purpose: a whole-domain cover would leave the background material unused.

| painted config | np=1 | np=2 | np=4 |
|---|---|---|---|
| 2D 50x50 identity: 2 regions = background, st=2 | 7 (Dirichlet-cell 6) | — | 7 |
| 2D 30x30 overlap: masked box covered by a background copy, st=2 ratio 0.5 | 5 | — | — |
| 2D 50x50 absorbing sourceless block (sigma_t 10, sigma_s 1, q 0) over st=2 ratio 0.5 | 5 | 5 | — |
| 2D 60x60 cold box: no face inflow, zero source outside a central 0.1x0.1 region | 5 | 5 | — |
| 1D multigroup 4 groups t05, double-density region 0.4-0.6 | 6, 6, 6, 6 | 6, 6, 6, 6 | — |

The cold box could not move: zero inflow leaves the painted source as the only rhs, so
the isotropic normalisation scales b uniformly, which a relative residual never sees.

## 3D verification
The 2D story, one axis wider, and the same deliberate absence: there are **no 3D
baselines** — `tests/verify_3dk` compares the operator itself, `make baselines` stays
1D + multigroup only.

`verify_3dk` runs the checks of `verify_2dk`, both S2 (8 angles) and S4 (24 — a 3D
level-symmetric set has twice the ordinates of the same-order 2D set), each under both
vacuum treatments (see "Ghost-flux vacuum treatment" below):

1. **Pure-streaming closed form**, `psi = x + y + z` on an 8x6x4 grid over a 1x2x3 box —
   all three extents distinct, so no axis mix-up can hide. Residual ~8e-15 against the
   1e-12 tolerance, then the solve.
2. **Shell operator vs a reference matrix** on 4x3x2, natural ordering, serial only, with
   its own independent boundary classification as in 2D. Three BC configurations apiece:
   all-vacuum, all-reflect, and mixed with reflect on **left + front + bottom** — three
   reflective faces meeting at one corner, so the checked matrix carries the single,
   double AND triple cosine flips (the triple is new behaviour 2D cannot exercise) plus
   vacuum-wins edges and corners against the other three faces. Agrees to 0.0, i.e.
   bitwise, in all six runs.

Mind the face names in 3D: `bottom`/`top` are the **z** faces (PETSc's box-mesh "Face
Sets" convention), the y faces are `front`/`back` — see `docs/problem_files.md`.

**Parallel.** A 3D DMDA splits one direction at `-n 2` and two at `-n 4`; the recipes run
the closed form and the painting identity at both, and `-n 8` (the first genuinely 3D
processor grid) was checked by hand when the backend landed but stays out of the recipes —
CI runners have few cores. The infinite-medium check is the decomposition-independent
parallel oracle, exactly as in 1D/2D (`cube_10_inf_medium.json`, ~1e-13 against 1e-9).

## 3D iteration counts
Measured 2026-08-02 on the reference build at capture. CI has since run green on the
sensitive arches (64-bit indices, OpenMP Kokkos) with these pins as committed, so no
per-arch slack was needed. Re-measured 2026-09-25 on the local opt arch under the
ghost-flux default; moved counts carry their Dirichlet-cell value in brackets.

| config | np=1 | np=2 |
|---|---|---|
| 10^3, st=2 (ratio 1) | 6 (Dirichlet-cell 5) | 6 (Dirichlet-cell 5) |
| 10^3, st=2, ratio 0.5 | 4 | 4 |
| 10x5x5 (dx, dy, dz all differ), st=2 | 6 | 6 |
| 15^3 (same shape, 1.5x finer), st=2 | 6 | 6 |
| 10^3, streaming-only pmat, st=2 | 6 | 6 |
| 10^3 S4, st=2 | 6 | 6 |
| 10^3, left+front+bottom reflect, ratio 0.5 | 5 | 5 |
| 10^3 all-reflect infinite medium (rtol 1e-12) | 10 | 10 |
| 10^3 identity: 2 regions = background, st=2 | 6 (Dirichlet-cell 5) | 6 (np=4; Dirichlet-cell 5) |
| 10^3 overlap: masked box covered by a background copy, st=2 ratio 0.5 | 4 | — |
| 10^3 absorbing sourceless block over ratio 0.5 | 4 | 4 |
| 10^3 cold cube: no face inflow, zero source outside a central 0.2^3 region | 4 | 4 |
| 10^3 multigroup 4 groups t05 | 5, 5, 5, 5 | 5, 5, 5, 6 (Dirichlet-cell 5, 5, 5, 5) |

Every cube dropped from 20^3 to 10^3 on 2026-08-02 to bring CI down (see "Recipe cost"
below), and the mesh-independence twin from 30^3 to 15^3 to keep its 1.5x ratio; the
counts above are the re-measured references, one lower than the 20^3 ones almost
throughout. Mesh independence was a weaker statement after the resize: under
Dirichlet-cell 10^3 to 15^3 moved 5 to 6, where 20^3 to 30^3 held flat at 6. Under the
ghost-flux default 10^3 rises to 6, so 10^3 to 15^3 is flat at 6 again (and 2D's
40x20 to 80x40 pair is now the one that moves by one).

The cold cube's source region had to widen from 0.1^3 to 0.2^3: at 10 cells the old
0.45/0.55 bounds land exactly on cell centres, where membership is a floating-point coin
toss. Any resize has to re-check that — a bound `b` is fragile at `n` cells whenever
`b*n` is a half-integer.

The painting identity was re-checked bitwise
against the uniform 10^3 history at np=1 when the pins were re-captured. The absorbing
block was eyeballed via `-flux_vtk` (sigma_t 10 painted mid-cube, flux ~3.3 over the
block against ~10 in the surrounding medium) — those numbers are from the 20^3 file,
before the 2026-08-02 resize, and have not been re-measured at 10^3.

## Unstructured DG0 verification
The unstructured backend (`UnstructuredDG0` + `StreamingTermDG0`, Phase 6a) has no
baselines either, for the 2D/3D reason: `tests/verify_plexk` compares the operator
itself. It runs serially in `run_check` and at `-n 2` and `-n 4` in
`run_tests_short_parallel`, and every comparison in it is parallel-safe — nothing assumes
rank 0 owns anything, or that either backend numbers its rows naturally.

1. **FD twin, 2D quads.** On a uniform quad box DG0 upwind IS the structured upwind
   stencil (`|mu| dy / (dx dy) = |mu| / dx`), so each case is built on `StructuredFD2D`
   and on the plex, and compared: (a) the assembled matrices, brought onto one ordering
   by an explicit permutation matrix P (not `MatPermute`, which wants matching local
   sizes and the two decompositions differ), `||A_plex - P^T A_fd P||_inf <= 1e-12
   ||A_fd||_inf` (measured ~1e-15 to 3e-15); (b) the BC row mask and reflect flags **exactly** and the Dirichlet values to
   1e-14; (c) the rhs (`UboltFillInflow` + `UboltFillSource` + `UboltZeroReflectRows`)
   to 1e-14; (d) the matrix-free scatter on the same permuted random vector, 1e-14;
   (e) the composed `TransportOperator::diagonal()` against the plex's own
   `MatGetDiagonal`, **bitwise** (0.0); (f) a full solve at rtol 1e-13, the permuted
   solutions to 1e-9 (two different Krylov histories, so no tighter). Cases: 4x3 S2 and
   S4 vacuum, 5x4 S2 reflect left + bottom, 5x4 S4 mixed (reflect left + top, vacuum
   right + bottom), 6x4 S2 with inflow 1.0 on the left windowed to y in [0.5, 1.5] and
   0.3 on the bottom (the corner winning-face rule and the window), and a painted 4x3 S4
   — painted by box on both sides, and again with the plex side painted through a
   "Cell Sets" label written onto the same cells (the path a Gmsh physical group takes,
   so `paint_cell_sets` is held to the same twin).
2. **FD twin, 3D hexes**, the same (a)-(f) against `StructuredFD3D`: 3x2x2 S2 vacuum,
   3x3x2 S4 reflect left + front + bottom (the three-face corner), 4x2x3 S2 with inflow
   on the left windowed in (y, z).
3. **Simplex meshes** (triangles 6^2 = 72 cells, tets 3^3 = 162 cells — a simplex box
   has 2 triangles or 6 tets per box cell), which have no twin: (a) the infinite medium,
   every face reflective, flat to 1e-10 at `1 / (sum_weights (sigma_t - sigma_s))`
   (measured ~1e-14) — which pins reflection on the axis-aligned faces of triangles and
   tets and the scatter reading BC rows; (b) vacuum, a pure absorber and a source only in
   the central box [1/4, 3/4]^dim: converged, inside the exact discrete bounds
   `0 <= psi <= q / (sum_weights sigma_t)` to 1e-12 (DG0 upwind is monotone and the
   infinite medium is an upper solution), and the scalar flux integrated per unit source
   within 25% of the quad/hex mesh's at the same nominal h (measured 3% in 2D, 5% in
   3D). A sign or scaling error fails it; it is not a convergence claim.
4. **Layout and geometry invariants** on every mesh: the volumes sum to the box's to
   1e-12 relative, every cell's outward area-weighted normals close to 1e-12 (a normal
   that failed to flip outward shows up as 2 nA), the cell count is the box's on
   quads/hexes, and on the simplex boxes every "Face Sets" id sits on the box face the
   structured `FACE_*` constant of that id names, exactly (simplex and tensor boxes get
   the label from different PETSc code paths).
5. **Error paths**, through `PetscReturnErrorHandler`: a reflection partner that is
   itself a REFLECTIVE row (reflect on both x faces of a one-cell-wide box), `cell_sets`
   on a mesh with no "Cell Sets" label, a 2D quadrature on a 3D mesh, a `.vts` name on
   the plex, reflect on the slanted hypotenuse of `tests/meshes/tri_slanted.msh` (a
   4-triangle right triangle), and a boundary condition on a "Face Sets" value no face
   carries. All six must be rejected. The POSITIVE twin of the slanted case is checked
   too: reflect on that mesh's axis-aligned bottom with the hypotenuse a vacuum face
   must be accepted (some reflection partners there are Dirichlet rows, the
   symmetry-reduced geometry a file mesh exists for), converge, and stay inside the
   discrete bounds - found by the debug-arch sweep of 23 Sep 2026, which the first cut
   rejected with a misleading "single-cell-wide" message.
6. **VTU output**: writes a `.vtu` into the current directory, reads it back on rank 0
   after a barrier and requires the `<Piece NumberOfCells>` entries to sum to the global
   cell count — PETSc's writer writes every cell of a rank's local mesh unless a "vtk"
   label says otherwise, so an overlap cell written by two ranks would push it over —
   then deletes it. The file is `verify_plexk_tmp_<size>.vtu`, so nothing is left behind.
7. **Ghost-flux vacuum treatment.** Every FD twin case of 1 and 2 bar the "Cell Sets" one
   again under `"ghost_flux"`, the same (a)-(f): the rhs comparison now carries the ghost
   inflow (the windowed 2D case has both corner faces feeding the origin cell, each on
   its own window), and the mixed configs pin the reflect-wins corners and the mirror over
   the axis-aligned vacuum face. Then, with no twin: (a) on streaming + removal with
   sigma_t varying cell to cell, `||(VA)^T - P(VA)P||_F / ||VA||_F <= 1e-13` (measured
   ~1e-16), P the opposite-ordinate swap found by the driver's own cosine search and V
   the cell volumes per row. This is the property a half-quadrature preconditioner
   applied transposed rests on. It is not bitwise: the diagonal's outflow sum equals the
   opposite direction's inflow sum only because the cell's `nA_f` close, which is to
   rounding. It runs on 6^2 triangles S2, on 3^3 tets S4, and on
   `tests/meshes/square_irregular_tri.msh` S6. That file is a 4x4 triangulation with the
   interior nodes perturbed, so the areas vary by +-17%. There the UNWEIGHTED
   `||A^T - PAP|| / ||A||` is required to be above 1e-3 (it is ~0.07), because the
   generated simplex boxes have equal volumes and would not tell the two identities
   apart. (b) inflow `psi_in` on every face and source `sigma_t psi_in` against
   `psi = psi_in` through `UboltFillInflow` + `UboltFillSource`, residual below
   `1e-12 ||b||` (measured ~1e-15), which pins the `|Omega . nA_f| / V_c` weights on
   faces of every orientation. (c) `meshes/tri_slanted.msh` in ghost mode: no
   Dirichlet rows left, reflective rows still present, the solve converged and in
   bounds.

**Why the twin comparison is to rounding and not bitwise.** The two backends reach the
same coefficient through different arithmetic — the FD stencil writes `|mu| / dx`, DG0
writes `(Omega . nA_f) / V_c` with the area-weighted normal and the volume coming out of
`DMPlexComputeCellGeometryFVM` — so `|mu| dy / (dx dy)` and `|mu| / dx` agree to the last
bit or two, not exactly. The structure (which rows are BC rows, which slots are live) is
exact, and is checked exactly.

**The permutation, by centroid.** Neither backend's global numbering is assumed. The
join key is each cell's natural lexicographic index — on the FD side from its (i, j[, k]),
on the plex side `floor(x / dx)` etc. from its centroid — and it is used ONLY as a key:
each side's row of its local cell c is its OWN `rstart + c * n_angles + a`, the property
each backend's layout check asserts (`CheckDALayout`, `CheckPlexLayout`). The DMDA's row
must come from its own layout rather than from the natural index, because a 2D/3D DMDA
numbers patch-lexicographically in parallel (see "DMDA layout" below); the plex's global
numbering comes from its global section and the simple partitioner, and is unrelated to
either. The key goes through a Vec indexed by key — the plex ranks write their row base
into it, the FD ranks read theirs back — and out of it come the permutation matrix and a
`VecScatter` for the vectors, whatever the two decompositions are.

**Parallel.** `-n 2` splits both twins, `-n 4` is the first run where the DMDA splits in
two directions and the plex's simple partition cuts the cell points into four
contiguous ranges; both pass with the same measured differences as serial.

## Unstructured iteration counts
Measured 2026-09-22 on the opt arch (`arch-linux-c-opt`), `-ksp_max_it 400
-ksp_converged_reason`. Swept 2026-09-25 in the 64-bit and OpenMP CI images:
only the two `plex_box_50_st2` streaming-only rows moved (9 to 10). The simplex-box
rows (`plex_tri_*`, `plex_tet_*`) do NOT run in CI: generating them needs PETSc's
triangle / (c)tetgen, which the CI images lack, so `tests/Makefile` skips them there
(`PETSC_HAVE_TRIANGLE`, `PETSC_HAVE_TETMESHER`) and `verify_plexk` skips its
generated-simplex checks. Triangles stay covered in CI through the Gmsh mesh files.
Re-measured 2026-09-25 under the ghost-flux default and **pinned on the opt count,
exactly**, as every pin now is; a moved count carries its Dirichlet-cell value in
brackets. The structured twin is the same file without `"type": "unstructured"`, run
with the same options.

| recipe | plex np=1 | plex np=2 | structured twin np=1 / np=2 | notes |
|---|---|---|---|---|
| `plex_box_50_st2` (50x50 quads, ratio 1) | 7 | 7 | `box_50_st2`: 7 / 7 (Dirichlet-cell 6 / 6) | the twins agree now; see "the twin difference" below |
| `plex_box_50_st2`, `-precon_stream -ksp_pc_side right` | 7 (Dirichlet-cell 9) | 8 (Dirichlet-cell 9) | 7 / 7 (Dirichlet-cell 9 / 9) | Dirichlet-cell: hair-trigger, 10 on the 64-bit and OpenMP CI arches |
| `plex_box_50_st2`, `-matfree_removal -ksp_pc_side right` | 7 (Dirichlet-cell 9) | 8 (Dirichlet-cell 9) | 7 / 7 (Dirichlet-cell 9 / 9) | same count as the line above |
| `plex_box_50_reflect_lb` | 6 | 6 | `box_50_reflect_lb`: 6 / 6 | |
| `plex_tri_30_st2` (1800 triangles, S4, ratio 0.5) | 5 | 5 | — | |
| `plex_cube_10_st2` (1000 hexes, ratio 1) | 6 (Dirichlet-cell 5) | 6 (Dirichlet-cell 5) | `cube_10_st2`: 6 / 6 (Dirichlet-cell 5 / 5) | |
| `plex_tet_6_st2` (1296 tets, ratio 0.5) | 5 | 5 | — | |
| `plex_square_msh` (8 Gmsh triangles) | 5 (Dirichlet-cell 4) | 4 | — | |
| `plex_tri_30_inf_medium_ghost`, `-check_inf_medium -ksp_rtol 1e-12` (ghost-flux right + top, reflect left + bottom; measured 2026-09-25) | 12 | 12 | — | |
| the same, `-matfree_removal` | 33 | — | — | |
| `plex_tet_6_inf_medium_ghost`, `-check_inf_medium -ksp_rtol 1e-12` (ghost-flux right + back + top; measured 2026-09-25) | 11 | 11 | — | |
| `plex_decades4`, `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 4, 6, 15, 29 | 4, 6, 15, 29 | `box_decades4`: 4, 6, 15, 29 both | Dirichlet-cell 4, 6, 15, 30 on both sides |
| `plex_decades4`, `-matfree_removal -precon_ref_shift` (default k = 2) | 7, 8, 27, 51 | 7, 8, 27, 51 | `box_decades4`: 7, 8, 27, 51 both | Dirichlet-cell plex 7, 9, 24, 48 serial and 7, 8, 24, 48 otherwise |

A multigroup row pins the max over its groups (29, 51), as everywhere else.

**The twin difference is a finding, not a bug.** On a uniform quad/hex box the plex matrix
IS the structured one to ~1e-15 (verify_plexk, above), but its rows are in a different
order — plex point order under the simple partitioner, not DMDA order — and PCAIR is not
permutation-invariant: its CF splitting and its approximate inverses depend on the order
it walks the rows. So a twin can land an iteration away. Under ghost-flux the only twin
rows that differ are the streaming-only pmat pair at np=2 (plex 8, structured 7); the
Dirichlet-cell measurements below had two others, both on the edge of rtol on BOTH sides
(the rtol margins are Dirichlet-cell-era numbers and were not re-measured):
- `plex_box_50_st2`: the structured solve clears rtol at iteration 6 with 1.4% to spare
  (5.30e-4 against a target of 5.38e-4); the plex one misses at iteration 6 by 0.4%
  (5.398e-4 against 5.378e-4 — even the preconditioned `r_0` differs in the fourth
  digit, which is the permutation acting on PCAIR) and converges at 7 with a wide margin
  (0.07 of rtol). The pin of 7 is safe; another arch could see 6, which passes.
- `plex_decades4` default k, group 1 serial: 9 against the structured 8, missing rtol at
  8 by 9%; at np=2 the plex matches the structured 8 (0.94 of rtol). The recipe pins the
  group max, 48, which is unaffected. Under ghost-flux group 1 is 8 on both sides.

The partitioner moved the same edge under Dirichlet-cell: `plex_box_50_st2` at `-n 2`
was 7 under the default simple partitioner and 6 under `-petscpartitioner_type parmetis`
(tri and tet: 5 under both). Under ghost-flux it is 7 under both. That is why the backend defaults to `simple` — deterministic, and available on
every CI image, where ParMETIS may not be — and why no recipe passes a partitioner.

**Hair-trigger pins** (Dirichlet-cell era, not re-measured under ghost-flux), the final
residual as a fraction of rtol at the pinned iteration
(np=1 / np=2), i.e. where another arch could need one more: the streaming-only pmat and
matfree pair (0.87 / 0.71), the two `plex_decades4` group-3 solves (0.94 / 0.94, and
group 1 of default k at np=2, 0.94), `plex_cube_10_st2` (0.76), `plex_square_msh`
(0.60 / 0.64). The structured twins of the first two carry exactly that +1 in their
pins then (10, 31, 49), and the CI sweep did ask for 10 on the first. The rest cleared
rtol by 6x or more.

**Second-arch sweep** (2026-09-23, Dirichlet-cell era, `arch-linux-c-debug` against a
debug-built PFLARE main): build with `-Wall -Werror`, `check`, `tests_short` and `tests` all exit 0; every
plex recipe and every structured twin converges in EXACTLY the opt count above at np 1
and np 2 (per group too), with the same rtol margins to three digits; `-malloc_dump`
reports nothing unfreed on `verify_plexk` (np 1, 2, 3) and on the plex solves with and
without `.vtu` output. One structured recipe is the sharpest edge in the whole suite:
`box_50_st2` at np 2 cleared rtol at its then pin of 6 by 0.013% (it takes 7 under
ghost-flux).

**Checked by hand, not recipes** (2026-09-22, np 1 and 2): every plex problem with
`-check_matfree -matfree_removal` — matvec differences 2e-16 to 4e-16 against 1e-13,
composed diagonal **0.0** against the assembled one on every file, triangles and tets
included; `plex_box_50_st2` and `plex_tri_30_st2` with `-ubolt_coo_two_call` — the
`-ksp_monitor` history is byte-identical to the one-call run at both rank counts; every
plex problem with `-flux_vtk x.vtu` at np 1, 2, 4 — the file's `NumberOfCells` totals the
global cell count (2500, 1800, 1000, 1296, 8, 900) and it carries `scalar_flux`,
`sigma_t`, `source`; and `-precon_dsa` on a plex file fails with PETSC_ERR_SUP (the DSA
correction is a DMDA diffusion operator).

## Matrix-free removal (`-matfree_removal`)

`RemovalTerm` can be applied matrix-free instead of assembled
(`RemovalTerm::set_matrix_free`, `-matfree_removal` on the driver). The assembled
matrix then carries **streaming only**, which does not depend on the group: it is
assembled once before the sweep, no group refills anything, and PCAIR sets up once for
the whole sweep. It is off by default, so every recipe, count and baseline above is
untouched — all 24 baselines reproduce bitwise, which is what the default half of the
change was verified against. The recipes below are NEW pins.

Because the assembled matrix already *is* a streaming-only pmat, this mode **implies**
`-precon_stream`; an explicit `-precon_stream` alongside it is redundant and quietly
ignored (it would build a second copy of the same values). `-diag_scale` is a checked
error in this mode — it scales the assembled operator, which here is the streaming part
alone, so the matrix-free removal and scatter would stay unscaled. And because it
implies `-precon_stream`, it inherits that flag's unsupported combination with
`-precon_dsa` (see the DSA section below) — measured identical on
`cube_diffusive.json`: 35 iterations either way alone, neither converging in 300 with
`-precon_dsa` added. No recipe combines them, as none does for `-precon_stream`.

**The counts are the `-precon_stream` counts; the residual histories are not.** Same
operator, same pmat, same Jacobi diagonal — but the assembled path sums streaming and
removal into one matrix entry and multiplies once where the matrix-free path multiplies
twice and adds, so the two differ in the last bits and diverge from there. That is why
the matfree recipes are *pinned at their twin's pin* and get no baseline of their own:
the count is the invariant, the history is not. Every twin below measured exactly the
same count as its `-precon_stream` line, serial and at `-n 2`, under Dirichlet-cell and
again under the ghost-flux default (2026-09-25, local opt; the Dirichlet-cell counts in
brackets where they moved, and the plex pair is in "Unstructured iteration counts").

| config | `-precon_stream` / `-matfree_removal`, np=1 | np=2 |
|---|---|---|
| 1D slab st=2 | 8 / 8 (Dirichlet-cell 9 / 9) | 8 / 8 (Dirichlet-cell 9 / 9) |
| 1D multigroup 4 groups t05 | 9, 9, 9, 8 / same (Dirichlet-cell 11, 11, 11, 9) | 9, 9, 9, 8 / same (Dirichlet-cell 11, 11, 11, 9) |
| 2D 50x50 st=2 | 7 / 7 (Dirichlet-cell 9 / 9) | 7 / 7 (Dirichlet-cell 9 / 9) |
| 2D 80x40 st=2 | 7 / 7 (Dirichlet-cell 9 / 9) | — |
| 3D 10^3 st=2 | 6 / 6 | 6 / 6 |

The infinite-medium check runs in this mode too — `slab_inf_medium.json` in 7 serial
and 8 at `-n 2`, `box_50_inf_medium.json` in 24 both. Those are the streaming-only
pmat's counts rather than the default pc's 10, and they are what `-precon_stream`
measures on the same files. The solution lands at 1.4e-13 to 7.6e-12 against the 1e-9
tolerance; the drift from the default path's ~1e-13 is the pmat and not the matrix-free
apply, since `-precon_stream` measures the same errors to four digits. These three pins
carried one iteration of slack (measured + 1) until the 2026-09-25 re-pin, and sit on the
measured number (7, 8, 24) since; all-reflective, so the ghost-flux switch did not move
them.

**`-check_matfree` is the sharp oracle**, and it is what makes the mode trustworthy
rather than merely green. It builds BOTH operators on whatever problem file is being
run — the same discretisation and the same streaming term, with only which half of
`RemovalTerm` is live differing — and per group applies both shells to three random
vectors, then compares the composed diagonal against `MatGetDiagonal` of the fully
assembled matrix. The assembled operator is refilled per group and the matrix-free one
is never touched after its single assembly, so the per-group behaviour is in the check
and not just the operator at one group. Two tolerances, deliberately different:

- the matvec, `max|y_f - y_a| / max|y_a|` against **1e-13**. Measured 2.5e-16 to
  4.2e-16 in 1D, 2D and 3D, vacuum and reflective, serial and at `-n 2` — rounding and
  nothing else, and a mistake in the apply is many orders larger.
- the diagonal, `max|d_f - d_a|` against **0.0**, i.e. bitwise. Measured 0.0 everywhere,
  and it has to be: each term's `add_diagonal` writes the same expression its
  `assemble_add` writes into the diagonal slot, the composition sums them in the order
  the COO values are summed in, and the BC rows get the same 1.0. That identity is the
  whole reason the removal shell PC may compose its diagonal instead of reading it off a
  matrix, so it is asserted rather than assumed.

It is independent of `-matfree_removal` — it says the same thing whichever mode the
solve around it runs in — and it is in `make check`, on the 1D multigroup file.

## DSA (the diffusion correction, `-precon_dsa`)

`DSAPrecon` adds a third stage to the composite preconditioner: a cell-centred
diffusion operator on the same grid, restricted from and prolonged back onto the
ordinates (`include/ubolt/dsa.hpp`). It is off by default, so every recipe,
count and baseline above is untouched — the DSA recipes below are NEW pins.

The regime it exists for is the one nothing else in the preconditioner
addresses, and which no test problem demonstrated until now. The diffusive
problem files are deliberately the same physics in every dimension — `Sigma_t
100` over cells of width 0.1, so ten mean free paths per cell, at a scattering
ratio of 0.99, every face vacuum so the diffusion operator is nonsingular
whatever the ratio, all at S4: `slab_diffusive.json` is 100 cells over a length
of 10, `box_diffusive.json` 50x50 over 5x5 and `cube_diffusive.json` 10^3 over
1^3. Only the dimension changes between them, which is what makes the counts
comparable. Re-measured 2026-09-25 (local opt) under the ghost-flux default, which
every row with a vacuum face runs in; the Dirichlet-cell count follows in brackets where
it moved. Rows without a recipe at that np (np=2 of the st=2, additive, CG and yreflect
rows) were measured by hand.

| config | np=1 | np=2 |
|---|---|---|
| 1D diffusive slab, no DSA (the reference) | 23 (Dirichlet-cell 20) | 23 (Dirichlet-cell 20) |
| 1D diffusive slab, `-precon_dsa` | 11 | 11 (Dirichlet-cell 10) |
| 1D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 10 | 10 |
| 1D slab st=2, `-precon_dsa` | 5 | 5 |
| 2D diffusive box, no DSA (the reference) | 29 | 29 |
| 2D diffusive box, `-precon_dsa` | 11 | 11 |
| 2D diffusive box, `-precon_dsa -pc_composite_type additive` | 14 (Dirichlet-cell 23) | 14 (Dirichlet-cell 23) |
| 2D diffusive box, `-precon_dsa -dsa_ksp_type cg -dsa_ksp_max_it 5` | 11 | 11 |
| 2D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 10 | 10 |
| 2D box st=2, `-precon_dsa` | 5 | 5 |
| 3D diffusive cube, no DSA (the reference) | 18 (Dirichlet-cell 21) | 18 (Dirichlet-cell 21) |
| 3D diffusive cube, `-precon_dsa` | 8 (Dirichlet-cell 10) | 8 (Dirichlet-cell 10) |
| 3D diffusive cube, Y faces reflective, anisotropic box, `-precon_dsa` | 12 (Dirichlet-cell 10) | 12 (Dirichlet-cell 11) |
| 3D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 9 | 9 |
| 3D cube st=2, `-precon_dsa` | 4 | 4 |

Read each dimension's first two rows together: that pair IS the test, and
halving the count is what the correction buys — 23 to 11 in 1D, 29 to 11 in 2D,
18 to 8 in 3D (20 to 11, 29 to 11 and 21 to 10 under Dirichlet-cell). **The
corrected count is 8 to 11 in every dimension** while the uncorrected one is not,
so what the correction removes is exactly the part that varies with dimension.
That is the claim DSA makes, and it is the number to beat for anything that comes
after (see the numerical-diffusion note in `TODO.md`).

The remaining rows say DSA does not break what already worked — st=2 goes 6 to
5 in 1D, 7 to 5 in 2D and 6 to 4 in 3D, and the infinite media still land on
the exact constant (1D 2.3e-13 serial and 7.1e-12 at np=2, 2D 3.4e-14 and
8.3e-14, 3D 6.9e-14 and 1.1e-13, against the 1e-9 tolerance). Those runs are
also the reflective-branch check: every face reflective means every face is a
zero-Neumann one in the diffusion operator, and the singularity guard is
satisfied by the files' absorption. The solution is unchanged by the
correction, as it must be — `box_50_st2.json` at rtol 1e-12 agrees to 1.3e-12
relative with and without `-precon_dsa`.

**The 3D `yreflect` row is the face-convention pin**, and the only recipe that
can catch a y/z mix-up. In 3D `bottom`/`top` are the **Z** faces and the Y ones
are `front`/`back` (PETSc's box convention — 2D calls the Y faces bottom/top,
which is the trap). A DSA operator that swapped those two axes would put a
Marshak condition where the transport has reflection and a zero-Neumann one
where it has vacuum. `cube_diffusive_yreflect.json` is deliberately
anisotropic, 20x10x5 over 2 x 1 x 0.5 with the Y faces reflective, because a
cubic box hides the swap behind its symmetry: measured by swapping the two
axes in `DSAPrecon::create` by hand, it cost 10 -> 13 iterations there under
Dirichlet-cell (and 12 -> 15 on the Z-reflective mirror of the same file), so the
pin at 11 then failed on it. The file takes 12 under ghost-flux and is pinned at 12;
the swap has not been re-measured there, so whether it still fails the pin is open.

The two 2D generality rows pin the CLI rather than a regime. The shell is added
to the composite BEFORE `KSPSetFromOptions`, so the paper's additive
combination is just `-pc_composite_type additive` (slower here — 14 against 11,
23 under Dirichlet-cell — which is why multiplicative is the default order, but it must keep working),
and the inner diffusion solve is reachable under its own prefix, so
`-dsa_ksp_type cg -dsa_ksp_max_it 5` replaces the single PCGAMG application
with five CG iterations. That buys nothing on this problem (11 either way), and
that is the point: the default inexact solve is already enough.

Until 2026-09-25 these pins carried one iteration of deliberate slack (measured
count + 1), because PCGAMG was new to this test matrix and its aggregation looked
like the most arch-sensitive thing in the suite; CI ran green with it. The ghost-flux
re-pin put them on the local opt count like every other pin, so a DSA recipe is the
first place to look if CI asks for +1.

**`-precon_stream` + `-precon_dsa` is not supported and no recipe combines
them.** In 1D and 2D the combination does not converge in 300 iterations — but
neither does `-precon_stream` alone on those files. A streaming-only pmat
against `Sigma_t 100` is the pre-existing strong-removal problem (the Phase 5
open question in `TODO.md`), not something DSA made worse or was expected to
fix; the correction is being added to a preconditioner that is already failing
on the hyperbolic part. 3D is the one case where the two can be told apart, and
it is worth knowing about: `cube_diffusive.json` with `-precon_stream` alone
converges in 47, and adding `-precon_dsa` to it does **not** converge in 300.
That is a genuine interaction rather than an inherited failure. The obvious
suspect is that a DSA present switches the composite's residual updates onto
the amat (see below), so the stages after index 0 are handed a residual built
with the full operator while PCAIR is still set up on a streaming-only pmat —
but that has not been established, and the combination is unsupported either
way. Open, and recorded in `TODO.md`.

**`-diag_scale` + `-precon_dsa` is not special-cased and not recommended.** The
two are mechanically compatible — nothing errors — but scaling the assembled
operator breaks the `R A P` consistency the `/sum_weights` prolongation scaling
relies on, so the diffusion operator is no longer the right coarse model for
the system being solved. No recipe combines them.

**Composite residual updates come from the AMAT when a DSA is present.** A
multiplicative `PCComposite` forms the residual it hands each later stage from
*pmat* by default, and pmat carries no scattering (it is the assembled
streaming/removal matrix, or a streaming-only one). So once PCAIR has inverted
it, the residual reaching the DSA shell is both tiny and free of the one thing
DSA corrects — measured on `slab_diffusive.json`, the moment reaching the
diffusion solve is ~1e-6 of the residual and the iteration count does not move
at all. `TransportSolver::create` therefore calls `PCSetUseAmat` when, and only
when, it is given a DSA, before `KSPSetFromOptions` so `-pc_use_amat false`
still wins. Without a DSA nothing changes, which is why all 24 baselines still
reproduce byte-for-byte.

### Literature benchmark pins

Four 2D problems replicate configurations from the DSA literature and pin the
heterogeneous regimes nothing above reaches. Full parameter sweeps, the source
papers' own numbers and the deviations from them live in an external report
(`dsa_benchmarks_report.pdf` + `reproduce_dsa_benchmarks.py`, kept outside this
repo; the harness also generated `box_random8.json` — its `_comment` records the
seed and draw order, so the file is reproducible from the harness alone). Each
runs both ways, reference and `-precon_dsa`, serial and `-n 2`, in
`run_tests_short_*`; all four cost ~1 s serial.

| problem | regime pinned | np=1 | np=2 |
|---|---|---|---|
| `box_crooked_pipe.json` (28x20) | discontinuous D through the harmonic face mean (Southworth et al. crooked pipe, Table I set 2) | 113 / 28 (Dirichlet-cell 123 / 72) | 114 / 28 (Dirichlet-cell 95 / 69) |
| `box_layers.json` (40x40) | alternating thick/thin layers (the Warsa mixed regime) | 24 / 12 (Dirichlet-cell 25 / 13) | 24 / 11 (Dirichlet-cell 25 / 12) |
| `box_lattice.json` (56x56) | scattering ratio exactly 1 with painted pure absorbers | 9 / 5 | 9 / 5 |
| `box_random8.json` (32x32) | fixed-seed blockwise random thick/thin/absorber mix | 19 / 8 (Dirichlet-cell 27 / 11) | 19 / 8 (Dirichlet-cell 27 / 11) |

Counts are reference / `-precon_dsa`, measured on the local **opt** arch under the
ghost-flux default (2026-09-25) and pinned exactly; until then the pins sat at
measured + 1 like the other DSA pins, and CI ran green with that. Under
Dirichlet-cell the crooked pipe was the one file whose reference count moved
with the rank count (123 vs 95 — PCAIR on a strongly heterogeneous operator);
under ghost-flux the two are 113 and 114, and its DSA count is 28 at both.

The crooked pipe's counts were re-measured on 2026-08-04 when inflow went per
face: the file used to be driven by a painted unit-`Source` strip in the first
pipe column (a material `pipe_src`, retired with its paint box), because UBOLT
had only one global inflow value and could not drive one face. It is now driven
the way the paper does it — an inward isotropic source at the inlet, here
`"left": {"type": "vacuum", "inflow": 1.0, "window": [2.0, 3.0]}`, the pipe
mouth only. `dy = 5/20 = 0.25`, so the window covers exactly the four boundary
cells `j = 8..11` (centres 2.125–2.875) the strip covered. The magnitude is
only relative in a linear fixed-source solve, so `1.0` angle-integrated is the
faithful replacement for the retired unit angle-integrated `Source` strip.

### Checks that are not recipes
A recipe passes on exit 0, so the two guards are checked BY HAND:

- **Void guard**: `./transportk -problem problems/slab_st0.json -precon_dsa`
  must fail with the `Sigma_t > 0 in every cell` message —
  `D = 1/(3 Sigma_t)` is undefined in a void, and the fix is a region-masked
  diffusion operator (future work, as in the paper) rather than a fudged
  coefficient.
- **Singularity guard**: an all-reflective problem whose within-group `Sigma_s`
  equals its `Sigma_t` must fail with the pure-Neumann message. This is the same
  constraint the transport operator carries (see the reflective section above),
  which is why no shipped problem file is in that state.

## The reference-shifted streaming pmat (`-precon_ref_shift`)

`-matfree_removal` leaves the assembled matrix carrying streaming ONLY, and the
pmat defaults to it. That is fine while the removal is weak and useless once it
is strong: PCAIR is then set up on something the solve is not.
`-precon_ref_shift` puts a representative removal back — the pmat becomes
`L + alpha_g * D_ref`, with `D_ref` group 0's per-cell `Sigma_t` and `alpha_g`
the log-mean of `Sigma_t(g)/Sigma_t(0)` over the mesh — in `k` shared copies,
`k` from `-precon_ref_k` or from the default rule in `RefShiftPmats`. It is off
by default, so every recipe, count and baseline above is untouched; these are
NEW pins. It requires `-matfree_removal` (a checked error otherwise), and `k`
pmats means `k` PCAIR hierarchies, which is the mode's whole cost.

**The claim is an identity, not an improvement.** When the alphas are covered
exactly — one bin per distinct ratio, which the driver reports on stderr as
`worst mismatch 1` — `alpha_g * D_ref` IS `Sigma_t(g)` cell by cell, the pmat is
the full one, and the run reproduces the DEFAULT (full-pmat) iteration counts
exactly. Out of `k` matrices assembled once, with no assembly anywhere in the
group sweep. Six problem files say so independently, under Dirichlet-cell and again
under the ghost-flux default (2026-09-25, local opt; Dirichlet-cell in brackets where it
moved):

| exact-coverage identity | ref-shift count | the full-pmat count it reproduces |
|---|---|---|
| `slab_st2.json` (1 group, so k = 1 always) | 6 | 6 |
| `slab_mg4_t05.json`, `-precon_ref_k 2` (two distinct ratios) | 6, 6, 6, 6 | 6, 6, 6, 6 |
| `slab_decades4.json`, `-precon_ref_k 4` | 4, 8, 14, 22 (Dirichlet-cell 20) | 4, 8, 14, 22 |
| `box_decades4.json`, `-precon_ref_k 4` | 4, 6, 15, 29 (Dirichlet-cell 30) | 4, 6, 15, 29 |
| `slab_diffusive.json` (1 group), no DSA / `-precon_dsa` | 23 / 11 (Dirichlet-cell 20 / 11) | 23 / 11 |
| `box_diffusive.json` (1 group), no DSA / `-precon_dsa` | 29 / 11 | 29 / 11 |

Mind the pc side: these run with the DEFAULT (left) preconditioning, not the
`-ksp_pc_side right` every `-precon_stream` and `-matfree_removal` recipe
carries. That is the point — the pmat is a full one again, so the comparison is
with the full-pmat recipes, not with the streaming-only ones.

### The decades problem files
`slab_decades4.json` and `box_decades4.json` are the new problem files, sharing
`materials/decades4.json`: four groups at `Sigma_t` 0.1, 1, 10 and 100 with
within-group scattering ratios 0.5, 0.6, 0.9 and 0.99 and downscatter into the
next group down only (so the group sweep is still exact), over cells of width
0.1 — 0.01, 0.1, 1 and 10 mean free paths per cell. A 5x density-scaled copy of
the material is painted over a block in the middle, so the problem is
heterogeneous while every group RATIO stays identical in both materials:
`alpha_g * D_ref` is exact per group there, which no scalar shift could be. The
1D file is 100 cells over a length of 10, the 2D one 30x30 over 3x3 — the same
physics one axis wider, in the way the diffusive trio is.

The alphas are 1, 10, 100 and 1000, three decades, which is the point: covering
them exactly costs four hierarchies and the default rule settles for two.
Per-group counts, re-measured 2026-09-25 on the local opt arch under the ghost-flux
default, the Dirichlet-cell counts in brackets where they moved. Serial and `-n 2` agree
on every row measured at both (all but the bare-L and k = 1 rows) except the 2D k = 2
DSA row's third group (9 serial, 8 at `-n 2`); under Dirichlet-cell the exception was
the DSA runs' third group (6 serial, 5 at `-n 2`). The two bare-L rows without DSA run
`-ksp_pc_side right` as their Dirichlet-cell measurement did; the bare-L + DSA row runs
right-preconditioned too:

| `slab_decades4.json` | per-group iterations |
|---|---|
| default pc (full pmat, the reference) | 4, 8, 14, 22 (Dirichlet-cell 4, 8, 14, 20) |
| `-matfree_removal` alone (bare streaming pmat) | 9, 25, 172, **DIVERGED_ITS (300)** (Dirichlet-cell 10, 26, 208) |
| `-precon_ref_shift -precon_ref_k 4` (exact) | 4, 8, 14, 22 (Dirichlet-cell 4, 8, 14, 20) |
| `-precon_ref_shift` (the default rule: k = 2, worst mismatch 3.16) | 13, 11, 28, 44 (Dirichlet-cell 13, 11, 26, 42) |
| `-precon_ref_shift -precon_ref_k 1` (one shared pmat, mismatch 31.6) | 94, 21, 21, 68 (Dirichlet-cell 115, 22, 23, 76) |
| `-precon_dsa` on the full pmat | 4, 4, 5, 11 (Dirichlet-cell 4, 4, 6, 11) |
| `-matfree_removal -precon_dsa` (bare L + DSA) | 11, 30, **DIVERGED_ITS (300)** |
| `-precon_ref_shift -precon_ref_k 4 -precon_dsa` | 4, 4, 5, 11 (Dirichlet-cell 4, 4, 6, 11) |
| `-precon_ref_shift -precon_dsa` (k = 2) | 11, 11, 7, 10 (Dirichlet-cell 11, 11, 7, 13) |

| `box_decades4.json` | per-group iterations |
|---|---|
| default pc (full pmat, the reference) | 4, 6, 15, 29 (Dirichlet-cell 4, 6, 15, 30) |
| `-matfree_removal` alone (bare streaming pmat) | 5, 16, 82, **DIVERGED_ITS (300)** (Dirichlet-cell 6, 17, 89) |
| `-precon_ref_shift -precon_ref_k 4` (exact) | 4, 6, 15, 29 (Dirichlet-cell 4, 6, 15, 30) |
| `-precon_ref_shift` (the default rule: k = 2) | 7, 8, 27, 51 (Dirichlet-cell 7, 8, 24, 48) |
| `-precon_ref_shift -precon_ref_k 4 -precon_dsa` | 4, 4, 6, 11 |
| `-precon_ref_shift -precon_dsa` (k = 2; not a recipe) | 7, 8, 9, 10 (Dirichlet-cell 7, 8, 8, 179) |

Read each table's second row against its third: that pair IS the test. The bare
streaming pmat does not converge on the thick group at all, and the shift is
what makes the mode usable there. **The bare-L rows are deliberately NOT
recipes** — a pin whose expected outcome is divergence would have to be shaped
as a failure, and a recipe's pass/fail is its exit code. They are recorded here
instead, and they are what makes the ref-shift pins mean something.

The mismatch cost is the other thing these tables pin down: at a worst mismatch
of 3.16 the thick group costs 44 against the exact 22 in 1D and 51 against 29 in
2D (Dirichlet-cell: 42 against 20, 48 against 30), i.e. roughly a factor of two. That is worse than the campaign's 1.3-1.6x for
a factor-3 mismatch, and the difference is that the campaign measured one
mismatched group on a homogeneous problem where these have two mismatched groups
at once on a heterogeneous one at 10 mean free paths per cell.

### DSA on a shifted pmat
DSA cannot attach to a bare streaming pmat — structurally, not by degree. Both
`-matfree_removal -precon_dsa` rows above diverge, as `-precon_stream
-precon_dsa` already does (see the DSA section). On an EXACTLY covered shifted
pmat it lands on the full-pmat-plus-DSA counts exactly, 11 on the thick group in
every table above, which is the re-attachment claim.

On a MISMATCHED shifted pmat it was fragile under Dirichlet-cell: 1D at k = 2 was
fine (13 on the thick group against the 42 it took without DSA) while 2D at k = 2
took **179** on the group where `-precon_ref_k 4` takes 11. **The DSA recipes
therefore pin exact coverage only.** Under the ghost-flux default the fragility is
gone on these files — 10 on the thick group in 1D (against 44 without DSA) and 10 in
2D (np 1 and 2), i.e. slightly better than exact coverage — which says the 179 lived
in the Dirichlet boundary rows rather than in the mismatch itself. That is one
problem pair, not a claim; the recipes still pin exact coverage only, and the
amat/pmat interaction recorded for `-precon_stream -precon_dsa` in `TODO.md` is
still open.

### Streaming-only groups
A group whose `Sigma_t` is identically zero has no ratio to any reference — and
needs none, because its operator IS the bare streaming matrix. `RefShiftPmats`
gives every such group one shared unshifted bin whose pmat is the streaming
matrix itself (reference-counted, not copied — it costs no memory beyond its
hierarchy), picks the reference as the first group with removal in every cell,
and bins the rest as usual. Only a group that is zero in SOME cells is refused:
a single ratio cannot represent a field that is removal here and void there.

`slab_decades4_stream0.json` is `slab_decades4.json` with the top group made
pure streaming (`Sigma_t[0] = 0` in both materials, and with it every scatter
out of that group — a zero total admits none, so group 0 is a decoupled
streaming solve):

| `slab_decades4_stream0.json` (ghost-flux; Dirichlet-cell in brackets) | per-group iterations |
|---|---|
| default pc (full pmat, the reference) | 1, 8, 14, 22 (1, 8, 14, 20) |
| `-matfree_removal` alone (bare streaming pmat, left-preconditioned) | 1, 22, 111, **DIVERGED_ITS (300)** (1, 22, 116) |
| `-precon_ref_shift` (default rule — exact here) | 1, 8, 14, 22 (1, 8, 14, 20) |

Group 0 solves in one iteration either way — the bare streaming pmat IS its
exact operator — and the reference falls to group 1, whose three alphas span
two decades, within the default rule's exact reach. The single-group corner is
`slab_st0.json`: every group streaming-only, no reference at all, one unshifted
bin, and the count must equal the matrix-free run without the flag (same pmat,
same everything).

### Pins and slack

| recipe | np=1 | np=2 | pin |
|---|---|---|---|
| 1D `slab_st2`, ref-shift (1 group) | 6 | — | 6 |
| 1D `slab_mg4_t05`, `-precon_ref_k 2` | 6 | 6 | 6 |
| 1D `slab_decades4`, `-precon_ref_k 4` | 22 (Dirichlet-cell 20) | 22 | 22 |
| 1D `slab_decades4`, default k | 44 (Dirichlet-cell 42) | 44 | 44 |
| 1D `slab_decades4`, `-precon_ref_k 4 -precon_dsa` | 11 | 11 | 11 |
| 2D `box_decades4`, `-precon_ref_k 4` | 29 (Dirichlet-cell 30) | 29 | 29 |
| 2D `box_decades4`, default k | 51 (Dirichlet-cell 48) | 51 | 51 |
| 2D `box_decades4`, `-precon_ref_k 4 -precon_dsa` | 11 | 11 | 11 |
| 2D `box_diffusive`, ref-shift, no DSA | 29 | — | 29 |
| 2D `box_diffusive`, ref-shift + DSA | 11 | — | 11 |
| 1D `slab_decades4_stream0`, default k | 22 (Dirichlet-cell 20) | 22 | 22 |
| 1D `slab_st0`, ref-shift (all streaming-only) | 1 | — | 1 |

Every pin sits on the local opt count since the 2026-09-25 ghost-flux re-pin. Before it,
the exact-coverage 1D pins sat on the measured count and everything else carried one
iteration of slack: the DSA rows for the PCGAMG reason the other DSA pins did, the 2D
rows for the reason the other 2D pins did, and the two default-k rows because a PCAIR
hierarchy set up on a deliberately mismatched operator looked like the most
arch-sensitive setup in the suite. These are the rows to check first if CI asks for +1.

The parallel variants are not just duplicates here. The alphas are a log-mean
over every cell, so they come off an MPI reduction, and the binning is then
computed redundantly on every rank; a `-n 2` run that did not reproduce the
serial count would mean the ranks had disagreed about which pmat a group
belongs to.

### Checks that are not recipes
Three guards, all of which exit non-zero, so all of which are checked BY HAND:

- `./transportk -problem problems/slab_mg4_t05.json -precon_ref_shift` must fail
  with the "run the two together" message: the shift repairs the pmat
  `-matfree_removal` leaves behind, and every other mode already has its removal.
- `./transportk -problem problems/slab_mg4_t05.json -matfree_removal -precon_ref_k 2`
  must fail with the "it needs `-precon_ref_shift`" message rather than silently
  ignoring the number.
- the mixed-void guard: give `materials/decades4_stream0.json`'s material 1 a
  POSITIVE group-0 `Sigma_t` (leave material 0's at zero) and run
  `slab_decades4_stream0.json` with the flags — it must fail with the "zero in
  some cells and positive in others" message. A group zero EVERYWHERE is the
  handled streaming-only case above; a group zero SOMEWHERE is a void region,
  which no single ratio can represent — same stance, and the same eventual fix
  (a region-masked operator), as `-precon_dsa`'s void guard above.

## Source and inflow conventions
Both rhs knobs are **isotropic, angle-integrated strengths**, divided by the
quadrature's `sum_weights` on their way onto the ordinates — `/2` in 1D, `/4 pi` in 2D
and 3D — so the same number means the same physics in every dimension. A material's
`Source` is written that way by `UboltFillSource`; a vacuum face's `inflow` is written
that way by the backend, into the per-row Dirichlet values `UboltFillInflow` lays onto
b. Until Aug 2026 `Source` was the literal per-angle rhs entry; the 2026-08-02 baseline
re-capture (the six mg t05/stream logs) and the pin re-measure above are that change
landing.

`inflow` followed a month later. It was one GLOBAL, PER-ANGLE value (a `VecSet` over
the whole rhs, default 1.0), and testing.md used to record that as deliberate — the
incoming ANGULAR flux is per-ordinate by nature. That stance fell when inflow went per
face (2026-08-04): a per-face knob sitting next to a per-face `Source` with the
opposite scaling is an inconsistency nobody can carry in their head, and the papers the
DSA benchmarks reproduce all prescribe an isotropic incident source, which is the
angle-integrated quantity. The top-level `"inflow"` key is gone; a file carrying it
errors with a migration message.

The migration was byte-for-byte. `sum_weights` is the quadrature's exact analytic
measure, not a floating sum (`set_weights(w_h, 2.0)` in 1D, `set_weights(w_h,
4.0 * PETSC_PI)` in 2D and 3D), so writing the exact double of that measure into every
previously-default file makes `inflow / sum_weights` come back to exactly 1.0 — the old
per-angle default — and the rhs is bitwise what it was. Every file that had relied on
the default got `{"type": "vacuum", "inflow": 2.0}` (1D) or `{"type": "vacuum",
"inflow": 12.566370614359172}` (2D and 3D, the round-trip decimal of `4.0 * PETSC_PI`)
on every vacuum face; the cold files (`inflow: 0.0`) simply dropped the key, since 0 is
the new default. Verified by re-running every problem file's `-ksp_monitor` log before
and after and diffing: identical everywhere except `box_crooked_pipe.json`, the one
deliberate physics change (below).

The retired `slab_1dk` had neither knob — its source and inflow were one per-angle
`VecSet(b, 1.0)` by design — and its baselines survive it because `Source 2.0` with
`inflow 2.0` rebuilds that rhs exactly (the 1D single-group problem files all carry
that pair), which was verified byte-for-byte on the unification and again on this
migration.

Two old recipes died with the option surface, both strictly subsumed: the "runtime
phase space sizes" slab run (every size is runtime-from-file on every run now) and the
"multigroup with 1 group" run (a 1-group file IS the single-group path in the unified
driver — there is no separate code path left to compare).

## Recipe cost
CI runs the whole suite on four arches in parallel (opt/debug/64-bit/OpenMP), so the wall
time is the slowest job — the debug one. The suite is what dominates it, and within the
suite 3D dominated everything else: on the reference debug build a single 20^3 cube run
cost 21-34 s against ~2 s for a 50x50 2D run, because cost grows faster than cell count
(20x10x10, at a quarter of 20^3's cells, took 3.5 s to its 21 s).

That is what the 2026-08-02 resize addressed. Measured over the 43 serial recipes of
`run_check` + `run_tests_short_serial`, on the reference debug build:

| | before | after |
|---|---|---|
| the 9 resized recipes in that set | 145 s | 19 s |
| all 43 (the rest are 1D, unchanged) | 200 s | 77 s |

`make check && make tests` end to end after the resize is **272 s** on that build. The
before-figure for the whole suite was never captured cleanly (it ran well past 20
minutes), so the 43-recipe subset above is the like-for-like measurement; the full suite
gains more than the 2.6x it shows, because it also carries the parallel duplicates of
every 3D recipe and the 30^3 mesh-independence run, the single most expensive line in it.

Keep new recipes cheap by default. A 3D recipe wants a reason to be bigger than 10^3, and
the discretisation checks (`verify_2dk`/`verify_3dk`) make their point on 4x3x2 grids —
resolution is not what a pinned iteration count or an operator comparison is testing.

### The OpenMP job and rank/thread oversubscription
The runners have 2 vCPUs and the OpenMP job sets `OMP_NUM_THREADS=2`, which is right for
a serial recipe and wrong for every `mpiexec -n 2` one: 2 ranks x 2 threads is 4 threads
on 2 cores, and the four `-n 4` recipes make it 8. Left alone that cost the job **862 s
of test time against the opt job's 53 s** — the same build, the same problems, a 16x
gap that lived entirely in the parallel recipes (the serial ones were, if anything,
faster than opt).

Two defaults compounded to produce it. OpenMPI confines each rank to a single core at
small rank counts, so a rank's two threads shared one core; and libgomp spins at
parallel-region barriers, so the descheduled thread burned that core rather than
yielding. The result is a per-kernel-launch penalty, not a per-element one, which is why
a 1D slab converging in one iteration took 36 s there against 0.2 s under opt, and why
shrinking meshes barely moved that job while it cut the debug job nearly in half.

`dockerfiles/Dockerfile_kokkos` now sets `OMPI_MCA_hwloc_base_binding_policy=none`,
`OMP_PROC_BIND=false` and `OMP_WAIT_POLICY=PASSIVE`. The job still runs 2 threads per
rank, so it still tests threaded execution under MPI; it just no longer pins them onto
one core or spins while waiting. If a future recipe set makes that job the long pole
again, the next lever is `OMP_NUM_THREADS=1`, which trades the threading coverage away.

## Adding a test problem
1. Write a problem file in `tests/problems/` (schema: `docs/problem_files.md`,
   walkthrough: `docs/problem_setup.md`), with a `"_comment"` saying what it pins.
   Values that are not exact dyadics get 17 significant digits so the file round-trips
   to the intended doubles.
2. Add invocation lines to the appropriate `run_*` recipe in `tests/Makefile`: an
   `@echo` label plus the literal `./transportk -problem problems/<file> -options`
   line, serial and `-n 2` variants, with `-ksp_max_it` pinned to the observed
   converged count.
   `mesh.type` picks the backend: `"structured"` (the default) is the DMDA finite
   difference ones, `"unstructured"` the DG0 plex one (2D/3D, a box or a mesh file,
   no `-precon_dsa`); an unstructured quad/hex box is the natural twin of a structured
   file, and its count goes next to the structured one in "Unstructured iteration
   counts".
3. No output files from any recipe: `output.flux_vtk` (and the `-flux_vtk` override)
   are fine on a problem file you run by hand but must not appear in anything a `run_*`
   recipe names — a test run leaves nothing behind.

## Adding a test driver
Should a second driver ever be needed (the next `verify_*`, say):
1. Add the executable name to `TEST_TARGETS` (and `CHECK_TARGETS` if it belongs in
   `make check`) in the top `Makefile`. The two lists are identical today; the split
   exists for the day a driver is too slow for `check`.
2. Nothing to do for `.gitignore` — `tests/*k` covers every driver binary, since every
   translation unit is a Kokkos one and the executables all end in `k`.
3. Driver rules: exit non-zero unless `KSPGetConvergedReason() > 0`; diagnostics to
   stderr so `-ksp_monitor` stdout stays capture-clean; problem definition through
   `ProblemSpec`, not new physics options.

## DMDA layout (Phases 3 and 4)
Every discretisation backend builds its layout from a DMDA, which is also what decides the
parallel decomposition. There is no second layout path and no option to select one: Phase 3a carried
a hand-rolled twin behind `-ubolt_use_dm` purely to verify the DM extraction, and 3b deleted
it. What replaced the twin as the check is `tests/baselines/` — all 24 logs reproduce
bitwise, which is what 3a and 3b were each verified against.

The DM chooses its own distribution. PETSc's default 1D DMDA split is
`M/size + ((M % size) > rank)` (`src/dm/impls/da/da1.c`), the same formula
`PetscSplitOwnership` uses, so this is the same decomposition UBOLT had when `PhaseSpace`
computed it — 3a confirmed that empirically at np=1,2,3 (including the uneven 334/333/333
split) before 3b handed the decision over.

`CheckDALayout` in `src/structured_fd_1dk.kokkos.cxx` asserts at setup that the DM's sizes
match the phase space and that its global numbering really is the angle-fastest,
contiguous one every COO index is written against. That property holds for 1D DMDA (PETSc
ordering coincides with natural ordering) but it is load-bearing, and a violation would
otherwise surface as a wrong answer rather than an error.

**2D is where that stops being free.** A 2D DMDA numbers each rank's patch contiguously
and *lexicographically within the patch*, so the global natural ordering `j * n_cells_x + i`
is wrong as soon as more than one rank splits a direction. `StructuredFD2D` therefore takes
its COO **columns** from the DM's local-to-global map (which covers the ghost nodes, so a
column can point into a neighbour's patch) rather than computing them, and its own
`CheckDALayout` asserts the weaker property that actually holds: sizes; angle-fastest
contiguous dof; and this rank's owned nodes numbered contiguously from its `rstart` in the
patch's own lexicographic order. That last one is what makes
`local cell index = (j - ys) * xm + (i - xs)` true, which is the indexing the per-cell
xsection views and `RemovalTerm`'s `r / n_angles` both depend on.

**Ordering constraint (Phase 3b).** `PhaseSpace::create` no longer decides the
decomposition — it leaves `local_cells` as `PETSC_DECIDE` and the backend's `create()`
fills it in. So the discretisation must be created *before* anything sized off the phase
space. Everything that reads `local_cells`/`local_rows()` calls `ps.check_decomposed()`
first, which fails with `PETSC_ERR_ARG_WRONGSTATE` and a pointed message rather than
letting a Kokkos view be allocated with a negative extent further downstream.

## np=3 caveat
From Phase 1a the parallel decomposition is decided in cells (rows = local_cells *
n_angles) — by `PetscSplitOwnership` then, by the DMDA since Phase 3, which is the same
split. The pre-refactor code used PETSC_DECIDE over rows, which can split mid-cell
(e.g. 1000 cells x 4 angles on 3 ranks) — so np=3 results legitimately differ from the
pre-refactor binary (np=3, st=2 converges in the same 5 iterations as serial).
Baselines and tests use np=1,2 only, where the decompositions coincide.
