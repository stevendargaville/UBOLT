# Testing in UBOLT

This file is the testing POLICY. Its companions: `docs/dev/verification.md` (what every
check that is not a pinned count proves), `docs/dev/iteration_counts.md` (the pins, one
table per backend) and `docs/dev/history.md`, Testing history (the dated count tables and
the measurements behind each pin). The section map at the end says where each heading
this file used to carry went.

## Pass/fail contract
- Pass/fail is the process exit code — no output diffing, no log grepping (this must stay
  true so CI can just run `make tests`, which it does — see
  `.github/workflows/ci_build.yml`). `make` stops on the first non-zero exit.
- Every solve recipe in `tests/Makefile` pins `-ksp_max_it` to the baseline iteration
  count, so an iteration-count regression fails the run. A pin is the count measured on
  the local opt arch (`arch-linux-c-opt`), exactly, with no per-arch slack: every pin was
  re-set that way 2026-09-25 for the ghost-flux vacuum default (see "Switching the
  default" in `docs/dev/history.md`). The same recipes run on every CI arch
  (opt/debug/64-bit/OpenMP), so CI flags any arch that needs one more, and at that point
  the pin takes the max over the arches. The first CI run after the re-pin did that for
  four recipes: the 1D ghost-flux and 2D all-reflective `-matfree_removal` inf-medium
  checks (64-bit, 17 -> 18 and 24 -> 25), the 2D ghost-flux inf-medium check at np=2
  (OpenMP, 11 -> 12) and the crooked pipe at np=2 (OpenMP, 114 -> 117). Every recipe
  carrying such a CI +1 today is marked in the note column of
  `docs/dev/iteration_counts.md`. Older per-arch notes in the history ("pinned 10") are
  the history of the pre-2026-09-25 policy.
- Multigroup caveat: `-ksp_max_it` is one option for the whole group sweep, so a
  multigroup recipe pins the **max over the groups**. A single group getting slower
  without exceeding that max will not fail the recipe — the multigroup baselines below
  are what pin the per-group counts.
- The drivers return 1 unless `KSPGetConvergedReason() > 0` (since Phase 1a), so a
  non-converged solve fails the recipe on its own — no `-ksp_error_if_not_converged`
  or `-on_error_abort` is needed for pass/fail (CI sets `-on_error_abort` in
  `PETSC_OPTIONS` anyway, belt and braces against a hang inside PETSc). Driver
  diagnostics go to stderr so they never appear in a captured baseline. The four st=2
  `-diag_scale` lines of `capture_baselines` carry `|| true` (see "Single-group
  baselines" below for which of them still fail to converge).
- Targets: `make check` (fast sanity, 18 runs: the five `verify_*` drivers and 13 solves)
  < `make tests_short` < `make tests` (`tests_short`, then `run_tests_serial` and
  `run_tests_parallel`). Parallel variants use `$(MPIEXEC) -n 2` — plus `-n 4` runs of
  the two painting identities and of the `verify_*` drivers — and are skipped under
  MPIUNI, including `run_check`'s single parallel line. `tests_short` is most of the
  suite (353 of the 370 recipe lines `make tests` runs); the name is historical.
- The `verify_*` drivers are the exception to "pass/fail is a pinned iteration count":
  they are discretisation and quadrature checks and their exit codes come from tolerances
  on numerical checks (`docs/dev/verification.md`). They print what they measured against
  the tolerance, so a run that is drifting towards failure is visible before it fails.

## Pins: where they live and how they are set
- **`tests/Makefile` is the only source of truth.** `docs/dev/iteration_counts.md` mirrors
  it, one table per backend; the narrative docs quote counts only as dated measurements
  (history), never as the pin. Change the Makefile first, then the table row.
- **The CI-arch rule.** Pin the local opt count exactly. Do not pre-emptively add slack
  and do not sweep the CI arches by hand before pushing: CI runs every recipe on the opt,
  debug, 64-bit and OpenMP images and goes red on any arch that needs one more. Then
  sweep that image's affected recipes with their pins lifted, raise exactly the ones it
  needs (the pin becomes the max over the arches), and record the local count and the
  image in the note column of `iteration_counts.md`. The counts that clear rtol on their
  last iteration by a percent or two are the ones that do this: a perturbation far too
  small to be a regression (a different compiler, index size or thread count) decides
  whether that iteration counts (the 2026-08-04 quadrature fix showed it; history, "2D
  iteration counts").
- **How to re-pin** after a change that is meant to move counts:
  1. Re-measure every affected recipe with its pin lifted (run the recipe's command line
     by hand with a large `-ksp_max_it`, e.g. 400, and `-ksp_converged_reason`), on the
     local opt arch, on the branch AND on main, so every move is attributed to the change
     rather than to the environment.
  2. Pin each moved recipe on the branch's local opt count. A pin that carries a CI +1
     and whose local count did not move keeps its pin.
  3. Record the moves as a dated before/after table under the feature's section of
     `docs/dev/history.md`, Testing history, and update `iteration_counts.md`.
  4. A count that went UP must be explained by the change. A pin is never raised to make
     an unexplained regression pass.
- **Adding a recipe**: see "Adding a test problem" below; the new row goes into its
  backend's table in `iteration_counts.md`.

## Bitwise capture for a numerical refactor
A change that claims to be numerically inert (a refactor, a move of code between
translation units, a kernel rewrite that keeps the arithmetic) has to show it, not argue
it. The procedure the Sep 2026 refactor rounds used:
1. On the starting commit, on the arch you will verify on, capture
   `PETSC_OPTIONS=-ksp_monitor make tests_short` to a log outside the tree. Every recipe's
   command line is echoed before its `KSP Residual norm` lines, so the log is keyed by
   recipe.
2. On the change, the same command on the same arch, to a second log.
3. Diff the `KSP Residual norm` lines recipe by recipe. They must be IDENTICAL, byte for
   byte. A non-empty diff on a run means the change is not bit-identical: find the cause,
   and if it is a floating-point reordering (a sum or a reduction in a different
   order), undo it or take it out of the "inert"
   claim and re-pin it as a numerical change.
4. The `verify_*` drivers print every measurement they check; run the ones the change
   touches serially and at `-n 2` on both commits and diff their output.
5. Where the change reaches the 1D path, the 24 logs in `tests/baselines/` are the
   longer-lived oracle (below). Run the `capture_baselines` command lines by hand with the
   output redirected to a scratch directory - NOT `make baselines`, which overwrites the
   tracked files - and diff against the baseline of the same np.
6. Pin-lifted histories are the stronger form for a feature whose recipes might move:
   the DSA void-masking and void-bridging changes compared every `-precon_dsa` recipe,
   pin lifted, with `-ksp_monitor`, against main's, and found them identical.

## Baselines (tests/baselines/)
Captured `-ksp_monitor -ksp_converged_reason` logs. They are the ground truth for a
refactor: identical numerics means the residual history diffs clean against these files
(iteration counts exactly, residuals to ~1e-14).

24 logs in two families: 16 single-group (`slab1d_*`, Phase 0) and 8 multigroup
(`slab1dmg_*`, Phase 2). `make baselines` captures both. A refactor that claims to be
numerically inert must diff clean against all 24. All of them are 1D — the 2D/3D/plex checks are
sharper and need no baseline, see `docs/dev/verification.md`.

Since the driver unification (Aug 2026) every capture line runs `transportk` on a
problem file (`tests/problems/`); the log names keep their `slab1d`/`slab1dmg` history.
The unification itself was verified the strong way before the old drivers
(`slab_1dk`, `slab_1d_mgk`, `box_2dk`) were deleted: all 24 logs byte-for-byte from the
new driver, plus the t0 structural identity, the painting identity at np=1 and np=4, and
the 1D/2D infinite-medium checks.

### Single-group baselines
Re-captured 2026-09-25 for the switch to the ghost-flux vacuum treatment as the default
(the capture history, with every older count, is in `docs/dev/history.md`, "Single-group
baseline iteration counts"; the current counts are in `docs/dev/iteration_counts.md`). Before that, 2026-08-04 for the quadrature precision fix, then
2026-08-01 after the Dirichlet-row fix to the matrix-free scatter, 2026-07-31 after the
negative-angle upwind sign fix, and before *that* they
came from the pre-refactor code (the single-file `UBOLTk.kokkos.cxx`); Phase 1a and Phase
1b each reproduced all 16 bitwise, which is what those phases were verified against. The
`-ubolt_coo_two_call` assembly fallback reproduces them bitwise.

- Full matrix: {default pc, `-precon_stream -ksp_pc_side right`} x {`slab_st0.json`,
  `slab_st2.json`} x {np 1, 2} x {`-diag_scale` off, on}. File naming:
  `slab1d_<default|stream>_st<0|2>_np<1|2>[_ds].log`.
- Capture is capped at `-ksp_max_it 200`: all four st=2 diag_scale configs are pathological
  and a 200-iteration residual history is a strong enough fingerprint.
  Non-converged baselines are still valid fingerprints for a refactor diff. Under
  Dirichlet-cell none of those four converged, so all four capture lines carry
  `|| true`; under ghost-flux the default-pc pair scrapes in at 174/173 and the
  streaming-pmat pair hits the cap. The `|| true` is still on all four lines (harmless on
  the pair that converges), and the comment above `capture_baselines` in `tests/Makefile`
  still says none of the four converges, which is the Dirichlet-cell-era statement.
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

### Multigroup baselines (Phase 2)
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
change (`docs/dev/history.md`, "Source and inflow conventions"), when the two t0 logs deliberately did not move.

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
   converged count on the local opt arch (see "Pins" above), and add the row to its
   backend's table in `docs/dev/iteration_counts.md`. What the count is compared against
   (the reference without the feature, the twin on another backend) goes, dated, under
   the feature's section of `docs/dev/history.md`, Testing history.
   `mesh.type` picks the backend: `"structured"` (the default) is the DMDA finite
   difference ones, `"unstructured"` the DG plex one (2D/3D, a box or a mesh file,
   `mesh.order` 1 for DG1, `mesh.discretisation` `"cg_supg"` for CG-SUPG); an unstructured quad/hex box is the
   natural twin of a structured file, and a DG1 or CG file is the twin of the DG0 one of
   the same name, which is what its count is read against.
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
column can point into a neighbour's patch) rather than computing them, and
`CheckDALayout<DIM>` (shared by 2D and 3D, in `src/structured_fd_commonk.hpp`) asserts the weaker property that actually holds: sizes; angle-fastest
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


## Section map (where this file's old headings went, Sep 2026)
Code comments, `tests/Makefile` and older docs cite "docs/dev/testing.md, <heading>". Those
headings now live here:

| old heading in testing.md | now in |
|---|---|
| Pass/fail contract; Baselines; Single-group baselines; Multigroup baselines (Phase 2); Recipe cost; The OpenMP job; Adding a test problem / driver; DMDA layout; np=3 caveat | this file |
| Single-group baseline iteration counts | `history.md` (current counts: `iteration_counts.md`) |
| Quadrature verification; 2D verification (Phase 4); Unequal quadrature weights; 3D verification | `verification.md` |
| Reflective boundary conditions | the infinite-medium oracle and the singularity constraint: `verification.md`; the counts: `history.md` |
| Ghost-flux vacuum treatment | the checks: `verification.md`; the counts, "Switching the default (2026-09-25)" and "Ghost-flux reflective faces (2026-09-26)": `history.md` |
| Painted regions (MaterialSpec) | the painting-identity and overlap oracles: `verification.md`; the counts: `history.md` |
| Unstructured DG verification | `verification.md` (with checks 12 and 13, which were under "DSA with voids") |
| CG-SUPG verification and iteration counts ("CG-SUPG") | checks 1-8: `verification.md`, "CG-SUPG verification"; the counts, the LU split, the zeta and cell-size sweeps and the CG DSA: `history.md` |
| 2D iteration counts; 3D iteration counts; Unstructured iteration counts | `history.md` |
| Matrix-free removal (`-matfree_removal`) | the mode and its counts (the `-precon_stream` twins agree on count, not history): `history.md`; `-check_matfree`: `verification.md` |
| DSA (the diffusion correction, `-precon_dsa`); Discretisation-consistent D; Literature benchmark pins; DSA with voids | `history.md`; the not-a-recipe singularity guard: `verification.md` |
| The reference-shifted streaming pmat (`-precon_ref_shift`); The decades problem files; DSA on a shifted pmat; Streaming-only groups; Voids; Pins and slack | `history.md`; `-check_ref_shift` and the not-a-recipe guards: `verification.md` |
| Source and inflow conventions | `history.md` (the conventions themselves: `docs/problem_files.md`) |

Topics cited from code by name: the all-reflective singular constraint -
`verification.md`, "Reflective boundary conditions"; the Marshak-coefficient scan (0.25-1.0
moves no DSA count by more than 1) - `history.md`, "Switching the default"; a matfree
recipe equal in count to its `-precon_stream` twin - `history.md`, "Matrix-free removal".
