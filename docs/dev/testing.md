# Testing in UBOLT

## Pass/fail contract
- Pass/fail is the process exit code — no output diffing, no log grepping (this must stay
  true so CI can later just run `make tests`). `make` stops on the first non-zero exit.
- Every recipe in `tests/Makefile` pins `-ksp_max_it` to the baseline iteration count, so
  an iteration-count regression fails the run.
- Multigroup caveat: `-ksp_max_it` is one option for the whole group sweep, so a
  multigroup recipe pins the **max over the groups**. A single group getting slower
  without exceeding that max will not fail the recipe — the multigroup baselines below
  are what pin the per-group counts.
- The drivers return 1 unless `KSPGetConvergedReason() > 0` (since Phase 1a), so a
  non-converged solve fails the recipe on its own — no `-ksp_error_if_not_converged`
  or `-on_error_abort` needed. Driver diagnostics go to stderr so they never appear in a
  captured baseline. The two non-converging `capture_baselines` lines carry `|| true`.
- Targets: `make check` (fast sanity, 5 runs) < `make tests_short` < `make tests`.
  Parallel variants use `$(MPIEXEC) -n 2` — plus one `-n 4` run, see the 2D section — and
  are skipped under MPIUNI.
- `tests/verify_2dk` is the exception to "pass/fail is a pinned iteration count": it is a
  discretisation check and its exit code comes from tolerances on two numerical checks
  (below). It prints what it measured against the tolerance, so a run that is drifting
  towards failure is visible before it fails.

## Baselines (tests/baselines/)
Captured `-ksp_monitor -ksp_converged_reason` logs. They are the ground truth for a
refactor: identical numerics means the residual history diffs clean against these files
(iteration counts exactly, residuals to ~1e-14).

24 logs in two families: 16 single-group (`slab1d_*`, Phase 0) and 8 multigroup
(`slab1dmg_*`, Phase 2). `make baselines` captures both. A refactor that claims to be
numerically inert must diff clean against all 24. All of them are 1D — the 2D checks are
sharper and need no baseline, see below.

### Single-group baselines
Re-captured 2026-07-31 after the negative-angle upwind sign fix (see below). Before that
they came from the pre-refactor code (the single-file `UBOLTk.kokkos.cxx`), and Phase 1a
and Phase 1b each reproduced all 16 of them bitwise — which is what the phases were
verified against. The `-ubolt_coo_two_call` assembly fallback also reproduces them bitwise.

- Full matrix: {default pc, `-precon_stream -ksp_pc_side right`} x {`-max_exponent` 0, 2}
  x {np 1, 2} x {`-diag_scale` off, on}. File naming:
  `slab1d_<default|stream>_me<0|2>_np<1|2>[_ds].log`.
- Capture is capped at `-ksp_max_it 200`: the me=2 diag_scale configs are pathological
  (see table) and a 200-iteration residual history is a strong enough fingerprint.
  Non-converged baselines are still valid fingerprints for a refactor diff.
- Re-capture with `make baselines` — only do this deliberately (i.e. when the reference
  behavior itself is being intentionally changed), never to make a failing test pass.
- Environment matters for exact reproduction: these were captured with a debug PETSc main
  build (`arch-linux-c-debug`), gcc 13, OpenMPI, np as named. Different
  compilers/optimization may shift trailing digits; iteration counts should still match.
- np=1 and np=2 logs differ in the trailing digits (parallel reduction order); compare a
  log against the baseline for the *same* np, never across np.

## Single-group baseline iteration counts
Captured 2026-07-31, capped at `-ksp_max_it 200`. The pre-fix column is what the
pre-refactor code did, kept because Phases 1a and 1b were verified against it:

| config | np=1 | np=2 | pre-fix (np=1, np=2) |
|---|---|---|---|
| default pc, me=0 | 1 | 1 | 1, 1 |
| default pc, me=2 | 5 | 5 | 10, 10 |
| default pc, me=0, diag_scale | 1 | 1 | 1, 1 |
| default pc, me=2, diag_scale | 175 | 173 | DIVERGED_ITS (6305 uncapped) |
| precon_stream, me=0 | 1 | 1 | 1, 1 |
| precon_stream, me=2 | 10 | 10 | 16, 16 |
| precon_stream, me=0, diag_scale | 3 | 3 | 3, 3 |
| precon_stream, me=2, diag_scale | DIVERGED_BREAKDOWN (90) | DIVERGED_BREAKDOWN (90) | DIVERGED_ITS |

History of these baselines:
- **Negative-angle upwind sign** (FIXED after Phase 1b, this re-capture): the streaming
  term wrote the upwind neighbour coefficient as `-mu/dx`, correct for `mu > 0` but the
  wrong sign for `mu < 0`, so those rows were `|mu|/dx (psi_i + psi_i+1)` instead of
  `|mu|/dx (psi_i - psi_i+1)`. Fixing it makes the interior streaming rows sum to zero,
  and every well-behaved config converges in fewer iterations (me=2 went 10 -> 5, the
  streaming-pmat me=2 16 -> 10). Phase 1 deliberately preserved the bug so the refactor
  could be verified bit-for-bit, then fixed it in one commit of its own.
- **Parallel out-of-bounds bug in the matrix-free scatter** (FIXED, Phase 1a-pre): the
  original code reshaped the LOCAL vector with the GLOBAL `N_CELLS` and looped kernels
  over `N_CELLS`, so at np>1 it read past the local `x`/`sigma_s` arrays and wrote past
  the end of local `y`, sending `default pc, me=2, np=2` to NaN.
- **diag_scale + strong removal is still pathological** (175 its, or breakdown): diagonal
  scaling makes the removal shell PC an identity and degrades the composite. The sign fix
  improved it a lot but did not cure it. Known current behavior, not a target — excluded
  from pass/fail recipes; revisit when the diag_scale semantics are looked at again.

## Multigroup baselines (Phase 2)
`slab1dmg_<default|stream>_me<0|2>_g4_t<0|05>_np<1|2>.log` — `slab_1d_mgk` with 4 groups,
`t0`/`t05` being `-sigma_transfer` 0.0/0.5. Each log is the whole group sweep: one KSP
history after another, in group order. That is what pins the per-group iteration counts
and the downscatter source arithmetic, neither of which a single `-ksp_max_it` can reach.

The `t0` logs are exactly four byte-for-byte copies of the matching single-group
`slab1d_default_me2_np<n>.log`. That is the Phase 2 uncoupled-groups verification, and it
is worth re-checking as a structural property rather than only diffing the file — a change
that broke both files the same way would still diff clean.

| multigroup config | per-group iterations (np=1 and np=2) |
|---|---|
| default pc, me=0, transfer 0.5 | 1, 1, 1, 1 |
| default pc, me=2, transfer 0.0 | 5, 5, 5, 5 (= single group, four times) |
| default pc, me=2, transfer 0.5 | 5, 6, 6, 6 |
| default pc, me=2, transfer 2.0 | 5, 7, 6, 6 (recipe only, not captured) |
| precon_stream, me=2, transfer 0.5 | 12, 13, 13, 11 |

Group 0 matches the single-group count because its rhs is just the external source; the
later groups carry a downscatter source as well. The last group has no outgoing transfer,
so its `sigma_t` is lower than the others' — that is why the counts are not uniform.

With `-precon_stream` the streaming-only pmat does not depend on the group, so PCAIR is
set up once for the whole sweep; with the default pc the assembled matrix is refilled per
group and PCAIR re-runs its setup each time. Both are exercised by the recipes.

## 2D verification (Phase 4)
There are **no 2D baselines**, and that is deliberate. A residual-history baseline is an
indirect fingerprint of the numerics; in 2D `tests/verify_2dk` compares the operator
itself, so it is a strictly sharper oracle and there is nothing left for a baseline to
catch. `make baselines` is still 1D + multigroup only.

`verify_2dk` runs two checks, both S2 (4 angles) and S4 (12), and exits non-zero on either:

1. **Pure-streaming closed form.** `psi = x + y` is linear, and first-order upwind
   differencing is exact on a linear function, so the *discrete* operator has the PDE's
   solution: `mu dpsi/dx + eta dpsi/dy = mu + eta`, with `psi = x + y` prescribed on the
   inflow boundaries. Two halves — the residual `||A psi_exact - b||_inf` (no solver
   tolerance in the way; ~4e-15 against a 1e-12 tolerance) and then the solve, which says
   the system really does have that solution and is nonsingular. Run on a 16x12 grid over
   a 1x2 box, so nothing about it is symmetric in x and y and a mix-up cannot hide.
2. **Shell operator vs a reference matrix.** `MatComputeOperator` on the *shell*, so the
   matrix-free scatter is included, against a matrix built entry by entry in the driver on
   a 4x3 grid. Currently agrees to 0.0, i.e. bitwise. Serial only: the reference is
   written in the natural ordering, which is what a 2D DMDA uses on one rank.
   The reference deliberately reimplements the inflow test rather than reading the
   discretisation's Dirichlet mask — a check that asked the code under test which rows it
   thought were boundary rows would agree with it by construction.
   It also models the scatter on the Dirichlet rows, because that is what the code does;
   see the Phase 4 postscript in `TODO.md`.

**Parallel.** A 2D DMDA at `-n 2` splits in one direction only, so `-n 4` is the first
decomposition with a genuinely 2D processor grid and the one where the patch-lexicographic
global numbering is least like the natural one. Both are in `run_tests_short_parallel`.

## 2D iteration counts (`box_2dk`)
`-max_exponent` sets `sigma_t`; `-sigma_scatter` sets `sigma_s`, defaulting to the same
value (scattering ratio 1). Ratio 1 is the hardest case for a preconditioner that does
nothing about the scattering, and in 2D it is mesh independent only on a square grid —
see the research note in `TODO.md`. Recipes cover ratio 0.5 on every shape and ratio 1 on
square grids only.

| config | np=1 | np=2 |
|---|---|---|
| 50x50, me=2, ratio 0.5 | 8 | 8 |
| 60x30 (dx != dy), me=2, ratio 0.5 | 8 | 8 |
| 80x20 in a 1x0.25 box (dx == dy), me=2, ratio 0.5 | 9 | 10 |
| 120x60, me=2, ratio 0.5 | 8 | 8 |
| 60x60 S4, me=2, ratio 0.5 | 8 | 8 |
| 120x60, streaming-only pmat, me=2, ratio 0.5 | 20 | — |
| 50x50, me=0 (pure streaming) | 3 | 3 |
| 100x100, me=0 | 3 | 3 |
| 50x50, me=2, ratio 1 | 18 | 20 |
| 100x100, me=2, ratio 1 | 21 | 24 |
| 50x50, streaming-only pmat, me=2, ratio 1 | 46 | 54 |
| 30x30 S4, me=2, ratio 1 | 28 | 28 |

## Adding a test driver
1. Add the executable name to `TEST_TARGETS` (and `CHECK_TARGETS` if it belongs in
   `make check`) in the top `Makefile`.
2. Nothing to do for `.gitignore` — `tests/*k` covers every driver binary, since every
   translation unit is a Kokkos one and the executables all end in `k`.
3. Add invocation lines to the appropriate `run_*` recipe in `tests/Makefile`:
   an `@echo` label plus the literal `./exe -options` line, serial and `-n 2` variants,
   with `-ksp_max_it` pinned to the observed converged count.
4. Driver rules: exit non-zero unless `KSPGetConvergedReason() > 0`; no output files.

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
pre-refactor binary (np=3, me=2 converges in the same 5 iterations as serial).
Baselines and tests use np=1,2 only, where the decompositions coincide.
