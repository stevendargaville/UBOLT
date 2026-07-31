# Testing in UBOLT

## Pass/fail contract
- Pass/fail is the process exit code — no output diffing, no log grepping (this must stay
  true so CI can later just run `make tests`). `make` stops on the first non-zero exit.
- Every recipe in `tests/Makefile` pins `-ksp_max_it` to the baseline iteration count, so
  an iteration-count regression fails the run.
- Until the drivers check `KSPGetConvergedReason` themselves (Phase 1a), recipes add
  `-ksp_error_if_not_converged -on_error_abort` so a non-converged solve aborts with a
  non-zero exit code. Phase 1a moves this into the driver (exit 1 on reason <= 0) and the
  two options can then be dropped from the recipes.
- Targets: `make check` (fast sanity, 2 runs) < `make tests_short` < `make tests`.
  Parallel variants use `$(MPIEXEC) -n 2` and are skipped under MPIUNI.

## Baselines (tests/baselines/)
Captured `-ksp_monitor -ksp_converged_reason` logs from the pre-refactor code
(the single-file `UBOLTk.kokkos.cxx`, now `tests/slab_1dk.kokkos.cxx` unmodified).
They are the ground truth for the Phase 1 refactor: identical numerics means the residual
history of the refactored driver diffs clean against these files (iteration counts exactly,
residuals to ~1e-14).

- Full matrix: {default pc, `-precon_stream -ksp_pc_side right`} x {`-max_exponent` 0, 2}
  x {np 1, 2} x {`-diag_scale` off, on}. File naming:
  `slab1d_<default|stream>_me<0|2>_np<1|2>[_ds].log`.
- Capture is capped at `-ksp_max_it 200`: two diag_scale configs are pathological under
  the current code (see table) and a 200-iteration residual history is a strong enough
  fingerprint. Non-converged baselines are still valid fingerprints for the refactor diff.
- Re-capture with `make baselines` — only do this deliberately (i.e. when the reference
  behavior itself is being intentionally changed), never to make a failing test pass.
- Environment matters for exact reproduction: these were captured with a debug PETSc main
  build (`arch-linux-c-debug`), gcc 13, OpenMPI, np as named. Different
  compilers/optimization may shift trailing digits; iteration counts should still match.

## Baseline iteration counts
Captured 2026-07-31 from the pre-refactor code (capture capped at `-ksp_max_it 200`):

| config | np=1 | np=2 |
|---|---|---|
| default pc, me=0 | 1 | 1 |
| default pc, me=2 | 10 | 10 |
| default pc, me=0, diag_scale | 1 | 1 |
| default pc, me=2, diag_scale | DIVERGED_ITS (6305 uncapped) | DIVERGED_ITS |
| precon_stream, me=0 | 1 | 1 |
| precon_stream, me=2 | 16 | 16 |
| precon_stream, me=0, diag_scale | 3 | 3 |
| precon_stream, me=2, diag_scale | DIVERGED_ITS | DIVERGED_ITS |

Known issues captured in these baselines:
- **Parallel out-of-bounds bug in the matrix-free scatter** (FIXED, Phase 1a-pre): the
  original code reshaped the LOCAL vector with the GLOBAL `N_CELLS` and looped kernels
  over `N_CELLS`, so at np>1 it read past the local `x`/`sigma_s` arrays and wrote past
  the end of local `y`, sending `default pc, me=2, np=2` to NaN. After the fix (local
  cell counts everywhere in the scatter path) np=1 residual histories were verified
  bitwise identical, every np=2 baseline except the two previously-NaN default/me=2 logs
  was bitwise unchanged on re-capture, and the repaired config converges in 10 iterations
  matching serial.
- **diag_scale + strong removal is pathological** (6305 its / divergence): diagonal
  scaling makes the removal shell PC an identity and degrades the composite. Known
  current behavior, not a target — excluded from pass/fail recipes; revisit when the
  diag_scale semantics are looked at again.

## Adding a test driver
1. Add the executable name to `TEST_TARGETS` (and `CHECK_TARGETS` if it belongs in
   `make check`) in the top `Makefile`.
2. Add the executable to `.gitignore`.
3. Add invocation lines to the appropriate `run_*` recipe in `tests/Makefile`:
   an `@echo` label plus the literal `./exe -options` line, serial and `-n 2` variants,
   with `-ksp_max_it` pinned to the observed converged count.
4. Driver rules: exit non-zero unless `KSPGetConvergedReason() > 0`; no output files.

## np=3 caveat
From Phase 1 the parallel decomposition is decided in cells (rows = local_cells * n_angles).
The pre-refactor code used PETSC_DECIDE over rows, which can split mid-cell (e.g. 1000
cells x 4 angles on 3 ranks) — so np=3 results legitimately differ from the pre-refactor
binary. Baselines and tests use np=1,2 only, where the decompositions coincide.
