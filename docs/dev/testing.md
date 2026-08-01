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
  captured baseline. The four non-converging `capture_baselines` lines carry `|| true`.
- Targets: `make check` (fast sanity, 2 runs) < `make tests_short` < `make tests`.
  Parallel variants use `$(MPIEXEC) -n 2` and are skipped under MPIUNI.

## Baselines (tests/baselines/)
Captured `-ksp_monitor -ksp_converged_reason` logs. They are the ground truth for a
refactor: identical numerics means the residual history diffs clean against these files
(iteration counts exactly, residuals to ~1e-14).

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

### Multigroup baselines (added in Phase 2)
`slab1dmg_<default|stream>_me<0|2>_g4_t<0|05>_np<1|2>.log` — `slab_1d_mgk` with 4 groups,
`t0`/`t05` being `-sigma_transfer` 0.0/0.5. Each log is the whole group sweep: one KSP
history after another, in group order, so it pins the per-group iteration counts and the
downscatter source arithmetic that no single `-ksp_max_it` can.

The `t0` logs are exactly four byte-for-byte copies of the matching single-group
`slab1d_default_me2_np<n>.log`. That is the Phase 2 uncoupled-groups verification, and
it is worth re-checking directly rather than only diffing the file.

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

## Baseline iteration counts
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

## Adding a test driver
1. Add the executable name to `TEST_TARGETS` (and `CHECK_TARGETS` if it belongs in
   `make check`) in the top `Makefile`.
2. Nothing to do for `.gitignore` — `tests/*k` covers every driver binary, since every
   translation unit is a Kokkos one and the executables all end in `k`.
3. Add invocation lines to the appropriate `run_*` recipe in `tests/Makefile`:
   an `@echo` label plus the literal `./exe -options` line, serial and `-n 2` variants,
   with `-ksp_max_it` pinned to the observed converged count.
4. Driver rules: exit non-zero unless `KSPGetConvergedReason() > 0`; no output files.

## np=3 caveat
From Phase 1a the parallel decomposition is decided in cells (rows = local_cells *
n_angles). The pre-refactor code used PETSC_DECIDE over rows, which can split mid-cell
(e.g. 1000 cells x 4 angles on 3 ranks) — so np=3 results legitimately differ from the
pre-refactor binary (np=3, me=2 converges in the same 5 iterations as serial).
Baselines and tests use np=1,2 only, where the decompositions coincide.
