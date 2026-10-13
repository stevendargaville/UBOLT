# Pinned iteration counts

`tests/Makefile` is the ONLY source of truth for a pin: every solve recipe passes `-ksp_max_it <pin>`. This file tabulates those pins, one table per backend, as the Makefile has them on 2026-10-13, generated from its recipe lines (420 `transportk` lines, 257 distinct problem + option combinations; the `verify_*` lines have no pin, see `docs/dev/verification.md`). If a number here disagrees with the Makefile, the Makefile wins and this row is stale: re-pin in the Makefile first, then copy the number here.

How to read the tables:
- A pin is the count measured on the local opt arch (`arch-linux-c-opt`), exactly, unless the note column says a CI image needed more (the pin policy is in `docs/dev/testing.md`).
- Every count is a right-preconditioned one (`TransportSolver`'s default since 2026-10-13: each solve stops on the true residual). Every pin was re-set then; the old (left-preconditioned) -> new table is in `docs/dev/history.md`, "Right preconditioning by default". The CI +1 notes of the recipes that moved were dropped with it (their pins are the new local counts until CI says otherwise).
- A multigroup recipe pins the MAX over its groups: `-ksp_max_it` is one option for the whole sweep. The per-group counts are in the history tables.
- An upscatter recipe (note "upscatter") pins two numbers, written `k / n`: `-ksp_max_it k`, the MAX over every inner solve of every sweep (under the default adaptive inner rtol, `-outer_inner_gamma 0.01`, usually a fast group's, solved once from zero at the inner rtol; the thermal solves are loose early and warm started later), and `-outer_max_it n`, the number of sweeps over the thermal groups for the error estimate change * rho / (1 - rho) to reach `-outer_rtol`, the first sweep included. Recipes without `-ksp_rtol` / `-outer_rtol` run at the upscatter defaults (inner 1e-10, outer 1e-8); the `-outer_inner_gamma 0` rows run every solve at the inner rtol and keep the pre-adaptive pins. The "moments store" rows (`-outer_store moments`) keep only each group's scalar flux and current between sweeps and start a thermal solve from a reconstructed psi; their sweep counts are the angular twins', and `-ksp_max_it` matches the twin's or drops by one where the max was a thermal solve (the oracles); the reconstruction V-cycles are not in either number (stderr reports them).
- "suite": `check` is `run_check`, `short` is `run_tests_short_serial` / `_parallel`, `full` is `run_tests_serial` / `_parallel` (run by `make tests` after `tests_short`). "needs triangle / tetmesher": a generated simplex box, skipped where PETSc has no triangle / (c)tetgen - the CI images have neither.
- np=4 appears only on the painting identities (and on the verify drivers).
- Options are the recipe's own; `-problem problems/<problem>.json`, `-ksp_max_it` and `-outer_max_it` are left out.

The measurements behind these numbers - what each count was before and after the change that moved it, the no-DSA references, the LU splits, the sweeps - are dated narrative in `docs/dev/history.md`, Testing history:
- 1D/2D/3D structured: "Single-group baseline iteration counts", "2D iteration counts", "3D iteration counts", "Reflective boundary conditions: counts", "Ghost-flux vacuum treatment: counts" (and its "Switching the default", "Ghost-flux reflective faces"), "Painted regions", "Matrix-free removal".
- DSA on any backend: "DSA (the diffusion correction, `-precon_dsa`)" and its "Discretisation-consistent D", "Literature benchmark pins", "DSA with voids".
- `-precon_ref_shift` on any backend: "The reference-shifted streaming pmat" and its "The decades problem files", "DSA on a shifted pmat", "Streaming-only groups", "Voids", "Pins and slack".
- Unstructured DG0 / DG1: "Unstructured iteration counts" (twins, element-block scaling, DG1, plex DSA, the simplex creep, hair-trigger pins).
- CG-SUPG: "CG-SUPG verification and iteration counts: the counts" (the LU split, the zeta and cell-size sweeps, the CG DSA).

## The 24 baseline logs (`tests/baselines/`, captured 2026-10-13)

Not pins: `make baselines` captures these at `-ksp_max_it 200` and a refactor diffs against them (`docs/dev/testing.md`). Final iteration count of each log, np=1 / np=2 (multigroup: per group). Re-captured 2026-10-13 for right preconditioning: the 10 `stream` logs were already right preconditioned and are byte-identical; the 14 `default` logs record the true residual now (2026-09-25 counts: `slab1d_default_st2` 6, `_ds` 174 / 173, the st2 multigroup logs 6 per group):

| log | np=1 | np=2 |
|---|---|---|
| `slab1d_default_st0` | 1 | 1 |
| `slab1d_default_st2` | 5 | 5 |
| `slab1d_default_st0_ds` | 1 | 1 |
| `slab1d_default_st2_ds` | DIVERGED_ITS (200) | DIVERGED_ITS (200) |
| `slab1d_stream_st0` | 1 | 1 |
| `slab1d_stream_st2` | 8 | 8 |
| `slab1d_stream_st0_ds` | 2 | 2 |
| `slab1d_stream_st2_ds` | DIVERGED_ITS (200) | DIVERGED_ITS (200) |
| `slab1dmg_default_st0_g4_t05` | 1, 1, 1, 1 | 1, 1, 1, 1 |
| `slab1dmg_default_st2_g4_t0` | 5, 5, 5, 5 | 5, 5, 5, 5 |
| `slab1dmg_default_st2_g4_t05` | 5, 5, 5, 5 | 5, 5, 5, 5 |
| `slab1dmg_stream_st2_g4_t05` | 9, 9, 9, 8 | 9, 9, 9, 8 |

## 1D structured (`StructuredFD1D`)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `slab_decades4` | `-matfree_removal -precon_ref_shift` | 44 | 44 | - | short |  |
| `slab_decades4` | `-matfree_removal -precon_ref_shift -precon_dsa` | 12 | 13 | - | short |  |
| `slab_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 23 | 23 | - | short |  |
| `slab_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4 -precon_dsa` | 6 | 6 | - | short |  |
| `slab_decades4_stream0` | `-matfree_removal -precon_ref_shift` | 23 | 23 | - | short |  |
| `slab_decades4_stream0` | `-precon_dsa` | 6 | - | - | short |  |
| `slab_decades4_void` | `-matfree_removal -precon_ref_shift` | 44 | 44 | - | short |  |
| `slab_decades4_void` | `-matfree_removal -precon_ref_shift -precon_dsa` | 12 | - | - | short |  |
| `slab_decades4_void` | `-matfree_removal -precon_ref_shift -precon_ref_k 4 -check_ref_shift` | 23 | - | - | short |  |
| `slab_diffusive` |  | 23 | 23 | - | short |  |
| `slab_diffusive` | `-precon_dsa` | 6 | 6 | - | short |  |
| `slab_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12` | 10 | 11 | - | short |  |
| `slab_inf_medium` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 19 | 26 | - | short |  |
| `slab_inf_medium` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 9 | 10 | - | short |  |
| `slab_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 8 | 8 | - | short |  |
| `slab_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 18 | - | - | short | np=1: local opt 17, the 64-bit CI image 18 |
| `slab_mg2_coarse` |  | 9 | - | - | short |  |
| `slab_mg4_t0` |  | 5 | 5 | - | short |  |
| `slab_mg4_t05` |  | 5 | 5 | - | check/short |  |
| `slab_mg4_t05` | `-check_matfree` | 5 | 5 | - | check/short |  |
| `slab_mg4_t05` | `-matfree_removal` | 9 | 9 | - | short |  |
| `slab_mg4_t05` | `-matfree_removal -precon_ref_shift -precon_ref_k 2` | 5 | 5 | - | check/short |  |
| `slab_mg4_t05` | `-precon_stream` | 9 | 9 | - | short |  |
| `slab_mg4_t05` | `-ubolt_coo_two_call` | 5 | - | - | short |  |
| `slab_mg4_t05_dense` |  | 6 | 6 | - | short |  |
| `slab_mg4_t05_reflect_left` |  | 8 | - | - | short |  |
| `slab_mg4_t05_st0` |  | 1 | 1 | - | short/full |  |
| `slab_mg4_t2` |  | 5 | 5 | - | short/full |  |
| `slab_s2_coarse` |  | 9 | 9 | - | short |  |
| `slab_st0` |  | 1 | 1 | - | short/full |  |
| `slab_st0` | `-diag_scale` | 1 | 1 | - | short/full |  |
| `slab_st0` | `-diag_scale -precon_stream` | 2 | - | - | full |  |
| `slab_st0` | `-matfree_removal -precon_ref_shift` | 1 | - | - | short |  |
| `slab_st0` | `-precon_dsa` | 1 | - | - | short |  |
| `slab_st0` | `-precon_stream` | 1 | - | - | full |  |
| `slab_st2` |  | 5 | 5 | - | check/short |  |
| `slab_st2` | `-matfree_removal` | 8 | 8 | - | short |  |
| `slab_st2` | `-matfree_removal -precon_ref_shift` | 5 | - | - | short |  |
| `slab_st2` | `-precon_dsa` | 5 | - | - | short |  |
| `slab_st2` | `-precon_stream` | 8 | 8 | - | check/short |  |
| `slab_st2` | `-sub_1_pc_air_print_stats_timings` | 5 | - | - | short |  |
| `slab_st2` | `-ubolt_coo_two_call` | 5 | 5 | - | short |  |
| `slab_st2_dirichlet_cell` |  | 6 | - | - | short |  |
| `slab_st2_reflect_left` |  | 8 | - | - | short |  |
| `slab_st2_reflect_left` | `-ubolt_coo_two_call` | 8 | - | - | short |  |
| `slab_up3_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11` | 13 / 19 | 13 / 19 | - | check (np=2: short) | upscatter |
| `slab_up3_inf_medium` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11` | 8 / 19 | - | - | short | upscatter |
| `slab_up3_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_store moments` | 13 / 19 | 13 / 19 | - | short | upscatter, moments store |
| `slab_up3_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_store moments -outer_recon none` | 13 / 19 | 13 / 19 | - | short | upscatter, moments store |
| `slab_up3_layers` |  | 12 / 34 | 12 / 34 | - | short | upscatter |
| `slab_up3_layers` | `-precon_dsa` | 8 / 34 | - | - | short | upscatter |
| `slab_up3_layers` | `-outer_store moments` | 12 / 34 | 12 / 34 | - | short | upscatter, moments store |
| `slab_void_gap` |  | 23 | - | - | short |  |
| `slab_void_gap` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 23 | - | - | short |  |
| `slab_void_gap` | `-matfree_removal -precon_ref_shift -precon_dsa` | 7 | 7 | - | short |  |
| `slab_void_gap` | `-precon_dsa` | 7 | 7 | - | short |  |
| `slab_void_gap` | `-precon_dsa -dsa_void_bridge 0` | 8 | - | - | short |  |

## 2D structured (`StructuredFD2D`)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `box_30_overlap` |  | 5 | - | - | short |  |
| `box_30_s4_reflect_lb` |  | 6 | - | - | short |  |
| `box_30_s4_st2` |  | 6 | 6 | - | short |  |
| `box_40x20_st2` |  | 6 | 6 | - | short |  |
| `box_50_absorber` |  | 5 | 5 | - | short |  |
| `box_50_identity` |  | 6 | - | 6 | short |  |
| `box_50_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 11 | - | short |  |
| `box_50_inf_medium` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 30 | - | - | short |  |
| `box_50_inf_medium` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 9 | 9 | - | short |  |
| `box_50_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 12 | 12 | - | check/short |  |
| `box_50_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 26 | - | - | short |  |
| `box_50_ratio05` |  | 5 | 5 | - | short |  |
| `box_50_reflect_lb` |  | 6 | 5 | - | check/short |  |
| `box_50_reflect_lb` | `-check_matfree` | 6 | - | - | short |  |
| `box_50_st0` |  | 3 | 3 | - | short |  |
| `box_50_st2` |  | 6 | 6 | - | check/short |  |
| `box_50_st2` | `-check_matfree` | - | 6 | - | short |  |
| `box_50_st2` | `-matfree_removal` | 7 | 7 | - | short |  |
| `box_50_st2` | `-precon_dsa` | 5 | - | - | short |  |
| `box_50_st2` | `-precon_stream` | 7 | 7 | - | short |  |
| `box_50_st2` | `-ubolt_coo_two_call` | 6 | 6 | - | short |  |
| `box_50_st2_dirichlet_cell` |  | 8 | 8 | - | short |  |
| `box_50_st2_dirichlet_cell` | `-precon_dsa` | 7 | - | - | short |  |
| `box_60_cold` |  | 5 | 5 | - | short |  |
| `box_60_s4_st2` |  | 6 | 6 | - | full |  |
| `box_60_st0` |  | - | 3 | - | full |  |
| `box_60_st2` |  | 6 | 6 | - | full |  |
| `box_80x20_st2` |  | 5 | - | - | short |  |
| `box_80x40_st2` |  | 6 | 6 | - | full |  |
| `box_80x40_st2` | `-matfree_removal` | 7 | - | - | full |  |
| `box_80x40_st2` | `-precon_stream` | 7 | - | - | full |  |
| `box_crooked_pipe` |  | 231 | 233 | - | short |  |
| `box_crooked_pipe` | `-precon_dsa` | 9 | 9 | - | short |  |
| `box_crooked_pipe` | `-precon_dsa -dsa_consistent_d 0` | 30 | - | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift` | 52 | 52 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_dsa` | 9 | 9 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 3` | 30 | 30 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 30 | 30 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4 -precon_dsa` | 6 | 6 | - | short |  |
| `box_decades4_void` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 35 | 35 | - | short |  |
| `box_decades4_void` | `-matfree_removal -precon_ref_shift -precon_dsa` | 7 | - | - | short |  |
| `box_decades4_void` | `-matfree_removal -precon_ref_shift -precon_ref_k 1` | 68 | - | - | short |  |
| `box_diffusive` |  | 29 | 29 | - | short |  |
| `box_diffusive` | `-matfree_removal -precon_ref_shift` | 29 | - | - | short |  |
| `box_diffusive` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | - | - | short |  |
| `box_diffusive` | `-precon_dsa` | 6 | 6 | - | short |  |
| `box_diffusive` | `-precon_dsa -dsa_consistent_d 0` | 12 | - | - | short |  |
| `box_diffusive` | `-precon_dsa -dsa_consistent_d_power 1` | 6 | - | - | short |  |
| `box_diffusive` | `-precon_dsa -dsa_ksp_type cg -dsa_ksp_max_it 5` | 4 | - | - | short |  |
| `box_diffusive` | `-precon_dsa -pc_composite_type additive` | 8 | - | - | short |  |
| `box_lattice` |  | 9 | 9 | - | short |  |
| `box_lattice` | `-precon_dsa` | 5 | 5 | - | short |  |
| `box_layers` |  | 29 | 29 | - | short |  |
| `box_layers` | `-precon_dsa` | 13 | 13 | - | short |  |
| `box_random8` |  | 24 | 24 | - | short |  |
| `box_random8` | `-precon_dsa` | 7 | 7 | - | short |  |
| `box_s8_reflect_coarse` |  | 6 | 6 | - | short |  |
| `box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11` | 2 / 27 | 2 / 27 | - | short | upscatter |
| `box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_inner_gamma 0` | 12 / 27 | - | - | short | upscatter, every solve at the inner rtol |
| `box_up2_inf_medium` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11` | 1 / 27 | - | - | short | upscatter |
| `box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_store moments` | 2 / 27 | 2 / 27 | - | short | upscatter, moments store |
| `box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_store moments -outer_recon none` | 2 / 27 | 2 / 27 | - | short | upscatter, moments store |
| `box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -precon_stream -outer_store moments` | 5 / 27 | - | - | short | upscatter, moments store: every reconstruction stops on the safeguard |
| `box_up3_layers` |  | 11 / 16 | 11 / 16 | - | short | upscatter |
| `box_up3_layers` | `-outer_inner_gamma 0` | 15 / 15 | - | - | short | upscatter, every solve at the inner rtol |
| `box_up3_layers` | `-matfree_removal -precon_ref_shift` | 15 / 16 | - | - | short | upscatter |
| `box_up3_layers` | `-outer_store moments` | 11 / 16 | 11 / 16 | - | short | upscatter, moments store |
| `box_up3_layers` | `-outer_store moments -outer_recon 2` | 11 / 16 | - | - | short | upscatter, moments store |
| `box_up3_layers` | `-matfree_removal -precon_ref_shift -outer_store moments` | 15 / 16 | - | - | short | upscatter, moments store on shared hierarchies |
| `box_void_channel` |  | 31 | - | - | short |  |
| `box_void_channel` | `-matfree_removal -precon_ref_shift` | 31 | - | - | short |  |
| `box_void_channel` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | 6 | - | short |  |
| `box_void_channel` | `-precon_dsa` | 6 | 6 | - | short |  |
| `box_void_channel` | `-precon_dsa -dsa_void_bridge 0` | 8 | - | - | short |  |

## 3D structured (`StructuredFD3D`)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `cube_10_absorber` |  | 4 | 4 | - | short |  |
| `cube_10_cold` |  | 4 | 4 | - | short |  |
| `cube_10_identity` |  | 6 | - | 6 | short |  |
| `cube_10_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12` | 10 | 10 | - | short |  |
| `cube_10_inf_medium` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 8 | 8 | - | short |  |
| `cube_10_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 11 | - | short |  |
| `cube_10_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 22 | - | - | short |  |
| `cube_10_mg4_t05` |  | 6 | 6 | - | short |  |
| `cube_10_overlap` |  | 4 | - | - | short |  |
| `cube_10_ratio05` |  | 4 | 5 | - | short |  |
| `cube_10_reflect_3faces` |  | 5 | 5 | - | check/short |  |
| `cube_10_reflect_3faces` | `-check_matfree` | 5 | - | - | short |  |
| `cube_10_s4_st2` |  | 6 | 6 | - | short |  |
| `cube_10_st2` |  | 6 | 6 | - | check/short |  |
| `cube_10_st2` | `-matfree_removal` | 6 | 6 | - | short |  |
| `cube_10_st2` | `-precon_dsa` | 4 | - | - | short |  |
| `cube_10_st2` | `-precon_stream` | 6 | 6 | - | short |  |
| `cube_10_st2` | `-ubolt_coo_two_call` | 6 | 6 | - | short |  |
| `cube_10_st2_dirichlet_cell` |  | 6 | - | - | short |  |
| `cube_10x5x5_st2` |  | 7 | 7 | - | short |  |
| `cube_15_st2` |  | 6 | 6 | - | full |  |
| `cube_diffusive` |  | 18 | 18 | - | short |  |
| `cube_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `cube_diffusive` | `-precon_stream -precon_dsa` | 32 | 32 | - | short |  |
| `cube_diffusive_yreflect` | `-precon_dsa` | 5 | - | - | short |  |
| `cube_void_duct` |  | 22 | - | - | short |  |
| `cube_void_duct` | `-matfree_removal -precon_ref_shift` | - | 22 | - | short |  |
| `cube_void_duct` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 22 | - | - | short |  |
| `cube_void_duct` | `-matfree_removal -precon_ref_shift -precon_dsa` | 5 | - | - | short |  |
| `cube_void_duct` | `-precon_dsa` | 5 | 5 | - | short |  |
| `cube_void_duct` | `-precon_dsa -dsa_void_bridge 0` | 7 | - | - | short |  |

## Unstructured DG0 (`UnstructuredDG`, order 0)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `plex_box_50_reflect_lb` |  | 5 | 5 | - | short |  |
| `plex_box_50_st2` |  | 6 | 6 | - | check/short |  |
| `plex_box_50_st2` | `-matfree_removal` | 7 | 7 | - | short |  |
| `plex_box_50_st2` | `-precon_stream` | 7 | 7 | - | short |  |
| `plex_box_50_st2_dirichlet_cell` |  | 8 | - | - | short |  |
| `plex_box_50_st2_dirichlet_cell` | `-precon_dsa` | 7 | - | - | short |  |
| `plex_box_diffusive` |  | 29 | - | - | short |  |
| `plex_box_diffusive` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | 6 | - | short |  |
| `plex_box_diffusive` | `-precon_dsa` | 6 | 6 | - | short |  |
| `plex_box_diffusive` | `-precon_dsa -pc_composite_type additive` | 8 | - | - | short |  |
| `plex_box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11` | 2 / 27 | 2 / 27 | - | short | upscatter |
| `plex_box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_store moments` | 2 / 27 | 2 / 27 | - | short | upscatter, moments store |
| `plex_box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_store moments -outer_recon none` | 2 / 27 | 2 / 27 | - | short | upscatter, moments store |
| `plex_box_up3_layers` |  | 11 / 16 | 11 / 16 | - | short | upscatter |
| `plex_box_up3_layers` | `-outer_store moments` | 11 / 16 | 11 / 16 | - | short | upscatter, moments store |
| `plex_box_void_channel` |  | 30 | - | - | short |  |
| `plex_box_void_channel` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | 6 | - | short |  |
| `plex_box_void_channel` | `-precon_dsa` | 6 | 6 | - | short |  |
| `plex_box_void_channel` | `-precon_dsa -dsa_void_bridge 0` | 8 | - | - | short |  |
| `plex_cube_10_st2` |  | 6 | 6 | - | short |  |
| `plex_cube_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `plex_decades4` | `-matfree_removal -precon_ref_shift` | 52 | 52 | - | short |  |
| `plex_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 30 | 30 | - | short |  |
| `plex_decades4_void` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 31 | 31 | - | short |  |
| `plex_decades4_void` | `-matfree_removal -precon_ref_shift -precon_dsa` | 7 | - | - | short |  |
| `plex_square_msh` |  | 4 | 4 | - | short |  |
| `plex_tet_12_absorber_dirichlet_cell` | `-ksp_rtol 1e-12` | 7 | - | - | short (needs tetmesher) |  |
| `plex_tet_6_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 11 | - | short (needs tetmesher) |  |
| `plex_tet_6_st2` |  | 5 | 5 | - | short (needs tetmesher) |  |
| `plex_tet_diffusive` | `-precon_dsa` | 6 | 6 | - | short (needs tetmesher) |  |
| `plex_tri_30_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 12 | - | short (needs triangle) |  |
| `plex_tri_30_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 35 | - | - | short (needs triangle) |  |
| `plex_tri_30_inf_medium_ghost` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 9 | - | - | short (needs triangle) |  |
| `plex_tri_30_st2` |  | 5 | 5 | - | short (needs triangle) |  |
| `plex_tri_diffusive` |  | 34 | - | - | short (needs triangle) |  |
| `plex_tri_diffusive` | `-precon_dsa` | 6 | 6 | - | short (needs triangle) |  |

## Unstructured DG1 (`UnstructuredDG`, order 1)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `plex_box_30_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12` | 12 | 12 | - | short |  |
| `plex_box_30_inf_medium_dg1` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 42 | - | - | short |  |
| `plex_box_30_inf_medium_dg1` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 11 | - | - | short |  |
| `plex_box_50_reflect_lb_dg1` |  | 6 | 6 | - | short |  |
| `plex_box_50_st2_dg1` |  | 7 | 7 | - | check/short |  |
| `plex_box_50_st2_dg1` | `-check_matfree` | 7 | - | - | short |  |
| `plex_box_50_st2_dg1` | `-matfree_removal` | 9 | 8 | - | short |  |
| `plex_box_50_st2_dg1` | `-precon_stream` | 9 | - | - | short |  |
| `plex_box_50_st2_dg1` | `-ubolt_coo_two_call` | 7 | - | - | short |  |
| `plex_box_diffusive_dg1` |  | 34 | - | - | short |  |
| `plex_box_diffusive_dg1` | `-precon_dsa` | 7 | 7 | - | short |  |
| `plex_box_up2_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11` | 2 / 27 | - | - | short | upscatter |
| `plex_box_up2_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11 -outer_store moments` | 2 / 27 | - | - | short | upscatter, moments store |
| `plex_box_up3_layers_dg1` |  | 12 / 17 | 12 / 17 | - | short | upscatter |
| `plex_box_up3_layers_dg1` | `-outer_store moments` | 12 / 17 | 12 / 17 | - | short | upscatter, moments store |
| `plex_box_void_channel_dg1` |  | 37 | - | - | short |  |
| `plex_box_void_channel_dg1` | `-matfree_removal -precon_ref_shift -precon_dsa` | 10 | - | - | short |  |
| `plex_box_void_channel_dg1` | `-precon_dsa` | 10 | 10 | - | short |  |
| `plex_box_void_channel_dg1` | `-precon_dsa -dsa_void_bridge 0` | 10 | - | - | short |  |
| `plex_cube_10_st2_dg1` |  | 6 | 6 | - | short |  |
| `plex_cube_diffusive_dg1` |  | 32 | - | - | short |  |
| `plex_cube_diffusive_dg1` | `-precon_dsa` | 8 | 8 | - | short |  |
| `plex_decades4_dg1` | `-matfree_removal -precon_ref_shift` | 65 | - | - | short |  |
| `plex_decades4_dg1` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 36 | 36 | - | short |  |
| `plex_square_msh_dg1` |  | 5 | 5 | - | short |  |
| `plex_tet_6_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12` | 12 | 12 | - | short (needs tetmesher) |  |
| `plex_tet_6_st2_dg1` |  | 5 | 5 | - | short (needs tetmesher) |  |
| `plex_tet_diffusive_dg1` | `-precon_dsa` | 10 | - | - | short (needs tetmesher) |  |
| `plex_tri_30_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12` | 13 | - | - | short (needs triangle) |  |
| `plex_tri_30_st2_dg1` |  | 5 | 5 | - | short (needs triangle) |  |
| `plex_tri_diffusive_dg1` | `-precon_dsa` | 8 | - | - | short (needs triangle) |  |

## CG-SUPG (`UnstructuredCG`)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `cg_box_20_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 11 | - | short |  |
| `cg_box_50_reflect_lb` |  | 5 | - | - | short |  |
| `cg_box_50_reflect_lb` | `-precon_dsa` | 4 | - | - | short |  |
| `cg_box_50_st2` |  | 7 | 7 | - | check/short |  |
| `cg_box_up2_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12 -outer_rtol 1e-11` | 2 / 27 | 2 / 27 | - | short | upscatter |
| `cg_box_up3_layers` |  | 12 / 17 | 12 / 17 | - | short | upscatter |
| `cg_box_diffusive` |  | 38 | - | - | short |  |
| `cg_box_diffusive` | `-precon_dsa` | 6 | 6 | - | short |  |
| `cg_box_void_channel` |  | 48 | 48 | - | short |  |
| `cg_box_void_channel` | `-precon_dsa` | 8 | 8 | - | short |  |
| `cg_box_void_channel` | `-precon_dsa -dsa_void_bridge 0` | 7 | - | - | short |  |
| `cg_cube_10_st2` |  | 6 | - | - | short |  |
| `cg_cube_diffusive` |  | 47 | - | - | short |  |
| `cg_cube_diffusive` | `-precon_dsa` | 6 | - | - | short |  |
| `cg_decades4` |  | 42 | 42 | - | short |  |
| `cg_decades4` | `-precon_dsa` | 6 | - | - | short |  |
| `cg_slab_saaf_ls_void` |  | 4 | - | - | short |  |
| `cg_slab_saaf_ls_void` | `-diag_scale` | 4 | - | - | short |  |
| `cg_slab_saaf_ls_void` | `-precon_block_scale 0` | 4 | - | - | short |  |
| `cg_slab_saaf_ls_void` | `-supg_zeta 2` | 4 | - | - | short |  |
| `cg_slab_thin_thick` |  | 4 | 4 | - | short |  |
| `cg_square_msh` |  | 4 | 4 | - | short |  |
| `cg_tet_6_st2` |  | 5 | 5 | - | short (needs tetmesher) |  |
| `cg_tet_diffusive` |  | 38 | - | - | short (needs tetmesher) |  |
| `cg_tet_diffusive` | `-precon_dsa` | 6 | 6 | - | short (needs tetmesher) |  |
| `cg_tri_30_st2` |  | 5 | - | - | short (needs triangle) |  |
| `cg_tri_diffusive` |  | 35 | - | - | short (needs triangle) |  |
| `cg_tri_diffusive` | `-precon_dsa` | 6 | - | - | short (needs triangle) |  |
