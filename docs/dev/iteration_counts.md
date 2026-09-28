# Pinned iteration counts

`tests/Makefile` is the ONLY source of truth for a pin: every solve recipe passes `-ksp_max_it <pin>`. This file tabulates those pins, one table per backend, as the Makefile has them on 2026-09-28, generated from its recipe lines (371 `transportk` lines, 227 distinct problem + option combinations; the `verify_*` lines have no pin, see `docs/dev/verification.md`). If a number here disagrees with the Makefile, the Makefile wins and this row is stale: re-pin in the Makefile first, then copy the number here.

How to read the tables:
- A pin is the count measured on the local opt arch (`arch-linux-c-opt`), exactly, unless the note column says a CI image needed more (the pin policy is in `docs/dev/testing.md`).
- A multigroup recipe pins the MAX over its groups: `-ksp_max_it` is one option for the whole sweep. The per-group counts are in the history tables.
- "suite": `check` is `run_check`, `short` is `run_tests_short_serial` / `_parallel`, `full` is `run_tests_serial` / `_parallel` (run by `make tests` after `tests_short`). "needs triangle / tetmesher": a generated simplex box, skipped where PETSc has no triangle / (c)tetgen - the CI images have neither.
- np=4 appears only on the painting identities (and on the verify drivers).
- Options are the recipe's own; `-problem problems/<problem>.json` and `-ksp_max_it` are left out.

The measurements behind these numbers - what each count was before and after the change that moved it, the no-DSA references, the LU splits, the sweeps - are dated narrative in `docs/dev/history.md`, Testing history:
- 1D/2D/3D structured: "Single-group baseline iteration counts", "2D iteration counts", "3D iteration counts", "Reflective boundary conditions: counts", "Ghost-flux vacuum treatment: counts" (and its "Switching the default", "Ghost-flux reflective faces"), "Painted regions", "Matrix-free removal".
- DSA on any backend: "DSA (the diffusion correction, `-precon_dsa`)" and its "Discretisation-consistent D", "Literature benchmark pins", "DSA with voids".
- `-precon_ref_shift` on any backend: "The reference-shifted streaming pmat" and its "The decades problem files", "DSA on a shifted pmat", "Streaming-only groups", "Voids", "Pins and slack".
- Unstructured DG0 / DG1: "Unstructured iteration counts" (twins, element-block scaling, DG1, plex DSA, the simplex creep, hair-trigger pins).
- CG-SUPG: "CG-SUPG verification and iteration counts: the counts" (the LU split, the zeta and cell-size sweeps, the CG DSA).

## The 24 baseline logs (`tests/baselines/`, captured 2026-09-25)

Not pins: `make baselines` captures these at `-ksp_max_it 200` and a refactor diffs against them (`docs/dev/testing.md`). Final iteration count of each log, np=1 / np=2 (multigroup: per group):

| log | np=1 | np=2 |
|---|---|---|
| `slab1d_default_st0` | 1 | 1 |
| `slab1d_default_st2` | 6 | 6 |
| `slab1d_default_st0_ds` | 1 | 1 |
| `slab1d_default_st2_ds` | 174 | 173 |
| `slab1d_stream_st0` | 1 | 1 |
| `slab1d_stream_st2` | 8 | 8 |
| `slab1d_stream_st0_ds` | 2 | 2 |
| `slab1d_stream_st2_ds` | DIVERGED_ITS (200) | DIVERGED_ITS (200) |
| `slab1dmg_default_st0_g4_t05` | 1, 1, 1, 1 | 1, 1, 1, 1 |
| `slab1dmg_default_st2_g4_t0` | 6, 6, 6, 6 | 6, 6, 6, 6 |
| `slab1dmg_default_st2_g4_t05` | 6, 6, 6, 6 | 6, 6, 6, 6 |
| `slab1dmg_stream_st2_g4_t05` | 9, 9, 9, 8 | 9, 9, 9, 8 |

## 1D structured (`StructuredFD1D`)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `slab_decades4` | `-matfree_removal -precon_ref_shift` | 44 | 44 | - | short |  |
| `slab_decades4` | `-matfree_removal -precon_ref_shift -precon_dsa` | 11 | 11 | - | short |  |
| `slab_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 22 | 22 | - | short |  |
| `slab_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4 -precon_dsa` | 5 | 5 | - | short |  |
| `slab_decades4_stream0` | `-matfree_removal -precon_ref_shift` | 22 | 22 | - | short |  |
| `slab_decades4_stream0` | `-precon_dsa` | 5 | - | - | short |  |
| `slab_decades4_void` | `-matfree_removal -precon_ref_shift` | 44 | 44 | - | short |  |
| `slab_decades4_void` | `-matfree_removal -precon_ref_shift -precon_dsa` | 11 | - | - | short |  |
| `slab_decades4_void` | `-matfree_removal -precon_ref_shift -precon_ref_k 4 -check_ref_shift` | 23 | - | - | short |  |
| `slab_diffusive` |  | 23 | 23 | - | short |  |
| `slab_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `slab_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12` | 10 | 11 | - | short |  |
| `slab_inf_medium` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 7 | 8 | - | short |  |
| `slab_inf_medium` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 9 | 10 | - | short |  |
| `slab_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 8 | 8 | - | short |  |
| `slab_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 18 | - | - | short | np=1: local opt 17, the 64-bit CI image 18 |
| `slab_mg2_coarse` |  | 9 | - | - | short |  |
| `slab_mg4_t0` |  | 6 | 6 | - | short |  |
| `slab_mg4_t05` |  | 6 | 6 | - | check/short |  |
| `slab_mg4_t05` | `-check_matfree` | 6 | 6 | - | check/short |  |
| `slab_mg4_t05` | `-matfree_removal -ksp_pc_side right` | 9 | 9 | - | short |  |
| `slab_mg4_t05` | `-matfree_removal -precon_ref_shift -precon_ref_k 2` | 6 | 6 | - | check/short |  |
| `slab_mg4_t05` | `-precon_stream -ksp_pc_side right` | 9 | 9 | - | short |  |
| `slab_mg4_t05` | `-ubolt_coo_two_call` | 6 | - | - | short |  |
| `slab_mg4_t05_dense` |  | 6 | 6 | - | short |  |
| `slab_mg4_t05_reflect_left` |  | 8 | - | - | short |  |
| `slab_mg4_t05_st0` |  | 1 | 1 | - | short/full |  |
| `slab_mg4_t2` |  | 6 | 6 | - | short/full |  |
| `slab_s2_coarse` |  | 9 | 9 | - | short |  |
| `slab_st0` |  | 1 | 1 | - | short/full |  |
| `slab_st0` | `-diag_scale` | 1 | 1 | - | short/full |  |
| `slab_st0` | `-diag_scale -precon_stream -ksp_pc_side right` | 2 | - | - | full |  |
| `slab_st0` | `-matfree_removal -precon_ref_shift -ksp_pc_side right` | 1 | - | - | short |  |
| `slab_st0` | `-precon_dsa` | 1 | - | - | short |  |
| `slab_st0` | `-precon_stream -ksp_pc_side right` | 1 | - | - | full |  |
| `slab_st2` |  | 6 | 6 | - | check/short |  |
| `slab_st2` | `-matfree_removal -ksp_pc_side right` | 8 | 8 | - | short |  |
| `slab_st2` | `-matfree_removal -precon_ref_shift` | 6 | - | - | short |  |
| `slab_st2` | `-precon_dsa` | 5 | - | - | short |  |
| `slab_st2` | `-precon_stream -ksp_pc_side right` | 8 | 8 | - | check/short |  |
| `slab_st2` | `-sub_1_pc_air_print_stats_timings` | 6 | - | - | short |  |
| `slab_st2` | `-ubolt_coo_two_call` | 6 | 6 | - | short |  |
| `slab_st2_dirichlet_cell` |  | 6 | - | - | short |  |
| `slab_st2_reflect_left` |  | 8 | - | - | short |  |
| `slab_st2_reflect_left` | `-ubolt_coo_two_call` | 8 | - | - | short |  |
| `slab_void_gap` |  | 23 | - | - | short |  |
| `slab_void_gap` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 23 | - | - | short |  |
| `slab_void_gap` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | 6 | - | short |  |
| `slab_void_gap` | `-precon_dsa` | 6 | 6 | - | short |  |
| `slab_void_gap` | `-precon_dsa -dsa_void_bridge 0` | 6 | - | - | short |  |

## 2D structured (`StructuredFD2D`)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `box_30_overlap` |  | 5 | - | - | short |  |
| `box_30_s4_reflect_lb` |  | 6 | - | - | short |  |
| `box_30_s4_st2` |  | 6 | 6 | - | short |  |
| `box_40x20_st2` |  | 6 | 6 | - | short |  |
| `box_50_absorber` |  | 5 | 5 | - | short |  |
| `box_50_identity` |  | 7 | - | 7 | short |  |
| `box_50_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12` | 10 | 10 | - | short |  |
| `box_50_inf_medium` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 25 | - | - | short | np=1: local opt 24, the 64-bit CI image 25 |
| `box_50_inf_medium` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 9 | 9 | - | short |  |
| `box_50_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 12 | - | check/short | np=2: local opt 11, the OpenMP CI image 12 |
| `box_50_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 26 | - | - | short |  |
| `box_50_ratio05` |  | 5 | 5 | - | short |  |
| `box_50_reflect_lb` |  | 6 | 6 | - | check/short |  |
| `box_50_reflect_lb` | `-check_matfree` | 6 | - | - | short |  |
| `box_50_st0` |  | 3 | 3 | - | short |  |
| `box_50_st2` |  | 7 | 7 | - | check/short |  |
| `box_50_st2` | `-check_matfree` | - | 7 | - | short |  |
| `box_50_st2` | `-matfree_removal -ksp_pc_side right` | 7 | 7 | - | short |  |
| `box_50_st2` | `-precon_dsa` | 5 | - | - | short |  |
| `box_50_st2` | `-precon_stream -ksp_pc_side right` | 7 | 7 | - | short |  |
| `box_50_st2` | `-ubolt_coo_two_call` | 7 | 7 | - | short |  |
| `box_50_st2_dirichlet_cell` |  | 6 | 6 | - | short |  |
| `box_50_st2_dirichlet_cell` | `-precon_dsa` | 5 | - | - | short |  |
| `box_60_cold` |  | 5 | 5 | - | short |  |
| `box_60_s4_st2` |  | 6 | 6 | - | full |  |
| `box_60_st0` |  | - | 3 | - | full |  |
| `box_60_st2` |  | 7 | 7 | - | full |  |
| `box_80x20_st2` |  | 5 | - | - | short |  |
| `box_80x40_st2` |  | 7 | 7 | - | full |  |
| `box_80x40_st2` | `-matfree_removal -ksp_pc_side right` | 7 | - | - | full |  |
| `box_80x40_st2` | `-precon_stream -ksp_pc_side right` | 7 | - | - | full |  |
| `box_crooked_pipe` |  | 113 | 117 | - | short | np=2: local opt 114, the OpenMP CI image 117 |
| `box_crooked_pipe` | `-precon_dsa` | 8 | 8 | - | short |  |
| `box_crooked_pipe` | `-precon_dsa -dsa_consistent_d 0` | 28 | - | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift` | 51 | 51 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_dsa` | 8 | 8 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 3` | 29 | 29 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 29 | 29 | - | short |  |
| `box_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4 -precon_dsa` | 5 | 5 | - | short |  |
| `box_decades4_void` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 29 | 29 | - | short |  |
| `box_decades4_void` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | - | - | short |  |
| `box_decades4_void` | `-matfree_removal -precon_ref_shift -precon_ref_k 1` | 68 | - | - | short |  |
| `box_diffusive` |  | 29 | 29 | - | short |  |
| `box_diffusive` | `-matfree_removal -precon_ref_shift` | 29 | - | - | short |  |
| `box_diffusive` | `-matfree_removal -precon_ref_shift -precon_dsa` | 5 | - | - | short |  |
| `box_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `box_diffusive` | `-precon_dsa -dsa_consistent_d 0` | 11 | - | - | short |  |
| `box_diffusive` | `-precon_dsa -dsa_consistent_d_power 1` | 5 | - | - | short |  |
| `box_diffusive` | `-precon_dsa -dsa_ksp_type cg -dsa_ksp_max_it 5` | 4 | - | - | short |  |
| `box_diffusive` | `-precon_dsa -pc_composite_type additive` | 7 | - | - | short |  |
| `box_lattice` |  | 9 | 9 | - | short |  |
| `box_lattice` | `-precon_dsa` | 5 | 5 | - | short |  |
| `box_layers` |  | 24 | 24 | - | short |  |
| `box_layers` | `-precon_dsa` | 11 | 11 | - | short |  |
| `box_random8` |  | 19 | 19 | - | short |  |
| `box_random8` | `-precon_dsa` | 7 | 6 | - | short |  |
| `box_s8_reflect_coarse` |  | 6 | 6 | - | short |  |
| `box_void_channel` |  | 26 | - | - | short |  |
| `box_void_channel` | `-matfree_removal -precon_ref_shift` | 26 | - | - | short |  |
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
| `cube_10_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 22 | - | - | short | np=1: local opt 21, the 64-bit CI image 22 |
| `cube_10_mg4_t05` |  | 5 | 6 | - | short |  |
| `cube_10_overlap` |  | 4 | - | - | short |  |
| `cube_10_ratio05` |  | 4 | 4 | - | short |  |
| `cube_10_reflect_3faces` |  | 5 | 5 | - | check/short |  |
| `cube_10_reflect_3faces` | `-check_matfree` | 5 | - | - | short |  |
| `cube_10_s4_st2` |  | 6 | 6 | - | short |  |
| `cube_10_st2` |  | 6 | 6 | - | check/short |  |
| `cube_10_st2` | `-matfree_removal -ksp_pc_side right` | 6 | 6 | - | short |  |
| `cube_10_st2` | `-precon_dsa` | 4 | - | - | short |  |
| `cube_10_st2` | `-precon_stream -ksp_pc_side right` | 6 | 6 | - | short |  |
| `cube_10_st2` | `-ubolt_coo_two_call` | 6 | 6 | - | short |  |
| `cube_10_st2_dirichlet_cell` |  | 5 | - | - | short |  |
| `cube_10x5x5_st2` |  | 6 | 6 | - | short |  |
| `cube_15_st2` |  | 6 | 6 | - | full |  |
| `cube_diffusive` |  | 18 | 18 | - | short |  |
| `cube_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `cube_diffusive` | `-precon_stream -precon_dsa` | 25 | 25 | - | short |  |
| `cube_diffusive_yreflect` | `-precon_dsa` | 5 | - | - | short |  |
| `cube_void_duct` |  | 21 | - | - | short |  |
| `cube_void_duct` | `-matfree_removal -precon_ref_shift` | - | 21 | - | short |  |
| `cube_void_duct` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 21 | - | - | short |  |
| `cube_void_duct` | `-matfree_removal -precon_ref_shift -precon_dsa` | 5 | - | - | short |  |
| `cube_void_duct` | `-precon_dsa` | 5 | 5 | - | short |  |
| `cube_void_duct` | `-precon_dsa -dsa_void_bridge 0` | 6 | - | - | short |  |

## Unstructured DG0 (`UnstructuredDG`, order 0)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `plex_box_50_reflect_lb` |  | 6 | 6 | - | short |  |
| `plex_box_50_st2` |  | 7 | 7 | - | check/short |  |
| `plex_box_50_st2` | `-matfree_removal -ksp_pc_side right` | 7 | 7 | - | short |  |
| `plex_box_50_st2` | `-precon_stream -ksp_pc_side right` | 7 | 7 | - | short |  |
| `plex_box_50_st2_dirichlet_cell` |  | 6 | - | - | short |  |
| `plex_box_50_st2_dirichlet_cell` | `-precon_dsa` | 5 | - | - | short |  |
| `plex_box_diffusive` |  | 29 | - | - | short |  |
| `plex_box_diffusive` | `-matfree_removal -precon_ref_shift -precon_dsa` | 5 | 5 | - | short |  |
| `plex_box_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `plex_box_diffusive` | `-precon_dsa -pc_composite_type additive` | 7 | - | - | short |  |
| `plex_box_void_channel` |  | 26 | - | - | short | np=1: local opt 25 (np=2 26), pinned 26 for the gnu_opt CI image |
| `plex_box_void_channel` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | 6 | - | short |  |
| `plex_box_void_channel` | `-precon_dsa` | 6 | 6 | - | short |  |
| `plex_box_void_channel` | `-precon_dsa -dsa_void_bridge 0` | 8 | - | - | short |  |
| `plex_cube_10_st2` |  | 6 | 6 | - | short |  |
| `plex_cube_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `plex_decades4` | `-matfree_removal -precon_ref_shift` | 51 | 51 | - | short |  |
| `plex_decades4` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 29 | 29 | - | short | np=1: local opt 28, the OpenMP CI image 29 |
| `plex_decades4_void` | `-matfree_removal -precon_ref_shift -check_ref_shift` | 27 | 27 | - | short |  |
| `plex_decades4_void` | `-matfree_removal -precon_ref_shift -precon_dsa` | 6 | - | - | short |  |
| `plex_square_msh` |  | 4 | 4 | - | short |  |
| `plex_tet_12_absorber_dirichlet_cell` | `-ksp_rtol 1e-12` | 6 | - | - | short (needs tetmesher) |  |
| `plex_tet_6_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 11 | - | short (needs tetmesher) |  |
| `plex_tet_6_st2` |  | 5 | 5 | - | short (needs tetmesher) |  |
| `plex_tet_diffusive` | `-precon_dsa` | 5 | 5 | - | short (needs tetmesher) |  |
| `plex_tri_30_inf_medium_ghost` | `-check_inf_medium -ksp_rtol 1e-12` | 12 | 12 | - | short (needs triangle) |  |
| `plex_tri_30_inf_medium_ghost` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 33 | - | - | short (needs triangle) |  |
| `plex_tri_30_inf_medium_ghost` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 9 | - | - | short (needs triangle) |  |
| `plex_tri_30_st2` |  | 5 | 5 | - | short (needs triangle) |  |
| `plex_tri_diffusive` |  | 35 | - | - | short (needs triangle) |  |
| `plex_tri_diffusive` | `-precon_dsa` | 5 | 5 | - | short (needs triangle) |  |

## Unstructured DG1 (`UnstructuredDG`, order 1)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `plex_box_30_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12` | 12 | 12 | - | short |  |
| `plex_box_30_inf_medium_dg1` | `-matfree_removal -check_inf_medium -ksp_rtol 1e-12` | 42 | - | - | short |  |
| `plex_box_30_inf_medium_dg1` | `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 11 | - | - | short |  |
| `plex_box_50_reflect_lb_dg1` |  | 6 | 6 | - | short |  |
| `plex_box_50_st2_dg1` |  | 7 | 7 | - | check/short |  |
| `plex_box_50_st2_dg1` | `-check_matfree` | 7 | - | - | short |  |
| `plex_box_50_st2_dg1` | `-matfree_removal -ksp_pc_side right` | 9 | 8 | - | short |  |
| `plex_box_50_st2_dg1` | `-precon_stream -ksp_pc_side right` | 9 | - | - | short |  |
| `plex_box_50_st2_dg1` | `-ubolt_coo_two_call` | 7 | - | - | short |  |
| `plex_box_diffusive_dg1` |  | 34 | - | - | short |  |
| `plex_box_diffusive_dg1` | `-precon_dsa` | 6 | 7 | - | short |  |
| `plex_box_void_channel_dg1` |  | 34 | - | - | short |  |
| `plex_box_void_channel_dg1` | `-matfree_removal -precon_ref_shift -precon_dsa` | 10 | - | - | short |  |
| `plex_box_void_channel_dg1` | `-precon_dsa` | 10 | 10 | - | short |  |
| `plex_box_void_channel_dg1` | `-precon_dsa -dsa_void_bridge 0` | 10 | - | - | short |  |
| `plex_cube_10_st2_dg1` |  | 7 | 7 | - | short |  |
| `plex_cube_diffusive_dg1` |  | 32 | - | - | short |  |
| `plex_cube_diffusive_dg1` | `-precon_dsa` | 7 | 7 | - | short |  |
| `plex_decades4_dg1` | `-matfree_removal -precon_ref_shift` | 66 | - | - | short |  |
| `plex_decades4_dg1` | `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 36 | 36 | - | short |  |
| `plex_square_msh_dg1` |  | 5 | 5 | - | short |  |
| `plex_tet_6_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12` | 12 | 12 | - | short (needs tetmesher) |  |
| `plex_tet_6_st2_dg1` |  | 5 | 5 | - | short (needs tetmesher) |  |
| `plex_tet_diffusive_dg1` | `-precon_dsa` | 9 | - | - | short (needs tetmesher) |  |
| `plex_tri_30_inf_medium_dg1` | `-check_inf_medium -ksp_rtol 1e-12` | 13 | - | - | short (needs triangle) |  |
| `plex_tri_30_st2_dg1` |  | 5 | 5 | - | short (needs triangle) |  |
| `plex_tri_diffusive_dg1` | `-precon_dsa` | 8 | - | - | short (needs triangle) |  |

## CG-SUPG (`UnstructuredCG`)

| problem | options | np=1 | np=2 | np=4 | suite | note |
|---|---|---|---|---|---|---|
| `cg_box_20_inf_medium` | `-check_inf_medium -ksp_rtol 1e-12` | 11 | 11 | - | short |  |
| `cg_box_50_reflect_lb` |  | 6 | - | - | short |  |
| `cg_box_50_reflect_lb` | `-precon_dsa` | 4 | - | - | short |  |
| `cg_box_50_st2` |  | 7 | 7 | - | check/short |  |
| `cg_box_diffusive` |  | 39 | - | - | short |  |
| `cg_box_diffusive` | `-precon_dsa` | 5 | 5 | - | short |  |
| `cg_box_void_channel` |  | 47 | 46 | - | short |  |
| `cg_box_void_channel` | `-precon_dsa` | 8 | 8 | - | short |  |
| `cg_box_void_channel` | `-precon_dsa -dsa_void_bridge 0` | 7 | - | - | short |  |
| `cg_cube_10_st2` |  | 6 | - | - | short |  |
| `cg_cube_diffusive` |  | 49 | - | - | short | np=1: local opt 48, the CI images 49 |
| `cg_cube_diffusive` | `-precon_dsa` | 6 | - | - | short |  |
| `cg_decades4` |  | 41 | 42 | - | short |  |
| `cg_decades4` | `-precon_dsa` | 5 | - | - | short |  |
| `cg_slab_saaf_ls_void` |  | 4 | - | - | short |  |
| `cg_slab_saaf_ls_void` | `-diag_scale` | 4 | - | - | short |  |
| `cg_slab_saaf_ls_void` | `-precon_block_scale 0` | 4 | - | - | short |  |
| `cg_slab_saaf_ls_void` | `-supg_zeta 2` | 3 | - | - | short |  |
| `cg_slab_thin_thick` |  | 4 | 4 | - | short |  |
| `cg_square_msh` |  | 4 | 4 | - | short |  |
| `cg_tet_6_st2` |  | 5 | 5 | - | short (needs tetmesher) |  |
| `cg_tet_diffusive` |  | 39 | - | - | short (needs tetmesher) |  |
| `cg_tet_diffusive` | `-precon_dsa` | 5 | 5 | - | short (needs tetmesher) |  |
| `cg_tri_30_st2` |  | 5 | - | - | short (needs triangle) |  |
| `cg_tri_diffusive` |  | 35 | - | - | short (needs triangle) |  |
| `cg_tri_diffusive` | `-precon_dsa` | 5 | - | - | short (needs triangle) |  |
