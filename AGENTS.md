# UBOLT

## What it is
A fixed-source Boltzmann (SN radiation transport) solver library, `libubolt`, on PETSc + Kokkos + PFLARE.
GPU-native: the streaming/removal operator is assembled on device through PETSc's COO interface
(MATAIJKOKKOS), the scattering is applied matrix-free (MatShell + Kokkos), and the preconditioner is a
PCComposite of a removal block-Jacobi shell + PCAIR on the streaming (+ an optional DSA stage).
Energy groups are solved one at a time in a group Gauss-Seidel sweep: one preallocated matrix, a
values-only refill per group. The long-form design is `docs/architecture.md`.

## Build
1. `export PETSC_DIR=... PETSC_ARCH=...` (both required; tool shells may not inherit them). PFLARE is
   `$PFLARE_DIR`, default `../PFLARE` next to this repo.
2. In the top directory: `make -j3 build_tests` (builds `lib/libubolt.{so,a}` first; `make` alone builds
   only the library). Needs PETSc >= 3.25 with Kokkos; in practice PETSc main from 3 Sep 2026 on, for
   `MatGetState`'s `MatState` form used by `verify_plexk` (no 3.25.x release has it).
3. Fix every compile warning: CI (`.github/workflows/ci_build.yml`, `dockerfiles/Dockerfile_kokkos`, on
   the `stevendargaville/petsc_kokkos` opt/debug/64-bit/OMP images) builds PFLARE main, then UBOLT with
   `-Wall -Werror`, then runs `check` + `tests`.
4. PETSc's rules write no `.d` files, so header dependencies are declared by hand: `$(OBJS)` depends on
   `$(UBOLT_HEADERS)`, and `$(HEADER_STAMP)` deletes stale driver objects/binaries (PETSc's
   `% : %.kokkos.cxx` rule links every prerequisite, so drivers cannot list headers). Add
   prerequisites, never flags.

## Test
- `make check` (fast, 18 runs) < `make tests_short` < `make tests`, from the top directory. Run once and
  trust `make`'s exit code: 0 is a pass. Do not re-run to grep the output.
- Solve recipes pin `-ksp_max_it` in `tests/Makefile`, the only source of truth for pins
  (`docs/dev/iteration_counts.md` mirrors it). Pin the local opt arch's count; CI flags an arch needing +1.
- Every `verify_*` driver fails on numerical tolerances instead of an iteration count.
- `tests/baselines/` (24 logs, 1D) are never regenerated casually. A numerically inert change proves it
  with the bitwise-capture procedure in `docs/dev/testing.md`.

## Rules every change keeps
- Methods return `PetscErrorCode`; every PETSc call is wrapped in `PetscCall` (PETSc marks it `nodiscard`).
- Construction is `create(...)`, not a constructor; a class has `destroy()` iff it owns a PETSc handle.
- Exported classes are `PETSC_VISIBILITY_PUBLIC`, exported free functions `PETSC_EXTERN` (PETSc builds
  with `-fvisibility=hidden`).
- Every TU is a Kokkos one, `src/Xk.kokkos.cxx`. No `KOKKOS_LAMBDA` in `include/ubolt/`.
- Kokkos view typedefs live only in `include/ubolt/types.hpp`; state every layout that matters.
- Every Kokkos policy, `deep_copy` and gemm runs on `PetscGetKokkosExecutionSpace()`; fence before the
  host reads a d2h result (`docs/dev/kokkos.md`).
- No new host-device transfers or device allocations inside the group sweep or an apply; kernels capture
  view copies, never `this`.
- Construction order: the discretisation's `create` decides the decomposition and writes `local_cells`
  (and `n_basis`) into the `PhaseSpace`; everything sized off it comes after and calls
  `ps.check_decomposed()`.
- Rows are `(cell * n_basis + basis) * n_angles + angle`; xsections are per material entry
  (local cell; local element on CG), `Discretisation::n_material_entries()`.
- Terms write through the backend's slot maps (`CooPattern`), never raw indices: off-diagonals in the
  backend's order, the diagonal LAST on DG0/structured/CG (own slot i at DG1), `-1` nulls a slot.
- Never `DMCreateMatrix`: `Discretisation::create_matrix` preallocates the COO pattern PCAIR must see.
- BC contract: a term writes nothing on a row flagged in `BoundaryInfo::is_bc_row_d`, in `assemble_add`
  AND `apply_add`. Under the default ghost-flux there are no flagged rows; only Dirichlet-cell has them.
- A diagonal-carrying term's `add_diagonal` is the same arithmetic, in the same order, as the diagonal its
  `assemble_add` writes: bitwise (`-check_matfree` pins it).
- Rhs builders are not `OperatorTerm`s (`GroupSource`). The source fills ADD, as does `UboltFillInflow`
  under ghost-flux, so the rhs is zeroed then filled inflow first.
- No floating-point reordering in a "refactor" without re-capturing (`docs/dev/testing.md`, bitwise capture).
- The vendored `src/external/nlohmann/json.hpp` is included only by `src/problem_speck.kokkos.cxx`,
  never from `include/ubolt/`.
- `src/sn_lqn_table.hpp` is generated (`python3 src/sn_lqn_table.py`): never edit it, never include it
  from `include/ubolt/`.
- Geometry and direction cosines stay on concrete classes; base-typed code stays dimension-independent.

## Codebase map: `include/ubolt/` + `src/`
| File | What it is | The one thing to know |
|---|---|---|
| `phase_space.hpp` | `PhaseSpace`: cell/angle/basis counts | `n_groups` is metadata, not in the row count; the backend fills `local_cells`/`n_basis` |
| `sn_quadrature.hpp` | `AngularQuadrature` base + `SNQuadrature` (1D), `SNQuadrature2D`/`3D`, `UboltAngularIntegral` | Base = weights, `w_d`, host directions, reflection maps; 1D is Gauss-Legendre any even order, 2D/3D level-symmetric S2-S18 |
| `bc_spec.hpp` | `BCSpec`, `BCFace`, `BCType`, `VacuumTreatment` | Keyed by "Face Sets" ids (`FACE_*`); `GHOST_FLUX` default, `DIRICHLET_CELL` opt-in |
| `material_spec.hpp` | `MaterialSpec`, `MaterialSourceTable`, `UboltFillSource`, `UboltFillCellSource`, `UboltCheckMaterialIds` | Dense material ids 0..n-1; which cells are which material is painted by the backend |
| `material_regions.hpp` | `MaterialInterval1D`, `MaterialBox2D`/`3D` | Centroid membership, inclusive, later regions win |
| `plex_mesh_spec.hpp` | `PlexMeshSpec` | A box built in code (quad/hex or `simplex`) or a mesh file |
| `problem_spec.hpp` | `ProblemSpec`, `BackendKind` | JSON reader (`src/problem_speck`); schema in `docs/problem_files.md` |
| `coo_pattern.hpp` | `CooPattern`, `BoundaryInfo`, `UboltFillInflow`, `UboltZeroReflectRows` | Slot maps + BC rows; `ghost_inflow_d` is the ghost-flux rhs |
| `discretisation.hpp` | `Discretisation` base + `BoundaryRows` | The backend contract (5 steps in the header); `set_uniform_pattern`/`set_pattern` take a `BoundaryRows` |
| `structured_fd_1d/2d/3d.hpp` | `StructuredFD1D`/`2D`/`3D` on a DMDA | 2D/3D share `src/structured_fd_commonk.hpp` (`CheckDALayout<DIM>`); 3D bottom/top are Z |
| `plex_discretisation.hpp` | `PlexDiscretisation` | Mesh creation + painting shared by both plex backends; plumbing in `src/plex_commonk.hpp` (`CheckPlexStratumLayout`) |
| `unstructured_dg.hpp` | `UnstructuredDG`, DG0/DG1 upwind on DMPlex | Two-stage: `create_mesh`, `PhaseSpace::create` off `n_global_cells()`, then `create(ps, quad, bcs, order)`; DG1 is ghost-flux only |
| `unstructured_cg.hpp` | `UnstructuredCG`, P1/Q1 CG-SUPG | Rows per VERTEX, xsections per local ELEMENT; weak BCs, no BC rows; `add_weighted_load` |
| `operator_term.hpp` | `OperatorTerm` | `assembled`/`matrix_free`, `set_group`, `assemble_add`/`apply_add`, `has_diagonal`/`add_diagonal` |
| `terms.hpp` | `StructuredStreamingTerm` (`StreamingTerm1D`/`2D`/`3D`), `StreamingTermDG0`/`DG1`, `RemovalTerm`, `ScatteringTerm` | DG streaming takes no quadrature (reads the backend's); `RemovalTerm` can be matrix-free |
| `terms_cg.hpp` | `SUPGTermCG`, `ScatteringTermCG`, `GroupTransferCG`, `UboltFillSourceCG` | CG siblings of the generic terms: SUPG couples a vertex's star and depends on angle |
| `multigroup.hpp` | `GroupXSections`, `GroupSource`, `GroupTransfer` | Tables LayoutRight, group slowest; `GroupSource` = `add_external` / `set_scalar_flux` / `add_transfer` |
| `transport_operator.hpp` | `TransportOperator` | Assembled terms in one matrix + matrix-free terms behind a MatShell; `set_group`, `assemble`, `diagonal` |
| `transport_solver.hpp` | `TransportSolver` | KSP + composite PC (0 removal, 1 PCAIR, 2 DSA); `refresh()` after each group's refill |
| `block_inverse.hpp` | `ElementBlockInverse` | Inverse (cell, angle) n_basis blocks off the device CSR; the removal stage and `-precon_block_scale` |
| `ref_shift.hpp` | `RefShiftPmats` | k pmats `L + alpha * D_ref` per void-support class; the driver keeps one solver per bin |
| `dsa.hpp` | `DSAPrecon` | Caller-owned, `set_group()` per group; operators in `src/dsa_structuredk`, `dsa_plexk` (DG0+DG1), `dsa_cgk` behind private `DSAOperator` (`src/dsa_operatork.hpp`) |
| `flux_output.hpp` | `UboltWriteScalarFluxVTK`, `UboltCellField` | One writer for every backend: `.vts`/`.vtr` on a DMDA, `.vtu` on a plex (CG writes point data) |
| `types.hpp` | Every Kokkos view typedef | Include first: it pulls in `petscvec_kokkos.hpp` |
| `ubolt.hpp` | Umbrella header | Include before `<petscksp.h>` and friends |

## Codebase map: `tests/`
| Path | What it is / checks |
|---|---|
| `transportk.kokkos.cxx` | THE solve driver and the library's example: any backend, dimension and group count; physics from `-problem`; `BuildBackend` is its one per-backend switch; the group sweep lives here |
| `verify_quadraturek` | The generated quadrature sets against their defining moment conditions |
| `verify_2dk`, `verify_3dk` | Structured discretisation: pure-streaming closed form, shell vs reference matrix, library rhs, `A^T = P A P` under ghost-flux |
| `verify_plexk` | DG0 plex vs its structured twin to rounding, DG1 order/identities, `ElementBlockInverse`, plex DSA incl. voids; serial and -n 2/4 (index of checks in its header) |
| `verify_cgk` | CG-SUPG element tables, constant exactness, second order, two slab benchmarks, CG DSA; serial and -n 2/4 |
| `problems/`, `problems/materials/` | The JSON problem files the recipes name, and shared materials files |
| `meshes/` | Three Gmsh 2.2 `.msh` files, named by problem files and read by `verify_plexk`/`verify_cgk` |
| `baselines/` | 24 reference `-ksp_monitor` logs, 1D only; `make baselines` overwrites them |
| `Makefile` | Literal recipes with pinned `-ksp_max_it`: `run_check`, `run_tests_short_*`, `run_tests_*`, `capture_baselines` |

## Driver knobs (`tests/transportk`)
| Option | What it does | Refused / notes |
|---|---|---|
| `-problem <file.json>` | The problem: mesh, backend, SN order, materials, regions, BCs, output | Required |
| `-precon_stream` | Build PCAIR on a streaming-only pmat | Ignored under `-matfree_removal` (implied); refused on cg_supg |
| `-matfree_removal` | Removal applied matrix-free; the assembled matrix is streaming only, assembled once | Refused on cg_supg and with `-diag_scale` |
| `-precon_ref_shift` | Per-bin pmats `L + alpha * D_ref` (`RefShiftPmats`) | Needs `-matfree_removal`; refused on cg_supg |
| `-precon_ref_k <k>` | Shifted bins per support class; 0/unset = default rule | Needs `-precon_ref_shift` |
| `-precon_dsa` | Add the DSA stage (composite index 2) | Refused with `-diag_scale` |
| `-dsa_consistent_d`, `-dsa_consistent_d_power` | Blend upwind numerical diffusion into D (on by default, p = 1.5) | Structured and DG0 only |
| `-dsa_mip_penalty` | DG1 interior-penalty constant (4) | DG1 only |
| `-dsa_void_sigma_t`, `-dsa_void_bridge`, `-dsa_void_d` | Void threshold (0), bridge (default) vs mask, fixed void D | CG never masks |
| `-dsa_ksp_*`, `-dsa_pc_*` | The inner diffusion solve (default PREONLY + GAMG) | |
| `-precon_block_scale` | PCAIR on `D^{-1} pmat` (`ElementBlockInverse`) | Default ON on DG and CG, OFF structured |
| `-diag_scale` | Diagonally scale the assembled operator and rhs | Refused with `-matfree_removal` and `-precon_dsa` |
| `-supg_zeta <z>` | Override `mesh.supg_zeta` | cg_supg only |
| `-check_inf_medium` | Solution vs the infinite-medium constant (1e-9) | Box mesh, no paint, absorption, unwindowed vacuum faces |
| `-check_matfree` | Matrix-free vs assembled removal: matvec (1e-13), diagonal (exact) | Refused on cg_supg |
| `-check_ref_shift` | Each shifted pmat vs its group's full operator (1e-12) | Needs `-precon_ref_shift` |
| `-flux_vtk <file>` | Overrides `output.flux_vtk`; multigroup writes `_g<g>` per group | Extension checked against the backend before the solve |
| `-ubolt_coo_two_call` | Debug: one `MatSetValuesCOO` per term (read by `TransportOperator`) | Bitwise the same as the default |
cg_supg refuses `-matfree_removal`, `-precon_stream`, `-precon_ref_shift` and `-check_matfree`: it has no
group-independent streaming term (tau depends on sigma_t). PETSc options (`-ksp_*`, `-sub_1_pc_air_*`) pass through.

## Where to read more
- `TODO.md`: current state, open items, roadmap, where the research notes live.
- `docs/architecture.md`: the long-form design - data flow, backends, slot conventions, BC treatments,
  the group sweep and new-physics seams, the preconditioner, quadratures.
- `docs/dsa.md`: DSA derivations per backend, voids.
- `docs/problem_files.md` (schema) and `docs/problem_setup.md` (walkthrough for a new problem).
- `docs/dev/kokkos.md`: before touching any `*.kokkos.cxx`, COO assembly, MatShell code, a new backend, or
  a multi-dimensional view.
- `docs/dev/testing.md` (policy, pins, bitwise capture, baselines), `docs/dev/verification.md` (what each
  check proves), `docs/dev/iteration_counts.md` (every pin), `docs/dev/history.md` (history only).
