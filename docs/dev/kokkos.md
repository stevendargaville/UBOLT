# Kokkos conventions in UBOLT

## Views
- All view typedefs live in one place, `include/ubolt/types.hpp`. Use them — don't declare
  ad-hoc `Kokkos::View` types. `PetscScalar2DConstKokkosView` and the host-unmanaged 2D
  views are deliberately `LayoutRight` so that a 1D Vec reshapes to (cells, angles) with
  angle contiguous (angle-fastest dof ordering).
- Kokkos translation units are named `Xk.kokkos.cxx` — the `.kokkos.cxx` suffix triggers
  PETSc's Kokkos build rules, and the trailing `k` on the stem is the PFLARE convention.
- No `KOKKOS_LAMBDA` in public headers: kernels stay in `src/*.kokkos.cxx` so consumers of
  libubolt compile with a plain C++ compiler.

## Include order
- `petscvec_kokkos.hpp` must be the FIRST PETSc header in a C++ translation unit (it
  `#error`s otherwise). `types.hpp` pulls it in, so ubolt headers that need views include
  `ubolt/types.hpp` before anything else, and `ubolt/ubolt.hpp` lists it first — include
  the umbrella header before `<petscksp.h>` and friends.
- Only Kokkos-using files include `types.hpp`: keeping Kokkos out of the plain `.cxx`
  translation units means they don't need the Kokkos device compiler.
- PETSc compiles C++ with `-fvisibility=hidden` but strips it from its Kokkos rule, so
  which rule compiled a file would otherwise decide whether its symbols are exported.
  Public declarations are therefore tagged `PETSC_EXTERN` (which is
  `extern "C" __attribute__((visibility("default")))` in C++, so the library exports
  unmangled names — public functions can't be overloaded).

## COO assembly discipline
The streaming/removal operator is assembled with the PETSc COO interface so that value
fill runs on device (MATAIJKOKKOS dispatches `MatSetValuesCOO` to the GPU):
- `MatSetPreallocationCOO` happens ONCE on the host: build `oor`/`ooc` index arrays there.
  Repeated assembly (per group, per parameter change) is values-only:
  fill a device view with a `Kokkos::parallel_for` and call `MatSetValuesCOO`.
- Dirichlet trick: a `-1` row/col index in the preallocation means "ignore this entry".
  Boundary rows keep only their diagonal (set to 1 by the streaming fill); the ordering of
  entries per row is fixed at preallocation time and value fills must match it exactly
  (currently: off-diagonal first, diagonal second, for every row regardless of angle sign).
- Phase 1 generalizes this to slot maps: `row_slot_offset_d` (CSR-shaped COO slot ranges
  per row) and `diag_slot_d` (which slot is the diagonal). Assembled terms write through
  the slot maps, never raw COO positions.
- Contract: assembled terms must contribute NOTHING to rows flagged in the Dirichlet row
  mask (`is_dirichlet_row_d` from Phase 1). Today `add_removal` re-zeroes those diagonals
  by hand — that implicit coupling is exactly what the mask replaces.

## MatShell rules
- `MatShellSetVecType` from the assembled matrix's vectype is MANDATORY after
  `MatCreateShell` — without it the shell creates host vectors and every Kokkos view
  access silently round-trips through host.
- Matrix-free terms keep persistent scratch views (e.g. `scalar_flux_d`); never allocate
  device memory inside an apply.
- Input views must be `const` views (`VecGetKokkosView` with a const view type) so PETSc
  doesn't mark the vector dirty.
- Mind device-view scope: views captured by a MatShell context must outlive the KSP solve
  (the current driver keeps them alive in a block scope around the solve).

## pflare / GPU dispatch
- pflare dispatches on the matrix type at RUNTIME: PCAIR only takes its Kokkos/GPU paths
  when handed MATAIJKOKKOS matrices and VECKOKKOS vectors. UBOLT sets these types in code;
  when a DM creates objects (Phase 3+) run with
  `-dm_mat_type aijkokkos -dm_vec_type kokkos`.
- `PFLARE_KOKKOS_DEBUG=1` makes pflare run CPU and Kokkos paths side by side and abort on
  mismatch — useful when a solve misbehaves only on device.
