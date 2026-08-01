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

## Layout: CPU vs GPU
The default layout of `Kokkos::View<T**, DefaultMemorySpace>` is **not the same on both
backends** — `LayoutRight` on a host space, `LayoutLeft` on CUDA/HIP. Any multi-dimensional
view whose element ordering matters must therefore state its layout explicitly, and every
one in `types.hpp` that matters does:

- `PetscScalar2DConstKokkosView` — `LayoutRight`, so a 1D Vec reshapes to (cell, angle)
  with angle contiguous. This is *correctness*: the dof ordering is angle-fastest, and the
  default layout would transpose the meaning of the reshape on a GPU build.
- `PetscScalar2DRightKokkosView` / `PetscScalar3DRightKokkosView` (the multigroup xsection
  tables) — `LayoutRight` with cell as the **fastest** extent, so fixing the group index
  or pair slices out a contiguous run of `local_cells`. That is the right choice on both
  backends for the same reason it is one array: a stride-1 walk vectorises on a CPU and
  coalesces on a GPU. Reversing it to cell-slowest would put consecutive threads
  `n_groups` (or `n_groups^2`) elements apart.
- Views with a trailing extent of 1 (`w_d_`, `scalar_flux_d_`) are identical under either
  layout, which is why they can keep the default typedef.

**Do not guard contiguity with `std::is_assignable`.** Kokkos sets `is_assignable_layout`
whenever the *destination* rank is 1, so assigning a strided rank-1 subview to a
`LayoutRight` view compiles cleanly and then aborts at run time with "View assignment must
have compatible layouts". Test the layout instead — a non-contiguous subview comes back as
`Kokkos::LayoutStride`:

```cpp
using Slice = decltype(Kokkos::subview(SomeView(), 0, 0, Kokkos::ALL()));
static_assert(!std::is_same_v<typename Slice::array_layout, Kokkos::LayoutStride>, "...");
```
`src/multigroupk.kokkos.cxx` does this for both xsection tables.

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
- `MatSetPreallocationCOO` happens ONCE on the host, in the discretisation backend: it
  builds the `oor`/`ooc` index arrays and keeps them, so every matrix built from the same
  discretisation (`StructuredFD1D::create_matrix`) shares the sparsity. Repeated assembly
  (per group, per parameter change) is values-only: terms fill a device view with a
  `Kokkos::parallel_for` and `TransportOperator` calls `MatSetValuesCOO`.
- Dirichlet trick: a `-1` row/col index in the preallocation means "ignore this entry", so
  boundary rows keep only their diagonal. The ordering of entries per row is fixed at
  preallocation time (1D: off-diagonal first, diagonal second, every row regardless of
  angle sign) and value fills must match it exactly.
- Terms address entries through the `CooPattern` slot maps — `row_slot_offset_d`
  (CSR-shaped COO slot ranges per row) and `diag_slot_d` (which slot is the diagonal) —
  never through raw COO positions.
- Contract: assembled terms must contribute NOTHING to rows flagged in
  `BoundaryInfo::is_dirichlet_row_d`. `TransportOperator::assemble_into` writes the
  identity on those rows after the terms have run, so a term never has to know what the
  boundary condition is.
- All the assembled terms add into ONE shared values array and go in with a single
  `MatSetValuesCOO(INSERT_VALUES)`. `-ubolt_coo_two_call` is the debug fallback: one call
  per term, the first INSERTing and the rest ADDing. Same per-entry arithmetic in the same
  order, so the two paths agree bit for bit — the test suite runs both.

## MatShell rules
- `MatShellSetVecType` from the assembled matrix's vectype is MANDATORY after
  `MatCreateShell` — without it the shell creates host vectors and every Kokkos view
  access silently round-trips through host.
- Matrix-free terms keep persistent scratch views (e.g. `scalar_flux_d`); never allocate
  device memory inside an apply.
- Input views must be `const` views (`VecGetKokkosView` with a const view type) so PETSc
  doesn't mark the vector dirty.
- Mind device-view scope: views captured by a MatShell context must outlive the KSP solve,
  and all of them must be gone before `PetscFinalize` takes Kokkos down (the driver keeps
  the whole problem in a block scope).
- Kernels in member functions must capture value copies of what they need, never `this` —
  a member access inside a `KOKKOS_LAMBDA` dereferences a host pointer on the device.
  Take local copies of the views at the top of the method (view copies are shallow).
- Don't define a `KOKKOS_LAMBDA` inside another lambda: nvcc rejects extended device
  lambdas there. Use a file-static free function taking the views instead.
- PETSc brings Kokkos up lazily, so anything that allocates device memory before the first
  Kokkos-typed PETSc object exists must call `PetscKokkosInitializeCheck()` first.

## pflare / GPU dispatch
- pflare dispatches on the matrix type at RUNTIME: PCAIR only takes its Kokkos/GPU paths
  when handed MATAIJKOKKOS matrices and VECKOKKOS vectors. UBOLT sets these types in code;
  when a DM creates objects (Phase 3+) run with
  `-dm_mat_type aijkokkos -dm_vec_type kokkos`.
- `PFLARE_KOKKOS_DEBUG=1` makes pflare run CPU and Kokkos paths side by side and abort on
  mismatch — useful when a solve misbehaves only on device.
