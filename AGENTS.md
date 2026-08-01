UBOLT: fixed-source Boltzmann (radiation transport) solver library on PETSc + Kokkos + PFLARE.
Architecture: assemble the streaming/removal operator (MATAIJKOKKOS via the PETSc COO
interface so assembly runs on device), apply scattering matrix-free (MatShell + Kokkos),
precondition with PCComposite = removal shell PC + PCAIR (pflare inverts streaming).
Energy groups are solved one at a time in a group Gauss-Seidel sweep, so the sparsity is
preallocated once and each group is a values-only refill of the same matrix.
The discretisation is DM-backed (a 1D DMDA so far): the DM owns the mesh, the layout and
the parallel decomposition, but deliberately creates no matrices or vectors itself.

Codebase map
- `tests/`: test drivers (they are also the examples) + a Makefile of literal run commands.
  `tests/slab_1dk.kokkos.cxx`: 1D slab single-group SN driver.
  `tests/slab_1d_mgk.kokkos.cxx`: the multigroup one (`-n_groups`, `-sigma_transfer`); it
  is also where the group sweep itself lives, until a second sweep strategy justifies
  promoting it into the library. `tests/baselines/`: captured reference `-ksp_monitor`
  logs — never regenerate casually (see `docs/dev/testing.md`).
- `src/` + `include/ubolt/`: the libubolt library. `PhaseSpace` (sizes only — `n_groups` is
  metadata on it, NOT part of the row count, and it does NOT decide the decomposition),
  `SNQuadrature` (+ `UboltAngularIntegral`, the shared angular integral), `StructuredFD1D`
  (the discretisation: owns a 1D DMDA and through it the mesh, the cell-based
  decomposition and the COO sparsity; hands out `CooPattern` + `BoundaryInfo`),
  `OperatorTerm` and the `Streaming`/`Removal`/`Scattering` terms,
  `GroupXSections` + `GroupTransfer` (multigroup xsection tables and the group-to-group
  source), `TransportOperator` (assembled terms in one matrix + matrix-free terms behind a
  MatShell), `TransportSolver` (KSP + the composite PC, plus `refresh()` for what the PC
  caches off the assembled matrix). `types.hpp` owns every Kokkos view typedef, `ubolt.hpp`
  is the umbrella header. Every translation unit is a Kokkos one, named `Xk.kokkos.cxx`
  (the suffix triggers PETSc's Kokkos build rules). See `TODO.md` for the roadmap and
  current phase.
- Library conventions: methods return `PetscErrorCode` and every PETSc call is wrapped in
  `PetscCall` (PETSc marks `PetscErrorCode` `nodiscard`, so unchecked calls warn);
  construction is a `create(...)` method, not a constructor, so it can return errors, and
  the objects that own PETSc handles have a matching `destroy()`. Exported classes are
  tagged `PETSC_VISIBILITY_PUBLIC` and exported free functions `PETSC_EXTERN`, because
  PETSc compiles with `-fvisibility=hidden`.
- Construction ORDER matters: `StructuredFD1D::create` is what decides the parallel
  decomposition (its DMDA does) and it writes `local_cells` back into the `PhaseSpace`,
  which `PhaseSpace::create` leaves as `PETSC_DECIDE`. So the discretisation comes first,
  and anything sized off the phase space comes after. Every `create` that reads
  `local_cells`/`local_rows()` calls `ps.check_decomposed()` to say so out loud —
  add that call to new ones.
- New physics = subclass `OperatorTerm` (assembled contribution into the shared COO values
  and/or matrix-free apply). Discretisation backends produce a `CooPattern` (slot maps) +
  `BoundaryInfo` (Dirichlet row mask); terms write through those, never raw indices.
  Anything that builds a right hand side rather than acting on the unknowns being solved
  for is NOT an `OperatorTerm` — see `GroupTransfer` — but it still owes the Dirichlet mask
  the same contract: leave flagged rows alone, they carry the boundary condition.
- PETSc source is at `$PETSC_DIR/$PETSC_ARCH`. Both env variables must be set.
  PFLARE is at `$PFLARE_DIR` (defaults to `/home/sdargavi/projects/PFLARE` in the Makefile).

Read only when the task needs it
- `TODO.md` — roadmap, current phase, per-phase verification criteria, research notes
- `docs/dev/testing.md` — before adding or modifying tests, and before touching baselines
- `docs/dev/kokkos.md` — before touching `*.kokkos.cxx`, COO assembly or MatShell code, and
  before adding any multi-dimensional view (layouts differ between host and device
  backends, and the obvious compile-time contiguity guard does not work)

Build
1. In top repo directory: `make -j3 build_tests` (PETSc >= 3.25 configured with Kokkos
   required). This builds `lib/libubolt.{so,a}` first; `make` on its own builds just the library.
2. Rule: fix all compile warnings (CI will build with `-Werror`).
3. PETSc's rules generate no `.d` files, so header dependencies are declared by hand in the
   Makefiles: `$(OBJS)` depends on `$(UBOLT_HEADERS)`, and the drivers are handled by
   `$(HEADER_STAMP)`, which deletes the stale driver objects/binaries so PETSc's own rules
   rebuild them. The drivers cannot just take the headers as prerequisites — PETSc's
   `% : %.kokkos.cxx` rule compiles and links in one go and passes every remaining
   prerequisite to the linker. Keep everything compiling through PETSc's rules with
   PETSc's configured flags; add prerequisites, never flags.

Tests
1. Run the test targets below once. Trust `make`'s exit code: 0 means all tests passed;
   any failure breaks the run with a non-zero code. Don't re-run to grep the output.
2. In top repo directory: `make check`
3. In top repo directory: `make tests_short`, full suite: `make tests`
4. Pass/fail = pinned `-ksp_max_it` per recipe: an iteration-count regression fails the run.
   Numerical refactors must additionally diff residual histories against `tests/baselines/`
   (see `docs/dev/testing.md`).
