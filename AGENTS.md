UBOLT: fixed-source Boltzmann (radiation transport) solver library on PETSc + Kokkos + PFLARE.
Architecture: assemble the streaming/removal operator (MATAIJKOKKOS via the PETSc COO
interface so assembly runs on device), apply scattering matrix-free (MatShell + Kokkos),
precondition with PCComposite = removal shell PC + PCAIR (pflare inverts streaming).

Codebase map
- `tests/`: test drivers (they are also the examples) + a Makefile of literal run commands.
  `tests/slab_1dk.kokkos.cxx`: 1D slab single-group SN driver. `tests/baselines/`: captured
  reference `-ksp_monitor` logs — never regenerate casually (see `docs/dev/testing.md`).
- `src/` + `include/ubolt/`: the libubolt library. `PhaseSpace` (sizes + the cell-based
  decomposition), `SNQuadrature`, `StructuredFD1D` (the discretisation: owns the mesh and
  the COO sparsity, hands out `CooPattern` + `BoundaryInfo`), `OperatorTerm` and the
  `Streaming`/`Removal`/`Scattering` terms, `TransportOperator` (assembled terms in one
  matrix + matrix-free terms behind a MatShell), `TransportSolver` (KSP + the composite
  PC). `types.hpp` owns every Kokkos view typedef, `ubolt.hpp` is the umbrella header.
  Every translation unit is a Kokkos one, named `Xk.kokkos.cxx` (the suffix triggers
  PETSc's Kokkos build rules). See `TODO.md` for the roadmap and current phase.
- Library conventions: methods return `PetscErrorCode` and every PETSc call is wrapped in
  `PetscCall` (PETSc marks `PetscErrorCode` `nodiscard`, so unchecked calls warn);
  construction is a `create(...)` method, not a constructor, so it can return errors, and
  the objects that own PETSc handles have a matching `destroy()`. Exported classes are
  tagged `PETSC_VISIBILITY_PUBLIC` and exported free functions `PETSC_EXTERN`, because
  PETSc compiles with `-fvisibility=hidden`.
- New physics = subclass `OperatorTerm` (assembled contribution into the shared COO values
  and/or matrix-free apply). Discretisation backends produce a `CooPattern` (slot maps) +
  `BoundaryInfo` (Dirichlet row mask); terms write through those, never raw indices.
- PETSc source is at `$PETSC_DIR/$PETSC_ARCH`. Both env variables must be set.
  PFLARE is at `$PFLARE_DIR` (defaults to `/home/sdargavi/projects/PFLARE` in the Makefile).

Read only when the task needs it
- `TODO.md` — roadmap, current phase, per-phase verification criteria, research notes
- `docs/dev/testing.md` — before adding or modifying tests, and before touching baselines
- `docs/dev/kokkos.md` — before touching `*.kokkos.cxx`, COO assembly or MatShell code

Build
1. In top repo directory: `make -j3 build_tests` (PETSc >= 3.25 configured with Kokkos
   required). This builds `lib/libubolt.{so,a}` first; `make` on its own builds just the library.
2. Rule: fix all compile warnings (CI will build with `-Werror`).
3. There is NO header dependency tracking (PETSc's basic rules generate no `.d` files), so
   an edit to a header does not rebuild the `.o` files that include it. After touching
   anything in `include/ubolt/`, `make clean` first. Getting this wrong is not a link
   error: a struct whose layout changed (e.g. adding a field to `PhaseSpace`) silently
   reads members at the wrong offset and shows up as a garbage-sized Kokkos allocation.

Tests
1. Run the test targets below once. Trust `make`'s exit code: 0 means all tests passed;
   any failure breaks the run with a non-zero code. Don't re-run to grep the output.
2. In top repo directory: `make check`
3. In top repo directory: `make tests_short`, full suite: `make tests`
4. Pass/fail = pinned `-ksp_max_it` per recipe: an iteration-count regression fails the run.
   Numerical refactors must additionally diff residual histories against `tests/baselines/`
   (see `docs/dev/testing.md`).
