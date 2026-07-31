UBOLT: fixed-source Boltzmann (radiation transport) solver library on PETSc + Kokkos + PFLARE.
Architecture: assemble the streaming/removal operator (MATAIJKOKKOS via the PETSc COO
interface so assembly runs on device), apply scattering matrix-free (MatShell + Kokkos),
precondition with PCComposite = removal shell PC + PCAIR (pflare inverts streaming).

Codebase map
- `tests/`: test drivers (they are also the examples) + a Makefile of literal run commands.
  `tests/slab_1dk.kokkos.cxx`: 1D slab single-group SN driver. `tests/baselines/`: captured
  reference `-ksp_monitor` logs — never regenerate casually (see `docs/dev/testing.md`).
- `src/` + `include/ubolt/`: the libubolt library — `quadrature` (SN quadrature),
  `streaming` (COO preallocation + streaming/removal fills), `matshell` (streaming/removal
  + matrix-free scatter apply), `precon` (composite removal shell PC + PCAIR); `types.hpp`
  owns every Kokkos view typedef, `ubolt.hpp` is the umbrella header. Kokkos translation
  units are named `Xk.kokkos.cxx` (the suffix triggers PETSc's Kokkos build rules).
  See `TODO.md` for the roadmap and current phase.
- Library conventions: public functions return `PetscErrorCode` and every PETSc call is
  wrapped in `PetscCall` (PETSc marks `PetscErrorCode` `nodiscard`, so unchecked calls
  warn); public declarations are tagged `PETSC_EXTERN` because PETSc compiles with
  `-fvisibility=hidden`.
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

Tests
1. Run the test targets below once. Trust `make`'s exit code: 0 means all tests passed;
   any failure breaks the run with a non-zero code. Don't re-run to grep the output.
2. In top repo directory: `make check`
3. In top repo directory: `make tests_short`, full suite: `make tests`
4. Pass/fail = pinned `-ksp_max_it` per recipe: an iteration-count regression fails the run.
   Numerical refactors must additionally diff residual histories against `tests/baselines/`
   (see `docs/dev/testing.md`).
