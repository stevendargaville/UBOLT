UBOLT: fixed-source Boltzmann (radiation transport) solver library on PETSc + Kokkos + PFLARE.
Architecture: assemble the streaming/removal operator (MATAIJKOKKOS via the PETSc COO
interface so assembly runs on device), apply scattering matrix-free (MatShell + Kokkos),
precondition with PCComposite = removal shell PC + PCAIR (pflare inverts streaming).
Energy groups are solved one at a time in a group Gauss-Seidel sweep, so the sparsity is
preallocated once and each group is a values-only refill of the same matrix.
The discretisation is DM-backed (1D and 2D DMDAs): the DM owns the mesh, the layout and
the parallel decomposition, but deliberately creates no matrices or vectors itself.

Codebase map
- `tests/`: test drivers (they are also the examples) + a Makefile of literal run commands.
  `tests/slab_1dk.kokkos.cxx`: 1D slab single-group SN driver.
  `tests/slab_1d_mgk.kokkos.cxx`: the multigroup one (`-n_groups`, `-sigma_transfer`); it
  is also where the group sweep itself lives, until a second sweep strategy justifies
  promoting it into the library.
  `tests/box_2dk.kokkos.cxx`: the 2D single-group driver (`-n_cells_x/-n_cells_y`,
  `-length_x/-length_y`, `-sigma_scatter`).
  `tests/verify_2dk.kokkos.cxx`: the 2D discretisation check — a pure-streaming closed
  form and the shell operator against a reference matrix. `tests/baselines/`: captured
  reference `-ksp_monitor` logs, 1D only — never regenerate casually
  (see `docs/dev/testing.md`).
- `src/` + `include/ubolt/`: the libubolt library. `PhaseSpace` (sizes only — `n_groups` is
  metadata on it, NOT part of the row count, and it does NOT decide the decomposition),
  `AngularQuadrature` (the dimension-independent part: n_angles, sum_weights and the
  weight matrix) with `SNQuadrature` / `SNQuadrature2D` under it, plus
  `UboltAngularIntegral`, the shared angular integral; `Discretisation` (the backend base:
  `create_matrix`, `coo_pattern`, `boundary_info`, `destroy`, `dm`, and
  `set_uniform_pattern` for a fixed-entries-per-row backend) with `StructuredFD1D` and
  `StructuredFD2D` under it (each owns a DMDA and through it the mesh, the cell-based
  decomposition and the COO sparsity, and adds only its own geometry — `dx()`, `dy()`);
  `OperatorTerm` and the `Streaming`/`Streaming2D`/`Removal`/`Scattering` terms,
  `GroupXSections` + `GroupTransfer` (multigroup xsection tables and the group-to-group
  source), `TransportOperator` (assembled terms in one matrix + matrix-free terms behind a
  MatShell), `TransportSolver` (KSP + the composite PC, plus `refresh()` for what the PC
  caches off the assembled matrix). `types.hpp` owns every Kokkos view typedef, `ubolt.hpp`
  is the umbrella header. Every translation unit is a Kokkos one, named `Xk.kokkos.cxx`
  (the suffix triggers PETSc's Kokkos build rules). See `TODO.md` for the roadmap and
  current phase.
- Dimension-independent vs not: everything that goes through the slot maps or the weights
  takes the `Discretisation` / `AngularQuadrature` base and works in any dimension
  (`TransportOperator`, `RemovalTerm`, `ScatteringTerm`, `GroupTransfer`). The geometry
  and the direction cosines stay on the concrete classes, because the only thing that
  reads them is the streaming term, which owns the upwind slot convention and so needs a
  per-dimension sibling anyway. Keep new abstractions on that line.
- Library conventions: methods return `PetscErrorCode` and every PETSc call is wrapped in
  `PetscCall` (PETSc marks `PetscErrorCode` `nodiscard`, so unchecked calls warn);
  construction is a `create(...)` method, not a constructor, so it can return errors, and
  the objects that own PETSc handles have a matching `destroy()`. Exported classes are
  tagged `PETSC_VISIBILITY_PUBLIC` and exported free functions `PETSC_EXTERN`, because
  PETSc compiles with `-fvisibility=hidden`.
- Construction ORDER matters: the discretisation's `create` is what decides the parallel
  decomposition (its DMDA does) and it writes `local_cells` back into the `PhaseSpace`,
  which `PhaseSpace::create` leaves as `PETSC_DECIDE`. So the discretisation comes first,
  and anything sized off the phase space comes after. Every `create` that reads
  `local_cells`/`local_rows()` calls `ps.check_decomposed()` to say so out loud —
  add that call to new ones.
- New physics = subclass `OperatorTerm` (assembled contribution into the shared COO values
  and/or matrix-free apply). Discretisation backends produce a `CooPattern` (slot maps) +
  `BoundaryInfo` (Dirichlet row mask); terms write through those, never raw indices.
  The Dirichlet contract binds BOTH halves of a term, `assemble_add` and `apply_add`:
  leave flagged rows alone, they carry the boundary condition, and the assembly has already
  put the identity there. Anything that builds a right hand side rather than acting on the
  unknowns being solved for is NOT an `OperatorTerm` — see `GroupTransfer` — but it owes
  the mask the same thing.
- PETSc source is at `$PETSC_DIR/$PETSC_ARCH`. Both env variables must be set.
  PFLARE is at `$PFLARE_DIR` (defaults to `/home/sdargavi/projects/PFLARE` in the Makefile).

Read only when the task needs it
- `TODO.md` — roadmap, current phase, per-phase verification criteria, research notes
- `docs/dev/testing.md` — before adding or modifying tests, and before touching baselines
- `docs/dev/kokkos.md` — before touching `*.kokkos.cxx`, COO assembly or MatShell code,
  before writing a new discretisation backend (it has the COO/slot-order/Dirichlet
  discipline a backend and its terms have to agree on), and before adding any
  multi-dimensional view (layouts differ between host and device backends, and the obvious
  compile-time contiguity guard does not work)

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
   `verify_2dk` is the exception — it fails on numerical tolerances instead.
   Numerical refactors must additionally diff residual histories against `tests/baselines/`
   (see `docs/dev/testing.md`).
