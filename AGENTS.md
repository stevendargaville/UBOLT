UBOLT: fixed-source Boltzmann (radiation transport) solver library on PETSc + Kokkos + PFLARE.
Architecture: assemble the streaming/removal operator (MATAIJKOKKOS via the PETSc COO
interface so assembly runs on device), apply scattering matrix-free (MatShell + Kokkos),
precondition with PCComposite = removal shell PC + PCAIR (pflare inverts streaming).
Energy groups are solved one at a time in a group Gauss-Seidel sweep, so the sparsity is
preallocated once and each group is a values-only refill of the same matrix.
The discretisation is DM-backed (1D, 2D and 3D DMDAs, and a 2D/3D DMPlex - upwind DG
or CG-SUPG on it): the DM owns the mesh (including its
coordinates, set by the backend for output), the layout and the parallel decomposition,
but deliberately creates no solver matrices or vectors itself.

Codebase map
- `tests/`: the drivers (they are also the examples) + a Makefile of literal run commands.
  `tests/transportk.kokkos.cxx`: THE solve driver, any dimension and group count — all
  physics comes from `-problem <file.json>` (schema: `docs/problem_files.md`), the CLI
  keeps only PETSc options and the strategy/verification knobs (`-precon_stream`,
  `-matfree_removal`, `-precon_ref_shift`, `-precon_dsa`, `-precon_block_scale`, `-diag_scale`,
  `-check_inf_medium`, `-check_matfree`, `-check_ref_shift`, `-flux_vtk` override). It is also where the group
  Gauss-Seidel sweep lives, until a second sweep strategy justifies promoting it into
  the library. It replaced the per-problem drivers (`slab_1dk`, `slab_1d_mgk`,
  `box_2dk`) in Aug 2026, verified byte-for-byte against all 24 baselines first.
  `tests/problems/`: the problem files the recipes name (+ `problems/materials/` for
  shared materials files); one file per distinct physics setup.
  `tests/verify_2dk.kokkos.cxx` / `tests/verify_3dk.kokkos.cxx`: the 2D/3D
  discretisation checks — a pure-streaming closed form and the shell operator against a
  reference matrix (the 3D mixed config puts three reflective faces around one corner),
  each under both vacuum treatments, plus the library-built boundary rhs against a
  constant solution and, under ghost-flux, the opposite-ordinate identity `A^T = P A P`
  (vacuum, mixed and all-reflect faces — it holds for any BC mix).
  `tests/verify_plexk.kokkos.cxx`: the unstructured backend's check, serial and -n 2/4 —
  the plex quad/hex box against its `StructuredFD2D`/`3D` twin to ROUNDING (matrix, BC
  rows, rhs, scatter, a solve; rows matched by centroid, each side's row from its OWN
  layout, in both vacuum treatments), the infinite medium on triangles/tets, the
  ghost-flux opposite-ordinate identity `(VA)^T = P(VA)P` (V = cell volumes; DG0 rows are
  per unit volume, so only the volume-weighted operator is exactly symmetric this way),
  geometry/Face Sets invariants, error
  paths, and the `.vtu` writing each cell once; and at DG1 (no twin) no BC rows, the same
  `(VA)^T = P(VA)P` with the reflective couplings inside it, a constant through the
  library rhs, a LINEAR field through the streaming matrix giving exactly `Omega . b` in
  interior cells, and second-order convergence against an exact SN solution; at both
  orders, the `ElementBlockInverse` of every operator those checks build (identity blocks
  in `D^{-1} A`, invariance under a row scaling, `apply()` against the scaled matrix, and a
  reuse `scale()` bumping the matrix state a PC compares, which fails only at -n 2+); and
  the plex DSA — its diffusion matrix = the twin's structured one times the cell volume,
  `P D^-1 R` to rounding with tight inner solves, and at DG1 the interior penalty matrix
  symmetric, a linear field mapped to `V sigma_a u` on interior cells to rounding, and a
  round trip through `apply()`; and the DSA's voids — the FD twin with its painted box
  made a void (the two operators agree bridged and masked), under the mask at DG0/DG1 a
  box with void high-x cells against the box of only the low-x ones (non-void rows
  identical, void rows V I, the correction equal there and zero in the void), and the
  bridge at DG0/DG1, 2D/3D, on a box split by a void slab (the chord 4 V / S, SPD by
  a dense Cholesky, a constant mapped to V sigma_a off the vacuum boundary - zero in
  the void - and a residual on one side corrected on the other, where the mask gives
  zero). `tests/meshes/`: mesh files the problem
  files name (a hand-written Gmsh 2.2 `.msh` today).
  `tests/verify_cgk.kokkos.cxx`: the CG-SUPG backend's check, serial and -n 2/4 — the
  element tables (partition of unity, linear exactness) on every cell shape, the
  layout, a constant exact to rounding through operator + rhs (voids, reflective
  faces, zeta 0.5 and 2), second order against an exact SN solution, the two
  literature slab benchmarks (the SAAF-LS void slab, arXiv 1605.05388, and Hammer/
  Morel/Wang's thin/thick slab, arXiv 1902.08729) against the EXACT SN solution of
  the same quadrature, error paths, and the CG DSA: its diffusion matrix = the
  transport restricted onto an isotropic flux, `m R A P`, to rounding (thick material +
  an unbridged void), the bridged voids' chord against closed forms, and the
  singular all-reflective case refused.
  `tests/verify_quadraturek.kokkos.cxx`: the quadrature sets themselves, against the
  moment conditions that define them — needed because they are generated, not tabulated.
  `tests/baselines/`: captured
  reference `-ksp_monitor` logs, 1D only — never regenerate casually
  (see `docs/dev/testing.md`).
- `src/` + `include/ubolt/`: the libubolt library. `PhaseSpace` (sizes only — `n_groups` is
  metadata on it, NOT part of the row count, and it does NOT decide the decomposition;
  `n_basis` IS part of it — spatial dofs per cell, 1 except DG1, written by the backend
  like `local_cells` — rows are `(cell * n_basis + basis) * n_angles + angle`, a (cell,
  basis) "node" is a contiguous run of angles, and the shared terms work per node with the
  xsection per cell, `row / rows_per_cell()`),
  `AngularQuadrature` (the dimension-independent part: n_angles, sum_weights and the
  weight matrix) with `SNQuadrature` / `SNQuadrature2D` / `SNQuadrature3D` under it
  (1D is a Gauss-Legendre rule generated by Newton at create() time, so ANY even order;
  2D/3D are the level-symmetric sets, even orders 2..18, whose constants come from the
  generated `src/sn_lqn_table.hpp` — emit it with `python3 src/sn_lqn_table.py`, never
  edit it, and never include it from `include/ubolt/`, the vendored-json rule. The 3D set
  folds nothing so it has twice the ordinates of the same-order 2D set, and beyond S4 the
  weights are NOT equal within a set), plus
  `UboltAngularIntegral`, the shared angular integral; `BCSpec` (boundary label id →
  BC family {vacuum, reflect}, plus the vacuum face's prescribed inflow and optional
  tangential window, plus the `VacuumTreatment` — `GHOST_FLUX`, the default since Sep
  2026: the inflow boundary cell keeps its full stencil, the outside-pointing slot is
  nulled and the inflow enters the rhs through the face flux
  (`BoundaryInfo::ghost_inflow_d`, added by `UboltFillInflow`; `UboltFillSource` ADDS
  for that reason); or `DIRICHLET_CELL`, opt-in via the problem file's
  `vacuum_treatment` (and the default before Sep 2026): an inflow boundary cell is an
  identity row carrying the inflow. Under ghost-flux a REFLECTIVE face is a face flux
  too — the outside-pointing slot points at the mirrored angle in the same cell — so
  there are NO BC rows, and a mixed reflect/vacuum corner takes each face's own ghost
  value (the first ghost-flux cut let reflect win there and the corner lost the vacuum
  inflow, an O(1) error; fixed Sep 2026); under Dirichlet-cell the reflective row is
  `psi(a) - psi(mirror) = 0` and vacuum wins a corner. Ghost-flux is what makes the
  streaming/removal block for `-Omega_d` exactly the transpose of the block for
  `Omega_d` on every row, for ANY mix of vacuum and reflective faces (the groundwork
  for transposed half-quadrature preconditioning; on the plex it is the
  volume-weighted `(VA)^T = P(VA)P`; not under Dirichlet-cell; streaming + removal
  only — the isotropic scatter is not symmetric once the weights differ; scope in
  `bc_spec.hpp`), and `DSAPrecon` takes either.
  Keyed the way DMPlex "Face Sets" ids are — the structured
  backends' `FACE_*` constants match PETSc's box-mesh convention, and a problem file's
  `boundary_conditions` names faces onto them); `MaterialSpec` (BCSpec's sibling
  for cell data: per-material, per-group xsections + external source — an isotropic
  strength, shared over the ordinates as `source / sum_weights` by `UboltFillSource` —
  DENSE indices 0..n-1
  because the tables reach device kernels — a DMPlex backend remaps "Cell Sets" labels to
  indices at paint time; which cells are which material is geometry, painted per-backend
  into a per-cell index view, then expanded through `GroupXSections::set_from_materials`
  and `UboltFillSource`, so terms never learn materials exist; a problem file's
  `regions` section is what gets painted); `ProblemSpec` (the JSON problem-definition
  reader filling MaterialSpec/BCSpec/paint lists/mesh sizes from one file — materials
  are a multigroup schema interoperable with other codes' materials files, path or
  inline, plus UBOLT's `Source[g]`
  extension; parsing lives in the one TU that includes the vendored
  `src/external/nlohmann/json.hpp`, which must NEVER be included from
  `include/ubolt/`); `Discretisation` (the backend
  base: `create_matrix`, `coo_pattern`, `boundary_info`, `destroy`, `dm`, and
  `set_uniform_pattern` for a fixed-entries-per-row backend, a wrapper over the general
  CSR-shaped `set_pattern` a variable-nnz one calls) with `StructuredFD1D`,
  `StructuredFD2D` and `StructuredFD3D` under it (each owns a DMDA and through it the
  mesh, the cell-based
  decomposition and the COO sparsity, and adds only its own geometry — `dx()`, `dy()`,
  `dz()`, and the material painting, `paint_intervals` / `paint_boxes` — NOTE the 3D
  `FACE_*` ids follow PETSc's convention, where bottom/top are the Z faces, not y as
  in 2D) and `UnstructuredDG` beside them (upwind DG on a DMPlex at order 0 or 1 — the
  `order` argument of `create`, `mesh.order` in a problem file — 2D/3D, any cell shape; the mesh from a `PlexMeshSpec` — a box built in code, quads/hexes or
  `simplex` triangles/tets, or a file PETSc reads. Construction is TWO-STAGE because the
  mesh decides the global cell count: `create_mesh`, then `PhaseSpace::create` off
  `n_global_cells()` — never n_x * n_y, a simplex box has 2 or 6 cells per box cell —
  then `create(ps, quad, bcs)`. Slots: `n_faces(c) + 1` per row, one per face in CONE
  order then the diagonal LAST, through `set_pattern`; a face slot is live only on an
  interior inflow face. Owned cells in point order are local cell k, asserted by
  `CheckPlexLayout`; face-adjacency one-cell overlap, "simple" partitioner by default.
  BCs are keyed by "Face Sets" values, which on a box ARE the `FACE_*` ids; a reflective
  face must be AXIS-ALIGNED (PETSC_ERR_SUP otherwise); materials by centroid
  `paint_boxes`, "Cell Sets" `paint_cell_sets`, or both layered via `paint_boxes_over`.
  DSA through `DSAPrecon`'s plex overload (a two-point-flux operator off the backend's
  host face data, `face_*_host()`; at DG1 an interior penalty operator in the DG1 space). DG1: a MODAL basis
  orthonormal on each cell (phi_0 = 1, the linear ones through the Cholesky factor of the
  cell's second moments from a fan of simplices, exact for planar faces), so the mass
  matrix is the identity and removal/scatter/source stay per node — the source lands on
  basis 0 only, and the scalar flux written out is the cell average (the slope rides along
  as extra fields); `(n_faces + 1) * n_basis` slots per row, own block LAST; GHOST-FLUX
  ONLY and NO BC rows at all — a reflective face is a face coupling to the mirrored
  angle's trace in the same cell, mirrored over that face's own axis);
  `UnstructuredCG` beside it (Phase 6b, `mesh.discretisation: "cg_supg"`: continuous
  P1/Q1 at the VERTICES with CONSISTENT SUPG - the whole residual tested with
  `v + tau Omega.grad v`, Wang's SAAF-tau from the first-order side - tau =
  min(1/sigma_t, h_Omega/zeta) per (element, angle, group), zeta a create() argument,
  0.5 by default, `mesh.supg_zeta`; rows (vertex, angle), so `ps.n_cells` is the global
  VERTEX count and `ps.local_cells` the owned vertices, while xsections and material ids
  are per LOCAL ELEMENT, overlap included - `GroupXSections::create(n_groups,
  n_elements)`; FEM-closure overlap, so a vertex's whole star is local; element tables M,
  G, K from PetscFE; weak boundary conditions with a lumped face mass and NO BC rows,
  reflective faces a mirror slot per axis; the plex plumbing shared with the DG backend
  in the internal `src/plex_commonk.hpp`). Its terms are siblings, not the generic ones,
  because the SUPG weight couples a vertex's star and depends on the angle
  (`include/ubolt/terms_cg.hpp`): `SUPGTermCG` (streaming + SUPG + removal + boundary,
  assembled, group-dependent), `ScatteringTermCG`, `GroupTransferCG` and
  `UboltFillSourceCG`, the last three through the backend's one weighted-load kernel;
  output `UboltWriteScalarFluxVTKCG` (point data). The driver refuses
  `-matfree_removal`/`-precon_stream`/`-precon_ref_shift`/`-check_matfree` (no
  group-independent streaming matrix); `-precon_dsa` works (`DSAPrecon`'s CG overload);
  `OperatorTerm` and the `Streaming`/`Streaming2D`/`Streaming3D`/`StreamingDG0`/
  `StreamingDG1`/`Removal`/`Scattering` terms (`StreamingTermDG0/DG1::create(ps, disc)`
  take NO quadrature: they read the ordinates off the backend, so the two cannot
  disagree; each checks the backend's order),
  `GroupXSections` + `GroupTransfer` (multigroup xsection tables and the group-to-group
  source), `TransportOperator` (assembled terms in one matrix + matrix-free terms behind a
  MatShell), `TransportSolver` (KSP + the composite PC, plus `refresh()` for what the PC
  caches off the assembled matrix), `DSAPrecon` (the OPTIONAL diffusion-synthetic
  acceleration stage of that composite, index 2 behind `-precon_dsa` — a cell-centred
  diffusion operator on the same grid, restricted from and prolonged back onto the
  ordinates, inverted inexactly under the `dsa_` prefix; NOT an `OperatorTerm`, it
  preconditions rather than contributes, and it is caller-owned with a per-group
  `set_group()` the driver makes because the solver has no group context. Geometry, so
  per-backend `create` overloads like the streaming term: a dof-1 DMDA twin + a star on
  the structured ones; on `UnstructuredDG` a two-point flux per face, assembled
  VOLUME-WEIGHTED (SPD where volumes vary, the restricted moment scaled by V to match),
  which on a quad/hex box is exactly V times the structured matrix. D is
  DISCRETISATION-CONSISTENT by default at DG0 and on the structured backends: each face
  blends the upwind scheme's numerical diffusion `m h` (m = the quadrature's half-range
  current, ~1/4) into the physical D as `(D^p + (m h)^p)^(1/p)`, p = 1.5
  (`-dsa_consistent_d 0` / `-dsa_consistent_d_power`) — so the structured `create`s
  need the concrete SN set, not just the `AngularQuadrature` base. At DG1 the diffusion
  unknown is DG1 too: the MIP interior penalty form (Wang & Ragusa) on the backend's
  modal basis, one unknown per (cell, basis) node, block size n_basis for GAMG, every
  node restricted and corrected; penalty constant `-dsa_mip_penalty`, 4 by default.
  On `UnstructuredCG` the diffusion unknown is per VERTEX, the transport's: a
  continuous P1/Q1 weak form on the backend's element tables (consistent mass, the
  lumped face mass times the half-range current on vacuum faces, per-ELEMENT
  `D = 1/(3 sigma_t)`), which is EXACTLY `m R A P` of the SUPG operator on an
  isotropic flux in thick cells — so no consistent-D blend there; the restricted
  moment is scaled by the lumped mass m_i, and restriction/prolongation are the
  identity in space.
  VOIDS (a cell with group `Sigma_t <= -dsa_void_sigma_t`, 0 by default) are BRIDGED
  by default, every backend and order: kept in the operator with no absorption and the
  free-flight `D = L / 3`, `L = 4 V / S` the voids' mean chord (one per group, summed
  over ranks; S = faces onto material or vacuum, not reflective ones), so the
  correction couples the regions a void separates; at DG1 a face touching a void takes
  the harmonic-D weighted interior penalty (plain MIP welds the material to the void).
  (On CG the voids are per element and bridged the same way; `-dsa_void_bridge 0`
  keeps the SUPG operator's own tensor `(1/W) sum_a w_a tau_a Omega_a Omega_a^T`
  there instead of masking, which degrades as h / zeta shrinks with the mesh.)
  `-dsa_void_bridge 0` MASKS them instead (an identity row, times V on the plex,
  restricted and corrected to zero, a face into it the (consistent-D) Marshak or MIP
  vacuum face for its neighbour), and an all-void or would-be-singular group falls back
  to the mask; per group, the void flagged in the staged D (0 masked, negative bridged
  on the plex) so it crosses ranks with the ghosting, and bit-for-bit the plain
  operator when there is no void),
  `ElementBlockInverse` (the inverse of each (cell, angle) n_basis x n_basis element
  block of a MATAIJKOKKOS matrix, read straight off its device CSR — strided by
  n_angles under layout A, so not PETSc's contiguous-block inverse — plus `D^{-1} x` and
  `D^{-1} A` on the SAME pattern, all on the device; at n_basis 1 it is the diagonal).
  `TransportSolver::create(..., block_scale)` uses it to wrap composite index 1 in a
  shell: PCAIR built on `D^{-1} pmat`, applied to `D^{-1} r` — a preconditioner-only
  change (operator, rhs, residual norms untouched), the inner PC keeping the `sub_1_`
  prefix. `-precon_block_scale`, default ON for the DG backend (both orders), OFF on the
  structured ones; it is what lets PCAIR coarsen DG1 at its default strong threshold.
  Composite index 0, the removal stage, is the same class on EVERY backend, read off the
  operator rather than pmat: the point Jacobi it always was at n_basis 1 (to the bit),
  block-Jacobi at DG1; under `-matfree_removal` `setup(A, diag)` takes the composed
  diagonal in place of the streaming-only matrix's (exact: removal has no off-diagonals),
  and it is kept on DG though it scales the same blocks (dropping it measured 1-2 worse),
  `RefShiftPmats` (the OTHER optional pmat strategy, behind `-precon_ref_shift` and only
  under `-matfree_removal`: k copies of the streaming matrix each carrying a
  REPRESENTATIVE removal `alpha_k * D_ref`, plus the group-to-bin map. VOIDS: the
  groups are sorted by support (the cells where `Sigma_t > 0`), one reference group
  per support class, binned per class (`-precon_ref_k` is per class) - the reference
  is zero in the class's voids, so a void cell's pmat row is the bare streaming row the
  operator has there, and an empty support is the one unshifted bin whose pmat IS the
  streaming matrix; only a negative `Sigma_t` is an error. It owns those
  matrices and nothing else — the driver keeps one `TransportSolver`, hence one PCAIR
  hierarchy, per bin, because the group loop is the driver's),
  `UboltWriteScalarFluxVTK` (the scalar flux — at DG1 the cell average — of a
  solution, plus any extra per-cell fields the caller hands over as `UboltCellField`s —
  the driver passes the group's `sigma_t` and its `source`, the latter expanded onto the
  cells by `UboltFillCellSource` — written through PETSc's VTK viewer onto a dof-1 twin
  of the backend's DM, dispatched on its type: `.vts`/`.vtr` on a DMDA, `.vtu` on a
  DMPlex (owned cells only, via a "vtk" label; on the CG backend its sibling
  `UboltWriteScalarFluxVTKCG` writes the flux as point data, `scalar_flux.nodal`); a
  problem file's `output.flux_vtk`, or `-flux_vtk` as the override). `types.hpp` owns every Kokkos view typedef, `ubolt.hpp`
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
  `BoundaryInfo` (BC row mask + reflect slots); terms write through those, never raw
  indices. The BC contract binds BOTH halves of a term, `assemble_add` and `apply_add`:
  leave flagged rows alone, they carry the boundary condition, and the assembly has
  already written them — the identity on a Dirichlet (vacuum) row, identity minus the
  mirrored angle on a reflective one. A term that sits on the diagonal also owes
  `has_diagonal()`/`add_diagonal(Vec)` the SAME arithmetic its `assemble_add` writes into
  the diagonal slot — that is what lets the removal shell PC compose its diagonal from
  the terms when a diagonal-carrying term is applied matrix-free (`-matfree_removal`)
  instead of reading `MatGetDiagonal` off a matrix that is then missing it.
  Anything that builds a right hand side rather than
  acting on the unknowns being solved for is NOT an `OperatorTerm` — see `GroupTransfer`
  — but it owes the mask the same thing; the rhs itself owes Dirichlet rows the per-row
  inflow — `UboltFillInflow` — and reflect rows a zero — `UboltZeroReflectRows`.
- PETSc source is at `$PETSC_DIR/$PETSC_ARCH`. Both env variables must be set.
  PFLARE is at `$PFLARE_DIR` (defaults to `../PFLARE`, alongside this repo, in the Makefile).

Read only when the task needs it
- `TODO.md` — roadmap, current phase, per-phase verification criteria, research notes
- `docs/problem_files.md` — the problem/materials JSON schema reference;
  `docs/problem_setup.md` — the walkthrough for writing a new problem file
- `docs/dev/testing.md` — before adding or modifying tests, and before touching baselines
- `docs/dev/kokkos.md` — before touching `*.kokkos.cxx`, COO assembly or MatShell code,
  before writing a new discretisation backend (it has the COO/slot-order/Dirichlet
  discipline a backend and its terms have to agree on), and before adding any
  multi-dimensional view (layouts differ between host and device backends, and the obvious
  compile-time contiguity guard does not work)

Build
1. In top repo directory: `make -j3 build_tests` (PETSc >= 3.25 configured with Kokkos
   required; in practice PETSc main from 3 Sep 2026 on, for `MatGetState`'s `MatState` form in
   `verify_plexk` - no 3.25.x release has it, and the CI base image tracks main). This builds `lib/libubolt.{so,a}` first; `make` on its own builds just the library.
2. Rule: fix all compile warnings (CI will build with `-Werror`).
   CI (`.github/workflows/ci_build.yml`) builds `dockerfiles/Dockerfile_kokkos` on the
   prebuilt `stevendargaville/petsc_kokkos` image (opt/debug/64-bit/OMP arches): it
   builds PFLARE main, then UBOLT with `-Wall -Werror`, then runs `check` + `tests`.
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
