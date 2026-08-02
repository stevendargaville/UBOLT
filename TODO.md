# UBOLT roadmap

Full plan and architecture rationale: see the approved plan (design discussion July 2026).
Each phase is a reviewable unit with its own verification. Do not start a phase before the
previous one's verification has passed and been reviewed.

## Phase 0 — Scaffolding + baseline capture (no behavior change)
- [x] Directory tree, top Makefile (library skeleton), tests/Makefile (PFLARE-style recipes)
- [x] Bookkeeping: CLAUDE.md, AGENTS.md, TODO.md, docs/dev/{testing,kokkos}.md
- [x] Move unmodified source to tests/slab_1dk.kokkos.cxx, builds as tests/slab_1dk
- [x] Capture baselines: {default pc, precon_stream} x {sigma_t 0, 2} x {np 1, 2}
      x {diag_scale off, on} into tests/baselines/; iteration table in docs/dev/testing.md
- [x] Pin per-recipe -ksp_max_it to baseline counts in tests/Makefile
- Verify: `make check` and `make tests` pass; baselines committed.
- Findings during capture (see docs/dev/testing.md): (i) parallel out-of-bounds bug in the
  matrix-free scatter (global N_CELLS used on local arrays) — np=2 default/st=2 goes NaN;
  fix scheduled as its own commit at the start of Phase 1a; (ii) diag_scale + strong
  removal is pathological (6305 its / divergence) — excluded from pass/fail recipes.

## Phase 1 — Pure refactor into libubolt (identical numerics)
- [x] 1a-pre (own commit): fix the parallel scatter out-of-bounds bug — ShellMatMultApply
      must reshape/loop over LOCAL cells (local_rows/N_ANGLES), and the sigma/scalar_flux
      views must be sized/filled with local counts; then re-capture the invalidated np=2
      default/st=2 baselines. Done: np=1 bitwise identical, only the two NaN logs changed
      on re-capture, repaired config converges in 10 its matching serial.
- [x] 1a: mechanical split of existing free functions into src/, driver calls them;
      #defines become -n_cells/-n_angles/-length options; driver exits nonzero unless
      KSPGetConvergedReason > 0. Done: all 16 baselines reproduce bitwise; the driver is
      the only thing that changed behaviourally. Beyond the mechanical move:
      (i) library functions return PetscErrorCode and wrap every PETSc call in PetscCall
      (the pre-refactor file emitted 89 nodiscard warnings; libubolt now builds clean),
      (ii) public declarations tagged PETSC_EXTERN since PETSc uses -fvisibility=hidden,
      (iii) the row decomposition is now decided in cells (PetscSplitOwnership over
      n_cells) rather than PETSC_DECIDE over rows — identical at np=1,2 and it fixes the
      mid-cell split, np=3 now converges in 10 its matching serial,
      (iv) the dead random-xsection block in the driver is commented out consistently
      (it computed values that were immediately overwritten).
- [x] 1b: introduce PhaseSpace, SNQuadrature, StructuredFD1D (CooPattern slot maps +
      is_dirichlet_row_d), StreamingTerm/RemovalTerm/ScatteringTerm, TransportOperator,
      TransportSolver; driver shrinks to ~60 lines; delete dead code.
      Done: driver is 76 code lines, all 16 baselines still bitwise identical, and so is
      the `-ubolt_coo_two_call` fallback. Notes:
      - Dirichlet rows are now the assembly's job, not a term's: terms skip flagged rows
        and TransportOperator writes the identity afterwards. `add_removal` re-zeroing
        those diagonals by hand is gone.
      - The streaming-only pmat comes from `assemble_subset`, so the MatDuplicate/MatCopy
        pair is gone too; both matrices are preallocated from the same CooPattern.
      - Kernels capture value copies of the views instead of the host context pointer, so
        the scatter apply is no longer host-backend-only (was a latent CUDA/HIP bug).
      - The library calls `PetscKokkosInitializeCheck()` before allocating device memory:
        it used to ride on `MatSetType(MATAIJKOKKOS)` happening first, which is no longer
        the first thing the driver does.
- Verify: residual histories diff-identical against tests/baselines/ for the full matrix,
  np=1,2 (np=3 differs by design: cell-based decomposition fixes the mid-cell split bug).
- Fidelity notes: single summed COO INSERT replaces INSERT-then-ADD (identical per-entry
  math; keep a two-call debug fallback); runtime sizes replace #defines.

## Phase 2 — Multigroup with sparsity reuse
- [x] n_groups in PhaseSpace; per-group xsections as explicit-LayoutRight views
      (sigma_t 2D as planned, sigma_s 3D — see the note below)
- [x] One Vec per group, outer group loop (Gauss-Seidel, downscatter-only first);
      per-group values-refill via assemble(), no re-preallocation
- [x] Driver tests/slab_1d_mgk.kokkos.cxx with -n_groups
- Verify: -n_groups 1 reproduces Phase 1 exactly; identical groups + zero transfer
  reproduce single-group answer per group; pinned per-group iterations.
  Done, all three: `-n_groups 1` is bitwise identical to `slab_1dk` for
  {default pc, precon_stream} x {st 0, 2}; the 4-group zero-transfer log is four
  byte-for-byte copies of the single-group log at np=1 and np=2; 8 multigroup baselines
  captured and the recipes pinned. All 16 single-group baselines still reproduce bitwise.
  Notes:
  - `n_groups` is metadata on PhaseSpace, NOT part of the row count. Groups are solved
    one at a time, so the sparsity, the CooPattern and the assembled matrix are the same
    for every group and the sweep only refills values through `assemble()`.
  - `sigma_t` is 2D (group, cell) as planned; `sigma_s` is 3D (from, to, cell) — the
    plan's "2D" predates needing the transfer blocks. Both are explicitly LayoutRight
    with the group indices slowest, so fixing the group(s) slices out a CONTIGUOUS
    per-cell view. That is what lets RemovalTerm/ScatteringTerm keep taking a plain 1D
    view: the sweep re-points them with `set_sigma_t`/`set_sigma_s` and they never learn
    that groups exist.
  - `GroupTransfer` builds the off-diagonal blocks' contribution into the rhs. It is not
    an OperatorTerm (it does not act on the unknowns being solved for) and it skips
    Dirichlet rows, the same contract the assembled terms have.
  - With downscatter only, the group system is block lower triangular, so ONE forward
    sweep is exact — there is no outer iteration yet. Upscatter is what forces it.
  - `GroupTransfer` caches each group's scalar flux (`set_scalar_flux`, called once when
    the group is solved) rather than integrating inside `add_source`. A group's scalar
    flux is fixed once it is solved and every group below it scatters from the same one,
    so integrating on demand costs G(G-1)/2 angular integrals per sweep instead of G — at
    16 groups that was 120 against 95 for the whole iterative solve. Upscatter will need
    the validity flag relaxed, since it wants the previous iterate's flux.
  - `TransportSolver::refresh()` added: the removal shell PC caches the inverse diagonal
    of the assembled matrix, and sigma_t changes under it every group. PCAIR needs no
    help (it tracks pmat's state), and with `-precon_stream` the streaming-only pmat is
    group-independent so PCAIR keeps one setup for the whole sweep.
  - The angular integral was factored out of `ScatteringTerm` into `UboltAngularIntegral`
    so the scatter and the group transfer do bit-identical arithmetic. Verified inert:
    all 16 single-group baselines unchanged.
- Deferred out of this phase: upscatter + the outer iteration it needs; spatially varying
  xsections — DONE Aug 2026, see the per-region materials postscript (`MaterialSpec` +
  `GroupXSections::set_from_materials`); promoting the
  group loop out of the driver into a MultigroupSolver (wait for a second sweep strategy).

## Phase 3 — DMDA adoption (behavior-preserving)
- [x] 3a: StructuredFD1D layout from a 1D DMDA (dof=n_angles, stencil 1); extraction to
      the same CooPattern/BoundaryInfo; kernels unchanged; -ubolt_use_dm option, hand
      layout stays as the verification twin.
      Done: both paths reproduce all 24 baselines bitwise at np=1,2, and agree with each
      other at np=3 (uneven 334/333/333 split). Notes:
      - The option is `-ubolt_use_dm`, not `-use_dm`: library-internal switches carry the
        `ubolt_` prefix (cf. `-ubolt_coo_two_call`); unprefixed options are the drivers'.
      - The two paths differ in exactly ONE place, where cell_start/local_cells come from
        (MPI_Exscan vs DMDAGetCorners). Everything downstream is shared, so the twin test
        exercises the DM extraction rather than a second copy of the sparsity logic.
      - The DM is NOT given DMSetFromOptions: that would expose -da_grid_x, which could
        resize the mesh out from under the PhaseSpace everything else is sized from.
      - StructuredFD1D now owns a PETSc handle, so it has a destroy(); both drivers call it.
      - The matrix still comes from MatSetPreallocationCOO, not DMCreateMatrix (below).
- [x] 3b: delete the hand-rolled path and the -ubolt_use_dm option, DMDA becomes the only
      way StructuredFD1D::create builds a layout, and it also owns the decomposition -
      PhaseSpace no longer calls PetscSplitOwnership.
      Done: all 24 baselines still reproduce bitwise. Notes:
      - StructuredFD1D::create now takes the PhaseSpace by non-const reference and FILLS
        local_cells from DMDAGetCorners. PhaseSpace::create leaves it PETSC_DECIDE.
      - That imposes an ordering constraint - the discretisation must be built before
        anything sized off the phase space - so PhaseSpace grew check_decomposed(), and
        the six creates that read local_cells/local_rows() (the three terms,
        TransportOperator, GroupXSections, GroupTransfer) call it. Without it the failure
        would be a Kokkos view allocated with a negative extent somewhere downstream
        rather than an error at the call that got the order wrong.
      - CheckDALayout lost its local_cells-vs-PhaseSpace check, which became tautological
        once the DM is the authority. The sizes and the angle-fastest/contiguous ordering
        checks stay - those are still real.
- Verify: DONE. 3a bitwise-compared -ubolt_use_dm against the hand layout; both phases
  reproduce all 24 tests/baselines/ logs bitwise at np=1,2, and 3a additionally agreed
  with the hand layout at np=3.
- DECIDED (Aug 2026): the hand-rolled layout was transitional, not a permanent second
  backend. It existed in 3a only as the bitwise oracle and went in 3b. It was 1D-only by
  construction, and Phase 4 already says there is no hand-layout twin in 2D, so carrying
  it past Phase 3 would have bought nothing and cost a path to keep alive.
- Design constraints, both about keeping "behavior-preserving" true:
  - Build the matrix with MatSetPreallocationCOO exactly as now, NOT DMCreateMatrix.
    DMDA preallocates from the stencil (3 points x dof couplings per row); the upwind
    operator has 2 entries per row. A different sparsity is a different matrix and PCAIR
    would see it. The DM supplies ownership and indices, not the matrix.
  - Let DMDACreate1d choose its own distribution (lx = NULL) rather than handing it
    PhaseSpace's. There was going to be a staged step for this, on the assumption the two
    might differ; they cannot. PETSc's default 1D DMDA split is
    `M/size + ((M % size) > rank)` (src/dm/impls/da/da1.c), the same formula
    PetscSplitOwnership uses. So the DM decided from 3a on, with a CheckDALayout assert
    holding the two to agreement through 3a and the np=2 twin diff as the empirical proof;
    3b made the DM the sole authority and dropped the now-tautological assert. This is what
    folded the old 3c into 3a/3b.
- Note: assembly uses global column indices and the matrix-free scatter is cell-local, so
  nothing in Phase 3 needs the DM's ghost exchange. The DM is bookkeeping here (layout and
  decomposition only); it starts paying for itself in Phase 4.

## Phase 4 — 2D structured (DMDA 2D)
- [x] 4a: the abstract Discretisation base (the decision below, taken as called).
      Done: all 24 baselines re-capture bitwise. `create_matrix`, `coo_pattern`,
      `boundary_info`, `destroy` and `dm` are on the base, with the state they read
      (comm, the PhaseSpace copy, the DM handle, the COO coordinate arrays).
      `TransportOperator::create` and `RemovalTerm::create` take the base;
      `StreamingTerm::create` keeps StructuredFD1D. One thing beyond the four methods:
      `set_uniform_pattern(slots_per_row, dirichlet_mask)` builds the slot maps, since 1D
      and 2D both lay out a fixed number of entries per row with the diagonal LAST — that
      convention is now stated once rather than in each backend.
- [x] 4b: StructuredFD2D (3-slot rows), 2D angle quadrature table
      NOTE: this said "4-slot rows" until Aug 2026. It is 3 — upwinding mu dpsi/dx and
      eta dpsi/dy separately contributes ONE neighbour per axis plus a diagonal, the
      direct generalisation of 1D's 2 (upwind neighbour + diagonal). The 5 points are what
      the DMDA star stencil preallocates, not what the operator touches; keep the two
      apart, which is the same reason create_matrix must not become DMCreateMatrix.
      Slot order convention, extending 1D's "off-diagonals first, diagonal last": upwind-x,
      upwind-y, diagonal. Which x/y neighbour is upwind is fixed per angle at preallocation
      from sign(mu)/sign(eta), exactly as 1D does it, so the value fills never branch on
      direction. A row with mu or eta exactly 0 takes a -1 in that slot, the same trick
      Dirichlet rows already use.
      Done. Notes:
      - The COO COLUMNS come from the DM's local-to-global map, not arithmetic. This is
        the piece 1D did not need: a 2D DMDA numbers each rank's patch contiguously and
        lexicographically WITHIN the patch, so `(j * n_cells_x + i)` is wrong the moment
        more than one rank splits a direction. The map covers the ghost nodes too, which
        is what lets a column point into a neighbour's patch.
      - `SNQuadrature2D` is the level-symmetric sets, S2 (4 ordinates) and S4 (12) — four
        quadrants of {1, 3}. XY geometry is symmetric about z = 0, so only xi > 0 is
        tracked and the weights are doubled; they sum to the full 4 pi and xi is never
        needed. A thin `AngularQuadrature` base carries n_angles/sum_weights/w_d, which is
        all ScatteringTerm and GroupTransfer ever wanted, so both work in either dimension
        unchanged. Same split as the Discretisation base: the direction cosines stay on
        the concrete class because only the streaming term reads them.
      - Dirichlet rows are the INFLOW ones: any node whose x-upwind OR y-upwind neighbour
        is outside the box, corners included through either test.
      - `sum_weights` is passed to the base rather than summed from the weights: it is the
        exact measure of the angular domain, and summing would have moved 1D's 2.0 by a
        rounding and broken the baselines.
- Verify: DONE, all three, in `tests/verify_2dk.kokkos.cxx` (+ `tests/box_2dk.kokkos.cxx`
  for the pinned counts).
  - Pure-streaming closed form: psi = x + y is linear and first-order upwind differencing
    is exact on a linear function, so the DISCRETE operator has the PDE's solution.
    Residual 4e-15 / 8e-15 (S2 / S4) against a 1e-12 tolerance, at np = 1, 2, 3 and 4, and
    the solve returns the closed form to 4e-14. Pins both upwind signs, dx against dy and
    the Dirichlet rows at once.
  - MatComputeOperator of the SHELL (so the matrix-free scatter is in it) against a
    reference matrix built entry by entry in the driver: max difference 0.0, i.e. bitwise,
    on 4x3 with S2 and S4. Serial, because the reference is written in natural ordering —
    which is exactly the difference StructuredFD2D's layout assert exists to police.
  - Pinned PCAIR iterations: recipes in tests/Makefile, counts in docs/dev/testing.md.
    (Both the counts and the reference matrix moved straight after Phase 4, when the
    postscript below fixed the Dirichlet bug this verification found.)
  - No 2D residual-history baselines were captured. In 1D they are the numerical-inertness
    oracle for a refactor; in 2D the reference-matrix check is a strictly sharper one (it
    compares every entry of the operator, not a residual history that happens to agree),
    so the baselines would only have added files to keep in sync.
  (No hand-layout twin in 2D: DMDA's PETSc ordering differs from natural ordering — and
  after Phase 3b there is no hand layout in 1D either, so the twin technique is retired
  for good.)
- Inherited from Phase 3: the DM already owns the decomposition and writes local_cells
  into the PhaseSpace, so StructuredFD2D slots into the same seam. Two things do NOT
  carry over unexamined: CheckDALayout's angle-fastest/contiguous assert (2D ordering is
  the thing that differs, so it has to be rewritten, not copied), and create_matrix's
  hand-built COO preallocation (3 slots per row, see above — NOT the DMDA stencil's 5).
  Both were done as called. The 2D assert is weaker on purpose and states what it needs:
  sizes; angle-fastest contiguous dof; and this rank's owned nodes numbered contiguously
  from its rstart in the PATCH's own lexicographic order. That last one is what makes
  "local cell index = (j - ys) * xm + (i - xs)" true, which the per-cell xsection views
  and RemovalTerm's `r / n_angles` both rely on.
- DECIDED (Aug 2026), the Discretisation base: taken exactly as proposed below and done in
  4a. Phase 6's checklist item is reworded to note it has already happened.
  - `create_matrix()`, `coo_pattern()`, `boundary_info()` — all that RemovalTerm and
    TransportOperator use, and all dimension-independent. This is the base class.
  - `dx()` — used ONLY by StreamingTerm, which needs a 2D sibling regardless (it owns the
    upwind slot convention and in 2D needs dx and dy). Dimension-specific terms can keep
    taking the concrete class, so the base does not have to grow a geometry interface.

## Phase 5 — Matrix-free removal / single-streaming-matrix experiment
- [ ] RemovalTerm::apply_add + -matfree_removal: Assembled{Streaming} +
      MatrixFree{Removal, Scattering}; one assembled matrix for all groups
- [ ] Terms get optional get_inv_diagonal(Vec) so RemovalPCShell composes its diagonal
      analytically instead of MatGetDiagonal(assembled)
- Verify: matvec equivalence vs assembled path on random vectors (<1e-13); solution norms
  match Phase 2; iteration counts recorded (not pinned) — preconditioner quality is the
  research question; findings recorded below.

## Phase 6 — DMPlex FEM backends
- [ ] DECISION POINT first: hand-written Kokkos DG/CG kernels over DMPlex (plan default)
      vs MFEM as a Discretisation backend — see Research notes below. Recommended spike
      before committing: assemble an MFEM DG advection matrix, convert to aijkokkos, feed
      to PCAIR, measure the interop cost.
- [ ] 6a DGUpwind: broken-PetscSection layout (cf. PFLARE tests/adv_dg_upwind.c) but
      extraction-only — flattened device views (cell->dof offsets, face->(cell-,cell+),
      normals/areas, volumes, boundary faces), variable-nnz COO pattern, Kokkos
      volume + face kernels
- [ ] 6b CGSUPG: PetscFE/PetscDS host-only for quadrature/tabulations copied to device
      once; volume kernels; Dirichlet via identity-row mechanism
- BCs: consume the existing `BCSpec` with real "Face Sets" label values (the structured
  backends' FACE_* ids already match the box-mesh convention, so square meshes carry
  over unchanged) — see the reflective-BC postscript below for the label/physics split
- [x] Introduce thin abstract Discretisation base — DONE in Phase 4a, as that phase's
      decision point predicted. This item survives only as the point where a DMPlex
      backend would widen it: `set_uniform_pattern` is the part that will not carry over
      (DG has a variable-nnz COO pattern), while `create_matrix` and the accessors should.
- Verify: DG0 on uniform mesh reproduces the FD upwind matrix; manufactured-solution
  convergence rates; pinned iterations.

## Phase 7 — deferred
- [ ] CI: clone PFLARE's docker model + docs/dev/ci.md
- [ ] Pn / wavelet angular discretizations (sibling structs to SNQuadrature + own terms).
      `UboltAngularIntegral` already has the right shape for this: it is written as a gemm
      against a (n_angles, n_moments) weight matrix that SN happens to use with
      n_moments = 1. Pn widens that column dimension and the integral becomes a genuine
      gemm — do NOT "simplify" it to a gemv on the grounds that N is currently 1.
- [ ] Performance passes (kernel fusion in the MatShell loop). The `-precon_stream`
      MatDuplicate this used to name is already gone: Phase 1b replaced the
      MatDuplicate/MatCopy pair with `assemble_subset`, which preallocates from the same
      CooPattern.
- [ ] The group-to-group transfer is applied one (g_from, g_to) pair at a time, so a sweep
      does O(G^2) small kernels. With the scalar fluxes now cached per group they sit in
      G contiguous (local_cells, 1) arrays, so the whole `sum_g' sigma_s(g',g) phi(g')`
      contraction could become one batched kernel — a real gemm per cell if the xsections
      were spatially constant, a batched one as they are. Only worth it at large G.

## Phase 1 postscript — negative-angle upwind sign (own commit, after 1b)
- [x] `StreamingTerm` wrote the upwind neighbour coefficient as `-mu/dx`: correct for
      `mu > 0`, wrong sign for `mu < 0`, so those rows were `|mu|/dx (psi_i + psi_i+1)`
      instead of `|mu|/dx (psi_i - psi_i+1)`. Found during 1b and deliberately preserved
      so Phase 1 could be verified bit-for-bit, then fixed on its own with a full baseline
      re-capture. Interior streaming rows now sum to zero and everything well-behaved
      converges faster (st=2: 10 -> 5 its, streaming pmat st=2: 16 -> 10). New counts in
      `docs/dev/testing.md`; recipes re-pinned. diag_scale + st=2 is still pathological.

## Phase 4 postscript — the matrix-free scatter ignored the Dirichlet mask (own commit)
- [x] `ScatteringTerm::apply_add` subtracted `sigma_s phi / sum_weights` from EVERY row,
      Dirichlet rows included. So the shell's boundary rows were not the identity the
      assembly wrote — they were `psi_r - (scatter at that node) = incoming`, and the
      prescribed inflow value was polluted by the scattering source in that cell.
      Found in Phase 4b while writing the reference matrix in `tests/verify_2dk`. Present
      since Phase 1 and identical in 1D: the Dirichlet contract in `operator_term.hpp` was
      written for `assemble_add` only, so `apply_add` was never told about it. Fixed on its
      own with a full baseline re-capture, exactly as the Phase 1 upwind-sign fix was.
      What changed: the contract in `operator_term.hpp` now covers both halves of a term;
      `ScatteringTerm::create` takes a `Discretisation` so it can hold the mask; the apply
      skips flagged rows. The scalar flux is still integrated over EVERY angle, Dirichlet
      ones included — the prescribed incoming flux is a real part of the flux in that cell,
      so the interior angles must scatter off it. Only the write to a flagged row is
      dropped. `verify_2dk`'s reference matrix now says a Dirichlet row is the identity and
      nothing else, and the operator matches it bitwise.
      Everything with `sigma_s = 0` is unchanged; everything else moved. In 1D the
      well-behaved configs are a wash (st=2: 5 -> 6, streaming pmat st=2: 10 -> 9,
      multigroup streaming sweep max 13 -> 11) and the pathological diag_scale + st=2 pair
      got worse (175 its -> capped at 200; breakdown at 90 -> 150). In 2D it is a large
      win — see the research note below, which the fix rewrote.

## Phase 4 postscript — scalar flux VTK output (own commit, Aug 2026)
- [x] `UboltWriteScalarFluxVTK(ps, disc, quad, psi, filename)`: the scalar flux of a
      solution, written for inspection in ParaView/VisIt. Not a roadmap phase — added on
      request once there were 2D solutions worth looking at. How it works: the shared
      `UboltAngularIntegral` (so the written flux is bit-identical with what the terms
      integrate), onto a dof-1 `DMDACreateCompatibleDMDA` twin of the backend's DMDA, out
      through PETSc's VTK viewer, which does the parallel gather. Dimension-independent:
      it takes the `Discretisation` base and never sees the geometry.
      - A DMDA is a structured grid, so the formats are `.vts`/`.vtr` (the extension
        picks); `.vtu` is PETSc's DMPlex format and is rejected with an error saying so.
        A DMPlex backend (Phase 6) is where a `.vtu` path would appear.
      - The backends now set uniform coordinates on their DMDA at create — node (i, j) at
        (i dx, j dy), the `verify_2dk` convention — so the files carry real mesh
        positions. Nothing in the solve reads them, and the compatible DMDA carries them
        over to the flux vector for free. A direction with a single node keeps no
        coordinates (PETSc's uniform spacing divides by n - 1).
      - The dof-1 copy relies on the same layout property `CheckDALayout` asserts: a DMDA
        global vector holds the owned patch contiguously in patch-lexicographic order,
        which IS the local cell order the scalar flux view is indexed by.
      - Every solve driver exposes `-flux_vtk <file>`; the multigroup driver writes one
        file per group (`flux.vts` -> `flux_g0.vts`, ...), only after a full sweep.
      Verified: serial vs np=4 (a genuinely 2D processor grid) fields agree to 1e-13 at
      `-ksp_rtol 1e-12`, so the gather/ordering is exact; coordinates land on [0, L - dx]
      exactly; `make check`/`tests_short` pass and the baselines still reproduce bitwise
      (spot-checked np=1 single-group and np=2 multigroup), so the change is numerically
      inert — as it must be, since coordinates are the only thing the solve path even
      sees, and nothing reads them.

## Phase 4 postscript — reflective boundary conditions (own commit, Aug 2026)
- [x] Specular reflective BCs, selectable per boundary face, alongside the existing
      vacuum (prescribed-inflow) rows. Pulled forward from the unstructured phase because
      the selection API is where the two meet: `BCSpec` maps an integer boundary label id
      to a BC family ({vacuum, reflect}, default vacuum), keyed the way DMPlex
      "Face Sets" ids are. The structured backends' `FACE_*` constants match PETSc's
      box-mesh convention (1D left=1/right=2; 2D bottom=1/right=2/top=3/left=4), so the
      Phase 6 backends consume the SAME spec with real "Face Sets" values, on the
      adv_dg_upwind split: the label says which family a face belongs to, the sign of
      direction dot normal (the upwind test) decides what happens per row.
      How a reflective row works — no sparsity growth anywhere:
      - The row is `psi(cell, a) - psi(cell, a') = 0`, `a'` the mirrored angle. The
        backend repurposes one of the row's otherwise-nulled off-diagonal slots for the
        partner column (same cell, so always rank-local) and records it in
        `BoundaryInfo::reflect_slot_d`; `SetBoundaryRows` (was `SetDirichletIdentity`)
        writes the identity plus the -1.0. Terms are untouched: the mask (renamed
        `is_bc_row_d`, it now covers both BC kinds) means what it always meant.
      - The quadratures grew host-side reflection maps, built by SEARCH over the cosines
        with a PetscCheck, so a future non-symmetric set fails loudly rather than
        reflecting into a wrong angle.
      - Mirror rule at corners: any incoming vacuum face wins (Dirichlet); a direction
        incoming through two reflective faces flips both cosines — still one partner.
        A reflective face on a single-cell-wide direction is a checked setup error.
      - The rhs owes reflect rows a zero — `UboltZeroReflectRows`, called by the drivers
        after filling b (and before `-diag_scale`).
      Verified three ways (docs/dev/testing.md): the vacuum default reproduces all 24
      baselines bitwise; `verify_2dk`'s reference matrix runs all-vacuum/all-reflect/mixed
      x S2/S4 and agrees bitwise, pinning the slots, partners and corner rule per entry;
      and `-check_inf_medium` (all faces reflective, uniform source) hits the exact
      `1/(sigma_t - sigma_s)` to ~1e-13 in serial and parallel — decomposition
      independent, so it is the parallel oracle for the partner columns. All-reflect with
      scattering ratio 1 is singular, hence `-sigma_scatter` on `slab_1dk` and the
      mg-keeps-a-vacuum-face rule.

## Phase 4 postscript — per-region materials and sources (own commit, Aug 2026)
- [x] `MaterialSpec` — BCSpec's sibling for cell data: per-material, per-group xsections
      and an external source (the per-angle rhs value, no sum_weights normalisation).
      The split mirrors BCSpec exactly: the spec says what a material MEANS, and which
      cells are which material is geometry, so painting the per-local-cell material index
      view lives on the concrete backends — `StructuredFD1D::paint_intervals` /
      `StructuredFD2D::paint_boxes` (background + shapes in order, later wins, membership
      by cell centre) — and the Phase 6 backends will read a DMPlex "Cell Sets" label
      into the same view. Unlike BCSpec the tables have to reach device kernels, so
      materials are DENSE indices 0..n-1, not arbitrary label ids: the unstructured
      backend remaps its label values to indices at paint time, on the host, the same
      place BCSpec is consulted at create.
      - Expansion goes through the surfaces the terms already consume:
        `GroupXSections::set_from_materials` fills the per-cell tables and
        `UboltFillSource` writes the source into b's non-BC rows (Dirichlet rows keep
        the driver's inflow, `UboltZeroReflectRows` still owns the reflect rows). The
        terms never learn materials exist; both fills check the painted index range
        out loud.
      - Drivers: `box_2dk` takes `-n_regions` + `-region_<r>_box x0,x1,y0,y1` with
        per-region `-region_<r>_sigma_t/_sigma_s/_source`; `slab_1d_mgk` takes
        `-region_<r>_interval x0,x1` with a per-region `-region_<r>_density` scaling
        the whole group recipe (one knob, physically a density change) and
        `-region_<r>_source`. `-n_regions 0` is the uniform problem bit-for-bit.
      Verified: all 24 baselines reproduce bitwise through the new fill path (the mg
      driver now builds its xsections and rhs via MaterialSpec); the painting-identity
      recipes (regions carrying the background values) land exactly on the uniform
      counts, serial and on the -n 4 2D processor grid where the patch-lexicographic
      cell-centre mapping is least like the natural one; and new heterogeneous pins —
      a sourceless absorber in a scattering box (5 its, flux depression confirmed by
      eye via -flux_vtk) and a double-density mid-slab multigroup region (6 per group).
- [x] Follow-ons, own commits in the same PR:
      - `-inflow` on `box_2dk` and `slab_1d_mgk`: the vacuum faces' incoming flux,
        previously hard-coded at 1.0 by the same VecSet that filled the source (default
        unchanged, baselines re-verified bitwise). `-inflow 0` plus a painted source
        region is a cold box driven by that region alone — pinned as a recipe (5 its).
        `slab_1dk` deliberately keeps its single VecSet: it is the frozen baseline driver
        and predates the source/inflow split.
      - `-max_exponent` renamed to `-sigma_t` (now PetscReal) on all three solve drivers:
        the old name was a fossil of the original random `mantissa * 10^exponent`
        per-cell xsection, commented out before the first baselines were captured.
        Baseline logs and docs notation moved me<0|2> -> st<0|2>; a full recapture under
        the new option reproduced all 24 renamed logs byte-for-byte.

## Phase 4 postscript 2 — isotropic external source (own commit, Aug 2026)
- [x] `MaterialSpec::set_source` is now the angle-integrated (isotropic) strength Q, and
      `UboltFillSource` takes the `AngularQuadrature` and writes `Q / sum_weights` on
      each ordinate (Q/2 in 1D, Q/4pi in 2D) — the convention `sum_weights()` was
      documented for but the fill never used. Before this the source was the literal
      per-angle rhs entry, so the same `-region_<r>_source` value meant a different
      physical source in 1D and 2D. `-inflow` stays per-angle: it prescribes the
      incoming ANGULAR flux on Dirichlet rows. `slab_1dk` is untouched — its single
      VecSet is per-angle by design and the 16 single-group baselines stand.
      - `slab_1d_mgk` grew `-source` (`-inflow`'s sibling: the background material's
        isotropic strength, default 1.0). `-source 2` puts 1.0 on each 1D ordinate,
        exactly `slab_1dk`'s VecSet, which is what the t0 baseline captures now pass —
        so the t0-equals-single-group byte-for-byte check survives the convention change.
      - `box_2dk`'s `-check_inf_medium` expectation moved from `1/(sigma_t - sigma_s)`
        to `1/(sum_weights (sigma_t - sigma_s))` — the sharpest direct check of the new
        normalisation (unit isotropic source + unit absorption gives scalar flux 1
        exactly, verified to 1e-13).
      Verified: t0 captures with `-source 2` reproduce the previous logs byte-for-byte
      (files unchanged in the commit); the 6 t05/stream multigroup baselines re-captured
      deliberately (per-group counts moved within their max: t05 6,6,6,5 -> 6,6,6,6,
      stream 11,11,10,9 -> 11,11,11,9); painting identity still bitwise vs uniform at
      np=1 and np=4 (both now 7 its at np=4); cold box count unchanged (uniform rhs
      scaling is invisible to a relative residual); the two streaming-only pmat 2D
      serial references moved 8 -> 9 and 9 -> 10, pins keep their +1 CI slack (10, 11).
- **"Scattering ratio 1 in 2D degrades on non-square grids" — RESOLVED, it was the
  Dirichlet bug** (observed in Phase 4b, explained by the postscript fix above; recorded
  because the wrong conclusion is an easy one to reach again). The observation was real:
  with `sigma_s = sigma_t` the composite PC looked mesh independent on a square grid
  (50x50 in 18 iterations, 100x100 in 21) and not on any other shape (60x30 and 80x20 did
  not converge in 400), while dropping the ratio to 0.5 fixed it. The tempting reading —
  that this is the missing-DSA hole and therefore Phase 5's problem — was wrong. It was
  the matrix-free scatter writing to the Dirichlet rows: the amount it wrote is
  proportional to `sigma_s`, which is why the ratio looked like the variable, and it broke
  the boundary rows, which is why the shape of the boundary mattered. With the fix every
  one of those shapes converges in 5 to 7 iterations, mesh independently: 60x30 -> 6 and
  120x60 -> 7; 200x50 in a 1x0.25 box -> 5; S4 at 100x100 -> 6, down from 87. There is
  still no DSA and a genuinely diffusive problem will still want one, but nothing in the
  current test set demonstrates that — the lesson is to suspect the operator before the
  preconditioner when a count depends on something the operator should not care about.
  `-sigma_scatter` on `box_2dk`, added to investigate this, stays: varying the ratio
  independently of `sigma_t` is worth having.
- **Single assembled streaming matrix across all groups** (Phase 5 direction): apply
  removal + scatter matrix-free so only one assembled matrix is stored for all energy
  groups. Open question: an effective preconditioner for streaming-only pmat when removal
  is strong. Record iteration-count findings here as Phase 5 runs.
  Re-baseline that question before starting: `-precon_stream` is already a streaming-only
  pmat with strong removal, and the Aug 2026 Dirichlet fix moved those numbers. As of now
  it costs roughly a factor of 1.5 over the full pmat rather than the several-fold gap the
  note was written against — 1D st=2 goes 6 -> 9, the multigroup sweep max 6 -> 11, 2D
  50x50 6 -> 8 and 120x60 7 -> 9. Phase 5's cost/benefit looks different at that ratio, and
  the numbers in `docs/dev/testing.md` are the ones to argue from.
- **MFEM as Phase 6 backend**: MFEM owns mesh/FE spaces/integrators (ConvectionIntegrator +
  DGTraceIntegrator = DG upwind streaming out of the box; AMR; high-order; GPU full and
  partial assembly). Frictions: MFEM's PETSc bridge produces MATAIJ not MATAIJKOKKOS (pflare
  GPU path dispatches on aijkokkos); MFEM device model is CUDA/HIP/RAJA not Kokkos (two
  device runtimes in one binary); per-group values-refill trick needs plumbing across the
  hypre->PETSc boundary. Cleanest hybrid candidate: MFEM AssemblyLevel::ELEMENT computes
  element matrices on device, UBOLT scatters them into its own COO arrays and owns the
  global aijkokkos matrix. `--download-mfem` exists in PETSc configure.
- **libCEED** (parked): `--download-libceed` exists; good for high-order CG volume terms
  matrix-free; no DG face integrals; CUDA/HIP backends, not Kokkos-native.
- **Angle-major ordering** (vs current angle-fastest): per-angle contiguous blocks or
  MatNest of per-angle streaming blocks each with its own AIR — plausible PCAIR win.
  There is NO DofOrdering enum reserving this (an earlier note here claimed there was);
  angle-fastest is currently hardcoded, stated in PhaseSpace's header comment as
  `row = cell * n_angles + angle`. Since Phase 3 it is also asserted against the DM's
  global numbering in `CheckDALayout`, so that assert is the other place a change of
  ordering has to be taught about.
- **Kokkos layouts are not the same on host and device** (found in Phase 2): the default
  layout of a rank-2+ View is LayoutRight on a host space but LayoutLeft on CUDA/HIP, so
  anything whose element ordering matters must state its layout. Worse, the obvious
  compile-time contiguity guard does not work — Kokkos allows assigning a rank-1 view of
  ANY layout to another, so a strided slice compiles and then aborts at run time. Test for
  `Kokkos::LayoutStride` instead. Full writeup in `docs/dev/kokkos.md`; this will matter
  again for the Phase 4 2D slices and the Phase 6 flattened DMPlex views.
- **Latent bug fixed in Phase 1a**: PETSC_DECIDE row split can land mid-cell (np=3 with
  1000x4); the decomposition is decided in cells (PetscSplitOwnership over n_cells) from
  Phase 1a on. np=1,2 are unaffected (the splits coincide), np=3 now converges in 10 its
  matching serial where the pre-refactor binary did not.
