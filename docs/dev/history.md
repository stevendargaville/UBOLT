# UBOLT development history

What UBOLT did and measured: the finished roadmap narratives, moved verbatim from `TODO.md`, and the dated iteration-count tables, moved verbatim from `docs/dev/testing.md` (Sep 2026).
Open work is in `TODO.md` and the pins in force in `docs/dev/iteration_counts.md` - every number here is labelled with its era and is NOT a pin.

Reading notes, added when the text moved here (Sep 2026 docs restructure):
- A reference to "TODO.md" or to a phase / postscript / "the research note" in the text below means the section of the same name in THIS file; "docs/dev/testing.md, <heading>" means that heading in this file's Testing history, in `docs/dev/verification.md`, or in `docs/dev/testing.md` - the section map at the end of `docs/dev/testing.md` says which.
- An unticked checkbox below is still open and is tracked in `TODO.md`.
- Names are as they were when written. Renamed since: `StreamingTerm` -> `StreamingTerm1D`; `GroupTransfer::add_source` -> `add_transfer` (on the `GroupSource` base); `UboltFillElementSource` -> folded into `UboltFillCellSource`; `CheckPlexLayout` / `CheckCGLayout` -> `CheckPlexStratumLayout`; `set_uniform_pattern(slots_per_row, dirichlet_mask)` / `set_pattern` now take a `BoundaryRows` struct; `UnstructuredDG0` -> `UnstructuredDG` (with an `order`); the drivers `slab_1dk`, `slab_1d_mgk`, `box_2dk` -> `transportk`.

# Roadmap history (from TODO.md)

## Current state as written on 2026-09-28, before the restructure

Last landed: **the CG DSA** (`-precon_dsa` on `cg_supg`: a P1/Q1 diffusion operator
on the transport's vertices, exactly the SUPG operator restricted onto an isotropic
flux in thick cells; every diffusive CG problem 35-48 -> 5-6 as its DG0 twin, the
void channel 47 -> 8 - see the Phase 6 item); before it, **CG-SUPG** (`UnstructuredCG`,
Phase 6b - the Phase 6 item); before it,
**`-precon_ref_shift` on voids** (`RefShiftPmats` sorts the groups by
their void cells, one reference per pattern, zero in the void, so a void cell's pmat
row is the bare streaming row the operator has there; exact coverage stays the full
pmat, voids included - see the ticked item at the end of Phase 6); before it, **void
bridging in the DSA** (by default `-precon_dsa` keeps `Sigma_t = 0`
cells IN its diffusion operator with the free-flight `D = L / 3`, `L = 4 V / S` the
voids' mean chord, and at DG1 a harmonic-D weighted interior penalty on the faces
touching a void, so the correction couples the regions a void separates: the void
recipes 6 / 8 / 6 / 8 / 10 -> 6 / 6 / 5 / 6 / 10 serial; `-dsa_void_bridge 0` is the
mask; see the Phase 4 postscript 5 DSA notes); before it, **void masking in the DSA**
(`-precon_dsa` masks `Sigma_t = 0` cells out of its correction on every backend and
order instead of refusing them, a face into a void taking the same consistent-D
Marshak face as a vacuum boundary); before it, **a discretisation-consistent DSA diffusion coefficient** (the default
`DSAPrecon` blends the upwind scheme's numerical diffusion `m h` into every face D as
`(D^1.5 + (m h)^1.5)^(1/1.5)`, structured and DG0 plex; `-dsa_consistent_d 0` restores
the physical D - every diffusive recipe 11/11/8 -> 5, crooked pipe 28 -> 8, the
literature crooked-pipe sets 51-106 -> 17-21 and the eps sweep flat at 9-11, see the
Phase 4 DSA notes); before it, **DG1-consistent DSA** (at DG1 `DSAPrecon`'s plex overload builds the MIP
interior penalty diffusion operator in the DG1 space itself, every basis node restricted
and corrected; see the Phase 6 item); before it, **DSA on the plex backend** (a
volume-weighted two-point-flux diffusion operator, PR #12); before that,
**`ElementBlockInverse::scale` bumps its state directly** (the MPI wrapper's,
through `PetscObjectStateIncrease`, pinned in `verify_plexk` via `MatGetState` - which
needs PETSc main from 3 Sep 2026); before it, **the block-Jacobi removal stage** (composite index 0 inverts the
operator's element blocks through `ElementBlockInverse`, bitwise the old point Jacobi at
n_basis 1, block Jacobi at DG1); before it, **element-block-inverse scaling of PCAIR's
pmat** (`ElementBlockInverse`, `-precon_block_scale`, default on the DG backend at both
orders, PR #9), which lets PCAIR coarsen DG1 at its default strong threshold; before that, **ghost-flux reflective faces**
(the DG0 mixed-corner fix under the ghost-flux postscript: under `"ghost_flux"` a
reflective face is a face coupling to the mirrored angle in the same cell, in every
backend, so there are no BC rows and a reflect/vacuum corner takes both faces' ghost
values, PR #8), on top of linear DG (DG1) on the plex backend (PR #7), Phase 6a (DG0 on
DMPlex, PR #2) and the ghost-flux vacuum treatment as the default (PR #4). DSA now
works at both orders on the plex: at DG0 every quad/hex twin takes the structured DSA
count and simplices go 35 -> 10 / 31 -> 9; at DG1 (27 Sep 2026) the diffusive quad box,
hex cube, triangles and tets go 34 / 32 / 40 / 43 -> 6 / 7 / 8 / 9 (DG0+DSA now takes
5 on all four meshes' DG0 twins, through the consistent D). 6b CG-SUPG landed 28 Sep
2026 (the Phase 6 item: `mesh.discretisation: "cg_supg"`, consistent SUPG = SAAF-tau,
verified against the SAAF-LS void slab and the thin/thick slab benchmarks; its
follow-ups are under it; its two iteration gaps against DG0 and
Wang's cell size were investigated 28 Sep 2026, no code change, and the CG DSA
closed the diffusive one the same day). Next up: one of the open questions carried as checkboxes since 27 Sep 2026: per-group
cached Mat/KSP for the DSA (in the Phase 4 postscript 5 DSA notes; void masking, the
other, is done), and the follow-ups the 27 Sep 2026 round left (the simplex +1 - a
stronger cheap DSA inner solve was looked at and closed with no change; the list at the end of Phase 6 - the void-bridging
DSA operator and `-precon_ref_shift` on partly-void groups from it are done). Both Phase 6a things to watch are
closed (27 Sep 2026): simplex iteration counts creeping up with refinement is
explained and gone under the current defaults - it was PCAIR's row-relative R drop on
the old unscaled Dirichlet-cell pmat, which either ghost-flux or the element-block
scaling removes; simplices now sit at most one over quads/hexes, flat in n - and
triangle L-infinity below first order is not a defect: the code matches a Python
model of upwind DG0 to 7 digits, and the deficit comes from PETSc's simplex box being
a different, non-nested diagonal pattern at each n (nested refinement of one box gives
first order). Details in the Phase 6a items. The
half-quadrature transposed PC stays blocked on PFLARE's PCAIR `PCApplyTranspose`. The
reflective-face re-pins were swept in the 64-bit CI image (one +1,
`cube_10_inf_medium_ghost -matfree_removal` pinned 22); the block-scaled plex pins are
re-measured on top of it (see docs/dev/testing.md, "Unstructured iteration counts").
Both findings from regenerating the Phase 6a report are now fixed; the element-block
scaling left two follow-ups under the ghost-flux postscript: the block-Jacobi removal
stage is done (index 0 inverts the operator's element blocks through the same
`ElementBlockInverse`, bitwise the old point Jacobi at n_basis 1; kept on DG after
measuring that dropping it costs 1-2 iterations), and the scaled matrix's state bump is
now direct and tested (see that item).

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

## Phase 4.5 — 3D structured (DMDA 3D) (Aug 2026)
- [x] `StructuredFD3D` (4-slot rows: upwind-x, upwind-y, upwind-z, diagonal LAST — the
      DMDA star stencil's 7 points are what it preallocates, not what the operator
      touches, same discipline as 2D), `SNQuadrature3D`, `StreamingTerm3D`,
      `MaterialBox3D` + `paint_boxes`, `ProblemSpec` dimension 3, the driver's third
      branch (the old bare `else` is now `else if` with a SETERRQ default). Everything
      else — removal, scattering, group transfer, solver, VTK output — worked through
      the bases untouched, which is the Phase 4 seam paying out.
- Notes:
  - `SNQuadrature3D` folds nothing: 3D has no symmetry plane, so all 8 octants are real
    and xi is a real cosine — S2 is 8 ordinates and S4 is 24, twice the same-order 2D
    set. Three reflection maps (mu, eta, xi), built by the same FindOrdinate search,
    widened to three cosines with NULL for the ones a dimension lacks (verified inert:
    1D baselines re-run bitwise).
  - FACE ids follow PETSc's 3D box "Face Sets" convention (`plexcreate.c`,
    DMPlexSetBoxLabel_Internal): bottom/top are the Z faces (1/2), front/back the y
    faces (3/4), right/left the x faces (5/6). That means bottom/top CHANGE AXIS
    between 2D and 3D — flagged in the header, ProblemSpec and the docs, since it is
    the likeliest user-facing confusion.
  - A corner between three reflective faces composes all three flips — still one
    partner, one column, the same repurposed first slot.
- Verify: DONE — `tests/verify_3dk.kokkos.cxx` (closed form psi = x + y + z on 8x6x4
  over a 1x2x3 box at np 1/2/4/8, residual ~8e-15; reference matrix on 4x3x2 bitwise
  for vacuum/all-reflect/mixed, where mixed = left+front+bottom so the triple flip is
  in the checked matrix), the infinite-medium check (~1e-13, np 1 and 2), the painting
  identity (bitwise vs the uniform cube history), and pinned recipes for the cube
  problem files (counts in docs/dev/testing.md). No 3D baselines, the 2D precedent.
- [x] Follow-up sweep of the 3D pins over the CI arches: closed Aug 2026 — CI runs
      green with the pins as committed, so no per-arch slack was needed.

## Phase 5 — Matrix-free removal / single-streaming-matrix experiment
- [x] RemovalTerm::apply_add + -matfree_removal: Assembled{Streaming} +
      MatrixFree{Removal, Scattering}; one assembled matrix for all groups — DONE.
      `RemovalTerm::set_matrix_free` switches the halves (default unchanged, so all 24
      baselines still reproduce bitwise); `TransportOperator` partitions its terms at
      assemble() rather than add_term() so the switch can go either side of the add. The
      mode assembles ONCE before the sweep and no group refills anything, so it implies
      the streaming-only pmat (`-precon_stream` alongside it is redundant and ignored)
      and rules out `-diag_scale` (a checked error). Counts equal the `-precon_stream`
      twins exactly, 1D/2D/3D, serial and np=2; histories differ in the last bits by
      design — see docs/dev/testing.md.
- [x] Terms get optional get_inv_diagonal(Vec) so RemovalPCShell composes its diagonal
      analytically instead of MatGetDiagonal(assembled) — DONE as
      `OperatorTerm::has_diagonal()` + `add_diagonal(Vec)` (Streaming*/Removal override,
      Scattering deliberately does not — it has never been in the assembled matrix)
      composed by `TransportOperator::diagonal()`, BC rows 1.0. The solver uses it iff
      `diagonal_is_composed()`, so the assembled path still reads MatGetDiagonal. Each
      add_diagonal writes the same expression its assemble_add does, in the same order,
      so the composed diagonal is BITWISE the assembled one — pinned by `-check_matfree`
      (measured 0.0 everywhere; matvec 2.5e-16 to 4.2e-16 against 1e-13).
- [x] An effective preconditioner for the streaming-only pmat when removal is strong —
      DONE as `-precon_ref_shift` (+ `-precon_ref_k`), the Aug 2026 campaign's strategy A.
      `RefShiftPmats` (`include/ubolt/ref_shift.hpp`) builds k copies of the streaming
      matrix carrying `alpha_k * D_ref`, `D_ref` being group 0's per-cell `Sigma_t` and
      `alpha_g` the log-mean of `Sigma_t(g)/Sigma_t(0)`; the groups are clustered into k
      log-spaced bins and the driver keeps one `TransportSolver` (so one PCAIR hierarchy)
      per bin. Exact coverage — one bin per distinct ratio — reproduces the FULL pmat's
      iteration counts exactly on six problem files, out of k matrices assembled once and
      no assembly in the sweep at all; the new decades files are where the bare streaming
      pmat does not converge at all and the shift does. DSA re-attaches on it (exactly the
      full-pmat + DSA counts) where it cannot attach to a bare L. Counts, the bare-L
      comparison and the mismatch cost: `docs/dev/testing.md`. Since 27 Sep 2026 voids
      too: one reference per set of void cells, zero in the void (see the ticked item
      at the end of Phase 6).
- Verify: matvec equivalence vs assembled path on random vectors (<1e-13); solution norms
  match Phase 2; iteration counts recorded (not pinned) — preconditioner quality is the
  research question; findings recorded below.
- [x] Follow-up: DSA on a MISMATCHED reference-shifted pmat is fragile — 2D `box_decades4`
      at k = 2 took 179 iterations on the thick group where exact coverage takes 11.
      CLOSED 2026-09-25 by the ghost-flux default. The k = 1..4 sweep, with and without
      DSA, serial and np 2, on both decades files and the two `mg4_t05` files: every
      mismatched DSA run converges, including k = 1 at mismatch 31.6 (thick group 16 in
      1D, 19 in 2D, where no-DSA takes 68 and 83), and the default k = 2 + DSA takes
      10/11 against exact coverage's 11. The same `-precon_stream -precon_dsa` interaction
      went with it: on `cube_diffusive` Dirichlet-cell still reproduces 35 -> DIVERGED
      (300) with DSA added, and ghost-flux takes 44 -> 44. The likely mechanism is the DSA
      BC mask: under Dirichlet-cell every inflow boundary cell has its inflow ordinates
      masked out of both the restriction and the prolongation, so the diffusion solve sees
      a partial moment and corrects only half the angles in exactly those cells — an
      inconsistency a weak pmat (mismatched or streaming-only) cannot absorb. Ghost-flux
      empties the mask on an all-vacuum problem. Pinned: default-k + DSA on both decades
      files (serial + np 2) and `cube_diffusive -precon_stream -precon_dsa`. Table:
      docs/dev/testing.md, "DSA on a shifted pmat".
- [x] Follow-up (found in that sweep, fixed 2026-09-25): `RefShiftPmats::bin_alphas` broke
      TIES by rounding. On equally spaced log-alphas (the decades files at k = 3) which pair
      shared the merged bin depended on the MPI sum order of the log-means, so serial and
      np 2 built different pmats (2D: 4, 6, 27, 51 vs 7, 8, 15, 29). Widths now compare
      with a tie tolerance, and spare bins (the optimal-width greedy can need fewer than
      asked) are spent from the top, splitting off the highest distinct alpha, where the
      thick groups make an exact pmat pay most. No pinned count moved; k = 3 on
      `box_decades4` is now pinned at 29, serial and np 2.
- [x] Follow-up: sweep the `-precon_ref_shift` pins over the CI arches — SUPERSEDED
      2026-09-25: pins are now the local opt measurement and CI flags any arch that needs
      +1 (docs/dev/testing.md, "Pass/fail contract").
- Measurements behind the `RefShiftPmats` defaults (moved out of `ref_shift.hpp` /
  `ref_shiftk.kokkos.cxx` in the Sep 2026 refactor): the Phase 5 campaign (Aug 2026)
  measured iteration quality degrading gently with the mismatch ratio - a mismatch of 3
  costs roughly 1.3-1.6x the exact-ratio count, the whole useful window is about a
  factor of 10, and it is independent of the mesh and the angular order; hence
  `default_max_mismatch = 3`, `default_max_bins = 3`. Spare bins go to the TOP because
  the thick groups pay most for a mismatch: `box_decades4` at k = 3 takes 15, 29 on its
  two thick groups with them exact, against 27, 51 with them sharing a bin.

## Phase 6 — DMPlex backends
- [x] DECISION POINT (22 Sep 2026): hand-written Kokkos kernels over DMPlex, NOT MFEM, and
      no spike. Every MFEM friction in the research notes below (MATAIJ rather than
      MATAIJKOKKOS out of its PETSc bridge, a second device runtime alongside Kokkos, the
      per-group values-only refill needing plumbing across the hypre/PETSc boundary) fights
      the COO + slot-map + refill architecture the library is built on, so the spike would
      have measured what the notes already predict.
- [x] 6a `UnstructuredDG0`: DG0 (one dof per cell) upwind streaming on a DMPlex, 2D and 3D,
      box meshes built in code (quads/hexes or triangles/tets) or a mesh file read by
      PETSc (Gmsh). Broken section with n_angles dof per cell, so the row convention
      `row = cell * n_angles + angle` is unchanged and everything dimension-independent
      carries over untouched. Variable-nnz COO pattern: `n_faces(c) + 1` slots per row in
      cone order, diagonal last, through a new `Discretisation::set_pattern` that
      `set_uniform_pattern` now wraps. Geometry (cell volumes, outward area-weighted face
      normals, centroids) extracted on the host once by `DMPlexComputeCellGeometryFVM`
      into flat device views; `StreamingTermDG0` is the per-backend streaming sibling and
      reads the ordinates off the backend, never a quadrature of its own. BCs: the
      existing Dirichlet-cell contract, keyed by real "Face Sets" values (the box ids ARE
      the structured `FACE_*` ids); reflection needs an axis-aligned face. Materials:
      `paint_boxes` by centroid plus "Cell Sets" -> material. Output: `.vtu`. Not in this
      cut: DSA (a DMDA operator; its plex operator landed 27 Sep 2026, below), DG1+ (DG1
      landed later, under the ghost-flux postscript). The ghost-flux vacuum BC landed afterwards, on top of the structured
      ghost-flux commit (see the postscript below).
  - Measured (22 Sep 2026, opt arch, under the then-default Dirichlet-cell treatment;
    the pins have since run green on every CI arch, and were re-pinned for the
    ghost-flux default — the plex_box_50_st2 gap below closed there, both boxes take 7;
    table in docs/dev/testing.md): the quad/hex twins match their structured counts (reflect_lb
    6, cube 5, streaming-only pmat 9, ref-shift k=4 4/6/15/30) EXCEPT `plex_box_50_st2`,
    7 against 6 — the matrices agree to ~1e-15 but the rows are permuted and PCAIR is
    not permutation-invariant, and both solves sit on the rtol edge at iteration 6
    (structured clears by 1.4%, plex misses by 0.4%); the partitioner moves the same
    edge (parmetis gives 6 at np 2). Simplex: 1800 triangles S4 and 1296 tets S2, both
    ratio 0.5, converge in 5, the 8-triangle Gmsh file in 4; np 2 equals np 1 on every
    recipe but one group of the default-k ref-shift run. `-check_matfree` composes the
    diagonal to exactly 0.0 on every plex file and `-ubolt_coo_two_call` is
    history-identical.
  - Measured by the experiments pass (23 Sep 2026, opt arch; tables in the campaign
    report): DG0 is FIRST ORDER on every cell shape against the exact discrete-ordinates
    solution (pure absorber, reflective y faces, left inflow) — L2 orders 1.00 quads,
    1.02 triangles, 0.99 hexes, 0.99 tets — with plex quads reproducing the structured
    errors to every printed digit; triangles carry 0.62x the quad L2 error at the same n
    (12% better per cell), tets 8% better per cell than hexes; the Dirichlet-cell
    boundary treatment costs nothing visible in L2 (1.5% at n = 8, 0.05% at n = 256).
    Two things to watch: (i) the triangle L-INFINITY error converges below first order
    (0.72-0.91 between the finest levels), largest along the reflective walls (a guess
    that every square is split along the same diagonal is refuted: PETSc's simplex box
    mixes the two diagonal directions) — EXPLAINED 27 Sep 2026, not a defect: the
    scheme's pointwise sensitivity to the irregular diagonal pattern, see the item
    below; (ii) tets on the pure absorber at rtol 1e-12 take 6 -> 19 -> 22
    iterations for n = 8 -> 32 where hexes take 5 -> 7 (at the default rtol the counts
    are normal, 5-6). Iteration counts on simplices creep up ~1 per 4x refinement (2D
    ratio 1: 6, 7, 7, 8 at n = 30..240; tets 6, 7, 8) where quads/hexes match the
    structured backend exactly and are flat. (ii) is explained and gone under the
    current defaults (27 Sep 2026): PCAIR's relative R drop on an unscaled
    Dirichlet-cell pmat - see the item below. Without DSA a diffusive problem costs
    simplices 3.3-3.5x the structured+DSA count (36 against 11 on box_diffusive),
    quads/hexes 2.6x (29) — the size of the DSA gap this cut leaves open (closed at DG0
    on 27 Sep 2026, see the DSA item below). The plex path
    costs ~4% of a serial run in host-side create and 8-13% more peak memory than the
    DMDA on the same mesh; PCAIR setup is 85-95% of both. Debug-arch sweep: every count
    equals the opt pin at np 1 and 2, no leaks under -malloc_dump.
  - [x] Triangle L-infinity below first order (thing to watch (i) above, 0.72-0.91
    between the finest levels, largest along the reflective walls). That study ran
    under Dirichlet-cell, whose reflective rows carry an O(h) error the 26 Sep 2026
    ghost-flux reflective-face fix removed. CLOSED 27 Sep 2026 - NOT a defect, a
    property of upwind DG0 on PETSc's simplex box family. Re-run of the same E1 study
    (pure absorber, left inflow, reflective y, S4, exact SN cell averages, rtol 1e-12,
    opt arch; scripts + logs beside the Phase 6a report,
    `results/experiments_2026-09-27_tri_linf/`), ghost-flux default, orders between
    successive n = 8..256:

    | cell-average error | L1 | L2 | L-inf |
    |---|---|---|---|
    | quads (plex = FD to every digit) | 0.89 0.95 0.97 0.99 0.99 | 0.88 0.94 0.97 0.98 0.99 | 0.67 0.82 0.91 0.95 0.98 |
    | triangles (PETSc box) | 0.91 0.95 0.97 0.98 0.97 | 0.88 0.94 0.95 0.98 0.97 | 0.69 0.94 0.89 0.85 0.81 |
    | hexes S2, n = 8..32 | 0.93 0.96 | 0.91 0.96 | 0.76 0.88 |
    | tets S2, n = 8..32 | 0.87 0.93 | 0.87 0.93 | 0.74 0.87 |

    (triangles, n = 256: L1 3.99e-3, L2 6.07e-3, L-inf 2.78e-2; the Dirichlet-cell run
    had L-inf 3.21e-2.) Findings: (1) the maximum is no longer on the reflective walls:
    from n = 32 up it sits in a cell with NO boundary face, in the inflow boundary
    layer (x = 0.02-0.06, the S4 decay length is mu/sigma = 0.35), at a y that jumps
    between levels. (2) The code is not the cause: a 150-line Python model of textbook
    upwind DG0 with ghost-flux walls (each +-eta pair solved directly) reproduces
    UBOLT's triangle L1/L2/L-inf to all 7 printed digits at n = 16, 64 and 256.
    (3) Splitting the error into its x-column mean (the smooth part; the exact
    solution depends on x only) and the rest: the column mean is FIRST order on every
    mesh (0.94, 0.96, 0.97 at n = 256, 512, 1024 on the PETSc box); the L-inf deficit is
    all in the mesh-irregularity fluctuation. (4) PETSc's simplex box is Triangle's
    Delaunay of a lattice whose squares are all cocircular, so each diagonal is an
    arbitrary tie-break: ~70% one direction, in streaks along x, NOT nested, and its
    statistics drift with n ('/' fraction 0.22 -> 0.31, row-to-row correlation +0.08
    -> -0.25 from n = 16 to 1024) - successive levels are different mesh families.
    The model, to n = 1024 (2M triangles), L-inf orders 256 -> 512 -> 1024: PETSc box
    0.82, 0.81 (L2 0.97, 0.93); every square split the same way 0.96, 0.97; i.i.d.
    random diagonals 0.86, 0.88 (rising; the fluctuation's RMS goes at 0.97/0.98, the
    max of more cells is the extreme-value lag); NESTED regular refinement of PETSc's
    own n = 16 box 0.95, 0.96, 0.97 (from n = 64: 0.83, 0.90, 0.92). So on any
    consistent refinement family L-inf goes to first order; the PETSc-box sequence
    stays at ~0.8 because it is not one. This is the known picture for upwind DG0:
    only O(h^1/2) is guaranteed on general meshes (Johnson & Pitkaranta 1986, sharp by
    Peterson 1991), first order needs mesh structure (Cockburn, Dong & Guzman 2008),
    and the first order seen in L1/L2 is supraconvergence that is weakest pointwise.
    Consequences: a simplex convergence study should refine one coarse mesh (nested),
    not rebuild PETSc's box at each n; DG1 is second order on the same PETSc boxes (L2
    1.93-2.01, 25 Sep 2026 study), so there is nothing to fix at DG0. No code change,
    no regression test (no defect).
  - [x] Iteration counts creeping up on simplices (thing to watch (ii) above) -
    EXPLAINED and GONE under the current defaults (27 Sep 2026, opt arch). Re-measured
    (ghost-flux, element-block PCAIR scaling, block-Jacobi removal): the absorber at
    rtol 1e-12 takes tets 6 / 7 / 7 at n = 8 / 16 / 32 against hexes 5 / 6 / 7
    (was 6 / 19 / 22), triangles 4 / 5 / 6 / 6 / 7 against quads 8 / 8 / 6 / 6 / 7 at
    n = 8..128; at the default rtol, ratio 1, triangles 7 / 7 / 7 / 7 (n = 30..240,
    quads 6 / 6 / 7 / 7) and tets 6 / 7 / 7 (n = 6..24, hexes 5 / 6 / 6), ratio 0.5
    flat at 5 on both 2D meshes. DG1 (block-scaled, default threshold): absorber
    triangles 6 / 8 / 9 against quads 6 / 6 / 8 at n = 8..32, ratio 1 triangles
    7 / 8 / 8 against quads 7 / 7 / 7 at n = 30..120, tets 7 / 7 against hexes 6 / 7.
    CAUSE: the old study ran Dirichlet-cell with an unscaled pmat, and the old counts
    come back exactly with `"vacuum_treatment": "dirichlet_cell"` and
    `-precon_block_scale 0` (tets 6 / 19 / 22); EITHER ghost-flux OR the block scaling
    alone gives 6 / 7 / 7. The unscaled history is a steady 0.32 contraction per
    iteration against 0.012 scaled - not a tail. The culprit is PCAIR's R drop
    (`-pc_air_r_drop`, 1e-2 relative to the row): R = -A_cf A_ff^{-1} carries each F
    row's 1/diagonal, so it is NOT invariant under a row scaling, and next to the
    identity rows (diagonal 1) the transport couplings through interior F points
    (diagonal up to 84 on the n = 16 tets, and growing as 1/h) fall below the drop
    tolerance - hence the growth with n. Evidence: `-sub_1_pc_air_r_drop 0` (or 1e-4)
    restores 7 on the unscaled Dirichlet-cell n = 16 tets while `-pc_air_a_drop 0`,
    diagonally scaled polynomials, a Jacobi F inverse and sparsity order 2 all stay
    at 19; raising `r_drop` to 0.05 breaks the unscaled n = 8 tets (6 -> 16) and does
    nothing to a block-scaled run (7). The hierarchies with and without the scaling
    are the same size (19 levels, operator complexity 5.1 / 5.2), so it is not the
    coarsening. NOT cycles in the upwind graph: with the reflective same-cell
    couplings removed, the per-angle upwind graph of every PETSc simplex box measured
    (triangles n = 32, 128; tets n = 8, 16) has NO strongly connected component - it is
    exactly triangularisable, like the quad/hex ones; the reflective couplings close
    angle loops on quads/hexes just as on simplices. What remains is a constant +1
    on simplices at ratio 1 (not a creep): with an exact (LU) streaming inverse
    triangles/quads and tets/hexes all take 6, so it is PCAIR's approximation;
    `-sub_1_pc_air_strong_threshold 0.25` or `-sub_1_pc_air_inverse_sparsity_order 2`
    remove it at small n but it returns by n = 120 (triangles) / 24 (tets), and the
    counts sit on the rtol edge (both 2D meshes end iteration 6 at 1.7-1.8e-5), so
    no default was changed. Pinned by `plex_tet_12_absorber_dirichlet_cell`
    (Dirichlet-cell absorber at rtol 1e-12, 6; 12 unscaled). Side note: PCAIR's
    operator complexity grows with n on every mesh, most on hexes (12 / 20 / 27 on the
    absorber at n = 8 / 16 / 32, tets 4.2 / 5.5 / 6.6), with flat counts.
  - Seen in the report figures (23 Sep 2026): a quarter box with reflective faces differs
    from the full box's quadrant by O(h) — max 0.080 / 0.037 / 0.018 at n = 25 / 50 / 100
    on a flux of order 10, unchanged by rtol — on the STRUCTURED and the unstructured
    backend alike (the two agree to 2e-10 on the quarter problem). The reflective row
    equates the two ordinates INSIDE the boundary cell where the full box sees the
    mirror image one cell away: a first-order boundary approximation of the same kind
    as the Dirichlet-cell vacuum row, and the same ghost-flux treatment would remove
    both. Not a Phase 6a defect; recorded because "reflect = symmetry to solver
    tolerance" is the natural test to reach for and it is not true of this convention.
    REMOVED (26 Sep 2026) under ghost-flux, whose reflective faces are now face couplings
    to the mirrored angle: the quarter box equals the full box's quadrant to rounding
    (Python model of the FD operator, S2 and S4; see the ghost-flux postscript). Only
    `"dirichlet_cell"` still has it.
  - Found by the debug sweep and fixed (23 Sep 2026): the first cut rejected a reflective
    axis plane wherever it met a slanted vacuum face (the partner row is Dirichlet there,
    which is fine; only a REFLECTIVE partner is the unsupported single-cell-wide case),
    and it silently accepted a boundary condition on a "Face Sets" value no face carries.
    Both are now checked in `verify_plexk` on `tests/meshes/tri_slanted.msh`.
  - Found in the driver pass: the plex `.vtu` arrays were named `scalar_flux(null)` —
    PETSc's writer appends the section's field name, and a section with no fields gives
    it a null one. Fixed by giving the output twin's section one empty-named field.
- [x] DSA on the plex backend (27 Sep 2026). 6a left it out (`DSAPrecon` was a DMDA
      operator), and it was the biggest gap that cut left. `DSAPrecon::create` now has an
      `UnstructuredDG` overload: a cell-centred finite-volume diffusion operator, a
      two-point flux per face `T_f = A_f / (d_c / D_c + d_n / D_n)` (d the centroids'
      normal distances to the face, which the backend now stores per face slot next to
      its other host face data, `face_*_host()`), Marshak `A_f / (2 + d_c / D_c)` on
      vacuum faces, nothing on reflective ones, behind the same R/P. Assembled
      VOLUME-WEIGHTED (V_c times the structured per-unit-volume row), which keeps it SPD
      where volumes vary; the restricted moment is scaled by V to match. Host values into
      a COO pattern set once (`MatSetValuesCOO`), the neighbours' D through a VecScatter
      onto one entry per interior face slot. No DMDA twin, so the matrix is sized off the
      owned cells, whose global order is the transport rows'.
  - Verified (verify_plexk check 11): on every quad/hex FD twin case the plex diffusion
    matrix is V times the structured one to ~1e-15, and `P D^-1 R` agrees to rounding
    with tight inner solves, in both vacuum treatments, serial, -n 2 and -n 4.
  - Measured (opt arch, table in docs/dev/testing.md, "Unstructured iteration counts"):
    every quad/hex twin takes EXACTLY the structured DSA count, serial and np 2
    (box_diffusive 29 -> 11, ref-shift + DSA 11, additive 14, cube_diffusive 18 -> 8,
    box_50_st2_dirichlet_cell 5). Simplices close the 3.3-3.5x gap entirely: 5000
    triangles 35 -> 10, 6000 tets 31 -> 9 - at or below the structured+DSA count.
    (All physical-D counts: with the consistent D, 27 Sep 2026, every one of these
    diffusive files takes 5.)
  - DG1 did NOT get the same (superseded the same day by the interior penalty
    operator, the next item): the restriction sums basis 0's ordinates (the cell
    balance rows) and the prolongation corrects basis 0 only, which is an INCONSISTENT
    DSA for DG1 - its thick diffusion limit is a continuous linear discretisation, not
    this cell-centred one (the Warsa-Wareing-Morel result for DG in multi-D). Measured on
    the diffusive problems: quad box 34 -> 34 (nothing), hex cube 32 -> 25, triangles
    40 -> 26, tets 43 -> 21. A slope rebuilt from the neighbours' corrections
    (Green-Gauss onto the modal coefficients, Marshak face values on vacuum faces) was
    tried in the prolongation and made it WORSE on quads/hexes (34 -> 37, 25 -> 33) and
    barely better on simplices (26 -> 24, 21 -> 20), so it was dropped.
- [x] DG1-consistent DSA (27 Sep 2026): at DG1 `DSAPrecon`'s plex overload now builds
      the MIP (modified interior penalty) diffusion form of Wang & Ragusa (NSE 166, 2010)
      in the DG1 space itself, on the backend's orthonormal modal basis: one unknown per
      (cell, basis) node, every node restricted (V times its weighted ordinate sum -
      row (c, i) of the transport is the balance tested with phi_i) and every node
      corrected, the slopes included. kappa = max(C/2 (D_c/h_c + D_n/h_n), 1/4) on
      interior faces, max(C D_c/h_c, 1/4) on vacuum faces with MIP's 1/2 boundary
      terms (Marshak again), nothing on reflective ones; h = twice the centroid-to-face
      distance; C = `-dsa_mip_penalty`, 4. The face terms read the backend's exact face
      matrices and both cells' basis gradients (new host accessors: `face_own_host`,
      `face_up_host`, `basis_grad_host`, `face_neighbour_grad_host`, the last one
      covering overlap ghosts). The matrix carries block size n_basis, which lets GAMG
      aggregate a cell's nodes together.
  - Verified (verify_plexk check 11, DG1 half, replacing the "basis 0 = the DG0
    correction" check): symmetric (exactly), linear exactness on interior cells to
    ~1e-15 (catches a 10% error in one face coefficient), and a round trip through
    `apply()`, on the irregular triangle file and hexes, serial, -n 2 and -n 4.
  - Measured (opt arch, one GAMG V-cycle, no DSA -> old cell-average -> MIP): quad box
    34 -> 34 -> 6, hex cube 32 -> 25 -> 7, triangles 40 -> 26 -> 8, tets 43 -> 21 -> 9;
    np 2 and 4 within 1. Every one is now at or below the same mesh's DG0+DSA count
    (11 / 8 / 10 / 9, physical D; the consistent D later took DG0 to 5 on all four). With the inner solve exact every case is 5, so what remains is
    GAMG on the IP matrix: the block size took tets 16 -> 9 and triangles 11 -> 8, two
    V-cycles or 4 smoothing steps each take another 1-3 off at more cost per apply, and
    the GAMG threshold or no smoothed aggregation do nothing. The penalty barely moves
    anything (the 1/4 floor dominates in the thick limit); below C ~ 2 the form loses
    coercivity (NaNs on simplices at C = 1), and C = 2 buys 1 iteration on thin
    problems only, so 4 stays. The thin, c = 0.5 infinite-medium quad box went 9 -> 11
    (12 without DSA) - DSA has little to do there and the IP operator's thin-cell
    penalty is not the transport's; not pursued.
- [x] 6b CG-SUPG (28 Sep 2026): `UnstructuredCG`, `mesh.discretisation: "cg_supg"`.
      Decisions, from a literature pass (SN transport SUPG, Rattlesnake): CONSISTENT
      SUPG - the whole residual tested with `v + tau Omega.grad v`, which is Wang's
      SAAF-tau (NSE 176, 2014; Rattlesnake's CFEM "SAAF with a void treatment")
      written from the first-order side, so the exact solution satisfies the discrete
      equations for any tau; an inconsistent (streaming-only) SUPG is first order and
      adds a streamline diffusion ~h that ruins the thick limit. tau = min(1/sigma_t,
      h_Omega/zeta) per (element, angle, group): 1/sigma_t in thick cells is SAAF's
      diffusion limit, h/zeta covers thin cells and voids; zeta 0.5 (Rattlesnake's)
      by default, `mesh.supg_zeta` / `-supg_zeta`. The TODO's "Dirichlet via
      identity-row" was NOT taken: boundaries are weak with a lumped face mass and
      there are no BC rows, as DG1 (ghost-flux became the default after this line
      was written). PetscFE only on the host for the element tables (M, G, K), as
      planned; the kernels are row gathers over each vertex's star, no atomics.
      Rows (vertex, angle); xsections per local element; FEM-closure overlap. The
      plex plumbing is shared with the DG backend (`src/plex_commonk.hpp`, bitwise
      inert on all 91 plex recipes). Verified by `tests/verify_cgk`: constant exact
      to 1.3e-15 through operator and rhs (voids, reflective faces, every shape),
      second order (quads 1.95, triangles 1.96), and the two literature benchmarks
      against the exact SN solution - the SAAF-LS void slab (arXiv 1605.05388,
      order 2.05, 8-10 iterations to rtol 1e-13 at every refinement to 640 cells where
      GMRES + BoomerAMG on SAAF-tau took 801-8120) and the thin/thick slab (arXiv
      1902.08729, order 1.93; the flux at the interface vertex dips 33% (zeta 0.5) /
      35% (zeta 2) at their 8 cells per region). A global-balance check (a linear psi,
      outflow minus inflow) is what pins the weak boundary terms' scale - the first
      cut had the face coefficient divided by the area twice and passed everything
      else. Counts: the DG0 twins' except the diffusive quad box (39 vs 29) and the
      void channel (47 vs 26), investigated below. Tables in
      docs/dev/testing.md, "CG-SUPG verification and iteration counts".
  - [x] DSA for CG-SUPG. DONE 28 Sep 2026: `DSAPrecon`'s `UnstructuredCG` overload, a
    continuous P1/Q1 diffusion operator on the same vertices (restriction/prolongation
    the identity in space, the moment scaled by the lumped mass). It is EXACTLY the
    SUPG operator restricted onto an isotropic flux, `m R A P`, in thick cells
    (verify_cgk checks it to 4e-16): the SUPG term restricts to `(1/W) sum w tau Omega
    Omega^T : K` = `1/(3 sigma_t) K` when tau = 1/sigma_t, the vacuum face to the
    half-range current times the lumped face mass. D = 1/(3 sigma_t) per element
    (the SUPG tensor everywhere was worse in thin cells, 7 against 5 on the box;
    Marshak's 1/2 on vacuum faces costs 1-2); voids bridged by the chord L / 3 like
    every other backend (the SUPG tensor there degrades 7 / 9 / 11 with refinement,
    the chord holds 8). Every diffusive CG problem 35-48 -> 5-6, the DG0 twins' DSA
    counts; decades4 41 -> 5; the void channel 47 -> 8 against DG0's 6 - with both
    inner solves exact the channel is 7 against 4, so what is left is the diffusion
    model with SUPG in the void (no fixed void D does better). Wang's P0-projected
    DSA was not tried: the vertex operator already takes the DG0 counts and is the
    transport's own restriction. Tables in docs/dev/testing.md, "CG-SUPG".
  - [ ] `-matfree_removal`, `-precon_stream` and `-precon_ref_shift` on CG. All three
    assume a group-independent streaming matrix; with tau = min(1/sigma_t, h/zeta) the
    SUPG streaming part depends on the group. A sigma-independent tau (h/zeta only)
    would restore it at the cost of the thick diffusion limit; not done.
  - [x] The CG void channel takes 47 against DG0's 26 and the diffusive quad box 39
    against 29 (every other CG twin is within 1 of DG0). INVESTIGATED 28 Sep 2026, no
    code change - two different causes, neither a bug (an exact LU streaming inverse
    splits each count into discretisation and PCAIR; tables in docs/dev/testing.md,
    "CG-SUPG verification and iteration counts"):
    - Diffusive box: PCAIR's accuracy. Under LU CG ties DG0 (25 / 24). In thick cells
      tau = 1/sigma_t, so the SUPG removal term cancels the skew part of the streaming
      and the operator is dominated by sigma_t times the CONSISTENT mass matrix, not
      diagonally dominant on Q1 (off-diagonals 20/36 against 16/36; P1 triangles sit at
      equality, their gap is smaller, 35 against LU 25). A throwaway lumped-removal
      switch put AIR within 2 of LU, but lumping only the removal doubles the LU count
      (25 -> 50) and gives up SUPG consistency, so it is not the fix. No PCAIR knob
      does better than 35. The fix is the CG DSA above.
    - Void channel: mostly the discretisation (LU 38 against 21), SUPG streamline
      diffusion in the thin cells - any channel with sigma_t <= 1 behaves the same, and
      the LU count follows tau = h/zeta there: zeta 0.125 / 0.5 / 2 / 8 / 32 give LU
      45 / 38 / 27 / 24 / 23, AIR 52 / 47 / 41 / 53 / diverges (AIR needs the
      stabilisation the outer iteration pays for). The CG DSA (above) takes it to 8
      (DG0 + DSA: 6); a void tau not scaled by h/zeta is not tried (Wang's cell size
      tried, below: 46 against 47).
  - [ ] SUPG is not adjoint-consistent: no `A^T = P A P` on this backend, so the
    half-quadrature transposed PC (blocked on PFLARE anyway) would not carry over.
  - [x] h_Omega is the Tezduyar/Shakib element length along Omega; Wang's SAAF-tau uses
    a cell size. COMPARED 28 Sep 2026, kept h_Omega (throwaway switch, reverted; table
    in docs/dev/testing.md, "CG-SUPG verification and iteration counts"). On a square
    quad h_Omega = dx / max(|mu|, |eta|), so a fixed h only rescales tau per angle, and
    it behaves like a shift in zeta. On the void channel (AIR / LU, zeta 0.5) the
    shortest edge (= V^(1/dim) on quads) gives 46 / 35 against 47 / 38, the diameter
    48 / 37. The diffusive boxes do not change, because tau = 1/sigma_t there. With the
    shortest edge every exact-SN benchmark in verify_cgk is less accurate (coarsest
    mesh 2-18% worse RMS, finest within a few percent) and the slab counts go up at
    zeta 2. One iteration does not pay for that.
- BCs: consume the existing `BCSpec` with real "Face Sets" label values (the structured
  backends' FACE_* ids already match the box-mesh convention, so square meshes carry
  over unchanged) — see the reflective-BC postscript below for the label/physics split
- [x] Introduce thin abstract Discretisation base — DONE in Phase 4a, as that phase's
      decision point predicted. This item survives only as the point where a DMPlex
      backend would widen it: `set_uniform_pattern` is the part that will not carry over
      (DG has a variable-nnz COO pattern), while `create_matrix` and the accessors should.
- [x] Ghost-flux vacuum BC on the DG0 backend (Sep 2026, rebased onto the structured
      ghost-flux commit): a direction coming in only through vacuum faces keeps its
      physical row, the rhs takes `|Omega . nA_f| / V_c` times each incoming vacuum
      face's inflow; reflect wins mixed corners, mirrored over the reflective axes and
      any axis-aligned incoming vacuum face (so the box still matches its twin) -
      replaced 26 Sep 2026 by reflective faces as face couplings, see the DG0
      mixed-corner item under the ghost-flux postscript. The
      transpose groundwork: with V the cell volumes, `(V A)^T = P (V A) P` to rounding
      on any mesh (checked on triangles, tets and a perturbed-node mesh where the
      unweighted identity is off by 7%), so a half-quadrature PC built on `+Omega` serves
      `-Omega` on DG as `y = V^{-1} M^{-T} (V x)` - the only DG-specific piece that PC
      will need. Verified in `verify_plexk` check 7 and the `plex_*_inf_medium_ghost`
      recipes.
- Verify: DG0 on uniform quad/hex box meshes reproduces the FD upwind matrix to rounding
  (`tests/verify_plexk`, serial and parallel, the plex rows permuted onto the DMDA's by
  centroid); the infinite-medium closed form on triangles and tets through reflective
  faces; layout, geometry and error-path checks; pinned iterations on the plex twins of
  the structured recipes and on a Gmsh file with Cell Sets and Face Sets.
- Open questions left by the 27 Sep 2026 round (PRs #14, #15, #17; the findings are in
  their ticked items - the void-masking and consistent-D ones under the Phase 4
  postscript 5 DSA notes, the simplex one in the Phase 6a item above):
  - [x] A void-bridging DSA diffusion operator (27 Sep 2026): the void mask decouples
    void cells, so the correction did not couple regions a void separates (PR #14).
    Done as the default, `-dsa_void_bridge 0` keeping the mask: a void cell stays in
    the diffusion operator with no absorption and the free-flight `D = L / 3`, `L =
    4 V / S` the group's voids' mean chord (Cauchy / Behrens; `1 / (3 (Sigma_t + 1/L))`
    for a near-void under `-dsa_void_sigma_t`), and at DG1 the faces touching a void
    take the Ern-Stephansen-Zunino weighted interior penalty (harmonic face D). Beat
    the target (the tiny-`Sigma_t` workaround's 7 on the channel): 6 serial and np 2
    on the box and the plex channels, 5 on the duct, the slab gap 6 / 6 (np 2 7 -> 6),
    DG1 tied at 10 (7 with an exact inner solve, against the mask's 10 either way).
    Details and the scans behind the choice of D in the Phase 4 postscript 5 DSA
    notes and docs/dev/testing.md, "DSA with voids". Left: the chord is one number
    per group, not per connected void; DG1 is held at a tie by the one GAMG V-cycle.
  - [x] `-precon_ref_shift` refused a group that is void in only some cells - a check
    older than the DSA void mask, and separate from it (PR #14). DONE 27 Sep 2026:
    `RefShiftPmats` sorts the groups into SUPPORT classes (the cells where `Sigma_t >
    0`), each with its own reference - its first group - and bins each class on its own
    (`-precon_ref_k` is per class); the empty support is the streaming-only bin it
    already had. The reference is zero in the class's voids, so the shift is too and a
    void cell's pmat row is the bare streaming row, which is the operator's row there;
    the alphas are log-means over the support. A void in every group (a void material)
    leaves one class, and exact coverage is still the full pmat - the single-group void
    files reproduce the default counts exactly under ref-shift, with and without DSA
    (23 / 26 / 21 / 25 / 34, bridged DSA 6 / 6 / 5 / 6 / 10, where bare `-matfree_removal` takes
    277 or diverges), and so do the new `{slab,box,plex}_decades4_void.json`. Why
    classes rather than one reference with the voids masked out of the log-mean:
    `box_decades4_void`'s block is transparent in group 0 only, and forcing the four
    groups onto group 0's reference leaves the block unshifted for groups 1-3, taking
    the thick group 29 -> 801 iterations. Negative `Sigma_t` stays an error. New check
    `-check_ref_shift` (the driver): every group's pmat against its assembled
    streaming + removal operator, 9e-14 relative at exact coverage on the void files
    (the one-reference variant: 0.2, FAIL). All 35 pre-existing ref-shift recipes'
    residual histories are identical to main's. Counts and pins: docs/dev/testing.md,
    "Voids" under the reference-shifted pmat.
  - [x] GAMG on the DSA diffusion matrix is the remaining limit in the thick diffusion
    limit under the consistent D: with an exact inner solve eps 1e-4 takes 4, and two
    V-cycles (`-dsa_ksp_type richardson -dsa_ksp_max_it 2`) take 64^2 from 11 to 7, but
    double the DSA cost, so one V-cycle stays the default. A cheaper stronger inner
    solve was the item (PR #17). CLOSED 27 Sep 2026, no change: a sweep of the repo's
    serial `-precon_dsa` recipes found no cheap win. The GAMG knobs either cost as much
    as a second V-cycle (no aggressive coarsening doubles the operator complexity) or
    are mixed/neutral (threshold, W-cycle, pbjacobi). PFLARE GMRES-polynomial
    smoothers (arnoldi/newton, `pflareinv`) fail on heterogeneous problems on the raw
    matrix - the polynomial is built on unscaled A - and need a symmetric Jacobi
    scaling of the DSA matrix (plus the near null space rescaled to match); scaled,
    they beat Chebyshev by only ~5% at the same SpMV count, a margin that vanishes by
    degree 4, and a second V-cycle does as well. On these small problems (3-4 GAMG
    levels) the smoother is not where the gap to an exact solve is; if the DSA solve
    matters at scale, the clean route is a Jacobi-preconditioned polynomial inside
    PFLARE rather than scaling the matrix (GAMG is not exactly scaling-invariant).
  - [ ] A constant +1 iteration on simplices against quads/hexes at ratio 1 (not a
    creep): with an exact (LU) streaming inverse the counts match, so it is PCAIR's
    approximation. `-sub_1_pc_air_strong_threshold 0.25` or
    `-sub_1_pc_air_inverse_sparsity_order 2` remove it at small n but it returns at
    larger n, so the defaults stay (PR #15).

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
- [x] The inputs alongside the flux (Aug 2026): each file now also carries `sigma_t` and
      `source` for its group, so a file says what was solved without going back to the
      problem definition — the fastest check that a painted region landed where it was
      meant to. `UboltWriteScalarFluxVTK` grew an `(n_extra, UboltCellField *)` pair, a
      name plus a per-local-cell view, and queues each one onto the SAME dof-1 compatible
      DMDA; PETSc's VTK viewer holds every queued Vec and writes them as fields of the one
      file at destroy time (it duplicates what it is given, so ownership stays here).
      The writer stays free of materials: `sigma_t` is the slice `GroupXSections` already
      hands the removal term, and the source arrives as a per-cell view from
      `UboltFillCellSource`, `UboltFillSource`'s sibling — the isotropic strength as the
      spec holds it, NOT the `/ sum_weights` per-ordinate share and with no BC-row
      special case, because this is an output, not a rhs.
      Verified: `mg4_t05_dense` (painted mid-slab) writes 2.5/5.0 and 2.0/4.0 for g0/g3
      exactly as the materials file says; `box_50_absorber` writes sigma_t {2, 10} and
      source {0, 1} with the 10 x 10 block exactly on [0.4, 0.6]^2; both fields are
      bitwise identical serial vs np=2/np=3 while the flux agrees to solver tolerance, so
      the gather order is right; `make check` and `make tests` pass.

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
      serial references moved 8 -> 9 and 9 -> 10. The set of configs needing CI slack
      moved with the rhs: a full re-sweep of both CI images (64-bit, OMP) put the
      streaming-pmat serial pins back on their references (9, 10) and instead gave +1
      to the 50x50 ratio-1 serial solve and its two-call/painting-identity twins
      (64-bit), the np2 streaming pmat and the serial 100x100 S4 (OMP) — see
      docs/dev/testing.md.

## Phase 4 postscript 3 — problem-definition files, one driver (Aug 2026)
- [x] All physics now comes from a JSON problem file (`docs/problem_files.md`):
      dimension, mesh, angles, materials, painted regions, BCs, inflow, flux output.
      `ProblemSpec` reads it (rank 0 reads + broadcasts, nlohmann vendored at
      `src/external/nlohmann/json.hpp`, never included from public headers); materials
      are a multigroup schema interoperable with other codes' files — path or inline,
      fission fields accepted-ignored — plus UBOLT's `Source[g]` extension. The problem
      file is strict about unknown keys, the materials schema tolerant (files authored
      elsewhere carry fields UBOLT does not use).
- [x] `transportk` replaced `slab_1dk`, `slab_1d_mgk` and `box_2dk`: one driver, both
      dimensions, any group count; CLI keeps PETSc options + `-precon_stream`,
      `-diag_scale`, `-check_inf_medium`, `-flux_vtk`. The group sweep moved over
      verbatim and stays driver-side until a second sweep strategy justifies a
      MultigroupSolver (the Phase 2 note above stands). Recipes name problem files in
      `tests/problems/`; two old recipes died as strictly subsumed (runtime-sizes,
      1-group multigroup — see docs/dev/testing.md).
      Verified BEFORE deleting the old drivers: all 24 baselines byte-for-byte from
      `transportk` (the slab files carry `Source 2.0` + `inflow 1.0`, exactly
      `slab_1dk`'s VecSet rhs; the frozen-baseline-driver caveat is thereby retired);
      the t0 structural identity; painting identity bitwise at np=1 and np=4;
      1D/2D infinite-medium checks through the new per-group forward-substitution
      constant (~1e-13). `-check_inf_medium`'s guards moved with it: every face
      reflective, no paint, absorption in every group. The recipe pins carry the
      re-swept CI slack unchanged (same histories by construction, same maxes).

## Phase 4 postscript 4 — per-face inflow with an optional window (Aug 2026)
- [x] Inflow moved from ONE global per-angle value on the rhs to a per-face property of
      the boundary condition. A `boundary_conditions` entry is still the bare
      `"vacuum"`/`"reflect"` string, or now an object
      `{"type", "inflow", "window"}`; the top-level `"inflow"` key is gone and a file
      carrying it errors with a migration message rather than being reinterpreted.
      `BCSpec` holds a `BCFace` per label id; `BoundaryInfo` gained
      `dirichlet_value_d`, the per-row rhs value each backend computes ON THE HOST at
      create time (it already classifies every row with the geometry in hand), and
      `UboltFillInflow` lays it onto a zeroed b before `UboltFillSource`. Nothing about
      the matrix changed — a Dirichlet row is still the identity — so no `OperatorTerm`,
      `TransportOperator` or `DSAPrecon` code moved.
- [x] `inflow` is now an isotropic ANGLE-INTEGRATED strength, `inflow / sum_weights` per
      ordinate, the convention `Source` took in postscript 2; the "deliberately still
      per-angle" stance in docs/dev/testing.md fell with it (a per-face knob next to a
      per-face `Source` with the opposite scaling is not carryable, and the benchmark
      papers prescribe isotropic incident sources). The `window` restricts where on a
      face the inflow enters: one `[lo, hi]` pair per tangential axis, ascending
      global-axis order, membership by boundary-cell centre inclusive both ends — the
      same rule the paint lists follow. A row incoming through several vacuum faces
      takes the first in axis order x, y, z, which is the order `ClassifyRow` already
      checks.
- [x] Migration verified byte-for-byte: every previously-default file got the exact
      double of its quadrature measure (`2.0` in 1D, `12.566370614359172` in 2D/3D), so
      the division returns exactly 1.0 and the rhs is bitwise what it was; cold files
      just dropped the key, 0 being the new default. `-ksp_monitor` logs for all 49
      problem files before and after are identical except `box_crooked_pipe.json`.
- [x] `box_crooked_pipe.json` is the one deliberate physics change, and the point of the
      exercise: the paper drives it with an inward isotropic source at the pipe mouth,
      which UBOLT could not express, so the file faked it with a painted unit-`Source`
      strip (`pipe_src`, retired here with its paint box). It now carries
      `"left": {"type": "vacuum", "inflow": 1.0, "window": [2.0, 3.0]}` — the same four
      boundary cells the strip covered. Pins re-measured (123/72 serial, 94/69 at np=2)
      on the local opt arch; CI runs green with them (Aug 2026).

## Phase 4 postscript 5 — generalised quadratures at full precision (Aug 2026)
- [x] The quadratures were S2/S4 only, and their cosines were literals truncated to 7-10
      significant digits in a double code (`0.5773502692` for `1/sqrt(3)` is a relative
      error of 1.8e-10, `0.3500212` for the S4 level cosine 6e-8). Both are gone: the
      constants are now GENERATED, and the orders are no longer a pair.
- [x] **1D: any even order.** `SNQuadrature::create` builds the N point Gauss-Legendre
      rule by Newton on the Legendre recurrence, from the standard asymptotic seed —
      nothing tabulated, no upper bound. Only the N/2 positive roots are computed and the
      negative half is their exact NEGATION with a copy of the weight, which is what keeps
      the two `==` comparisons the reflection map rests on exact (`FindOrdinate`'s search
      and the weight-symmetry check) rather than nearly exact.
- [x] **2D/3D: the level-symmetric (LQn) sets, even 2 to 18**, from a generated table:
      `src/sn_lqn_table.py` (mpmath, 60 digits, checked in) emits `src/sn_lqn_table.hpp`,
      included ONLY from `src/sn_quadraturek.kokkos.cxx` — the vendored-json rule. The
      script's docstring has the derivation; the short version is that the defining axis
      moments collapse to a generalised Vandermonde system whose solution (the per-level
      weight sums) is unique for any mu_1, and the single leftover consistency condition,
      `sum_l (3 l - (n + 2)) y_l = 0`, is what fixes mu_1 — one scalar root-find at every
      order, landing on the published Lewis & Miller Table 4-1 value to its last tabulated
      digit at every order it reaches. The published values are seeds and a cross-check,
      nothing more.
- [x] **The weights are no longer equal within a set** above S4 — they are constant on the
      permutation classes of the level-index triple, 3 distinct values at S8 and 10 at S18.
      No consumer had to change: the backends read only the SIGN of a cosine, `w_d()` is
      always consumed through `UboltAngularIntegral`'s gemm, and DSA's `/sum_weights`
      prolongation is the isotropic projection rather than an equal-weight assumption. The
      public API of the three classes is unchanged.
- [x] **Where the table stops is computed, not chosen.** From S14 up the axis moments do
      not determine the weights uniquely (a family of dimension 1 at S14 and S16, 2 at
      S18, 3 at S20), and the script picks the member minimising the error in the MIXED
      even moments — one least-squares problem, unique, and the whole rule. It then walks
      the orders upwards and stops at the FIRST one whose member has a weight that is not
      strictly positive. That is S20, at -1.8e-5, so the table runs to S18 and the
      generated header prints that number as the reason. A negative weight is the
      level-symmetric family running out, which is the well-known reason SN codes cap LQn
      around here — a higher order wants a DIFFERENT family (product, Gauss-Chebyshev),
      which would be its own piece of work, not a different choice within this one.
- [x] `AngularQuadrature::set_weights` now CHECKS that the weights sum to the angular
      measure it is handed. That assumption was load-bearing and silent — the
      `-check_inf_medium` oracle rests on it — and a set that did not cover the domain it
      claimed would have surfaced as a wrong answer under an isotropic source.
- Verify: `tests/verify_quadraturek.kokkos.cxx`, in `make check` and `make tests_short`,
  serial and at `-n 2`. 1D Gauss exactness to degree 2N-1 at seven orders up to S32; the
  even axis moments at EVERY table order in 2D and 3D (~1e-15 against 1e-12, the real test
  of the table); counts, unit vectors, positivity, the reflection maps as
  weight-preserving involutions, 3D invariance under all six axis permutations, the 2D set
  being exactly the folded 3D one, the two closed-form cosines, and the error paths. Plus
  `box_s8_reflect_coarse.json`, the one recipe that puts unequal weights through a solve.
  Details in `docs/dev/testing.md`.
- All 24 baselines re-captured, on the same debug arch every previous capture used so the
  diff is like for like. Over the 20 non-pathological logs no iteration count moved, and
  the INITIAL residual — the fingerprint of the constants with no solver amplification in
  it — drifts by at most 2.2e-10, the size of the perturbation. Later iterations amplify
  it (under 1e-10 everywhere but the multigroup streaming-pmat pair, which reaches 2.2e-4
  by its eleventh); a converged run's final residual is at the rtol floor and not worth
  comparing relatively. The two pathological `diag_scale` logs moved where they break
  down, which nothing pins. Across the whole recipe suite exactly one count moved locally:
  2D 80x40 streaming-only pmat, 10 -> 9 on both local arches.
- [x] The streaming-only pmat recipes are the arch-fragile ones and now all carry a pin of
      10 against a reference count of 9. Tightening 80x40 onto its new local 9 was a
      mistake: CI then failed on its 50x50 SIBLING, which measures 9 locally both before
      and after the change but needs 10 on all three opt images. These three recipes clear
      rtol on their last iteration by a percent or two, so a perturbation far too small to
      be a regression decides whether that iteration counts — see the note in
      docs/dev/testing.md. Do not tighten them onto a local measurement.

### Research notes (the DSA notes, the Phase 5 notes, backend research)
(These notes sat under their own "Research notes" heading until Aug 2026, when a postscript inserted above it took the heading's place, so code comments cite them as "TODO.md Phase 4 postscript 5", "the Phase 4 DSA notes" or "TODO.md's DSA notes". The DSA notes are the **DSA, the diffusion half of the preconditioner** bullet; the Phase 5 notes are **Single assembled streaming matrix across all groups**; the `PCSetUseAmat` measurement is the last bullet.)

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
  120x60 -> 7; 200x50 in a 1x0.25 box -> 5; S4 at 100x100 -> 6, down from 87. The lesson
  is to suspect the operator before the preconditioner when a count depends on something
  the operator should not care about. (The "there is still no DSA, and nothing in the
  test set demonstrates the regime that wants one" half of this note is now closed —
  see the DSA note below.)
  `-sigma_scatter`, added to `box_2dk` to investigate this, survives the driver
  unification as a problem file's independent `Sigma_s`: varying the ratio
  independently of `Sigma_t` is worth having.
- **DSA, the diffusion half of the preconditioner** (Dargaville et al., JCP 518 (2024)
  113342, Section 3) — LANDED in 1D, 2D and 3D as `DSAPrecon` behind `-precon_dsa`, off by
  default. `G^-1 = P . D_diff^-1 . R`: masked 0th-moment
  restriction, a cell-centred `-div(D grad phi) + sigma_a phi` on the same grid (harmonic
  face D, Marshak on vacuum faces, zero Neumann on reflective ones), isotropic
  prolongation scaled by `1/sum_weights`. Inverted inexactly, KSPPREONLY + PCGAMG under
  the `dsa_` prefix (the paper's single BoomerAMG V-cycle is `-dsa_pc_type hypre` where
  hypre exists; the CI images have none). Sits at composite index 2, so the paper's
  additive combination is `-pc_composite_type additive` on the CLI.
  Findings so far, the baseline for the Phase-5-adjacent strategy work. The three
  diffusive files are deliberately the same physics — ten mean free paths per cell at a
  ratio of 0.99, S4, every face vacuum — so only the dimension changes between them:

  | serial / np=2 | no DSA | `-precon_dsa` |
  |---|---|---|
  | `slab_diffusive.json`, 1D, 100 cells | 20 / 20 | 11 / 10 |
  | `box_diffusive.json`, 2D, 50x50 | 29 / 29 | 11 / 11 |
  | `cube_diffusive.json`, 3D, 10^3 | 21 / 21 | 10 / 10 |

  - **The corrected count is 10 or 11 in every dimension and the uncorrected one is not**
    (20, 29, 21). What the correction removes is the part that varies with dimension,
    which is the claim DSA makes; that is the number anything later has to beat.
  - Easy problems are not hurt: st=2 goes 6 -> 5 in 1D, 7 -> 5 in 2D and 5 -> 4 in 3D, and
    the all-reflect infinite media still land on the exact constant (~1e-13) in all three.
  - The finding that cost the most to see: a multiplicative `PCComposite` builds the
    residual it hands later stages from **pmat**, which has no scattering in it, so the
    DSA shell was being handed a residual with its own target already removed — the
    moment reaching the diffusion solve was ~1e-6 of the residual and the count did not
    move at all. `PCSetUseAmat` when (and only when) a DSA is present is the fix. Any
    future preconditioner stage aimed at a matrix-free term inherits this problem.
  - Open: the streaming part of `R A P` is not the diffusion operator being built.
    First-order upwinding adds numerical diffusion of order `mean|mu| dx`, which at ten
    mean free paths per cell is several times the physical `D = 1/(3 sigma_t)`, so the
    correction is consistent with the PDE and not with the discretisation. A
    discretisation-consistent D (or a `D + c dx` fudge) is the obvious next experiment,
    and the table above is what to beat. DONE 27 Sep 2026 - see the ticked item below:
    the table is now 5 / 5 / 5.
  - CLOSED 2026-09-25 (a Dirichlet-cell artefact, see the Phase 5 follow-up on the
    mismatched ref-shift + DSA; ghost-flux takes 44 with and without DSA on
    `cube_diffusive`, and that is pinned). Original note: `-precon_stream` +
    `-precon_dsa` does not converge, and 3D says that is a real
    interaction rather than an inherited failure. In 1D and 2D neither the combination nor
    `-precon_stream` alone converges on the diffusive files, so nothing could be concluded
    there; on `cube_diffusive.json` `-precon_stream` alone converges in 47 and adding
    `-precon_dsa` diverges in 300. The obvious suspect is that a DSA switches the
    composite's residual updates onto the amat (the fix in the bullet above) while PCAIR
    is still set up on a streaming-only pmat — unestablished. This matters because Phase 5
    is exactly the streaming-only-pmat direction, so the two land on each other.
  - [x] Region masking for voids in the DSA diffusion operator (27 Sep 2026; `sigma_t <= 0`
    used to be a hard error, as in the paper's future work). A cell whose group `Sigma_t`
    is at or below `-dsa_void_sigma_t` (0 by default) is masked out of the correction on
    every backend and order: identity diffusion rows (V I on the plex, block by block at
    DG1), nothing restricted or corrected there, and a face into the void a Marshak (MIP
    vacuum) face for its neighbour - zero Neumann would leave a non-absorbing island in a
    void singular. Per group; the void rides the staged D as D = 0, so neighbours on
    other ranks see it; with no void the arithmetic is unchanged (every DSA recipe's
    residual histories identical to main, before and after merging the consistent D).
    Under the consistent D a face into a void is the blended vacuum face exactly, and
    the D = 0 flag is tested before any blend, so no `m h` coupling crosses it.
    Measured (opt, serial / np 2; no DSA -> DSA, physical D in brackets):
    `slab_void_gap` 23 -> 6 / 7 (12 / 13), `box_void_channel` 26 -> 8 / 8 (12),
    `cube_void_duct` 21 -> 6 / 6 (9), `plex_box_void_channel` 25 -> 8 / 8 (12), `_dg1`
    34 -> 10 / 10 - against 5 / 5 / 5 / 5 / 6 with no void; the unmasked "tiny Sigma_t"
    workaround takes 7 on the channel. `verify_plexk` check 12. Not done here: coupling
    the regions a void separates (done since - the next item), and `-precon_ref_shift`
    on a partially-void group (done since, at the end of Phase 6).
  - [x] Void BRIDGING in the DSA (27 Sep 2026, the default; `-dsa_void_bridge 0` is the
    mask above). A void cell stays in the diffusion operator, no absorption, with the
    FREE-FLIGHT coefficient: D = 1/(3 sigma_t) is <mu^2> times the flight 1/sigma_t,
    and in a void the flight is ended by the walls, whose mean is the chord L = 4 V / S
    (Cauchy; the Behrens void correction of reactor diffusion theory). V is the
    group's voids' volume, S the area of their faces onto material or a vacuum
    boundary (reflective faces mirror the flight on); D = L / 3, or 1/(3 (sigma_t +
    1/L)) for a near-void (Wigner's rational form); `-dsa_void_d` fixes it. Everything
    else is the plain operator's, so with no void the arithmetic is unchanged: all 63
    no-void `-precon_dsa` recipes pin-lifted with `-ksp_monitor` identical to main's.
    An all-void group, or one with no vacuum face and no absorption (pure Neumann once
    the voids are in), falls back to the mask. How D was chosen (opt, serial, fixed D
    swept 0.01-1e5): at DG0 the optimum scales with the void's width - a channel of
    height H wants D ~ 0.2 at H = 0.2, ~1 at H = 0.6 and 1.4 - and 1D wants D large
    (a planar gap is a perfect conductor in P1: partial currents cross it unchanged),
    but the count is flat within a factor ~3 of L / 3 in every case; D -> infinity
    (the tiny-Sigma_t workaround) loses one on the 2D channel. At DG1 the plain MIP
    form got WORSE with the void in (10 -> 13; its arithmetic penalty C/2 (D_c/h_c +
    D_n/h_n) welds the material's surface to the void, and even an exact inner solve
    only reached 9 at a hand-picked D); the weighted interior penalty (Ern, Stephansen &
    Zunino 2009: both averages and kappa off the face's harmonic D - the DG1 sibling
    of DG0's harmonic face D) on faces touching a void fixed it: 7 with LU, 10 (a tie)
    with the default GAMG V-cycle, 8-9 with two V-cycles or a GAMG threshold of 0.05
    where the mask stays at 10. Masking only the void's slope nodes, or the whole
    void's restriction / prolongation, changed nothing; the natural Robin form on a
    void's vacuum face moved the rate by two hundredths and no count. Measured (opt, serial /
    np 2, mask -> bridge): `slab_void_gap` 6 / 7 -> 6 / 6, `box_void_channel` 8 / 8 ->
    6 / 6, `cube_void_duct` 6 / 6 -> 5 / 5, `plex_box_void_channel` 8 / 8 -> 6 / 6,
    `_dg1` 10 / 10 -> 10 / 10. `verify_plexk` check 13; the mask keeps its recipes
    under `-dsa_void_bridge 0`.
  - [ ] Per-group cached DSA Mat/KSP instead of one refilled pair (a local change inside
    `DSAPrecon`, see its header; only worth it if a sweep revisits groups).
  - [x] DSA code restructure (28 Sep 2026, refactor W2, bit-identical: every
    `-precon_dsa` recipe's `-ksp_monitor` history and the verify_plexk / verify_cgk
    output unchanged, serial and np 2/3). `DSAPrecon` keeps the policy (restriction,
    prolongation, inner KSP, void census and bridge decision) in `src/dsak.kokkos.cxx`;
    each backend's operator is a private `DSAOperator` in its own TU
    (`dsa_structuredk`, `dsa_plexk` for DG0 + DG1, `dsa_cgk`); a generic
    `create(..., const Discretisation &, ...)` dispatches. Derivations moved out of the
    header into `docs/dsa.md`; the measurements the header quoted were already here
    (this item's siblings, the Phase 6 DG1 / CG DSA items) or in `docs/dev/testing.md`
    (the Marshak-coefficient scan: 0.25-1.0 moves no count by more than 1). Fewer
    transfers, none added: the weights are copied to the host once per create (they were
    copied once per plex face slot / CG vacuum face - millions of synchronous copies at
    create on a GPU); the void mask is uploaded only when it changes; the structured D
    staging and its ghost exchange run on host Vecs (were a VECKOKKOS round trip per
    group, two when bridging); CG's per-group D tensor is a host array, the device one
    only for an unbridged void's SUPG tensor; the cell-count reduction moved to create;
    and the volume / lumped-mass weight is folded into the restriction kernel (one
    launch fewer per apply). `destroy()` now resets the object.
  - **Literature benchmark study run (Aug 2026)** — six studies (crooked pipe per
    Southworth/Holec/Haut NSE 195, Warsa-style periodic horizontal interface, Brunner-style
    lattice, per-realization random media, Larsen diffusion-limit epsilon sweep, and the
    well-behaved classics: Adams-Larsen homogeneous ratio sweep + Castrianni-Adams
    iron-water), ~230 serial runs, with the report (`dsa_benchmarks_report.pdf`) and the
    reproduce harness (`reproduce_dsa_benchmarks.py`) kept outside this repo. Headlines:
    the multiplicative composite beats the
    classic sweep+DSA yardstick (rho = 0.2247c) by ~1.6x on the classics and is flat
    across eight decades of crooked-pipe heterogeneity with NO robustness cliff anywhere;
    where the no-DSA composite fails outright (diffusion limit below eps 1e-2, crooked
    pipe at cdt >= 1e3, 100-mfp interface layers) the DSA stage is what makes the problem
    solvable; random media are a strength (2.5x flat reduction, seed-to-seed spread
    compressed from ~20% to <= 2 iterations). The additive composite lost every one of
    100+ paired comparisons — settled as non-viable here.
  - [x] **A discretisation-consistent DSA diffusion coefficient** (27 Sep 2026, the
    "Open:" note above was the diagnosis and it was right). The targets as written were
    measured before ghost-flux; re-measured on main first, ghost-flux had already taken
    the eps sweep at 1e-4 from 67 to 17 and the crooked-pipe sets from ~200 to 51-106,
    so the gap left was the correction's D, not the boundary.
    - **What landed** (`DSAPrecon`, default on): every face D, Marshak faces included,
      becomes `(D^p + (m h)^p)^(1/p)`, p = 1.5, m = the quadrature's half-range current
      `sum_{Omega.n>0} w (Omega . n) / sum_w` (~1/4) and h the cell width across the
      face (on the plex, per face off the backend's ordinates and the normal, h the
      centroid-to-centroid distance; DG1 untouched, its penalty floor is the same 1/4).
      Derivation: for a linear flux in an infinite pure scatterer the upwind face
      current is EXACTLY `-(D + m h) grad phi`. `-dsa_consistent_d 0` is the physical
      D, `-dsa_consistent_d_power 1` the exact sum.
    - **Why p = 1.5, not the exact sum**: the sum is right in both limits but
      over-diffuses intermediate cells. A 1D Fourier analysis of step-differenced source
      iteration + DSA (Gauss S4) puts the optimal D below `D + m h` at tau = sigma_t h
      ~ 0.3-2 (and even below the physical D at tau < 0.3); the p = 1.5 blend is within
      ~0.03 of the optimal spectral radius at every tau in 0.1-100 and every ratio
      0.9-0.9999, p = 1 loses 0.02-0.05 at tau <= 1, p = 2 loses 0.1 at tau 1-2, ratio
      ~1. Measured: p = 1 cost +1-2 on the tau-0.5 classics (J box c = 1 9 -> 11, A box
      9 -> 11), p = 1.5 and 2 did not; on the hard cases the three agree within 1.
    - **Targets** (serial, rtol 1e-10, S4, one GAMG V-cycle; before = main under
      ghost-flux, "paper-era" = the Aug 2026 report):

      | problem | paper-era | before | after | exact-LU inner |
      |---|---|---|---|---|
      | eps sweep 64^2, eps 1 / 1e-1 / 1e-2 / 1e-3 / 1e-4 | 5/11/14/29/67 | 5/10/12/17/17 | 5/10/9/11/11 | eps 1e-4: 4 |
      | eps 1e-4, 128^2 | 59 | 17 | 11 | 4 |
      | crooked pipe 56x40, sets 1-5 at cdt 1e3 (set 5 sigma_a = 0) | 199/101/198/106/222 | 70/51/79/106/89 | 17/18/20/18/21 | set 5: 19, set 3 (sigma_a = 0): 17 |
      | crooked pipe set 1 cdt 1e3, 14x10 / 28x20 / 56x40 | 171/-/199 | 56/70/70 | 13/13/17 | 12/12/13 |
      | two-material pipe 56x40, sigma_pipe 1e-6 .. 100 | 77-98 | 25-48 | 11-19 | - |

      So the eps sweep is flat (9-11 from 1e-2 down, h-independent at 1e-4) as Haut et
      al.'s O(eps) theory says, and the crooked pipe is at 17-21 against the paper's
      ~30 (theirs is S8 DG, so like-for-like it is "at or below", not a claim of
      beating it). With an exact inner solve the eps 1e-4 case takes 4 and the pipe
      12-19, so the correction's FIDELITY gap is closed; what remains at eps 1e-4 is
      GAMG on the diffusion matrix (two V-cycles, `-dsa_ksp_type richardson
      -dsa_ksp_max_it 2`, take it 11 -> 7; not made a default - it doubles the DSA
      cost, and the pipe gains 2). That reverses the E4 conclusion recorded here before
      (a null exact-LU result was the physical D being wrong, not GAMG being fine).
    - **The rest of the external suite** (~110 problems, A-J + P5, serial, rtol 1e-10):
      nothing lost more than 1 (3 cases +1: lattice 56, J box c = 0.9, P5 box st1 c0.9
      s2); iron-water 16/13 -> 9/9 (60/120), layers with thick 1000-sigma layers 23-26 ->
      11-16, random media -1 to -3, P5 thick c = 0.99 boxes/slabs 20-21 -> 9, thin
      and intermediate problems unchanged. Repo pins: every diffusive file 11/11/8 -> 5
      in 1D/2D/3D and on every plex DG0 mesh (tri 10 -> 5, tet 9 -> 5), crooked pipe 28
      -> 8, additive composite 14 -> 7, `cube_diffusive -precon_stream -precon_dsa` 44 ->
      25; table in docs/dev/testing.md, "Discretisation-consistent D".
    - The heterogeneous-DSA fallback was NOT needed and was not tried.
  - Four benchmark problems promoted to pinned recipes (`box_crooked_pipe`, `box_layers`,
    `box_lattice`, `box_random8`, serial + np2 in `tests_short`, measured on the local
    opt arch, +1 slack) — CI runs green with these pins (Aug 2026).
    These files plus the external sweep set are the fixed benchmark suite for the Phase-5
    streaming/removal-separation comparison, the Phase-6 unstructured backends, and
    future parallel/GPU (LUMI) runs.
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
  ANSWERED for the strong-removal case by the Aug 2026 campaign's k reference-shifted
  hierarchies — landed as `-precon_ref_shift`, see the Phase 5 checklist above.
  - **CAMPAIGN RUN (Aug 2026)** — four-track investigation, ~500 runs; the full report
    (`phase5_streaming_pmat_report.pdf`) is kept outside this repo, data in
    `results_phase5*/`, code on `phase5-step0` + `phase5-track{1..4}` (worktrees
    `../UBOLT-track{1..4}`). All of it predates PR #13/#14/#15, so every number below is
    a finding and a direction, NEVER a pin — anything harvested is rebased and
    re-measured (PR #13 alone moved 2D counts by ±1). All under a standardized frame
    (right-preconditioned GMRES, rtol 1e-10, cap 300); the 1.5x note above is only true
    below ~0.1 mfp/cell — the degradation tracks mfp/cell (~3-4x at 0.1, ~20x at 1,
    DIVERGES at 10), and is otherwise flat in c, h and angle count. Findings:
    - `-precon_stream` iteration counts are mathematically identical to the
      matfree-removal build (same operator, same pmat), so all of this predated the
      implementation and bound it — which is why `-matfree_removal` landed with counts
      equal to its `-precon_stream` twins exactly.
    - **DSA cannot attach to a removal-free pmat — structural, not a bug**: the exact
      two-stage error operator (I - L^-1(L+D))(I - D^-1(L+D)) = I, so {Jacobi, AIR(L)}
      is neutral and DSA's gain turns AIR's over-correction into growth. Exact-LU
      cross-checked. DSA re-attaches iff removal enters AIR's operator.
    - **The winning design** (LANDED as `-precon_ref_shift`): pmat = L + alpha*D_ref
      with a PER-CELL reference removal (scalar shifts die on heterogeneity), k = 2-3
      hierarchies log-spaced over the group range (mismatch curve saturates; k=2 covers
      4 decades at <= 3.7x, exact ref recovers full-pmat counts EXACTLY incl.
      box_layers 61, box_lattice 17, box_random8 67), + the DSA stage per group
      (n_angles-fold smaller, in budget). The join is h-ROBUST where nothing else is:
      diffusive corner 23/21/16 over 50^2 -> 200^2 (improving!) vs FAIL for plain
      stream. Zero per-group AIR setups.
    - **Free default, one line of solver code** (NOT landed — branch `phase5-track1`,
      needs a rebase and a re-measure): bracket the AIR stage with a second removal
      Jacobi (`-precon_removal_repeat 1`, = symmetric_multiplicative). Never worse than
      plain stream anywhere, removes every divergence, beats the FULL pmat on 4/7 hard
      benchmarks (box_diffusive 49 vs 79, cube 23 vs 41, layers 46 vs 61, random8 46 vs
      67) and solves crooked pipe (103) where full pmat caps out. Its one hole: not
      h-robust in the diffusive corner (49/134/FAIL) — that is DSA's job, so it composes
      with rather than replaces the reference shift.
    - `PCSetUseAmat` should be ON whenever the pmat is not the full operator (improves
      plain stream 85 -> 58 on the cube; required for DSA) — still not the default, see
      the follow-up in the Phase 5 checklist. The removal shell diagonal must be
      composed as diag(L) + sigma_t with BC rows exactly 1.0, which is what
      `OperatorTerm::add_diagonal` landed as.
    - **Frozen AIR gridding seeded on L + per-group re-Galerkin costs 0..+2 its over
      SEVEN decades of sigma_t** (PFLARE reuse_sparsity keeps C/F, recomputes all
      values; seed direction matters — always seed from the streaming end; reuse
      quality proxied with zero-transfer mg files). Production path (no fine assembly)
      needs PFLARE work: a PCAIRSetDiagonalUpdate entry point + split level-1 Galerkin
      (R L P stored + R D_g P per group) + trimming the reuse store (26.9x nnz today).
      This is strategy B — if it lands, k drops toward 1 and it composes with
      `-precon_ref_shift`; if it fails, k = 2-3 stands.
      Also found: `-ksp_reuse_preconditioner` is silently a no-op for PCAIR (flag never
      reaches the internal shell) — reported.
    - **Matrix-free GMRES polynomial on the full OPERATOR** (PCPFLAREINV,
      `-pc_pflareinv_matrix_free` mandatory, Newton/Neumann bases): zero memory, setup
      = k+1 matvecs, beats PCAIR in the optically thick regime (cube 34 vs 41), order
      grows O(1/h) in the streaming-dominated regime. The GPU-cleanest fallback. On the
      streaming-only pmat polynomials are a dead end (converge to the wrong inverse).
    - **Per-angle AIR via strided fieldsplit** is an exact tie with monolithic at +1%
      memory (needs a `-pmat_block_size` knob, which lives on `phase5-step0` and is not
      on main, and coarse_eq_limit divided by n_angles); reflect coupling costs +2. A
      structural option, not an iteration win.
    - Dead ends, measured: Neumann/Woodbury correction around the shifted apply
      (diverges below s > T/2, net loss elsewhere); scalar shift on heterogeneous
      media; polynomials on the streaming pmat; DSA on any removal-free pmat.
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
- Measurement behind `PCSetUseAmat` under DSA (moved out of `transport_solverk.kokkos.cxx`
  in the Sep 2026 refactor): on `slab_diffusive.json`, with the composite forming its
  intermediate residuals from pmat, the moment reaching the diffusion solve was ~1e-6 of
  the residual and the count did not move at all; updating with AMAT (the shell, scatter
  included) took it from 19 iterations to 11.

## Phase 5 postscript — the ghost-flux vacuum treatment (own commit, Sep 2026)
Split out of the transposed-solves campaign (branch `claude/transpose-boltzmann-report-a40050`,
report outside the repo in `transpose_solves/`), which found that the opposite-ordinate
identity `A^T = P A P` (P swaps `(cell, Omega)` with `(cell, -Omega)`) holds exactly on
every interior row in 1D/2D/3D but fails on the Dirichlet-cell boundary rows, and that a
PCAIR built on half the ordinates applied transposed on the other half then costs 6-7x
the iterations in 2D/3D. The ghost-flux treatment removes the identity rows; the identity
is then exact on every row and the half-quadrature PC matched the full hierarchy within 1
iteration at ~50% setup/memory. The transposed applies, the probe driver and the solve
experiments stay on the campaign branch until PFLARE's PCAIR `PCApplyTranspose` merges.
- [x] Opt-in `"vacuum_treatment": "ghost_flux"`: `BCSpec::VacuumTreatment`,
      `BoundaryInfo::ghost_inflow_d`, per-axis GHOST rows in the three structured
      backends, reflect wins mixed corners (replaced 26 Sep 2026 by reflective faces as
      face couplings - see the DG0 mixed-corner item below), DSA refuses it. Default
      byte-identical.
- [x] Verified: `verify_2dk`/`verify_3dk` run closed form + reference matrix in both
      treatments, the library-built rhs against a constant solution, and `A^T = P A P`
      (0.0 relative, heterogeneous sigma_t, parallel too) under ghost-flux;
      `*_inf_medium_ghost.json` solves against the exact constant in 1D/2D/3D.
- [x] Swept the ghost-flux pins over the 64-bit and OpenMP CI images (two went up by 1).
- [x] Make ghost-flux the default — DONE 2026-09-25. It is the standard upwind FV inflow,
      differs from Dirichlet-cell by O(h) at the boundary cell, and is what DG does
      anyway. `"vacuum_treatment": "dirichlet_cell"` keeps the old numbers: its twins
      reproduce the previous 1D baselines byte for byte and the previous counts exactly.
      All 24 baselines re-captured on the debug arch; every recipe re-pinned to the max
      over local opt and the four CI images. `DSAPrecon`'s Marshak face needed NO
      rewrite: it already sits on the domain boundary, which is where ghost-flux puts
      the transport boundary, so the refusal was simply dropped (scaling the face
      coefficient over 0.25-1.0 moves no DSA count by more than 1). DSA got much
      better on the heterogeneous problems (crooked pipe 72 -> 28, random8 11 -> 8,
      cube_diffusive 10 -> 8) and worse on one (cube_diffusive_yreflect 10 -> 12);
      the ratio-1 workhorses cost one more iteration (box 6 -> 7, cube 5 -> 6), the
      streaming-only pmats one or two fewer, the default-k ref-shift runs two or three
      more. The structured/plex 50x50 twins now agree (7 and 7, where Dirichlet-cell
      gave 6 and 7). Full table: docs/dev/testing.md, "Switching the default".
- [x] Linear DG (DG1) upwind on the DMPlex backend (Sep 2026) — after ghost-flux became
      the default, because a DG face flux IS the ghost-flux inflow and there is no
      Dirichlet-cell analogue for a multi-dof cell. DECISION: layout A, rows
      `(cell * n_basis + basis) * n_angles + angle` — angle fastest, so a (cell, basis)
      "node" is a contiguous run of angles, over a per-(cell, angle) dof block (layout B,
      which would have made streaming's dense block contiguous but rewritten every
      dimension-independent term's stride). `PhaseSpace` gained `n_basis` (written by the
      backend, like `local_cells`), `local_nodes()` and `rows_per_cell()`; removal,
      scatter, `GroupTransfer`, `UboltAngularIntegral`, the ref-shift and the output work
      per node with the xsection per cell, unchanged at `n_basis` 1. What makes that
      valid is the basis: MODAL and ORTHONORMAL on each cell (phi_0 = 1, the linear ones
      orthonormalised through the Cholesky factor of the cell's second moments), so the
      mass matrix is the identity and removal/scatter stay diagonal per node, and a
      cell-constant source lands on basis 0 only. `UnstructuredDG0` became
      `UnstructuredDG` with an `order` (0 or 1; `mesh.order` in a problem file) and
      `StreamingTermDG1` beside `StreamingTermDG0`. Geometry: exact cell and face moments
      from a fan of simplices (planar faces required), per (cell, face) the two
      n_basis x n_basis face matrices, per cell the basis gradients. Boundaries: ghost-flux
      only, and NO BC rows at all — a reflective face is a face coupling to the mirrored
      angle's trace in the same cell, over that face's own axis, which removes DG0's
      composed-partner corner rule and its single-cell-wide rejection. Output: the cell
      average as `scalar_flux`, the slope as `scalar_flux_grad_*` fields. Verified
      (`verify_plexk` 8 and 9): zero BC rows; `(VA)^T = P(VA)P` to ~1e-15 with the
      reflective couplings in it; the constant through the library rhs; a linear field
      through the streaming matrix exact in interior cells on quads, hexes, triangles,
      tets and the irregular file; observed order 2.02 (quads) and 2.01 (triangles)
      against the exact SN solution DG0 is first order on. Pinned: every plex recipe's
      DG1 twin, serial and np 2 (docs/dev/testing.md, "Unstructured iteration counts" —
      DG1 costs 0 to 3 iterations more than DG0 at the default rtol).
- [x] DG1 + PCAIR: element-block-inverse scaling (Sep 2026). `ElementBlockInverse` reads
      the n_basis x n_basis (cell, angle) blocks straight off pmat's device CSR (layout A
      strides them by n_angles, so PETSc's contiguous-block MatInvertBlockDiagonal does
      not fit), inverts them per thread (Gauss-Jordan, partial pivoting, n_basis <= 4)
      and forms `D^{-1} pmat` on pmat's OWN pattern - every row of a block carries the
      same columns, checked on the device every call. `TransportSolver` wraps composite
      index 1 in a shell: PCAIR on `D^{-1} pmat`, applied to `D^{-1} r`. A
      preconditioner-only change - the operator, rhs, the residuals the KSP monitors
      and the other stages are untouched - and read off whatever pmat the solver has
      (assembled, streaming-only, reference-shifted, -diag_scale'd). DECISIONS: (i) the
      operator keeps its per-unit-volume rows. The suggestion was that the scaling could
      replace DG0's 1 / V_c; for what PCAIR sees it does (any per-row, or per-block,
      scaling cancels in D^{-1} A, so both orders now hand PCAIR a matrix with identity
      blocks and no volume factor), but taking 1 / V_c out of the OPERATOR would put V_c
      into removal, scatter and the source (the identity mass matrix is what keeps them
      per node) for nothing; (ii) it applies at both DG orders by default
      (`-precon_block_scale`, off on the structured backends, whose baselines predate
      it), so DG0 and DG1 go through the same stage - at DG0 it is the diagonal; (iii)
      the removal Jacobi stage (index 0) stays point-diagonal - since replaced by the
      element blocks too, see the next item. Measured on the TODO's
      DG1 quads (plex_box_50_st2_dg1 physics, S2, rtol 1e-10) at PCAIR's DEFAULT strong
      threshold: 14 / 18 / 21 AIR levels at n = 25 / 50 / 100 and 13 / 13 / 13
      iterations, against 28 / 58 / 119 and 14 / 16 / 17 unscaled, and 25 / 30 / 35 and
      13 / 14 / 14 under the old 0.25 stopgap, which every DG1 recipe dropped. DG0 on the
      same meshes: 12 / 15 / 18 levels, 12 / 11 / 12 iterations (12 / 14 / 19 and 12
      unscaled - row-relative strength of connection does not see a diagonal scaling).
      Verified (`verify_plexk` 10): identity blocks in `D^{-1} A`, invariance under a left
      row scaling, and `apply()` against the scaled matrix, to ~1e-15 on every DG0 and
      DG1 operator the ghost-flux and DG1 checks build, serial and -n 2/4
- [x] DG1: a block-Jacobi removal stage (follow-up to the element-block scaling, done
      26 Sep 2026). Composite index 0 used to invert the operator's POINT diagonal, which
      at DG1 drops the in-cell off-diagonals (the volume term -(Omega . grad phi_i) and
      the outflow face terms, as large as the diagonal). It now applies D_op^{-1} through
      the same `ElementBlockInverse` the block-scaled streaming stage uses, read off the
      OPERATOR (never pmat), so there is one local-inverse code path for both stages and
      every backend: at n_basis 1 a block is the diagonal entry and the stage is the old
      point Jacobi to the bit (FD and DG0 residual histories identical on all 247
      structured and plex DG0 recipe lines, serial and np 2, -diag_scale, -matfree_removal
      and DSA included). Under `-matfree_removal` the blocks are the assembled
      (streaming-only) matrix's with the operator's COMPOSED diagonal in place of their
      own - `ElementBlockInverse::setup(A, diag)` - which is exact because the removal is
      diagonal (identity mass matrix); `verify_plexk` 10 pins those blocks against the
      assembled operator's, BITWISE, on every DG0 and DG1 operator it builds.
      DECISION: keep the stage. The suggestion was that DG does not need it at all once
      PCAIR is block-scaled - in the default mode pmat IS the operator, so both stages
      invert the same D. Measured (opt arch, point / block / none, every plex recipe at
      np 1 and 2 plus 68 generated plex problems: a thickness sweep tau = sigma_t dx =
      0.1..100 at c = 0 / 0.5 / 0.9 on quads and triangles, the crooked pipe, box_diffusive,
      lattice, layers, random8, cube_diffusive and decades4, each in the default mode and
      under -matfree_removal -precon_ref_shift, plain -matfree_removal where tau <= 1):
      dropping the stage is never better than block by more than one iteration (crooked
      pipe DG1 362 against 363) and is worse by 1-2 in the removal-dominated and diffusive
      runs - a pure absorber at tau 10-100 takes 1 iteration with the stage and 2 without,
      box_diffusive on triangles DG1 40 -> 42, cube_diffusive 18 -> 19 (DG0) and 32 -> 34
      (DG1), the last decades4 group 29 -> 30 / 36 -> 37 under ref-shift. The stage costs
      one block apply per PC application against a full AIR V-cycle, so it stays. The
      block stage against the old point one at DG1: never worse in the default mode,
      better where it moves at all - crooked pipe 371 -> 363, box_layers 34 -> 33,
      cube_diffusive 33 -> 32, tau 1 absorber 3 -> 2; one +1 under plain -matfree_removal
      (tau 1, c 0.5, quads: 169 -> 170). No recipe pin moved
- [x] `ElementBlockInverse::scale` bumps the scaled MPI matrix's state (DONE 26 Sep
      2026). A PC rebuilds only if pmat's state changed since its last setup, and
      writing through `MatSeqAIJRestoreKokkosViewWrite` bumps the seq PARTS of an
      MPIAIJ matrix but not the wrapper PCAIR compares - so without a bump a parallel
      default-mode multigroup run silently reuses group 0's hierarchy for every group
      (still correct, just more iterations: exactly the failure nothing would flag).
      It was a no-op `MAT_FINAL_ASSEMBLY` (host-side row checks, and "rebuilds nothing"
      resting on PETSc internals); it is now `PetscObjectStateIncrease` straight off
      `petsc/private/petscimpl.h`, as PFLARE does - included only in the src TU, never
      from `include/ubolt/`. Pinned in `verify_plexk`'s element-block check: a
      `MAT_REUSE_MATRIX` scale must change `MatGetState`'s state and leave its
      nonzerostate alone. That is the `MatState` struct form (PETSc main, Sep 2026 -
      3.25 releases return a bare `PetscObjectState`), so it needs a PETSc newer than
      that. Verified the check bites (debug arch): with the bump removed it passes
      serially and fails every case at -n 2.
- [x] DG0 ghost-flux mixed corners (found regenerating the Phase 6a report, 25 Sep 2026;
      FIXED 26 Sep 2026, see the end of this item):
      where a reflective face meets a vacuum face, the ghost-flux "reflect wins" rule
      mirrors a direction coming in through both over both axes, so the corner cell never
      sees the vacuum face's inflow. On the quarter box (reflect left + bottom) against
      the full box's quadrant the error sits at the two mixed corners and does NOT shrink
      with h - max 2.07 / 2.15 / 2.19 at n = 25 / 50 / 100 on a flux of ~10 - and the
      deficit streams across the domain along the ordinates leaving the corner, while
      the mean halves (L2 still first order).
      Structured and plex agree to 1e-10, so it is the shared rule, not the plex backend.
      It also breaks the DG0 convergence study (pure absorber, left inflow meeting
      reflective top/bottom at (0,0) and (0,1)): the corner cell holds 3.11 where the
      exact flux is 6.26 - it loses the half of the inflow entering through both faces -
      so L-infinity is stuck at 3.14 for every n (0.028 at n = 256 under Dirichlet-cell)
      and the L2 order falls from 1.00 to 0.91, on FD, quads and triangles alike.
      Under Dirichlet-cell the corner took the vacuum inflow (vacuum wins) and the error
      was O(h) everywhere. DG1 has no such rule (per-face couplings) and its quarter box
      equals the full box to solver tolerance.
      FIX (26 Sep 2026): three candidates compared in a Python model of the 2D FD
      operator (S2/S4, n = 8..128, pure-absorber convergence study + quarter box against
      the full box + `A^T = P A P`): (A) mirror a mixed-corner row over its reflective
      axes only, (B) give a mixed-corner row its transport equation with the reflective
      face coupled to the mirrored angle, (C) do that on EVERY reflective row - DG1's
      rule. All three restore first-order L-infinity (0.97 at n = 128, was stuck at
      O(1)); only (C) makes the quarter box equal the full box to rounding (1e-14; A, B
      and Dirichlet-cell keep the O(h) reflective-row error of the 6a note) and keeps
      `A^T = P A P` on reflective rows. (C) landed in all four backends (1D, 2D, 3D FD
      and plex DG0), under ghost-flux only: a reflective inflow slot points at the
      mirrored angle in the same cell and the streaming term fills it; no BC rows are
      left, and the single-cell-wide reflect case (rejected under Dirichlet-cell) is
      simply accepted. Dirichlet-cell and every baseline are untouched (no baseline has
      a reflective face). Verified: `verify_2dk`/`verify_3dk` reference matrices,
      `A^T = P A P` exact on the mixed (and 2D all-reflect) configs, `verify_plexk`
      `(VA)^T = P(VA)P` with reflective low faces on triangles/tets/the irregular file,
      FD twins still to rounding. Ten pins moved against main, nine down (DSA
      infinite-medium runs by 1, `cube_diffusive_yreflect -precon_dsa` 12 -> 9,
      `plex_square_msh` 5 -> 4) and one up (`box_50_inf_medium_ghost -matfree_removal`
      25 -> 26); table in docs/dev/testing.md, "Ghost-flux reflective faces".
- [ ] The half-quadrature preconditioner and an `-adjoint` path — blocked on PFLARE's
      PCAIR `PCApplyTranspose`; see the campaign branch. Since 26 Sep 2026 the identity
      it rests on, `A^T = P A P` on streaming + removal under ghost-flux, holds for ANY
      mix of vacuum and reflective faces in every backend (checked in verify_2dk/3dk/
      plexk, and by hand in 1D) - reflective problems no longer need special handling. On the DG backend (either
      order) the transposed half needs the cell-volume similarity (`unstructured_dg.hpp`):
      `y = V^{-1} M^{-T} (V x)`, which reduces to the plain transpose on a uniform mesh.

# Testing history (from docs/dev/testing.md)

Every table below is dated by its heading and its own text. The pins in force are `tests/Makefile`'s, tabulated in `docs/dev/iteration_counts.md`; what the verify_* drivers check is in `docs/dev/verification.md`.

## Single-group baseline iteration counts (captures 2026-07-31 to 2026-09-25)
Captured 2026-09-25 under the ghost-flux default, capped at `-ksp_max_it 200`. The older
columns are kept because earlier phases were verified against them — "Dirichlet-cell" is
the 2026-08-04 capture, the last one before the switch (and what the opt-in
`"vacuum_treatment": "dirichlet_cell"` still reproduces byte for byte), "Dirichlet fix" is the 2026-08-01 capture,
before the quadrature constants were fixed; "sign fix" is 2026-07-31, before the scatter's
Dirichlet rows were fixed; and "pre-refactor" is what the original single file did:

| config | np=1 | np=2 | Dirichlet-cell (np=1, np=2) | Dirichlet fix (np=1, np=2) | sign fix (np=1, np=2) | pre-refactor |
|---|---|---|---|---|---|---|
| default pc, st=0 | 1 | 1 | 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| default pc, st=2 | 6 | 6 | 6, 6 | 6, 6 | 5, 5 | 10, 10 |
| default pc, st=0, diag_scale | 1 | 1 | 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| default pc, st=2, diag_scale | 174 | 173 | DIVERGED_ITS (200) | DIVERGED_ITS (200) | 175, 173 | DIVERGED_ITS |
| precon_stream, st=0 | 1 | 1 | 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| precon_stream, st=2 | 8 | 8 | 9, 9 | 9, 9 | 10, 10 | 16, 16 |
| precon_stream, st=0, diag_scale | 2 | 2 | 3, 3 | 3, 3 | 3, 3 | 3, 3 |
| precon_stream, st=2, diag_scale | DIVERGED_ITS (200) | DIVERGED_ITS (200) | DIVERGED_BREAKDOWN (60, 180) | DIVERGED_BREAKDOWN (150, 150) | DIVERGED_BREAKDOWN (90) | DIVERGED_ITS |

The st=0 rows were identical across every capture up to the ghost-flux switch, and had to
be: `sigma_s = 0` there, so neither the scatter nor its Dirichlet rows exist. The switch
moves the boundary rows themselves, so it is the first capture that moves them (only the
diag-scaled streaming pmat's count, 3 -> 2).

History of these baselines:
- **Ghost-flux became the default vacuum treatment** (2026-09-25, this re-capture): every
  vacuum inflow row is a physical row now, so every log moves from the first residual.
  Iteration counts: the default pc is unchanged; the streaming-only pmat improves by one
  (9 -> 8, and 3 -> 2 diag-scaled); the default-pc diag-scaled pair, pathological since
  the Dirichlet-row fix, converges again at 174/173, while the diag-scaled streaming pair
  goes from breakdown to the 200 cap - neither is a stable quantity and neither is pinned.
  The right-preconditioned (streaming-pmat) logs' INITIAL residual grows from 63 to 1313:
  with right preconditioning it is `||b||`, and the ghost rows' rhs is `|mu|/dx` times the
  inflow where a Dirichlet row's was the inflow itself, which at dx = 1e-3 makes the
  boundary rows dominate `||b||` and hence the rtol reference. The opt-in
  `"vacuum_treatment": "dirichlet_cell"` twin (`slab_st2_dirichlet_cell.json`) reproduces
  the previous capture's `default_st2` (np 1 and 2), `default_st2_ds` and `stream_st2`
  logs byte for byte on the same arch.
- **The quadrature constants were truncated to float precision** (FIXED 2026-08-04, this
  re-capture): the direction cosines were literals of 7 to 10 significant digits in a
  double code (`0.5773502692` for `1/sqrt(3)`, a relative error of 1.8e-10), and the
  quadratures now generate them — 1D by Newton on the Legendre polynomial at `create`
  time, 2D and 3D from a table generated at 60 decimal digits by `src/sn_lqn_table.py`.
  That perturbs every result in the trailing digits and nothing else, which is what the
  re-capture was checked against, on the same arch so the diff is like for like. Over the
  20 non-pathological logs: **no iteration count moved**, and the **initial** residual —
  the direct fingerprint of the constants, with no solver amplification in it — drifts by
  at most 2.2e-10, the size of the perturbation itself. Later iterations amplify that, as
  they must: through the non-converged iterations the drift stays under 1e-10 everywhere
  except the multigroup streaming-pmat pair, whose 11-iteration histories reach 2.2e-4 by
  the end. A run's FINAL residual is not a useful comparison at all — it sits at the rtol
  floor, so the change in it is 1e-13 to 1e-17 of that run's initial residual (3e-7 for
  the multigroup streaming pair) while its *relative* change can be anything.
  What DID move is the two `precon_stream, st=2, diag_scale` logs, from
  DIVERGED_BREAKDOWN at 150 to 60 (np=1) and 180 (np=2). Those are the pathological
  configs below: they break down rather than converge, so where they break down is not a
  stable quantity, and no recipe pins them. Across the whole recipe suite exactly one
  count moved — 2D 80x40 with a streaming-only pmat, 10 to 9 — see the 2D table.
- **The matrix-free scatter wrote to the Dirichlet rows** (FIXED after Phase 4, this
  re-capture): `ScatteringTerm::apply_add` subtracted the scattering source from every row
  including the boundary ones, so the shell's Dirichlet rows were not the identity the
  assembly had written and the prescribed inflow value was polluted. Present since Phase 1;
  the contract in `operator_term.hpp` only ever covered `assemble_add`. Found by
  `tests/verify_2dk`'s reference matrix. In 1D the effect is a wash — the well-behaved
  configs move by one iteration either way and the pathological diag_scale pair gets worse
  — but in 2D it was the whole of an apparent preconditioner problem (see `TODO.md`).
- **Negative-angle upwind sign** (FIXED after Phase 1b, this re-capture): the streaming
  term wrote the upwind neighbour coefficient as `-mu/dx`, correct for `mu > 0` but the
  wrong sign for `mu < 0`, so those rows were `|mu|/dx (psi_i + psi_i+1)` instead of
  `|mu|/dx (psi_i - psi_i+1)`. Fixing it makes the interior streaming rows sum to zero,
  and every well-behaved config converges in fewer iterations (st=2 went 10 -> 5, the
  streaming-pmat st=2 16 -> 10). Phase 1 deliberately preserved the bug so the refactor
  could be verified bit-for-bit, then fixed it in one commit of its own.
- **Parallel out-of-bounds bug in the matrix-free scatter** (FIXED, Phase 1a-pre): the
  original code reshaped the LOCAL vector with the GLOBAL `N_CELLS` and looped kernels
  over `N_CELLS`, so at np>1 it read past the local `x`/`sigma_s` arrays and wrote past
  the end of local `y`, sending `default pc, st=2, np=2` to NaN.
- **diag_scale + strong removal is still pathological** (175 its, or breakdown): diagonal
  scaling makes the removal shell PC an identity and degrades the composite. The sign fix
  improved it a lot but did not cure it. Known current behavior, not a target — excluded
  from pass/fail recipes; revisit when the diag_scale semantics are looked at again.

## 2D iteration counts (measured 2026-08-02, re-measured 2026-09-25 under the ghost-flux default)
The `st` in the table and the problem-file names is `Sigma_t` — the retired drivers took
it as `-sigma_t`, and until Aug 2026 that option was `-max_exponent`, a fossil of the
original driver's random per-cell xsection (`mantissa * 10^exponent` with the exponent
drawn up to it); that history is why anything older (commit messages, the history notes
above at their capture dates) says `me`. A file whose within-group `Sigma_s` equals its
`Sigma_t` is a scattering ratio of exactly 1 (no absorption). Counts below are with the
Dirichlet-row fix; before it, ratio 1 took 18 to 87 iterations on square grids and did not
converge at all on any other shape, which is what led to the fix — see the research note
in `TODO.md`. They were re-measured 2026-08-02 under the isotropic external source (see
the painted-regions section): only the two streaming-only pmat serial references moved,
8 -> 9 and 9 -> 10. Re-measured again 2026-09-25 under the ghost-flux default; where a
count moved, the Dirichlet-cell count follows it in brackets.

| config | np=1 | np=2 |
|---|---|---|
| 50x50, st=2 (ratio 1) | 7 (Dirichlet-cell 6) | 7 (Dirichlet-cell 6) |
| 50x50, st=2, ratio 0.5 | 5 | 5 |
| 40x20 (dx != dy), st=2 | 6 | 6 |
| 80x40 (same shape, 4x finer), st=2 | 7 (Dirichlet-cell 6) | 7 (Dirichlet-cell 6) |
| 80x20 in a 1x0.25 box (dx == dy), st=2 | 5 | 5 |
| 50x50, st=0 (pure streaming) | 3 | 3 |
| 60x60, st=0 | — | 3 |
| 60x60, st=2 | 7 | 7 |
| 50x50, streaming-only pmat, st=2 | 7 (Dirichlet-cell 9) | 7 (Dirichlet-cell 9) |
| 80x40, streaming-only pmat, st=2 | 7 (Dirichlet-cell 9) | — |
| 30x30 S4, st=2 | 6 | 6 |
| 60x60 S4, st=2 | 6 | 6 |
| 20x20 S8, left+bottom reflect, ratio 0.5 | 6 | 6 |

**Under Dirichlet-cell the three streaming-only pmat recipes were the arch-fragile ones,
and all three carried a pin of 10 against a reference count of 9.** They converged by
clearing rtol on the last iteration with only a percent or two to spare, so a
perturbation far too small to call a regression decided whether that last iteration
counted. The 2026-08-04 quadrature precision fix was exactly such a perturbation, and it
demonstrated the point twice: 80x40 serial measured 10 before it and 9 after on both
local arches, while 50x50 serial measured 9 locally both before and after but moved
9 -> 10 on all three CI images. Under ghost-flux the three take 7 and are pinned at 7,
the local count, per the contract — if CI asks for 8 on one arch, this is the history.

Mesh independent on every shape: 40x20 to 80x40 holds at 6 under Dirichlet-cell (6 to 7
under ghost-flux), and 200x50 in the 1x0.25 box (not a recipe) is 5, the same as 80x20
(a Dirichlet-cell measurement, not repeated).

These sizes were cut on 2026-08-02 to bring CI down (see "Recipe cost" below): the
"larger grid" family went 100x100 -> 60x60, the non-square mesh-independence pair
60x30/120x60 -> 40x20/80x40 with its 2x ratio intact, and the S4 reflective run joined
the plain S4 recipe at 30x30. The counts above are the re-measured references.

Until 2026-09-25 a pin was the max over the reference build and the two sensitive CI
arches (64-bit indices and the OpenMP Kokkos backend), re-swept 2026-08-02 in both CI images after the
isotropic-source change. Which configs sit one above the reference count changed with
it: the two streaming-only pmat serial recipes now measure the same everywhere (so
their pins equal the table), and instead the 50x50 ratio-1 SERIAL solve takes 7 under
64-bit — pinned 7 on all three recipes that run it (plain, `-ubolt_coo_two_call`, and
the painting identity, which is the same history by construction) — while under OpenMP
the np2 streaming-only pmat takes 10 (pinned 10) and the serial large-grid S4 takes 7
(pinned 7 — that slack was kept across the 2026-08-02 resize). Every other recipe
measures its reference count in both images; the resized recipes' pins are debug-arch
measurements, and CI runs green with them. The cost is
one iteration of slack on the reference build for those five recipes — the 1D
streaming-pmat baselines still pin their counts exactly. (That was the policy until
2026-09-25; since the ghost-flux re-pin every pin is the local opt count, so the 60x60 S4
serial pin is 6 and the 50x50 ratio-1 serial pins are 7 because the count itself is 7.)

**The 2026-08-04 quadrature precision fix was swept locally** (Dirichlet-cell era; opt,
then re-checked on debug) **and then against CI**, which is where the streaming-only pmat pins above got
their slack — the local sweep alone was not enough, and the note above records why. It has
NOT been swept in the sense of tightening every other pin back onto a per-arch maximum;
those are unchanged from before the fix, which is safe because the change only ever moves
a count by one and every other recipe passed on all four arches.

## Reflective boundary conditions: counts (Aug 2026; unmoved by the 2026-09-25 switch)

| reflective config | np=1 | np=2 |
|---|---|---|
| 1D, left reflect, st=2 (ratio 1) | 8 | — |
| 1D, left reflect, st=2, two-call assembly | 8 | — |
| 1D all-reflect infinite medium (rtol 1e-12) | 10 | 11 |
| 1D multigroup 4 groups t05, left reflect | 8 (max over groups) | — |
| 2D 50x50, left+bottom reflect, st=2 ratio 0.5 | 6 | 6 |
| 2D 30x30 S4, left+bottom reflect, ratio 0.5 | 6 | — |
| 2D 20x20 S8, left+bottom reflect, ratio 0.5 | 6 | 6 |
| 2D all-reflect infinite medium (rtol 1e-12) | 10 | 10 |

No row moved under the 2026-09-25 switch to the ghost-flux default: the all-reflect rows
have no vacuum face to change, and the mixed rows land on their Dirichlet-cell counts.

## Ghost-flux vacuum treatment: counts (2026-09-25 and 2026-09-26)

Counts, measured 2026-09-25 on the local opt arch and pinned on exactly that. The same
day's sweep in the 64-bit and OpenMP CI images measured one more on two rows (noted in
the table) and those pins first took the max; the re-pin for the default switch moved
them back onto the opt count, so these two are where CI would ask for +1:

| ghost-flux config (rtol 1e-12, `-check_inf_medium`) | np=1 | np=2 |
|---|---|---|
| 1D slab, both faces ghost vacuum | 8 | 8 |
| 1D slab, both faces ghost vacuum, `-matfree_removal` | 17 (pinned 18, 64-bit CI) | — |
| 2D 50x50, left+bottom reflect, right+top ghost vacuum | 11 (12 before 2026-09-26) | 11 (pinned 12, OpenMP CI) |
| 2D same, `-matfree_removal` | 26 (25 before 2026-09-26) | — |
| 3D 10^3, left+front+bottom reflect, others ghost vacuum | 11 | 11 |
| 3D same, `-matfree_removal` | 21 (pinned 22, 64-bit CI since 2026-09-26) | — |

### Switching the default (2026-09-25)
Ghost-flux replaced Dirichlet-cell as the default vacuum treatment; `"vacuum_treatment":
"dirichlet_cell"` keeps the old path. `*_dirichlet_cell.json` twins of `slab_st2`,
`box_50_st2`, `cube_10_st2` and `plex_box_50_st2` were added as recipes and reproduce the
pre-switch counts exactly (6, 6, 5, 7 - the plex one is 6 since element-block scaling
became the plex default, see "Unstructured iteration counts"; `box_50_st2_dirichlet_cell` also at np=2, 6, and
with `-precon_dsa`, 5), and the 1D baselines were re-captured — see "Single-group
baselines". Every pin was re-set onto the local opt count at the same time. The recipes
whose count moved (the max over groups for a multigroup recipe, as pinned; "and
`-matfree_removal`" means the right-preconditioned twin, which moved identically):

| recipe | Dirichlet-cell | ghost-flux |
|---|---|---|
| 1D `slab_st2`, `-precon_stream` and `-matfree_removal`, np=1 and 2 | 9 | 8 |
| 1D `slab_st0`, `-diag_scale -precon_stream`, np=1 | 3 | 2 |
| 1D `slab_mg4_t05`, `-precon_stream` and `-matfree_removal`, np=1 and 2 | 11 | 9 |
| 1D `slab_diffusive`, np=1 and 2 | 20 | 23 |
| 1D `slab_diffusive`, `-precon_dsa`, np=2 | 10 | 11 |
| 1D `slab_decades4`, ref-shift `-precon_ref_k 4`, np=1 and 2 | 20 | 22 |
| 1D `slab_decades4`, ref-shift default k, np=1 and 2 | 42 | 44 |
| 1D `slab_decades4_stream0`, ref-shift default k, np=1 and 2 | 20 | 22 |
| 2D `box_50_st2` (also `-ubolt_coo_two_call`; np=2 also `-check_matfree`), np=1 and 2 | 6 | 7 |
| 2D `box_50_identity`, np=1 | 6 | 7 |
| 2D `box_80x40_st2`, np=1 and 2 | 6 | 7 |
| 2D `box_50_st2`, `-precon_stream` and `-matfree_removal`, np=1 and 2 | 9 | 7 |
| 2D `box_80x40_st2`, `-precon_stream` and `-matfree_removal`, np=1 | 9 | 7 |
| 2D `box_decades4`, ref-shift `-precon_ref_k 4`, np=1 and 2 | 30 | 29 |
| 2D `box_decades4`, ref-shift default k, np=1 and 2 | 48 | 51 |
| 2D `box_diffusive`, `-precon_dsa -pc_composite_type additive`, np=1 | 23 | 14 |
| 2D `box_crooked_pipe`, np=1 | 123 | 113 |
| 2D `box_crooked_pipe`, np=2 | 95 | 114 |
| 2D `box_crooked_pipe`, `-precon_dsa`, np=1 | 72 | 28 |
| 2D `box_crooked_pipe`, `-precon_dsa`, np=2 | 69 | 28 |
| 2D `box_layers`, np=1 and 2 | 25 | 24 |
| 2D `box_layers`, `-precon_dsa`, np=1 | 13 | 12 |
| 2D `box_layers`, `-precon_dsa`, np=2 | 12 | 11 |
| 2D `box_random8`, np=1 and 2 | 27 | 19 |
| 2D `box_random8`, `-precon_dsa`, np=1 and 2 | 11 | 8 |
| 3D `cube_10_st2` (also `-ubolt_coo_two_call`), np=1 and 2 | 5 | 6 |
| 3D `cube_10_identity`, np=1 and 4 | 5 | 6 |
| 3D `cube_10_mg4_t05`, np=2 (its last group; serial stays 5) | 5 | 6 |
| 3D `cube_diffusive`, np=1 and 2 | 21 | 18 |
| 3D `cube_diffusive`, `-precon_dsa`, np=1 and 2 | 10 | 8 |
| 3D `cube_diffusive_yreflect`, `-precon_dsa`, np=1 | 10 | 12 |
| plex `plex_box_50_st2`, `-precon_stream` and `-matfree_removal`, np=1 | 9 | 7 |
| plex `plex_box_50_st2`, `-precon_stream` and `-matfree_removal`, np=2 | 9 | 8 |
| plex `plex_cube_10_st2`, np=1 and 2 | 5 | 6 |
| plex `plex_square_msh`, np=1 | 4 | 5 |
| plex `plex_decades4`, ref-shift `-precon_ref_k 4`, np=1 and 2 | 30 | 29 |
| plex `plex_decades4`, ref-shift default k, np=1 and 2 | 48 | 51 |

Every other recipe measures its Dirichlet-cell count, including every all-reflective one
(nothing to change there) and every `*_inf_medium_ghost` one (ghost-flux already). The
reflective ones then moved on 2026-09-26 - next subsection.

- The ratio-1 st=2 workhorses take one more in 2D and 3D (`box_50_st2`, `box_80x40_st2`,
  `cube_10_st2` and their identity/two-call copies, `plex_cube_10_st2`); 1D `slab_st2`,
  the ratio-0.5 files and `box_40x20_st2` / `box_60_st2` do not move. `plex_box_50_st2`
  stays at 7, so the structured and plex 50x50 boxes now agree.
- The streaming-only pmats improve everywhere: 9 -> 8 in 1D, 11 -> 9 on the multigroup
  slab, 9 -> 7 on the 2D boxes, 9 -> 7/8 on the plex box; 3D stays at 6.
- The mismatched default-k ref-shift gets worse by 2 or 3 (1D 42 -> 44, 2D 48 -> 51)
  while exact coverage moves with the full-pmat count it reproduces (1D 20 -> 22, 2D 30
  -> 29).
- DSA is much better on the heterogeneous problems — crooked pipe 72 -> 28 (and the
  unaccelerated serial/parallel gap, 123 against 95, is gone: 113 / 114), random8
  11 -> 8, `cube_diffusive` 10 -> 8, and the additive composite 23 -> 14 — but worse on
  `cube_diffusive_yreflect`, 10 -> 12. Off the recipes, the mismatched 2D ref-shift +
  DSA that took 179 on its thick group now takes 10 (see "DSA on a shifted pmat"),
  and `cube_diffusive -precon_stream -precon_dsa` converges in 44 where it diverged
  in 300 — both were Dirichlet-cell artefacts of the DSA boundary mask, and both are
  now pinned.
- DSA needed no change to its Marshak face for the switch: scaling that coefficient over
  0.25-1.0 moved no DSA count by more than 1.

### Ghost-flux reflective faces (2026-09-26)
The first ghost-flux cut left reflective faces as `psi(a) - psi(mirror a) = 0` rows and
let them WIN a mixed corner, mirrored over every axis the direction came in through - so
the corner cell never saw the vacuum face's inflow. Regenerating the Phase 6a report
found it: an O(1) error at the two mixed corners of the quarter box (2.07 / 2.15 / 2.19
at n = 25 / 50 / 100 on a flux of ~10) and L-infinity stuck at 3.14 for every n in the
DG0 convergence study. Three fixes were compared in a Python model of the 2D FD operator
(S2 and S4, n = 8 .. 128): (A) mirror a mixed-corner row over its reflective axes only,
(B) give a mixed-corner row its transport equation with the reflective face coupled to
the mirrored angle, (C) do that on EVERY reflective row. All three restore first-order
L-infinity in the convergence study; only (C) makes the quarter box equal the full box's
quadrant to rounding (A, B and Dirichlet-cell leave the O(h) boundary error 6a had
already recorded for reflective rows) and only (C) keeps `A^T = P A P` on the reflective
rows, because it leaves no BC rows at all. (C) is what landed, in every backend: a
reflective inflow face's upwind slot points at the mirrored angle in the same cell and
the streaming term fills it like any other upwind neighbour, exactly DG1's rule.
`"dirichlet_cell"` is untouched, and no baseline has a reflective face.

Checks: `verify_2dk`/`verify_3dk` check 2 builds the ghost-mode reflective rows this
way, and check 4 now runs on the mixed and all-reflect configs (plus the 1D identity by
hand, see check 4 above); `verify_plexk`
check 7 adds the low faces reflective to the `(VA)^T = P(VA)P` and constant-inflow checks
on triangles, tets and the irregular file, its slanted mesh has no BC rows under ghost
mode, and the single-cell-wide reflect box that Dirichlet-cell rejects must be accepted
under ghost-flux. The plex boxes still match their FD twins to rounding.

Every recipe with a reflective face was re-measured with its pin lifted, on this change
and on main, both on the local opt arch; these moved, and are pinned exactly (the two
recipes still carrying a CI +1 over the local count - `box_50_inf_medium_ghost` at np=2,
11 pinned 12, and `box_50_inf_medium -matfree_removal`, 24 pinned 25 - did not move and
keep their pins):

| recipe | before | after |
|---|---|---|
| 1D `slab_inf_medium`, `-precon_dsa`, np=1 | 10 | 9 |
| 2D `box_50_inf_medium_ghost`, np=1 | 12 | 11 |
| 2D `box_50_inf_medium_ghost`, `-matfree_removal`, np=1 | 25 | 26 |
| 2D `box_50_inf_medium`, `-precon_dsa`, np=1 and 2 | 10 | 9 |
| 3D `cube_10_inf_medium`, `-precon_dsa`, np=1 and 2 | 9 | 8 |
| 3D `cube_diffusive_yreflect`, `-precon_dsa`, np=1 | 12 | 9 |
| plex `plex_square_msh`, np=1 | 5 | 4 |
| plex `plex_tri_30_inf_medium_ghost`, np=1 | 12 | 11 |

Every other reflective recipe (the `*_reflect_lb` boxes, the 3D three-face corner, the
S8 box, the multigroup reflective slab, the other infinite-medium runs) kept its count,
and the DG1 recipes are untouched by construction. `-check_inf_medium` passes on every
infinite-medium recipe. CI then went red on the 64-bit image only, and a lifted-pin
sweep of every reflective recipe in that image found exactly one deviation:
`cube_10_inf_medium_ghost -matfree_removal` takes 22 there against 21 locally (unchanged
by this change locally), so it is pinned 22. Every other reflective recipe the image runs
fits its pin (the generated simplex boxes do not run in CI); the OpenMP image passed.

## Painted regions (MaterialSpec): counts (Aug 2026, re-measured 2026-08-02 and 2026-09-25)

| painted config | np=1 | np=2 | np=4 |
|---|---|---|---|
| 2D 50x50 identity: 2 regions = background, st=2 | 7 (Dirichlet-cell 6) | — | 7 |
| 2D 30x30 overlap: masked box covered by a background copy, st=2 ratio 0.5 | 5 | — | — |
| 2D 50x50 absorbing sourceless block (sigma_t 10, sigma_s 1, q 0) over st=2 ratio 0.5 | 5 | 5 | — |
| 2D 60x60 cold box: no face inflow, zero source outside a central 0.1x0.1 region | 5 | 5 | — |
| 1D multigroup 4 groups t05, double-density region 0.4-0.6 | 6, 6, 6, 6 | 6, 6, 6, 6 | — |

The cold box could not move: zero inflow leaves the painted source as the only rhs, so
the isotropic normalisation scales b uniformly, which a relative residual never sees.

## 3D iteration counts (measured 2026-08-02, re-measured 2026-09-25)
Measured 2026-08-02 on the reference build at capture. CI has since run green on the
sensitive arches (64-bit indices, OpenMP Kokkos) with these pins as committed, so no
per-arch slack was needed. Re-measured 2026-09-25 on the local opt arch under the
ghost-flux default; moved counts carry their Dirichlet-cell value in brackets.

| config | np=1 | np=2 |
|---|---|---|
| 10^3, st=2 (ratio 1) | 6 (Dirichlet-cell 5) | 6 (Dirichlet-cell 5) |
| 10^3, st=2, ratio 0.5 | 4 | 4 |
| 10x5x5 (dx, dy, dz all differ), st=2 | 6 | 6 |
| 15^3 (same shape, 1.5x finer), st=2 | 6 | 6 |
| 10^3, streaming-only pmat, st=2 | 6 | 6 |
| 10^3 S4, st=2 | 6 | 6 |
| 10^3, left+front+bottom reflect, ratio 0.5 | 5 | 5 |
| 10^3 all-reflect infinite medium (rtol 1e-12) | 10 | 10 |
| 10^3 identity: 2 regions = background, st=2 | 6 (Dirichlet-cell 5) | 6 (np=4; Dirichlet-cell 5) |
| 10^3 overlap: masked box covered by a background copy, st=2 ratio 0.5 | 4 | — |
| 10^3 absorbing sourceless block over ratio 0.5 | 4 | 4 |
| 10^3 cold cube: no face inflow, zero source outside a central 0.2^3 region | 4 | 4 |
| 10^3 multigroup 4 groups t05 | 5, 5, 5, 5 | 5, 5, 5, 6 (Dirichlet-cell 5, 5, 5, 5) |

Every cube dropped from 20^3 to 10^3 on 2026-08-02 to bring CI down (see "Recipe cost"
below), and the mesh-independence twin from 30^3 to 15^3 to keep its 1.5x ratio; the
counts above are the re-measured references, one lower than the 20^3 ones almost
throughout. Mesh independence was a weaker statement after the resize: under
Dirichlet-cell 10^3 to 15^3 moved 5 to 6, where 20^3 to 30^3 held flat at 6. Under the
ghost-flux default 10^3 rises to 6, so 10^3 to 15^3 is flat at 6 again (and 2D's
40x20 to 80x40 pair is now the one that moves by one).

The cold cube's source region had to widen from 0.1^3 to 0.2^3: at 10 cells the old
0.45/0.55 bounds land exactly on cell centres, where membership is a floating-point coin
toss. Any resize has to re-check that — a bound `b` is fragile at `n` cells whenever
`b*n` is a half-integer.

The painting identity was re-checked bitwise
against the uniform 10^3 history at np=1 when the pins were re-captured. The absorbing
block was eyeballed via `-flux_vtk` (sigma_t 10 painted mid-cube, flux ~3.3 over the
block against ~10 in the surrounding medium) — those numbers are from the 20^3 file,
before the 2026-08-02 resize, and have not been re-measured at 10^3.

## Unstructured iteration counts (2026-09-22 to 2026-09-27)
Measured 2026-09-22 on the opt arch (`arch-linux-c-opt`), `-ksp_max_it 400
-ksp_converged_reason`. Swept 2026-09-25 in the 64-bit and OpenMP CI images:
only the two `plex_box_50_st2` streaming-only rows moved (9 to 10). The simplex-box
rows (`plex_tri_*`, `plex_tet_*`) do NOT run in CI: generating them needs PETSc's
triangle / (c)tetgen, which the CI images lack, so `tests/Makefile` skips them there
(`PETSC_HAVE_TRIANGLE`, `PETSC_HAVE_TETMESHER`) and `verify_plexk` skips its
generated-simplex checks. Triangles stay covered in CI through the Gmsh mesh files.
Re-measured 2026-09-25 under the ghost-flux default and **pinned on the opt count,
exactly**, as every pin now is; a moved count carries its Dirichlet-cell value in
brackets. The structured twin is the same file without `"type": "unstructured"`, run
with the same options.

**Element-block scaling (2026-09-26).** Every plex row below now runs with PCAIR built on
the element-block-scaled pmat (`-precon_block_scale`, the plex default at both orders;
the structured twins keep it off). Re-measured on the opt arch and re-pinned on that
count; a count that moved carries its unscaled value as "unscaled N". Measured first on
the pre-reflective-face operators, where nothing got worse, and again after merging the
reflective-face change (2026-09-26), where exactly one row went up by one:
`plex_tri_30_inf_medium_ghost` serial, 11 unscaled to 12 (its count before that change).
At DG0 the blocks are the diagonal, and since PCAIR's strength of connection is
row-relative the hierarchy barely moves and no count moves by more than one; at DG1 it is what fixed the coarsening (see the DG1 table).

**Block-Jacobi removal stage (2026-09-26).** Composite index 0 now inverts the
operator's element blocks through the same `ElementBlockInverse`. At n_basis 1 (every
structured recipe and every DG0 plex one) that is the old point Jacobi to the bit: the
full `-ksp_monitor` history of all 247 such recipe lines, serial and np 2, is identical
before and after. At DG1 the histories move in the third digit but no pinned count does,
serial or np 2 - the tables below are unchanged. Where it does move a count (the crooked
pipe, box_layers, cube_diffusive, a tau-1 absorber) it goes down; see TODO.md.

| recipe | plex np=1 | plex np=2 | structured twin np=1 / np=2 | notes |
|---|---|---|---|---|
| `plex_box_50_st2` (50x50 quads, ratio 1) | 7 | 7 | `box_50_st2`: 7 / 7 (Dirichlet-cell 6 / 6) | the twins agree now; see "the twin difference" below |
| `plex_box_50_st2`, `-precon_stream -ksp_pc_side right` | 7 (Dirichlet-cell 9) | 7 (unscaled 8; Dirichlet-cell 9) | 7 / 7 (Dirichlet-cell 9 / 9) | Dirichlet-cell: hair-trigger, 10 on the 64-bit and OpenMP CI arches |
| `plex_box_50_st2`, `-matfree_removal -ksp_pc_side right` | 7 (Dirichlet-cell 9) | 7 (unscaled 8; Dirichlet-cell 9) | 7 / 7 (Dirichlet-cell 9 / 9) | same count as the line above |
| `plex_box_50_reflect_lb` | 6 | 6 | `box_50_reflect_lb`: 6 / 6 | |
| `plex_tri_30_st2` (1800 triangles, S4, ratio 0.5) | 5 | 5 | — | |
| `plex_cube_10_st2` (1000 hexes, ratio 1) | 6 (Dirichlet-cell 5) | 6 (Dirichlet-cell 5) | `cube_10_st2`: 6 / 6 (Dirichlet-cell 5 / 5) | |
| `plex_tet_6_st2` (1296 tets, ratio 0.5) | 5 | 5 | — | |
| `plex_square_msh` (8 Gmsh triangles) | 4 (5 from the ghost-flux switch to the reflective-face change of 2026-09-26) | 4 | — | |
| `plex_tri_30_inf_medium_ghost`, `-check_inf_medium -ksp_rtol 1e-12` (ghost-flux right + top, reflect left + bottom; re-measured 2026-09-26) | 12 (unscaled 11) | 12 | — | |
| the same, `-matfree_removal` | 33 | — | — | |
| `plex_tet_6_inf_medium_ghost`, `-check_inf_medium -ksp_rtol 1e-12` (ghost-flux right + back + top; measured 2026-09-25) | 11 | 11 | — | |
| `plex_tet_12_absorber_dirichlet_cell`, `-ksp_rtol 1e-12` (a 0.5-side box of 10368 tets, pure absorber, Dirichlet-cell; added 2026-09-27) | 6 (unscaled 12) | 6 (unscaled 12, not a recipe) | — | guards the element-block scaling against the old simplex creep, see below |
| `plex_decades4`, `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 4, 6, 15, 28 (unscaled 29; 29 in the OpenMP CI image, pinned 29) | 4, 6, 15, 29 | `box_decades4`: 4, 6, 15, 29 both | Dirichlet-cell 4, 6, 15, 30 on both sides |
| `plex_decades4`, `-matfree_removal -precon_ref_shift` (default k = 2) | 7, 8, 27, 51 | 7, 8, 27, 51 | `box_decades4`: 7, 8, 27, 51 both | Dirichlet-cell plex 7, 9, 24, 48 serial and 7, 8, 24, 48 otherwise |

**The simplex iteration creep (explained 2026-09-27).** The Phase 6a study saw tets on a
pure absorber at rtol 1e-12 take 6 / 19 / 22 at n = 8 / 16 / 32 (hexes 5 / 6 / 7) and
simplex counts rise ~1 per 4x refinement. Those counts come back exactly with
`"vacuum_treatment": "dirichlet_cell"` and `-precon_block_scale 0`; ghost-flux alone or
the block scaling alone gives 6 / 7 / 7. The cause is PCAIR's R drop (`-pc_air_r_drop`,
row-relative): R = -A_cf A_ff^{-1} carries each F row's 1/diagonal, so the drop is not
invariant under a row scaling, and beside the Dirichlet identity rows (diagonal 1) the
couplings through interior F points (diagonal ~1/h) are thrown away - `-sub_1_pc_air_r_drop
0` restores 7 on the unscaled n = 16 tets, no other PCAIR knob does. It is not cycles in the
upwind graph: PETSc's simplex boxes have none (per angle, reflective couplings aside).
`plex_tet_12_absorber_dirichlet_cell` shrinks the box to 0.5 so the effect shows at a
recipe-sized mesh. Under the defaults simplices sit at most one iteration over quads/hexes
and are flat in n (tables in TODO.md, Phase 6); with an exact streaming inverse the two
take the same count, so that +1 is PCAIR's approximation on simplices. A structured
`dirichlet_cell` problem runs unscaled by default and meets a milder form of the same
drop (the 128x128 S4 absorber at rtol 1e-12: 8 Dirichlet-cell, 7 with
`-precon_block_scale`, 7 under ghost-flux).

A multigroup row pins the max over its groups (29 - the OpenMP CI image's, local opt
is 28 - and 51), as everywhere else. The OpenMP image was swept on every plex recipe
when that pin went red (2026-09-26): it was the only one above local opt. Not in the
table: `plex_box_50_st2_dirichlet_cell` is 6 (unscaled 7, its structured twin 6).

**DG1** (`*_dg1.json`: the DG0 file with `"order": 1` in `mesh`, and no
`vacuum_treatment` - DG1 is ghost-flux only). Unscaled, PCAIR's coarsening stalls on
DG1 at its default strong threshold 0.5 - the AIR levels grow linearly with n (28 / 58 /
119 at n = 25 / 50 / 100 on the 50x50 box's physics) - which made the 50x50 DG1 boxes the
slowest lines in the suite and more than doubled the debug CI job; every DG1 line used to
pass `-sub_1_pc_air_strong_threshold 0.25` for that. With the element-block scaling (the
plex default since 2026-09-26) PCAIR builds 14 / 18 / 21 levels at the DEFAULT threshold
and the flag is gone. Measured on the opt arch and pinned on it; "0.25: N" is the count
under the old stopgap, where it differs (the old pins were the max over the opt, 64-bit
and OpenMP CI images, which put two rows one higher than local opt). The DG0 column is
the table above, for scale.

| recipe | DG1 np=1 | DG1 np=2 | DG0 np=1 |
|---|---|---|---|
| `plex_box_50_st2_dg1` (also `-ubolt_coo_two_call`, `-check_matfree`) | 7 (0.25: 7, 8 on 64-bit and OpenMP) | 7 | 7 |
| the same, `-precon_stream -ksp_pc_side right` | 9 (0.25: 10) | - | 7 |
| the same, `-matfree_removal -ksp_pc_side right` | 9 (0.25: 10) | 8 (0.25: 9, 10 in every CI image) | 7 |
| `plex_box_50_reflect_lb_dg1` | 6 | 6 | 6 |
| `plex_tri_30_st2_dg1` | 5 | 5 | 5 |
| `plex_cube_10_st2_dg1` | 7 | 7 | 6 |
| `plex_tet_6_st2_dg1` | 5 | 5 | 5 |
| `plex_square_msh_dg1` | 5 | 5 | 5 |
| `plex_box_30_inf_medium_dg1`, `-check_inf_medium -ksp_rtol 1e-12` (quads, so it runs in CI) | 12 (0.25: 13) | 12 (0.25: 13) | - |
| the same, `-matfree_removal` | 42 | - | - |
| `plex_tri_30_inf_medium_dg1`, `-check_inf_medium -ksp_rtol 1e-12` | 13 | - | 12 |
| `plex_tet_6_inf_medium_dg1`, `-check_inf_medium -ksp_rtol 1e-12` | 12 | 12 | 11 |
| `plex_decades4_dg1`, `-matfree_removal -precon_ref_shift -precon_ref_k 4` | 4, 7, 17, 36 | 4, 7, 17, 36 | 4, 6, 15, 29 |
| `plex_decades4_dg1`, `-matfree_removal -precon_ref_shift` (default k) | 8, 10, 41, 66 | - | 7, 8, 27, 51 |

The infinite-medium DG1 solves land at ~1e-13 against the 1e-9 tolerance, the slopes
included (the check wants the constant on basis 0 and zero on the rest).

**DSA** (`-precon_dsa` on the plex, 2026-09-27, opt arch, pinned on that count). On a
quad/hex box the plex diffusion operator is the structured one times the cell volume
(verify_plexk check 11), and every twin takes EXACTLY its structured count, serial and
np 2. The `*_diffusive` files are `box_diffusive` / `cube_diffusive` (ten mean free
paths per cell, ratio 0.99) with `"type": "unstructured"`, and `"simplex": true` or
`"order": 1` where named. At DG1 the diffusion operator is the MIP interior penalty form
in the DG1 space (27 Sep 2026; the rows below replaced the cell-average correction's 25 /
26 / 9 / 34, see TODO.md); it takes every diffusive problem further than DSA does at
DG0. The thin (c = 0.5) infinite-medium box barely needs DSA and went 9 -> 11.
The DG0 rows were re-measured 2026-09-27 for the consistent D (the physical-D count
follows as "phys"); the twins still take exactly the structured count, and the DG0
diffusive files now match or beat DG1 with DSA. DG1 ignores `-dsa_consistent_d`.

| recipe | plex np=1 | plex np=2 | structured twin np=1 | no DSA np=1 |
|---|---|---|---|---|
| `plex_box_diffusive`, `-precon_dsa` | 5 (phys 11) | 5 (phys 11) | 5 (phys 11) | 29 (structured 29) |
| the same, `-matfree_removal -precon_ref_shift -precon_dsa` | 5 (phys 11) | 5 (phys 11) | 5 (phys 11) | - |
| the same, `-precon_dsa -pc_composite_type additive` | 7 (phys 14) | - | 7 (phys 14) | - |
| `plex_box_50_st2_dirichlet_cell`, `-precon_dsa` | 5 | - | 5 | 6 |
| `plex_box_50_st2`, `-precon_dsa` (not a recipe) | 5 | - | 5 | 7 |
| `plex_cube_diffusive`, `-precon_dsa` | 5 (phys 8) | 5 (phys 8) | 5 (phys 8) | 18 (structured 18) |
| `plex_tri_diffusive` (5000 triangles), `-precon_dsa` | 5 (phys 10) | 5 (phys 10) | - | 35 |
| `plex_tet_diffusive` (6000 tets), `-precon_dsa` | 5 (phys 9) | 5 (phys 9) | - | 31 (not a recipe) |
| `plex_tri_30_inf_medium_ghost`, `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 9 | - | - | 12 |
| `plex_tet_6_inf_medium_ghost`, the same (not a recipe) | 9 | - | - | 11 |
| DG1 `plex_box_diffusive_dg1`, `-precon_dsa` | 6 | 7 | - | 34 |
| DG1 `plex_cube_diffusive_dg1`, `-precon_dsa` | 7 | 7 | - | 32 |
| DG1 `plex_tri_diffusive_dg1`, `-precon_dsa` | 8 | 8 (not a recipe) | - | 40 (not a recipe) |
| DG1 `plex_tet_diffusive_dg1`, `-precon_dsa` | 9 | 9 (not a recipe) | - | 43 (not a recipe) |
| DG1 `plex_box_30_inf_medium_dg1`, `-precon_dsa -check_inf_medium -ksp_rtol 1e-12` | 11 | - | - | 12 |

**The twin difference is a finding, not a bug.** On a uniform quad/hex box the plex matrix
IS the structured one to ~1e-15 (verify_plexk, above), but its rows are in a different
order — plex point order under the simple partitioner, not DMDA order — and PCAIR is not
permutation-invariant: its CF splitting and its approximate inverses depend on the order
it walks the rows. So a twin can land an iteration away. Under ghost-flux the only twin
rows that differed were the streaming-only pmat pair at np=2 (plex 8, structured 7 -
now 7 and 7, with the plex under element-block scaling, which also puts the plex through
a different preconditioner than its twin, so the twin counts are no longer a like-for-
like comparison); the
Dirichlet-cell measurements below had two others, both on the edge of rtol on BOTH sides
(the rtol margins are Dirichlet-cell-era numbers and were not re-measured):
- `plex_box_50_st2`: the structured solve clears rtol at iteration 6 with 1.4% to spare
  (5.30e-4 against a target of 5.38e-4); the plex one misses at iteration 6 by 0.4%
  (5.398e-4 against 5.378e-4 — even the preconditioned `r_0` differs in the fourth
  digit, which is the permutation acting on PCAIR) and converges at 7 with a wide margin
  (0.07 of rtol). The pin of 7 is safe; another arch could see 6, which passes.
- `plex_decades4` default k, group 1 serial: 9 against the structured 8, missing rtol at
  8 by 9%; at np=2 the plex matches the structured 8 (0.94 of rtol). The recipe pins the
  group max, 48, which is unaffected. Under ghost-flux group 1 is 8 on both sides.

The partitioner moved the same edge under Dirichlet-cell: `plex_box_50_st2` at `-n 2`
was 7 under the default simple partitioner and 6 under `-petscpartitioner_type parmetis`
(tri and tet: 5 under both). Under ghost-flux it is 7 under both. That is why the backend defaults to `simple` — deterministic, and available on
every CI image, where ParMETIS may not be — and why no recipe passes a partitioner.

**Hair-trigger pins** (Dirichlet-cell era, not re-measured under ghost-flux), the final
residual as a fraction of rtol at the pinned iteration
(np=1 / np=2), i.e. where another arch could need one more: the streaming-only pmat and
matfree pair (0.87 / 0.71), the two `plex_decades4` group-3 solves (0.94 / 0.94, and
group 1 of default k at np=2, 0.94), `plex_cube_10_st2` (0.76), `plex_square_msh`
(0.60 / 0.64). The structured twins of the first two carry exactly that +1 in their
pins then (10, 31, 49), and the CI sweep did ask for 10 on the first. The rest cleared
rtol by 6x or more.

**Second-arch sweep** (2026-09-23, Dirichlet-cell era, `arch-linux-c-debug` against a
debug-built PFLARE main): build with `-Wall -Werror`, `check`, `tests_short` and `tests` all exit 0; every
plex recipe and every structured twin converges in EXACTLY the opt count above at np 1
and np 2 (per group too), with the same rtol margins to three digits; `-malloc_dump`
reports nothing unfreed on `verify_plexk` (np 1, 2, 3) and on the plex solves with and
without `.vtu` output. One structured recipe is the sharpest edge in the whole suite:
`box_50_st2` at np 2 cleared rtol at its then pin of 6 by 0.013% (it takes 7 under
ghost-flux).

**Checked by hand, not recipes** (2026-09-22, np 1 and 2): every plex problem with
`-check_matfree -matfree_removal` — matvec differences 2e-16 to 4e-16 against 1e-13,
composed diagonal **0.0** against the assembled one on every file, triangles and tets
included; `plex_box_50_st2` and `plex_tri_30_st2` with `-ubolt_coo_two_call` — the
`-ksp_monitor` history is byte-identical to the one-call run at both rank counts; every
plex problem with `-flux_vtk x.vtu` at np 1, 2, 4 — the file's `NumberOfCells` totals the
global cell count (2500, 1800, 1000, 1296, 8, 900) and it carries `scalar_flux`,
`sigma_t`, `source`. (`-precon_dsa` on a plex file failed with PETSC_ERR_SUP then; it
has a plex operator since 2026-09-27, see "Unstructured iteration counts".)

## CG-SUPG verification and iteration counts: the counts (28 Sep 2026)

**Iteration counts** (28 Sep 2026, opt arch, default rtol; each `cg_*.json` is the
twin of the `plex_*.json` of the same name). `-precon_block_scale` keeps its
unstructured default ON, but on CG it changes nothing that matters:

| recipe | CG np=1 | CG np=2 | unscaled pmat | zeta 2 | DG0 twin |
|---|---|---|---|---|---|
| `cg_box_50_st2` | 7 | 7 | 7 | 7 | 7 |
| `cg_box_50_reflect_lb` | 6 | 6 | 6 | 6 | 6 |
| `cg_tri_30_st2` | 5 | 6 | 5 | 5 | 5 |
| `cg_cube_10_st2` | 6 | 6 | 6 | 6 | 6 |
| `cg_tet_6_st2` | 5 | 5 | 5 | 5 | 5 |
| `cg_square_msh` | 4 | 4 | 4 | 4 | 4 |
| `cg_decades4` (per group) | 4, 7, 16, 41 | 4, 7, 16, 42 | 4, 7, 16, 42 | 5, 7, 16, 41 | - |
| `cg_box_diffusive` | 39 | 39 | 39 | 39 | 29 |
| `cg_tri_diffusive` | 35 | 35 | 35 | 35 | 35 |
| `cg_box_void_channel` | 47 | 46 | 47 | 41 | 26 |
| `cg_slab_saaf_ls_void` | 4 | 4 | 4 | 3 | - |
| `cg_slab_thin_thick` | 4 | 4 | 4 | 3 | - |
| `cg_box_20_inf_medium`, `-check_inf_medium -ksp_rtol 1e-12` | 11 | 11 | - | - | - |

Above the DG0 twin: the diffusive quad box (39 against 29) and the void channel
(47 against 26). Investigated 28 Sep 2026 (opt arch, serial, no code change) by
swapping PCAIR for an exact inverse, `-precon_block_scale 0 -sub_1_pc_type lu`,
which splits each count into what the discretisation costs and what AIR costs:

| recipe | CG AIR | CG LU | DG0 AIR | DG0 LU |
|---|---|---|---|---|
| `*_box_diffusive` | 39 | 25 | 29 | 24 |
| `*_tri_diffusive` | 35 | 25 | 35 | 29 |
| `*_box_void_channel` | 47 | 38 | 25 | 21 |

- **Diffusive box: PCAIR.** Under LU the backends tie. With the scattering off (the
  same file, `Sigma_s` 0) AIR takes 5 iterations to 1e-10 on CG against 2 on DG0.
  In thick cells tau = 1/sigma_t, so the SUPG removal term cancels the skew part of
  the streaming, and what is left is sigma_t times the consistent mass matrix, which
  on Q1 is not diagonally dominant (P1 triangles sit at equality). A throwaway
  switch lumping that mass brought AIR to within 2 of LU (and the absorber to 2),
  but it doubles the LU count (25 -> 50) and breaks SUPG consistency. The PCAIR
  knobs (strong threshold, inverse sparsity order 2, `lair` Z, no drops, poly order
  12, inverse type, up-and-down smoothing, combinations) do no better than 35;
  zeta changes nothing (tau is 1/sigma_t there). Scanning sigma_t at c = 0.99, the gap
  is all AIR's from 10 mean free paths per cell up (CG AIR / LU 39 / 25, 26 / 15,
  18 / 10 at sigma_t 100 / 300 / 1000; DG0 29 / 24, 19 / 16, 12 / 10), while at 1-3 mean
  free paths per cell CG is slower under LU too (37 against 28, 40 against 33).
- **Void channel: the discretisation.** It is not void-specific (channel sigma_t
  0 / 0.01 / 0.1 / 1 all give 47 / 38). It follows tau = h/zeta in the thin
  cells, the SUPG streamline diffusion there:

  | zeta | 0.125 | 0.5 | 2 | 8 | 32 |
  |---|---|---|---|---|---|
  | CG LU | 45 | 38 | 27 | 24 | 23 |
  | CG AIR | 52 | 47 | 41 | 53 | diverges |

  AIR needs the stabilisation the outer iteration pays for. (Zeta 1000 reports
  a 1-iteration "convergence": AIR has blown up, preconditioned residual 1e17, and
  the true residual has not moved.) Candidates in TODO.md.

  Wang's cell size in place of h_Omega (28 Sep 2026, a throwaway switch, reverted):
  tau = min(1/sigma_t, h/zeta) with an angle-independent h. h_Omega = 2 / sum_j
  |Omega . grad phi_j(x_c)| is dx / max(|mu|, |eta|) on a square quad, so it is
  never below the edge, and a cell size only rescales tau per angle, which the
  zeta sweep already covers:

  | h | zeta 0.125 | 0.5 | 2 | 8 |
  |---|---|---|---|---|
  | h_Omega (current) | 52 / 45 | 47 / 38 | 41 / 27 | 53 / 24 |
  | shortest edge | 52 / 43 | 46 / 35 | 39 / 25 | diverges / 24 |
  | diameter | 53 / 45 | 48 / 37 | 43 / 27 | 69 / 24 |

  (AIR / LU on the void channel. The diffusive boxes do not move, 39 and 35, since tau
  is 1/sigma_t there. On quads V^(1/2) is the shortest edge.) In `verify_cgk`,
  h_Omega is the most accurate on every exact-SN benchmark. With the shortest edge
  the SAAF-LS void slab at 40 cells is 3.10e-3 against 2.97e-3 (zeta 0.5) and
  4.03e-3 against 3.49e-3 (zeta 2). The thin/thick slab at 16 cells is 8.09e-2
  against 7.82e-2. The pure-absorber order runs are 1-4% worse. The finest meshes
  agree to within a few percent, and the slab counts go up at zeta 2 (up to 14
  against 9). The diameter sits between the two. One iteration on the channel does
  not pay for that, so h_Omega stays.

**The CG DSA** (28 Sep 2026, opt arch; `-precon_dsa`, one GAMG V-cycle; identical at
1, 2 and 4 ranks on the diffusive boxes, tets, hex cube, decades4, reflect_lb and the
void channel). "LU" is `-dsa_pc_type lu`; the last columns are variants measured and
not adopted - the SUPG operator's own tensor as D everywhere, and Marshak's 1/2 in
place of the half-range current on the vacuum faces:

| recipe | none | DSA | DSA LU | DG0 twin DSA | SUPG-tensor D | Marshak 1/2 |
|---|---|---|---|---|---|---|
| `cg_box_50_st2` | 7 | 5 | 5 | 5 | 7 | 6 |
| `cg_box_50_reflect_lb` | 6 | 4 | 4 | - | 5 | 4 |
| `cg_tri_30_st2` | 5 | 4 | 4 | - | 4 | 4 |
| `cg_cube_10_st2` | 6 | 5 | 5 | - | 5 | 5 |
| `cg_tet_6_st2` | 5 | 4 | 4 | - | 4 | 4 |
| `cg_square_msh` | 4 | 3 | 3 | - | 3 | 3 |
| `cg_decades4` (per group) | 4, 7, 16, 41 | 4, 4, 5, 5 | 4, 4, 5, 4 | 4, 4, 5, 5 | 4, 4, 5, 5 | 4, 5, 5, 6 |
| `cg_box_diffusive` | 39 | 5 | 4 | 5 | 5 | 6 |
| `cg_tri_diffusive` | 35 | 5 | 4 | 5 | 5 | 7 |
| `cg_cube_diffusive` | 48 (49 on the CI images) | 6 | 5 | 5 | 6 | - |
| `cg_tet_diffusive` | 39 | 5 | 4 | 5 | 5 | - |
| `cg_box_void_channel` | 47 | 8 | 8 | 6 | 7 | 8 |
| `cg_slab_saaf_ls_void` | 4 | 4 | 4 | - | 4 | 4 |
| `cg_slab_thin_thick` | 4 | 4 | 4 | - | 4 | 4 |

(The Marshak and SUPG-tensor columns on the void channel were measured before the
void bridge, with the SUPG tensor in the void.) With both inner solves exact
(`-dsa_pc_type lu -precon_block_scale 0 -sub_1_pc_type lu`) CG ties DG0 on the
diffusive boxes, triangles, cube and thin box (4 / 4 / 4 / 5 each) - the DSA is
the transport's own restricted operator, so this was expected - but not on the
void channel, 7 against 4: the gap there is the diffusion model with SUPG in
the void, not an inner solve. Refining the diffusive boxes at fixed physics (25 /
50 / 100 / 200 cells a side): quads 5 / 5 / 6 / 6, triangles 5 / 5 / 6 / 6 (no
DSA 32-46). The void channel, unbridged (the SUPG tensor in the void), goes
6 / 7 / 9 / 11 because h / zeta shrinks with the mesh; BRIDGED (the default, D =
L / 3) it is 7 / 8 / 8 / 8. No fixed void D does better than 8 from 100 cells on
(scanned 0.05-1, and the SUPG tensor plus a fixed D), so the chord stays - it is
mesh-independent and the other backends' rule.

`-diag_scale` on a scattering
problem stalls (`cg_box_50_st2` does not converge in 500; the DG0 twin goes 7 ->
83): it scales the assembled operator but not the matrix-free scatter, the caveat
it has on every backend. Without scattering it is consistent and harmless (the void
slab, 4 -> 4, pinned).

## Matrix-free removal (`-matfree_removal`) (Aug 2026, re-measured 2026-09-25)

`RemovalTerm` can be applied matrix-free instead of assembled
(`RemovalTerm::set_matrix_free`, `-matfree_removal` on the driver). The assembled
matrix then carries **streaming only**, which does not depend on the group: it is
assembled once before the sweep, no group refills anything, and PCAIR sets up once for
the whole sweep. It is off by default, so every recipe, count and baseline above is
untouched — all 24 baselines reproduce bitwise, which is what the default half of the
change was verified against. The recipes below are NEW pins.

Because the assembled matrix already *is* a streaming-only pmat, this mode **implies**
`-precon_stream`; an explicit `-precon_stream` alongside it is redundant and quietly
ignored (it would build a second copy of the same values). `-diag_scale` is a checked
error in this mode — it scales the assembled operator, which here is the streaming part
alone, so the matrix-free removal and scatter would stay unscaled. And because it
implies `-precon_stream`, it inherits that flag's behaviour with `-precon_dsa`
(see the DSA section below) — measured identical on `cube_diffusive.json`. Under
Dirichlet-cell vacuum that was 35 iterations either way alone, and neither converged
in 300 with `-precon_dsa` added. Under the ghost-flux default it is 44 either way,
with or without `-precon_dsa`, serial and `-n 2`. The `-precon_stream -precon_dsa`
recipe pins that; this mode has no recipe of its own.

**The counts are the `-precon_stream` counts; the residual histories are not.** Same
operator, same pmat, same Jacobi diagonal — but the assembled path sums streaming and
removal into one matrix entry and multiplies once where the matrix-free path multiplies
twice and adds, so the two differ in the last bits and diverge from there. That is why
the matfree recipes are *pinned at their twin's pin* and get no baseline of their own:
the count is the invariant, the history is not. Every twin below measured exactly the
same count as its `-precon_stream` line, serial and at `-n 2`, under Dirichlet-cell and
again under the ghost-flux default (2026-09-25, local opt; the Dirichlet-cell counts in
brackets where they moved, and the plex pair is in "Unstructured iteration counts").

| config | `-precon_stream` / `-matfree_removal`, np=1 | np=2 |
|---|---|---|
| 1D slab st=2 | 8 / 8 (Dirichlet-cell 9 / 9) | 8 / 8 (Dirichlet-cell 9 / 9) |
| 1D multigroup 4 groups t05 | 9, 9, 9, 8 / same (Dirichlet-cell 11, 11, 11, 9) | 9, 9, 9, 8 / same (Dirichlet-cell 11, 11, 11, 9) |
| 2D 50x50 st=2 | 7 / 7 (Dirichlet-cell 9 / 9) | 7 / 7 (Dirichlet-cell 9 / 9) |
| 2D 80x40 st=2 | 7 / 7 (Dirichlet-cell 9 / 9) | — |
| 3D 10^3 st=2 | 6 / 6 | 6 / 6 |

The infinite-medium check runs in this mode too — `slab_inf_medium.json` in 7 serial
and 8 at `-n 2`, `box_50_inf_medium.json` in 24 both (the serial recipe pinned 25:
the 64-bit CI arch takes one more). Those are the streaming-only
pmat's counts rather than the default pc's 10, and they are what `-precon_stream`
measures on the same files. The solution lands at 1.4e-13 to 7.6e-12 against the 1e-9
tolerance; the drift from the default path's ~1e-13 is the pmat and not the matrix-free
apply, since `-precon_stream` measures the same errors to four digits. These three pins
carried one iteration of slack (measured + 1) until the 2026-09-25 re-pin, and sit on the
measured number (7, 8, 24) since; all-reflective, so the ghost-flux switch did not move
them.

(The `-check_matfree` paragraphs that followed here are in `docs/dev/verification.md`.)

## DSA (the diffusion correction, `-precon_dsa`) (Aug 2026 to 2026-09-27)

`DSAPrecon` adds a third stage to the composite preconditioner: a cell-centred
diffusion operator on the same grid, restricted from and prolonged back onto the
ordinates (`include/ubolt/dsa.hpp`). It is off by default, so every recipe,
count and baseline above is untouched — the DSA recipes below are NEW pins.
This section is the structured backends; the plex operator (2026-09-27, a
volume-weighted two-point flux, the structured matrix times V on a box) is
verified in "Unstructured DG verification", check 11, and its counts - the
`plex_*_diffusive` twins take exactly the structured ones below - are in
"Unstructured iteration counts", **DSA**.

The regime it exists for is the one nothing else in the preconditioner
addresses, and which no test problem demonstrated until now. The diffusive
problem files are deliberately the same physics in every dimension — `Sigma_t
100` over cells of width 0.1, so ten mean free paths per cell, at a scattering
ratio of 0.99, every face vacuum so the diffusion operator is nonsingular
whatever the ratio, all at S4: `slab_diffusive.json` is 100 cells over a length
of 10, `box_diffusive.json` 50x50 over 5x5 and `cube_diffusive.json` 10^3 over
1^3. Only the dimension changes between them, which is what makes the counts
comparable. Re-measured 2026-09-25 (local opt) under the ghost-flux default, which
every row with a vacuum face runs in; the Dirichlet-cell count follows in brackets where
it moved. The DSA rows were re-measured again 2026-09-27 for the discretisation-consistent
D (the default since then, see "Discretisation-consistent D" below); the physical-D count
they replaced follows as "phys" where it moved. Rows without a recipe at that np (np=2 of the st=2, additive, CG and yreflect
rows) were measured by hand.

| config | np=1 | np=2 |
|---|---|---|
| 1D diffusive slab, no DSA (the reference) | 23 (Dirichlet-cell 20) | 23 (Dirichlet-cell 20) |
| 1D diffusive slab, `-precon_dsa` | 5 (phys 11) | 5 (phys 11, Dirichlet-cell 10) |
| 1D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 9 | 10 |
| 1D slab st=2, `-precon_dsa` | 5 | 5 |
| 2D diffusive box, no DSA (the reference) | 29 | 29 |
| 2D diffusive box, `-precon_dsa` | 5 (phys 11) | 5 (phys 11) |
| 2D diffusive box, `-precon_dsa -pc_composite_type additive` | 7 (phys 14, Dirichlet-cell 23) | not re-measured (phys 14, Dirichlet-cell 23) |
| 2D diffusive box, `-precon_dsa -dsa_ksp_type cg -dsa_ksp_max_it 5` | 4 (phys 11) | not re-measured (phys 11) |
| 2D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 9 | 9 |
| 2D box st=2, `-precon_dsa` | 5 | 5 |
| 3D diffusive cube, no DSA (the reference) | 18 (Dirichlet-cell 21) | 18 (Dirichlet-cell 21) |
| 3D diffusive cube, `-precon_dsa` | 5 (phys 8, Dirichlet-cell 10) | 5 (phys 8, Dirichlet-cell 10) |
| 3D diffusive cube, Y faces reflective, anisotropic box, `-precon_dsa` | 5 (phys 9, Dirichlet-cell 10) | 12 before 2026-09-26, not re-measured (Dirichlet-cell 11) |
| 3D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 8 | 8 |
| 3D cube st=2, `-precon_dsa` | 4 | 4 |

Read each dimension's first two rows together: that pair IS the test, and
what the correction buys — 23 to 5 in 1D, 29 to 5 in 2D, 18 to 5 in 3D (with the
physical D it was 23 to 11, 29 to 11 and 18 to 8; 20 to 11, 29 to 11 and 21 to 10
under Dirichlet-cell). **The corrected count is 5 in every dimension** while the
uncorrected one is not, so what the correction removes is exactly the part that
varies with dimension. That is the claim DSA makes, and it is the number to beat
for anything that comes after.

### Discretisation-consistent D (the default since 2026-09-27)

The physical `D = 1/(3 sigma_t)` is the PDE's, not the discretisation's: first-order
upwind streaming adds a numerical diffusion `m h` per face (m = the quadrature's
half-range current, sum over outgoing ordinates of `w (Omega . n)` over the weight sum,
~1/4), exact for a linear flux in an infinite pure scatterer, and at ten mean free
paths per cell it is several times D. `DSAPrecon` now blends it into every face
coefficient, Marshak faces included, as `(D^p + (m h)^p)^(1/p)` with p = 1.5
(`-dsa_consistent_d_power`); `-dsa_consistent_d 0` is the old operator, pinned on
`box_diffusive` (11) and `box_crooked_pipe` (28), and power 1 (the exact sum) on
`box_diffusive` (5). Why 1.5: a 1D Fourier analysis of step-differenced source
iteration + DSA puts the optimal face D below `D + m h` for intermediate cells (tau =
sigma_t h ~ 0.3-2) and p = 1.5 tracks it to a few hundredths of spectral radius at
every tau and ratio, where the exact sum costs +1-2 on the literature classics at tau
0.5 (`J_box2d_60_st10_c1` 9 -> 11) and p = 2 over-shoots at ratio ~1. The recipe pins
that moved, serial / np 2 identical unless shown: every diffusive file 11/11/8 -> 5
(1D/2D/3D), the plex twins the same, triangles 10 -> 5, tets 9 -> 5, crooked pipe 28 ->
8, layers 12 -> 11, random8 8 -> 7 (np 2 8 -> 6), additive 14 -> 7, the CG inner solve
11 -> 4, `cube_diffusive -precon_stream -precon_dsa` 44 -> 25, the decades4 DSA lines
11 -> 5 at exact k and 10 -> 8 (2D) at the default k. Nothing got worse. The external
literature suite (~110 problems) is in `TODO.md`.

The remaining rows say DSA does not break what already worked — st=2 goes 6 to
5 in 1D, 7 to 5 in 2D and 6 to 4 in 3D, and the infinite media still land on
the exact constant (1D 2.3e-13 serial and 7.1e-12 at np=2, 2D 3.4e-14 and
8.3e-14, 3D 6.9e-14 and 1.1e-13, against the 1e-9 tolerance). Those runs are
also the reflective-branch check: every face reflective means every face is a
zero-Neumann one in the diffusion operator, and the singularity guard is
satisfied by the files' absorption. The solution is unchanged by the
correction, as it must be — `box_50_st2.json` at rtol 1e-12 agrees to 1.3e-12
relative with and without `-precon_dsa`.

**The 3D `yreflect` row is the face-convention pin**, and the only recipe that
can catch a y/z mix-up. In 3D `bottom`/`top` are the **Z** faces and the Y ones
are `front`/`back` (PETSc's box convention — 2D calls the Y faces bottom/top,
which is the trap). A DSA operator that swapped those two axes would put a
Marshak condition where the transport has reflection and a zero-Neumann one
where it has vacuum. `cube_diffusive_yreflect.json` is deliberately
anisotropic, 20x10x5 over 2 x 1 x 0.5 with the Y faces reflective, because a
cubic box hides the swap behind its symmetry: measured by swapping the two
axes in `DSAPrecon::create` by hand, it cost 10 -> 13 iterations there under
Dirichlet-cell (and 12 -> 15 on the Z-reflective mirror of the same file), so the
pin at 11 then failed on it. The file took 12 under the first ghost-flux cut and takes
9 since its reflective faces became face couplings (2026-09-26), pinned at 9; the swap
has not been re-measured there, so whether it still fails the pin is open.

The two 2D generality rows pin the CLI rather than a regime. The shell is added
to the composite BEFORE `KSPSetFromOptions`, so the paper's additive
combination is just `-pc_composite_type additive` (slower here — 14 against 11,
23 under Dirichlet-cell — which is why multiplicative is the default order, but it must keep working),
and the inner diffusion solve is reachable under its own prefix, so
`-dsa_ksp_type cg -dsa_ksp_max_it 5` replaces the single PCGAMG application
with five CG iterations. That buys nothing on this problem (11 either way), and
that is the point: the default inexact solve is already enough.

Until 2026-09-25 these pins carried one iteration of deliberate slack (measured
count + 1), because PCGAMG was new to this test matrix and its aggregation looked
like the most arch-sensitive thing in the suite; CI ran green with it. The ghost-flux
re-pin put them on the local opt count like every other pin, so a DSA recipe is the
first place to look if CI asks for +1.

**`-precon_stream` + `-precon_dsa` buys nothing, but no longer breaks
anything.** In 1D and 2D the combination does not converge in 300 iterations — but
neither does `-precon_stream` alone on those files. A streaming-only pmat
against `Sigma_t 100` is the pre-existing strong-removal problem (the Phase 5
open question in `TODO.md`), not something DSA made worse or was expected to
fix; the correction is being added to a preconditioner that is already failing
on the hyperbolic part. 3D is the one case where the two can be told apart, and
it was worth knowing about: under Dirichlet-cell vacuum `cube_diffusive.json`
with `-precon_stream` alone converges in 35 (47 before the isotropic-source
change), and adding `-precon_dsa` to it does **not** converge in 300. Under the
ghost-flux default both take **44** (serial and `-n 2`), with different residual
histories — the DSA stage is active, it just does not pay on a pmat with no
removal to attach to. The interaction was a Dirichlet-cell artefact, most likely
the DSA BC mask: every inflow boundary cell has its inflow ordinates masked out
of the restriction and the prolongation, so the diffusion solve sees a partial
moment there and corrects only half the angles, and a streaming-only pmat
cannot absorb that. Ghost-flux leaves the mask empty on an all-vacuum problem.
The 3D run is pinned (`-ksp_max_it 44`) as the regression test for it; the
1D/2D files are not, since they fail without DSA too.

**`-diag_scale` + `-precon_dsa` is not special-cased and not recommended.** The
two are mechanically compatible — nothing errors — but scaling the assembled
operator breaks the `R A P` consistency the `/sum_weights` prolongation scaling
relies on, so the diffusion operator is no longer the right coarse model for
the system being solved. No recipe combines them.

**Composite residual updates come from the AMAT when a DSA is present.** A
multiplicative `PCComposite` forms the residual it hands each later stage from
*pmat* by default, and pmat carries no scattering (it is the assembled
streaming/removal matrix, or a streaming-only one). So once PCAIR has inverted
it, the residual reaching the DSA shell is both tiny and free of the one thing
DSA corrects — measured on `slab_diffusive.json`, the moment reaching the
diffusion solve is ~1e-6 of the residual and the iteration count does not move
at all. `TransportSolver::create` therefore calls `PCSetUseAmat` when, and only
when, it is given a DSA, before `KSPSetFromOptions` so `-pc_use_amat false`
still wins. Without a DSA nothing changes, which is why all 24 baselines still
reproduce byte-for-byte.

### Literature benchmark pins

Four 2D problems replicate configurations from the DSA literature and pin the
heterogeneous regimes nothing above reaches. Full parameter sweeps, the source
papers' own numbers and the deviations from them live in an external report
(`dsa_benchmarks_report.pdf` + `reproduce_dsa_benchmarks.py`, kept outside this
repo; the harness also generated `box_random8.json` — its `_comment` records the
seed and draw order, so the file is reproducible from the harness alone). Each
runs both ways, reference and `-precon_dsa`, serial and `-n 2`, in
`run_tests_short_*`; all four cost ~1 s serial.

| problem | regime pinned | np=1 | np=2 |
|---|---|---|---|
| `box_crooked_pipe.json` (28x20) | discontinuous D through the harmonic face mean (Southworth et al. crooked pipe, Table I set 2) | 113 / 8 (phys D 28; Dirichlet-cell 123 / 72) | 114 (pinned 117, OpenMP CI) / 8 (phys D 28; Dirichlet-cell 95 / 69) |
| `box_layers.json` (40x40) | alternating thick/thin layers (the Warsa mixed regime) | 24 / 11 (phys D 12; Dirichlet-cell 25 / 13) | 24 / 11 (Dirichlet-cell 25 / 12) |
| `box_lattice.json` (56x56) | scattering ratio exactly 1 with painted pure absorbers | 9 / 5 | 9 / 5 |
| `box_random8.json` (32x32) | fixed-seed blockwise random thick/thin/absorber mix | 19 / 7 (phys D 8; Dirichlet-cell 27 / 11) | 19 / 6 (phys D 8; Dirichlet-cell 27 / 11) |

Counts are reference / `-precon_dsa`, measured on the local **opt** arch under the
ghost-flux default (2026-09-25) and pinned exactly; until then the pins sat at
measured + 1 like the other DSA pins, and CI ran green with that. Under
Dirichlet-cell the crooked pipe was the one file whose reference count moved
with the rank count (123 vs 95 — PCAIR on a strongly heterogeneous operator);
under ghost-flux the two are 113 and 114, and its DSA count is 28 at both.
The DSA column was re-measured 2026-09-27 for the discretisation-consistent D ("phys
D" is the count it replaced).

The crooked pipe's counts were re-measured on 2026-08-04 when inflow went per
face: the file used to be driven by a painted unit-`Source` strip in the first
pipe column (a material `pipe_src`, retired with its paint box), because UBOLT
had only one global inflow value and could not drive one face. It is now driven
the way the paper does it — an inward isotropic source at the inlet, here
`"left": {"type": "vacuum", "inflow": 1.0, "window": [2.0, 3.0]}`, the pipe
mouth only. `dy = 5/20 = 0.25`, so the window covers exactly the four boundary
cells `j = 8..11` (centres 2.125–2.875) the strip covered. The magnitude is
only relative in a linear fixed-source solve, so `1.0` angle-integrated is the
faithful replacement for the retired unit angle-integrated `Source` strip.

### DSA with voids

Until 27 Sep 2026 `-precon_dsa` refused any cell with `Sigma_t <= 0` (the "void
guard"; `D = 1/(3 Sigma_t)` is undefined there). Then those cells were MASKED out
of the correction, per group, on every backend and order (`DSAPrecon`'s header
has the design): identity diffusion rows (V I on the plex), nothing restricted
or corrected there, and a face into the void a Marshak / MIP-vacuum face for its
neighbour. Under the consistent D (the default, see above) that face is the
vacuum boundary face exactly, blend included - `(D_c^p + (m h)^p)^(1/p)` on
the real cell's side - and nothing crosses it: the void cell's D = 0 flag is
tested BEFORE any blend, so no `m h` coupling reaches a masked cell and the
matrix stays SPD. The threshold is `-dsa_void_sigma_t` (default 0). With no void
cell the masked code runs the unmasked arithmetic: every `-precon_dsa` recipe in
the Makefile (61, after merging the consistent D), pin-lifted with
`-ksp_monitor`, gave output identical to main's.

Since 27 Sep 2026 the voids are BRIDGED by default instead (`-dsa_void_bridge 0`
is the mask above): a void cell stays in the diffusion operator with no
absorption and the free-flight `D = L / 3`, `L = 4 V / S` the group's voids'
mean chord (S: faces onto material or a vacuum boundary, not reflective ones),
and at DG1 the faces touching a void take the harmonic-D weighted interior
penalty (`DSAPrecon`'s header has the design and why). An all-void group, or
one whose bridged operator would be pure Neumann, falls back to the mask. With
no void the arithmetic is unchanged: all 63 no-void `-precon_dsa` recipes,
pin-lifted with `-ksp_monitor`, identical to main's (bitwise residuals).

The void files are the diffusive ones (ten mean free paths per cell, ratio 0.99,
S4, all faces vacuum) with a `Sigma_t = 0` region painted in, run every way
(opt; np=1 / np=2):

| problem | void | no DSA np=1 | bridged DSA | masked DSA | bridged / masked, physical D | void-free DSA |
|---|---|---|---|---|---|---|
| `slab_void_gap.json` (100) | x in [4, 6], splits the slab | 23 | 6 / 6 | 6 / 7 | 10 / 10 vs 12 / 13 | 5 |
| `box_void_channel.json` (50x50) | channel in from the left face | 26 | 6 / 6 | 8 / 8 | 11 / 11 vs 12 / 12 | 5 |
| `cube_void_duct.json` (10^3) | 2x2-cell duct in from the left face | 21 | 5 / 5 | 6 / 6 | 8 / 9 vs 9 / 9 | 5 |
| `plex_box_void_channel.json` (quads, DG0) | as the box | 25 (np=2 26; pinned 26, gnu_opt CI) | 6 / 6 | 8 / 8 | 11 / 11 vs 12 / 12 | 5 |
| `plex_box_void_channel_dg1.json` (quads, DG1) | as the box | 34 | 10 / 10 | 10 / 10 | 10 / 10 vs 10 / 10 | 6 |

"Bridged" is the default, "masked" `-dsa_void_bridge 0`, both under the default
consistent D; "physical D" is `-dsa_consistent_d 0` (not pinned; DG1 has no
blend, so its columns agree). Pinned at the local opt counts: the bridged
serial and np=2, the masked serial (np=2 of the no-DSA references not pinned).
Bridging takes the channel and the duct within one of the void-free count;
at DG1 it ties the mask with the one GAMG V-cycle, and wins with a stronger
inner solve (LU: 10 -> 7; two V-cycles 8, `-dsa_pc_gamg_threshold 0.05` 9,
the mask 10 under all three). For scale: the channel
with an unmasked tiny `Sigma_t` (1e-3, the old workaround) takes 7, and masking
that one with `-dsa_void_sigma_t 1e-2` 8 (physical D: 12 both ways) - the
workaround's thin-region coupling is worth one iteration here, which is what a
void-bridging operator had to beat - bridging takes 6. The mask does not couple
the regions a void separates (the slab gap's two halves each see a Marshak
face); that coupling is left to the transport stages.

How the bridged D was chosen, from fixed `-dsa_void_d` sweeps (0.01 to 1e5,
serial, rate = mean residual reduction per iteration): at DG0 the best D grows
with the void's width - the channel wants ~0.2 at height 0.2 and ~1 at 0.6 and
1.4, the duct ~0.3 - while the slab gap wants D large (in P1 a planar gap is a
perfect conductor); every count is flat within a factor ~3 of `L / 3`
(0.13 / 0.34 / 0.67 on those channels), and D -> infinity (the tiny-`Sigma_t`
workaround) costs one on the channel. At DG1 plain MIP with the void in got
worse (10 -> 13: its arithmetic penalty welds the material to the void); the
weighted penalty fixed it. Neither masking the void's slope nodes nor its
whole restriction / prolongation changed a count.

A group void EVERYWHERE is now fully masked, so the correction is zero and the
count is the no-DSA one: `slab_st0 -precon_dsa` is pinned at 1, and
`slab_decades4_stream0 -precon_dsa` (void top group) at 5, its groups taking
1, 5, 5, 5 against 1, 8, 14, 22 without DSA (physical D: 1, 5, 6, 11).

(The `verify_plexk` check 12 and 13 paragraphs that stood here are in `docs/dev/verification.md`, under "Unstructured DG verification".)

The DSA (bridged by default, or masked) also composes with the reference-shifted pmat on the same void
files, since 27 Sep 2026 (`-matfree_removal -precon_ref_shift -precon_dsa`, the
same 6 / 6 / 5 / 6 / 10 as the full pmat with the bridged DSA): see "Voids" under the
reference-shifted pmat below.

(The "Checks that are not recipes" guard that closed this section is in `docs/dev/verification.md`.)

## The reference-shifted streaming pmat (`-precon_ref_shift`) (Aug 2026 to 2026-09-27)

`-matfree_removal` leaves the assembled matrix carrying streaming ONLY, and the
pmat defaults to it. That is fine while the removal is weak and useless once it
is strong: PCAIR is then set up on something the solve is not.
`-precon_ref_shift` puts a representative removal back — the pmat becomes
`L + alpha_g * D_ref`, with `D_ref` group 0's per-cell `Sigma_t` and `alpha_g`
the log-mean of `Sigma_t(g)/Sigma_t(0)` over the mesh (on a problem with voids,
the first group with the same void cells, over its non-void cells - see
"Voids" below) — in `k` shared copies,
`k` from `-precon_ref_k` or from the default rule in `RefShiftPmats`. It is off
by default, so every recipe, count and baseline above is untouched; these are
NEW pins. It requires `-matfree_removal` (a checked error otherwise), and `k`
pmats means `k` PCAIR hierarchies, which is the mode's whole cost.

**The claim is an identity, not an improvement.** When the alphas are covered
exactly — one bin per distinct ratio, which the driver reports on stderr as
`worst mismatch 1` — `alpha_g * D_ref` IS `Sigma_t(g)` cell by cell, the pmat is
the full one, and the run reproduces the DEFAULT (full-pmat) iteration counts
exactly. Out of `k` matrices assembled once, with no assembly anywhere in the
group sweep. Six problem files say so independently, under Dirichlet-cell and again
under the ghost-flux default (2026-09-25, local opt; Dirichlet-cell in brackets where it
moved):

| exact-coverage identity | ref-shift count | the full-pmat count it reproduces |
|---|---|---|
| `slab_st2.json` (1 group, so k = 1 always) | 6 | 6 |
| `slab_mg4_t05.json`, `-precon_ref_k 2` (two distinct ratios) | 6, 6, 6, 6 | 6, 6, 6, 6 |
| `slab_decades4.json`, `-precon_ref_k 4` | 4, 8, 14, 22 (Dirichlet-cell 20) | 4, 8, 14, 22 |
| `box_decades4.json`, `-precon_ref_k 4` | 4, 6, 15, 29 (Dirichlet-cell 30) | 4, 6, 15, 29 |
| `slab_diffusive.json` (1 group), no DSA / `-precon_dsa` | 23 / 11 (Dirichlet-cell 20 / 11) | 23 / 11 |
| `box_diffusive.json` (1 group), no DSA / `-precon_dsa` | 29 / 11 | 29 / 11 |

Mind the pc side: these run with the DEFAULT (left) preconditioning, not the
`-ksp_pc_side right` every `-precon_stream` and `-matfree_removal` recipe
carries. That is the point — the pmat is a full one again, so the comparison is
with the full-pmat recipes, not with the streaming-only ones.

### The decades problem files
`slab_decades4.json` and `box_decades4.json` are the new problem files, sharing
`materials/decades4.json`: four groups at `Sigma_t` 0.1, 1, 10 and 100 with
within-group scattering ratios 0.5, 0.6, 0.9 and 0.99 and downscatter into the
next group down only (so the group sweep is still exact), over cells of width
0.1 — 0.01, 0.1, 1 and 10 mean free paths per cell. A 5x density-scaled copy of
the material is painted over a block in the middle, so the problem is
heterogeneous while every group RATIO stays identical in both materials:
`alpha_g * D_ref` is exact per group there, which no scalar shift could be. The
1D file is 100 cells over a length of 10, the 2D one 30x30 over 3x3 — the same
physics one axis wider, in the way the diffusive trio is.

The alphas are 1, 10, 100 and 1000, three decades, which is the point: covering
them exactly costs four hierarchies and the default rule settles for two.
Per-group counts, re-measured 2026-09-25 on the local opt arch under the ghost-flux
default, the Dirichlet-cell counts in brackets where they moved. Serial and `-n 2` agree
on every row measured at both (all but the bare-L and k = 1 rows) except the 2D k = 2
DSA row's third group (9 serial, 8 at `-n 2`); under Dirichlet-cell the exception was
the DSA runs' third group (6 serial, 5 at `-n 2`). The two bare-L rows without DSA run
`-ksp_pc_side right` as their Dirichlet-cell measurement did; the bare-L + DSA row runs
right-preconditioned too:

| `slab_decades4.json` | per-group iterations |
|---|---|
| default pc (full pmat, the reference) | 4, 8, 14, 22 (Dirichlet-cell 4, 8, 14, 20) |
| `-matfree_removal` alone (bare streaming pmat) | 9, 25, 172, **DIVERGED_ITS (300)** (Dirichlet-cell 10, 26, 208) |
| `-precon_ref_shift -precon_ref_k 4` (exact) | 4, 8, 14, 22 (Dirichlet-cell 4, 8, 14, 20) |
| `-precon_ref_shift` (the default rule: k = 2, worst mismatch 3.16) | 13, 11, 28, 44 (Dirichlet-cell 13, 11, 26, 42) |
| `-precon_ref_shift -precon_ref_k 1` (one shared pmat, mismatch 31.6) | 94, 21, 21, 68 (Dirichlet-cell 115, 22, 23, 76) |
| `-precon_dsa` on the full pmat | 4, 4, 5, 11 (Dirichlet-cell 4, 4, 6, 11) |
| `-matfree_removal -precon_dsa` (bare L + DSA) | 11, 30, **DIVERGED_ITS (300)** |
| `-precon_ref_shift -precon_ref_k 4 -precon_dsa` | 4, 4, 5, 11 (Dirichlet-cell 4, 4, 6, 11) |
| `-precon_ref_shift -precon_dsa` (k = 2) | 11, 11, 7, 10 (Dirichlet-cell 11, 11, 7, 13) |

| `box_decades4.json` | per-group iterations |
|---|---|
| default pc (full pmat, the reference) | 4, 6, 15, 29 (Dirichlet-cell 4, 6, 15, 30) |
| `-matfree_removal` alone (bare streaming pmat) | 5, 16, 82, **DIVERGED_ITS (300)** (Dirichlet-cell 6, 17, 89) |
| `-precon_ref_shift -precon_ref_k 4` (exact) | 4, 6, 15, 29 (Dirichlet-cell 4, 6, 15, 30) |
| `-precon_ref_shift` (the default rule: k = 2) | 7, 8, 27, 51 (Dirichlet-cell 7, 8, 24, 48) |
| `-precon_ref_shift -precon_ref_k 4 -precon_dsa` | 4, 4, 6, 11 |
| `-precon_ref_shift -precon_dsa` (k = 2; not a recipe) | 7, 8, 9, 10 (Dirichlet-cell 7, 8, 8, 179) |

Read each table's second row against its third: that pair IS the test. The bare
streaming pmat does not converge on the thick group at all, and the shift is
what makes the mode usable there. **The bare-L rows are deliberately NOT
recipes** — a pin whose expected outcome is divergence would have to be shaped
as a failure, and a recipe's pass/fail is its exit code. They are recorded here
instead, and they are what makes the ref-shift pins mean something.

The mismatch cost is the other thing these tables pin down: at a worst mismatch
of 3.16 the thick group costs 44 against the exact 22 in 1D and 51 against 29 in
2D (Dirichlet-cell: 42 against 20, 48 against 30), i.e. roughly a factor of two. That is worse than the campaign's 1.3-1.6x for
a factor-3 mismatch, and the difference is that the campaign measured one
mismatched group on a homogeneous problem where these have two mismatched groups
at once on a heterogeneous one at 10 mean free paths per cell.

### DSA on a shifted pmat
DSA cannot attach to a bare streaming pmat — structurally, not by degree. The
bare-L + DSA row above diverges. Where a bare streaming pmat does converge
(`cube_diffusive`, 44), adding DSA leaves it at 44: it no longer breaks the solve
as it did under Dirichlet-cell, but it has nothing to attach to (see the DSA
section). On an EXACTLY covered shifted
pmat it lands on the full-pmat-plus-DSA counts exactly, 11 on the thick group in
every table above, which is the re-attachment claim.

On a MISMATCHED shifted pmat it was fragile under Dirichlet-cell: 1D at k = 2 was
fine (13 on the thick group against the 42 it took without DSA) while 2D at k = 2
took **179** on the group where `-precon_ref_k 4` takes 11, so the DSA recipes
pinned exact coverage only. Under the ghost-flux default the fragility is gone,
and the sweep below (2026-09-25, local opt, with the pins lifted) is why the
default-k + DSA runs are now recipes too (1D pinned at 11, 2D at 10, serial and
`-n 2`). Per-group iterations, `-n 2` in brackets where it differs:

| file | k (worst mismatch) | no DSA | `-precon_dsa` |
|---|---|---|---|
| `slab_decades4` | 1 (31.6) | 94, 21, 21, 68 | 52, 12, 9, 16 |
| `slab_decades4` | 2, the default (3.16) | 13, 11, 28, 44 | 11, 11, 7, 10 |
| `slab_decades4` | 3 (3.16) | 13, 11, 14, 22 | 11, 11, 5, 11 |
| `slab_decades4` | 4 (exact) | 4, 8, 14, 22 | 4, 4, 5, 11 |
| `box_decades4` | 1 (31.6) | 27, 15, 21, 83 (82) | 20, 11, 10, 19 |
| `box_decades4` | 2, the default (3.16) | 7, 8, 27, 51 | 7, 8, 9, 10 (7, 8, 8, 10) |
| `box_decades4` | 3 (3.16) | 7, 8, 15, 29 | 7, 8, 6, 11 |
| `box_decades4` | 4 (exact) | 4, 6, 15, 29 | 4, 4, 6, 11 |

Every mismatched DSA run converges, even one shared pmat at a mismatch of 31.6,
and on the thick group mismatched + DSA matches or beats exact coverage + DSA.
(k = 3 is a binning TIE on these files: the log-alphas are exactly equally
spaced, so any adjacent pair can share the one merged bin at the same worst
mismatch. As first measured, last-bit rounding in the parallel log-mean picked
the pair, so serial and `-n 2` built different pmats — 2D serial took 4, 6, 27,
51 and `-n 2` 7, 8, 15, 29. `RefShiftPmats`' binning (`BinAlphas`) now compares widths
with a tie tolerance, and spends any bin the optimal-width greedy pass leaves
spare from the top, splitting off the highest distinct alpha. So the thin pair
shares and the thick groups are exact on every rank count, and the rows above
are that binning. k = 3 on `box_decades4` is pinned at 29, serial and `-n 2`,
and the old rounding-decided choice fails that pin serially.) The `mg4_t05` files have at most a 1.12
mismatch and DSA takes 5-6 in 1D and 4-5 in 3D at every k. The 179 lived in
the Dirichlet-cell boundary rows, the same DSA-mask artefact as the
`-precon_stream -precon_dsa` divergence (see the DSA section).
`slab_decades4_stream0` + DSA was a clean error by design until the DSA's void
masking (27 Sep 2026); it now runs, the void group fully masked (see "DSA with
voids").

### Streaming-only groups
A group whose `Sigma_t` is identically zero has no ratio to any reference — and
needs none, because its operator IS the bare streaming matrix. `RefShiftPmats`
gives every such group one shared unshifted bin whose pmat is the streaming
matrix itself (reference-counted, not copied — it costs no memory beyond its
hierarchy), picks the reference as the first group with removal, and bins the
rest as usual. A group that is zero in only SOME cells was refused until 27 Sep
2026 - it is the "Voids" case below now.

`slab_decades4_stream0.json` is `slab_decades4.json` with the top group made
pure streaming (`Sigma_t[0] = 0` in both materials, and with it every scatter
out of that group — a zero total admits none, so group 0 is a decoupled
streaming solve):

| `slab_decades4_stream0.json` (ghost-flux; Dirichlet-cell in brackets) | per-group iterations |
|---|---|
| default pc (full pmat, the reference) | 1, 8, 14, 22 (1, 8, 14, 20) |
| `-matfree_removal` alone (bare streaming pmat, left-preconditioned) | 1, 22, 111, **DIVERGED_ITS (300)** (1, 22, 116) |
| `-precon_ref_shift` (default rule — exact here) | 1, 8, 14, 22 (1, 8, 14, 20) |

Group 0 solves in one iteration either way — the bare streaming pmat IS its
exact operator — and the reference falls to group 1, whose three alphas span
two decades, within the default rule's exact reach. The single-group corner is
`slab_st0.json`: every group streaming-only, no reference at all, one unshifted
bin, and the count must equal the matrix-free run without the flag (same pmat,
same everything).

### Voids
Since 27 Sep 2026 `RefShiftPmats` takes a group that is void in some cells.
The groups are sorted by SUPPORT - the cells where `Sigma_t > 0` - into
classes, each with its own reference (its first group) and its own binning
(`-precon_ref_k` is per class); the empty support is the streaming-only bin
above. The reference is zero in the class's voids, so the shift is too: a void
cell's pmat row is the bare streaming row, which is exactly the operator's row
there, and the log-mean runs over the support only. A void in every group - a
void material, the usual case - leaves one class, so exact coverage is still
the full pmat, voids included. With no void there is one class over the whole
mesh and the arithmetic is unchanged: every pre-existing ref-shift recipe's
residual history (35 recipes, serial and `-n 2`) is identical to main's.

Why classes, not one reference with the voids masked out of the log-mean:
`box_decades4_void.json` paints a block transparent in group 0 only
(`materials/decades4_void.json`'s material 3). Forced onto group 0's reference,
groups 1-3 get no removal in that block — 10 mean free paths per cell in the
thick group — and take 4, 9, 48, **801** where the classes take 4, 6, 16, 29
(the full-pmat counts).

(The `-check_ref_shift` paragraph that stood here is in `docs/dev/verification.md`.)

Counts (opt; per-group for the decades files, the maximum being the pin; `-n 2`
in brackets where it differs). "bare" is `-matfree_removal` alone,
left-preconditioned. The DSA columns are the default, bridged DSA (re-measured
after rebasing onto void bridging; under the mask, `-dsa_void_bridge 0`, the
single-group rows take 6 (7) / 8 / 6 / 8 / 10 and the decades exact rows
4, 5, 7, 8):

| problem | default pc / + DSA | bare | ref-shift | ref-shift + DSA |
|---|---|---|---|---|
| `slab_void_gap` | 23 / 6 | 277 (298) | 23 | 6 |
| `box_void_channel` | 26 / 6 | DIVERGED (300) | 26 | 6 |
| `cube_void_duct` | 21 / 5 | 45 | 21 | 5 |
| `plex_box_void_channel` | 25 (26) / 6 | DIVERGED (300) | 25 (26) | 6 |
| `plex_box_void_channel_dg1` | 34 (33) / 10 | DIVERGED (300) | 34 (33) | 10 |
| `slab_decades4_void`, k = 4 (exact) | 5, 8, 14, 23 / 4, 4, 5, 6 | 10, 26, 119, DIVERGED | 5, 8, 14, 23 | 4, 4, 5, 6 |
| `slab_decades4_void`, default k = 2 (3.16) | | | 14, 12, 29, 44 | 11, 11, 7, 6 |
| `slab_decades4_void`, k = 1 (31.6) | | | 90, 21, 21, 68 | 51, 12, 10, 18 (17) |
| `box_decades4_void`, default (4 bins, 2 classes, exact) | 4, 6, 16, 29 / 4, 4, 5, 6 | 5, 16, 75, DIVERGED | 4, 6, 16, 29 | 4, 4, 5, 6 |
| `box_decades4_void`, k = 1 (a bin per class, 10) | | | 4, 35, 16, 68 | 4, 21, 5, 10 |
| `plex_decades4_void`, default (exact) | 4, 6, 15, 27 / 4, 4, 5, 6 | 5, 16, 75, DIVERGED | 4, 6, 15, 27 | 4, 4, 5, 6 |
| `plex_decades4_void`, k = 1 (10) | | | 4, 34, 15, 67 | 4, 21, 5, 10 |

Every ref-shift column reproduces the default-pc column exactly where the
coverage is exact, with and without DSA, serial and `-n 2`, so the void-bridged
(and the masked) DSA composes with it unchanged. The void costs the mismatched runs nothing
either: `slab_decades4_void` at the default k takes 44 like `slab_decades4`.

### Pins and slack

| recipe | np=1 | np=2 | pin |
|---|---|---|---|
| 1D `slab_st2`, ref-shift (1 group) | 6 | — | 6 |
| 1D `slab_mg4_t05`, `-precon_ref_k 2` | 6 | 6 | 6 |
| 1D `slab_decades4`, `-precon_ref_k 4` | 22 (Dirichlet-cell 20) | 22 | 22 |
| 1D `slab_decades4`, default k | 44 (Dirichlet-cell 42) | 44 | 44 |
| 1D `slab_decades4`, `-precon_ref_k 4 -precon_dsa` | 11 | 11 | 11 |
| 2D `box_decades4`, `-precon_ref_k 4` | 29 (Dirichlet-cell 30) | 29 | 29 |
| 2D `box_decades4`, default k | 51 (Dirichlet-cell 48) | 51 | 51 |
| 2D `box_decades4`, `-precon_ref_k 4 -precon_dsa` | 11 | 11 | 11 |
| 2D `box_diffusive`, ref-shift, no DSA | 29 | — | 29 |
| 2D `box_diffusive`, ref-shift + DSA | 11 | — | 11 |
| 1D `slab_decades4_stream0`, default k | 22 (Dirichlet-cell 20) | 22 | 22 |
| 1D `slab_st0`, ref-shift (all streaming-only) | 1 | — | 1 |
| 1D `slab_void_gap`, ref-shift (+ `-check_ref_shift`) / + DSA | 23 / 6 | — / 6 | 23 / 6 |
| 2D `box_void_channel`, ref-shift / + DSA | 26 / 6 | — / 6 | 26 / 6 |
| 3D `cube_void_duct`, ref-shift (+ `-check_ref_shift`) / + DSA | 21 / 5 | 21 / — | 21 / 5 |
| plex `plex_box_void_channel`, ref-shift + DSA | 6 | 6 | 6 |
| plex `plex_box_void_channel_dg1`, ref-shift + DSA | 10 | — | 10 |
| 1D `slab_decades4_void`, `-precon_ref_k 4` (+ `-check_ref_shift`) | 23 | — | 23 |
| 1D `slab_decades4_void`, default k / + DSA | 44 / 11 | 44 / — | 44 / 11 |
| 2D `box_decades4_void`, default k (+ `-check_ref_shift`) / + DSA / k = 1 | 29 / 6 / 68 | 29 / — / — | 29 / 6 / 68 |
| plex `plex_decades4_void`, default k (+ `-check_ref_shift`) / + DSA | 27 / 6 | 27 / — | 27 / 6 |

Every pin sits on the local opt count since the 2026-09-25 ghost-flux re-pin. Before it,
the exact-coverage 1D pins sat on the measured count and everything else carried one
iteration of slack: the DSA rows for the PCGAMG reason the other DSA pins did, the 2D
rows for the reason the other 2D pins did, and the two default-k rows because a PCAIR
hierarchy set up on a deliberately mismatched operator looked like the most
arch-sensitive setup in the suite. These are the rows to check first if CI asks for +1.

The parallel variants are not just duplicates here. The alphas are a log-mean
over every cell, so they come off an MPI reduction, and the binning is then
computed redundantly on every rank; a `-n 2` run that did not reproduce the
serial count would mean the ranks had disagreed about which pmat a group
belongs to.

(The "Checks that are not recipes" guards that closed this section are in `docs/dev/verification.md`.)

## Source and inflow conventions (Aug 2026)
Both rhs knobs are **isotropic, angle-integrated strengths**, divided by the
quadrature's `sum_weights` on their way onto the ordinates — `/2` in 1D, `/4 pi` in 2D
and 3D — so the same number means the same physics in every dimension. A material's
`Source` is written that way by `UboltFillSource`; a vacuum face's `inflow` is written
that way by the backend, into the per-row Dirichlet values `UboltFillInflow` lays onto
b. Until Aug 2026 `Source` was the literal per-angle rhs entry; the 2026-08-02 baseline
re-capture (the six mg t05/stream logs) and the pin re-measure above are that change
landing.

`inflow` followed a month later. It was one GLOBAL, PER-ANGLE value (a `VecSet` over
the whole rhs, default 1.0), and testing.md used to record that as deliberate — the
incoming ANGULAR flux is per-ordinate by nature. That stance fell when inflow went per
face (2026-08-04): a per-face knob sitting next to a per-face `Source` with the
opposite scaling is an inconsistency nobody can carry in their head, and the papers the
DSA benchmarks reproduce all prescribe an isotropic incident source, which is the
angle-integrated quantity. The top-level `"inflow"` key is gone; a file carrying it
errors with a migration message.

The migration was byte-for-byte. `sum_weights` is the quadrature's exact analytic
measure, not a floating sum (`set_weights(w_h, 2.0)` in 1D, `set_weights(w_h,
4.0 * PETSC_PI)` in 2D and 3D), so writing the exact double of that measure into every
previously-default file makes `inflow / sum_weights` come back to exactly 1.0 — the old
per-angle default — and the rhs is bitwise what it was. Every file that had relied on
the default got `{"type": "vacuum", "inflow": 2.0}` (1D) or `{"type": "vacuum",
"inflow": 12.566370614359172}` (2D and 3D, the round-trip decimal of `4.0 * PETSC_PI`)
on every vacuum face; the cold files (`inflow: 0.0`) simply dropped the key, since 0 is
the new default. Verified by re-running every problem file's `-ksp_monitor` log before
and after and diffing: identical everywhere except `box_crooked_pipe.json`, the one
deliberate physics change (below).

The retired `slab_1dk` had neither knob — its source and inflow were one per-angle
`VecSet(b, 1.0)` by design — and its baselines survive it because `Source 2.0` with
`inflow 2.0` rebuilds that rhs exactly (the 1D single-group problem files all carry
that pair), which was verified byte-for-byte on the unification and again on this
migration.

Two old recipes died with the option surface, both strictly subsumed: the "runtime
phase space sizes" slab run (every size is runtime-from-file on every run now) and the
"multigroup with 1 group" run (a 1-group file IS the single-group path in the unified
driver — there is no separate code path left to compare).
