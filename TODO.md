# UBOLT roadmap

Full plan and architecture rationale: see the approved plan (design discussion July 2026).
Each phase is a reviewable unit with its own verification. Do not start a phase before the
previous one's verification has passed and been reviewed.

## Phase 0 — Scaffolding + baseline capture (no behavior change)
- [x] Directory tree, top Makefile (library skeleton), tests/Makefile (PFLARE-style recipes)
- [x] Bookkeeping: CLAUDE.md, AGENTS.md, TODO.md, docs/dev/{testing,kokkos}.md
- [x] Move unmodified source to tests/slab_1dk.kokkos.cxx, builds as tests/slab_1dk
- [x] Capture baselines: {default pc, precon_stream} x {max_exponent 0, 2} x {np 1, 2}
      x {diag_scale off, on} into tests/baselines/; iteration table in docs/dev/testing.md
- [x] Pin per-recipe -ksp_max_it to baseline counts in tests/Makefile
- Verify: `make check` and `make tests` pass; baselines committed.
- Findings during capture (see docs/dev/testing.md): (i) parallel out-of-bounds bug in the
  matrix-free scatter (global N_CELLS used on local arrays) — np=2 default/me=2 goes NaN;
  fix scheduled as its own commit at the start of Phase 1a; (ii) diag_scale + strong
  removal is pathological (6305 its / divergence) — excluded from pass/fail recipes.

## Phase 1 — Pure refactor into libubolt (identical numerics)
- [x] 1a-pre (own commit): fix the parallel scatter out-of-bounds bug — ShellMatMultApply
      must reshape/loop over LOCAL cells (local_rows/N_ANGLES), and the sigma/scalar_flux
      views must be sized/filled with local counts; then re-capture the invalidated np=2
      default/me=2 baselines. Done: np=1 bitwise identical, only the two NaN logs changed
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
  {default pc, precon_stream} x {me 0, 2}; the 4-group zero-transfer log is four
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
  xsections (the tables support it, only the constant setters are written); promoting the
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
- [ ] StructuredFD2D (4-slot rows), 2D angle quadrature table
- Verify: pure-streaming closed-form check; small-grid MatComputeOperator of the shell vs
  host reference (1e-12); pinned PCAIR iterations. (No hand-layout twin in 2D: DMDA's
  PETSc ordering differs from natural ordering — and after Phase 3b there is no hand
  layout in 1D either, so the twin technique is retired for good.)
- Inherited from Phase 3: the DM already owns the decomposition and writes local_cells
  into the PhaseSpace, so StructuredFD2D slots into the same seam. Two things do NOT
  carry over unexamined: CheckDALayout's angle-fastest/contiguous assert (2D ordering is
  the thing that differs, so it has to be rewritten, not copied), and create_matrix's
  hand-built COO preallocation (a 2D stencil is 5 points, the operator 4 slots + diagonal).

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
- [ ] Introduce thin abstract Discretisation base (only now that a second backend exists)
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
      converges faster (me=2: 10 -> 5 its, streaming pmat me=2: 16 -> 10). New counts in
      `docs/dev/testing.md`; recipes re-pinned. diag_scale + me=2 is still pathological.

## Research notes
- **Single assembled streaming matrix across all groups** (Phase 5 direction): apply
  removal + scatter matrix-free so only one assembled matrix is stored for all energy
  groups. Open question: an effective preconditioner for streaming-only pmat when removal
  is strong. Record iteration-count findings here as Phase 5 runs.
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
