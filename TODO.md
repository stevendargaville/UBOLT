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
- [ ] n_groups in PhaseSpace; per-group xsections as 2D LayoutRight views
- [ ] One Vec per group, outer group loop (Gauss-Seidel, downscatter-only first);
      per-group values-refill via assemble(), no re-preallocation
- [ ] Driver tests/slab_1d_mgk.kokkos.cxx with -n_groups
- Verify: -n_groups 1 reproduces Phase 1 exactly; identical groups + zero transfer
  reproduce single-group answer per group; pinned per-group iterations.

## Phase 3 — DMDA adoption (behavior-preserving)
- [ ] StructuredFD1D constructor from 1D DMDA (dof=n_angles, stencil 1); extraction to the
      same CooPattern/BoundaryInfo; kernels unchanged; -use_dm option, hand layout stays
- Verify: -use_dm vs hand layout bitwise-compare residual histories, np=1,2.
  (1D DMDA ordering coincides with natural ordering; assert dof-interleaving in setup.)

## Phase 4 — 2D structured (DMDA 2D)
- [ ] StructuredFD2D (4-slot rows), 2D angle quadrature table
- Verify: pure-streaming closed-form check; small-grid MatComputeOperator of the shell vs
  host reference (1e-12); pinned PCAIR iterations. (No hand-layout twin in 2D: DMDA's
  PETSc ordering differs from natural ordering.)

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
- [ ] Pn / wavelet angular discretizations (sibling structs to SNQuadrature + own terms)
- [ ] Performance passes (kernel fusion in the MatShell loop, avoid the -precon_stream
      MatDuplicate)

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
  MatNest of per-angle streaming blocks each with its own AIR — plausible PCAIR win,
  DofOrdering enum reserves the decision.
- **Latent bug fixed in Phase 1a**: PETSC_DECIDE row split can land mid-cell (np=3 with
  1000x4); the decomposition is decided in cells (PetscSplitOwnership over n_cells) from
  Phase 1a on. np=1,2 are unaffected (the splits coincide), np=3 now converges in 10 its
  matching serial where the pre-refactor binary did not.
