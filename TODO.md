# UBOLT roadmap

Full plan and architecture rationale: see the approved plan (design discussion July 2026;
not kept in the repo - the architecture as built is described in AGENTS.md).
Each phase is a reviewable unit with its own verification. Do not start a phase before the
previous one's verification has passed and been reviewed.

This file is what is open. The finished phases' narratives, every postscript, the research
notes and the pre-restructure "Current state" paragraph are in `docs/dev/history.md`,
verbatim; "Where the notes live" at the end maps the names code comments use onto it.

## Current state (updated 2026-09-28)

### Next up
- Extensions the refactor's seams now make local (not scheduled): upscatter (an outer
  iteration over `GroupSource::add_transfer` + a norm), a time-derivative term
  (`OperatorTerm` with a diagonal half and a rhs half), Pn/anisotropic scatter (widen
  `AngularQuadrature::w_d` plus a moment-to-discrete matrix), a new direction set (a
  subclass calling `set_weights` + `set_directions`) - see docs/architecture.md, "new
  physics seams".
- [ ] Per-group cached DSA Mat/KSP instead of one refilled pair (a local change inside
  `DSAPrecon`, see its header; only worth it if a sweep revisits groups). One of the two
  open questions carried as checkboxes since 27 Sep 2026 (void masking, the other, is
  done). History: the DSA notes.
- [ ] A constant +1 iteration on simplices against quads/hexes at ratio 1 (not a creep):
  with an exact (LU) streaming inverse the counts match, so it is PCAIR's approximation.
  `-sub_1_pc_air_strong_threshold 0.25` or `-sub_1_pc_air_inverse_sparsity_order 2`
  remove it at small n but it returns at larger n (by n = 120 triangles / 24 tets), so
  the defaults stay (PR #15). A stronger cheap DSA inner solve was looked at alongside and
  closed with no change. History: Phase 6, "Iteration counts creeping up on simplices".
- [ ] `-matfree_removal`, `-precon_stream` and `-precon_ref_shift` on CG. All three
  assume a group-independent streaming matrix; with tau = min(1/sigma_t, h/zeta) the
  SUPG streaming part depends on the group. A sigma-independent tau (h/zeta only) would
  restore it at the cost of the thick diffusion limit; not done. (The driver refuses the
  three on `cg_supg`.)
- [ ] SUPG is not adjoint-consistent: no `A^T = P A P` on the CG backend, so the
  half-quadrature transposed PC (blocked, below) would not carry over to it.

### Blocked
- [ ] The half-quadrature preconditioner and an `-adjoint` path — blocked on PFLARE's
      PCAIR `PCApplyTranspose`; see the campaign branch
      (`claude/transpose-boltzmann-report-a40050`, report outside the repo in
      `transpose_solves/`). Since 26 Sep 2026 the identity it rests on,
      `A^T = P A P` on streaming + removal under ghost-flux, holds for ANY mix of vacuum
      and reflective faces in every backend (checked in verify_2dk/3dk/plexk, and by hand
      in 1D) - reflective problems no longer need special handling. On the DG backend
      (either order) the transposed half needs the cell-volume similarity
      (`unstructured_dg.hpp`): `y = V^{-1} M^{-T} (V x)`, which reduces to the plain
      transpose on a uniform mesh. The campaign measured the half-quadrature PC within 1
      iteration of the full hierarchy at ~50% setup/memory (history: Phase 5 postscript).

### Landed, newest first
One line each; the full item is in docs/dev/history.md under the phase named.
- 2026-09-29 Clean-up refactor after the merge wave (branch
  `claude/ubolt-review-refactor-c95859`): 2D/3D structured backends on one template,
  `PlexDiscretisation` base, one quadrature type for the plex backends, DSA split into a
  policy TU + one operator TU per backend (a per-face-slot d2h weight copy at create
  removed), `OperatorTerm::set_group` / `GroupSource`, one structured streaming kernel,
  every Kokkos kernel on PETSc's execution space, driver `BuildBackend`, one flux writer,
  AGENTS.md / TODO.md / testing docs restructured. Bit-identical on all 341 short-suite runs.
- 2026-09-28 CG DSA (PR #22): a vertex P1/Q1 diffusion operator on `cg_supg`, exactly
  `m R A P` of the SUPG operator in thick cells; diffusive CG 35-48 -> 5-6, void channel
  47 -> 8. Phase 6.
- 2026-09-28 CG-SUPG, Phase 6b (PR #21): `UnstructuredCG`, consistent SUPG = SAAF-tau,
  weak BCs; SAAF-LS void slab order 2.05 in 8-10 iterations flat to 640 cells. Phase 6.
- 2026-09-27 `-precon_ref_shift` on partly-void groups (PR #19): one reference per
  support class; `box_decades4_void` thick group 29 (one shared reference: 801). Phase 6,
  end.
- 2026-09-27 Void bridging in the DSA, the default (PR #18, landed with #19): free-flight
  `D = L / 3`, `L = 4 V / S`; the void recipes 6 / 8 / 6 / 8 / 10 -> 6 / 6 / 5 / 6 / 10
  serial. Research notes, DSA.
- 2026-09-27 Void masking in the DSA (PR #14): `Sigma_t = 0` cells masked instead of
  refused; `box_void_channel` 26 -> 8. Research notes, DSA.
- 2026-09-27 Discretisation-consistent DSA D (PR #17): `(D^1.5 + (m h)^1.5)^(1/1.5)`;
  every diffusive recipe 11/11/8 -> 5, crooked pipe 28 -> 8, literature crooked-pipe sets
  51-106 -> 17-21. Research notes, DSA.
- 2026-09-27 DG1-consistent (MIP) DSA (PR #13): diffusive quad box / hex cube /
  triangles / tets 34 / 32 / 40 / 43 -> 6 / 7 / 8 / 9. Phase 6.
- 2026-09-27 DSA on the plex backend (PR #12): every quad/hex twin takes the structured
  DSA count; triangles 35 -> 10, tets 31 -> 9 (physical D). Phase 6.
- 2026-09-26 `ElementBlockInverse::scale` bumps its state directly (PR #11): pinned in
  `verify_plexk` via `MatGetState`, which needs PETSc main from 3 Sep 2026. Phase 5
  postscript.
- 2026-09-26 Block-Jacobi removal stage (PR #10): bitwise the old point Jacobi at
  n_basis 1; dropping the stage on DG costs 1-2. Phase 5 postscript.
- 2026-09-26 Element-block PCAIR scaling (PR #9): DG1 quads 14 / 18 / 21 AIR levels at
  n = 25 / 50 / 100 at the default threshold, against 28 / 58 / 119. Phase 5 postscript.
- 2026-09-26 Ghost-flux reflective faces, the DG0 mixed-corner fix (PR #8): no BC rows
  under ghost-flux, `A^T = P A P` for any BC mix; ten pins moved, nine down. Phase 5
  postscript.
- 2026-09-26 Linear DG (DG1) on the plex backend (PR #7): observed order 2.02 (quads) /
  2.01 (triangles). Phase 5 postscript.
- 2026-09-25 Ghost-flux the default vacuum treatment (PR #4): crooked pipe + DSA 72 -> 28,
  every pin re-set on the local opt count. Phase 5 postscript.
- 2026-09-25 Phase 6a, DG0 on DMPlex (PR #2; hand-written Kokkos, decided 22 Sep 2026):
  quad/hex twins match the structured counts. Phase 6.
- 2026-09-25 Ghost-flux vacuum treatment, opt-in (PR #1): `A^T = P A P` exact on every
  row. Phase 5 postscript.
- Aug 2026 `-precon_ref_shift` (Phase 5 campaign strategy A): exact coverage reproduces
  the full-pmat counts on six files. Phase 5.
- Aug 2026 `-matfree_removal` + the composed diagonal: counts equal the `-precon_stream`
  twins exactly, diagonal bitwise. Phase 5.
- Aug 2026 DSA in 1D/2D/3D (`-precon_dsa`): the diffusive trio 20 / 29 / 21 -> 11 / 11 /
  10. Research notes, DSA.
- Aug 2026 generated quadratures (any even 1D order, LQn to S18): Phase 4 postscript 5.
- Aug 2026 problem files + one driver `transportk`, per-face inflow with a window,
  isotropic source, materials, reflective BCs, VTK output: Phase 4 postscripts 1-4.
- Aug 2026 and before: Phases 0-4.5 (baselines, libubolt, multigroup, DMDA, 2D, 3D) and
  the upwind-sign and scatter-mask fixes (Phase 1 / Phase 4 postscripts).

### Closed questions
- CG iteration gaps against DG0 (28 Sep 2026, no code change): the diffusive quad box is
  PCAIR's accuracy (LU ties DG0), the void channel is the discretisation (SUPG streamline
  diffusion in thin cells); the CG DSA closed the diffusive one. Phase 6.
- Wang's cell size for tau (28 Sep 2026): compared, h_Omega kept (one iteration on the
  channel, less accurate on every exact-SN benchmark). Phase 6.
- Simplex iteration counts creeping up with refinement (27 Sep 2026, PR #15): PCAIR's
  row-relative R drop on the old unscaled Dirichlet-cell pmat; ghost-flux or the
  element-block scaling removes it; pinned by `plex_tet_12_absorber_dirichlet_cell`.
  Phase 6.
- Triangle L-infinity below first order (27 Sep 2026, PR #16): not a defect - PETSc's
  simplex box is a different, non-nested diagonal pattern at each n; nested refinement
  gives first order. Phase 6.
- A cheaper stronger DSA inner solve (the item PR #17 left; closed 27 Sep 2026): no cheap
  win, one GAMG V-cycle stays. Phase 6, open questions of the 27 Sep round.
- DSA on a mismatched reference-shifted pmat, and `-precon_stream -precon_dsa` diverging
  on `cube_diffusive` (2026-09-25): Dirichlet-cell artefacts of the DSA BC mask, gone
  under ghost-flux, both pinned. Phase 5.
- The reflective-row O(h) error against a full box's quadrant (26 Sep 2026): removed
  under ghost-flux, only `"dirichlet_cell"` keeps it. Phase 6.
- An effective preconditioner for the streaming-only pmat under strong removal (Aug 2026):
  answered by `-precon_ref_shift`. Research notes, single assembled streaming matrix.
- The additive DSA composite (Aug 2026): lost every one of 100+ paired comparisons,
  settled as non-viable. Research notes, DSA.
- "Scattering ratio 1 in 2D degrades on non-square grids" (Aug 2026): it was the
  scatter writing to the Dirichlet rows. Research notes.

## Roadmap
Each finished phase: one line of status, the full narrative in `docs/dev/history.md` under
the same heading.

### Phase 0 — Scaffolding + baseline capture (no behavior change)
Done: tree, Makefiles, the 24 baselines and their pins.

### Phase 1 — Pure refactor into libubolt (identical numerics)
Done (1a-pre, 1a, 1b), then the Phase 1 postscript (negative-angle upwind sign).

### Phase 2 — Multigroup with sparsity reuse
Done. Deferred out of it (carried into Phase 7 below): upscatter + the outer iteration it
needs, and promoting the group loop into a MultigroupSolver. Spatially varying xsections
landed Aug 2026 (the per-region materials postscript).

### Phase 3 — DMDA adoption (behavior-preserving)
Done (3a, 3b): the DMDA owns layout and decomposition; the hand-rolled layout is gone.

### Phase 4 — 2D structured (DMDA 2D)
Done (4a the `Discretisation` base, 4b `StructuredFD2D`), then postscripts: the scatter's
Dirichlet mask, VTK output, reflective BCs, materials, and postscripts 2-5 (isotropic
source, problem files + one driver, per-face inflow, generated quadratures). The research
notes (DSA, the Phase 5 campaign, backend research) are in history.md under Phase 4
postscript 5.

### Phase 4.5 — 3D structured (DMDA 3D) (Aug 2026)
Done.

### Phase 5 — Matrix-free removal / single-streaming-matrix experiment
Done: `-matfree_removal`, the composed diagonal, `-precon_ref_shift` (voids since 27 Sep
2026). The Phase 5 postscript - ghost-flux (default since 2026-09-25), DG1, element-block
scaling, the block-Jacobi removal stage, the scale state bump, the DG0 mixed corners - is
done; its one open item is the Blocked one above.

### Phase 6 — DMPlex backends
Done: 6a DG0, DG1, the plex DSA, the DG1 MIP DSA, the DSA's void masking and bridging,
ref-shift on partial voids, 6b CG-SUPG and the CG DSA. Open items: the simplex +1 and the
two CG items under Next up, and:
- [ ] The void-bridging DSA's leftovers (27 Sep 2026): the chord `L = 4 V / S` is one
  number per group, not per connected void; and at DG1 the bridge is held at a tie with
  the mask (10 / 10) by the one GAMG V-cycle - LU takes it to 7, two V-cycles to 8.
  History: Phase 6, open questions of the 27 Sep round, and the DSA notes.
- [ ] Whether the 3D DSA y/z face-swap still fails its pin: `cube_diffusive_yreflect`
  caught a swapped `DSAPrecon` axis under Dirichlet-cell (10 -> 13), but the swap has not
  been re-measured since the file went to 9 (reflective faces as face couplings,
  2026-09-26). History: Testing history, DSA.

### Phase 7 — deferred
- [ ] docs/dev/ci.md. The CI itself exists (`.github/workflows/ci_build.yml` builds
      `dockerfiles/Dockerfile_kokkos` on the prebuilt `stevendargaville/petsc_kokkos`
      image for the opt, debug, 64-bit and OpenMP arches, builds PFLARE main, then UBOLT
      with `-Wall -Werror`, then runs `check` + `tests`); what is missing is the write-up.
      (This item used to read "CI: clone PFLARE's docker model + docs/dev/ci.md"; the
      docker half is done.)
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
- [ ] Upscatter + the outer iteration it needs (deferred out of Phase 2). With downscatter
      only the group system is block lower triangular and one forward sweep is exact;
      upscatter will need `GroupTransfer`'s scalar-flux validity flag relaxed, since it
      wants the previous iterate's flux.
- [ ] Promote the group loop out of `tests/transportk` into a MultigroupSolver - wait for
      a second sweep strategy (deferred out of Phase 2; the driver keeps it until then).

## Parked directions
Measured but not scheduled; the numbers are in history.md's research notes (the Phase 5
campaign, Aug 2026 - every campaign number predates PRs #13-#15 and is a direction, never
a pin).
- `-precon_removal_repeat 1` (a second removal Jacobi around the AIR stage): never worse
  than plain stream, beats the full pmat on 4/7 hard benchmarks, not h-robust in the
  diffusive corner. On branch `phase5-track1`, needs a rebase and a re-measure.
- Strategy B, frozen AIR gridding seeded on L + per-group re-Galerkin (0..+2 iterations
  over seven decades of sigma_t): needs PFLARE work (a `PCAIRSetDiagonalUpdate` entry
  point, a split level-1 Galerkin, a trimmed reuse store). If it lands, k drops toward 1.
- `PCSetUseAmat` whenever the pmat is not the full operator (plain stream 85 -> 58 on the
  cube): still not the default.
- A matrix-free GMRES polynomial on the full operator (PCPFLAREINV): the GPU-cleanest
  fallback; beats PCAIR in the optically thick regime, order grows O(1/h) where streaming
  dominates.
- Per-angle AIR via strided fieldsplit: an exact tie with monolithic, a structural option
  (needs the `-pmat_block_size` knob on `phase5-step0`).
- Angle-major ordering (per-angle contiguous blocks or a MatNest of per-angle streaming
  blocks each with its own AIR): a plausible PCAIR win; `CheckDALayout` and PhaseSpace's
  `row = cell * n_angles + angle` are the places a change of ordering has to be taught.
- The DSA inner solve at scale: if it matters, a Jacobi-preconditioned polynomial inside
  PFLARE rather than scaling the DSA matrix.
- MFEM as a hybrid element-matrix source (AssemblyLevel::ELEMENT into UBOLT's own COO)
  and libCEED (parked) - both frictions recorded in the research notes.
- `-ksp_reuse_preconditioner` is silently a no-op for PCAIR (the flag never reaches the
  internal shell) - reported upstream.

## Where the notes live
Names that code comments and docs use for notes that are now in `docs/dev/history.md`:

| cited as | from | now |
|---|---|---|
| "TODO.md's DSA notes", "the Phase 4 DSA notes", "the Phase 4 postscript 5 DSA notes" | `include/ubolt/dsa.hpp`, `docs/dsa.md` | history.md, Phase 4 postscript 5 -> Research notes -> **DSA, the diffusion half of the preconditioner**; plus the Phase 6 DSA items |
| "TODO.md Phase 4 postscript 5" (the `PCSetUseAmat` measurement) | `src/transport_solverk.kokkos.cxx`, `docs/dsa.md` | history.md, Research notes, last bullet ("Measurement behind `PCSetUseAmat` under DSA") and the DSA bullet |
| "TODO.md Phase 5" (the ref-shift mismatch measurements) | `include/ubolt/ref_shift.hpp`, `src/ref_shiftk.kokkos.cxx` | history.md, Phase 5, "Measurements behind the `RefShiftPmats` defaults"; the campaign in the Research notes |
| "the Pn note in TODO.md" | `include/ubolt/sn_quadrature.hpp`, `src/sn_quadraturek.kokkos.cxx` | this file, Phase 7, the Pn item |
| "fixed Aug 2026, see TODO.md" (the scatter writing Dirichlet rows) | `tests/verify_2dk.kokkos.cxx` | history.md, "Phase 4 postscript — the matrix-free scatter ignored the Dirichlet mask" |
| "the Phase 6 item(s)", "the Phase 6a items" | `docs/dsa.md`, older text | history.md, Phase 6 |
