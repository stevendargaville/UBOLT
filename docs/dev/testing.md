# Testing in UBOLT

## Pass/fail contract
- Pass/fail is the process exit code — no output diffing, no log grepping (this must stay
  true so CI can just run `make tests`, which it does — see
  `.github/workflows/ci_build.yml`). `make` stops on the first non-zero exit.
- Every solve recipe in `tests/Makefile` pins `-ksp_max_it` to the baseline iteration
  count, so an iteration-count regression fails the run. The same recipes run on every
  CI arch (opt/debug/64-bit/OpenMP), so a pin is the max over those environments —
  today that differs from the reference count only for the two streaming-only pmat 2D
  recipes, see the 2D table.
- Multigroup caveat: `-ksp_max_it` is one option for the whole group sweep, so a
  multigroup recipe pins the **max over the groups**. A single group getting slower
  without exceeding that max will not fail the recipe — the multigroup baselines below
  are what pin the per-group counts.
- The drivers return 1 unless `KSPGetConvergedReason() > 0` (since Phase 1a), so a
  non-converged solve fails the recipe on its own — no `-ksp_error_if_not_converged`
  or `-on_error_abort` is needed for pass/fail (CI sets `-on_error_abort` in
  `PETSC_OPTIONS` anyway, belt and braces against a hang inside PETSc). Driver
  diagnostics go to stderr so they never appear in a captured baseline. The four
  non-converging `capture_baselines` lines carry `|| true`.
- Targets: `make check` (fast sanity, 6 runs) < `make tests_short` < `make tests`.
  Parallel variants use `$(MPIEXEC) -n 2` — plus one `-n 4` run, see the 2D section — and
  are skipped under MPIUNI, including `run_check`'s single parallel line.
- `tests/verify_2dk` is the exception to "pass/fail is a pinned iteration count": it is a
  discretisation check and its exit code comes from tolerances on two numerical checks
  (below). It prints what it measured against the tolerance, so a run that is drifting
  towards failure is visible before it fails.

## Baselines (tests/baselines/)
Captured `-ksp_monitor -ksp_converged_reason` logs. They are the ground truth for a
refactor: identical numerics means the residual history diffs clean against these files
(iteration counts exactly, residuals to ~1e-14).

24 logs in two families: 16 single-group (`slab1d_*`, Phase 0) and 8 multigroup
(`slab1dmg_*`, Phase 2). `make baselines` captures both. A refactor that claims to be
numerically inert must diff clean against all 24. All of them are 1D — the 2D checks are
sharper and need no baseline, see below.

Since the driver unification (Aug 2026) every capture line runs `transportk` on a
problem file (`tests/problems/`); the log names keep their `slab1d`/`slab1dmg` history.
The unification itself was verified the strong way before the old drivers
(`slab_1dk`, `slab_1d_mgk`, `box_2dk`) were deleted: all 24 logs byte-for-byte from the
new driver, plus the t0 structural identity, the painting identity at np=1 and np=4, and
the 1D/2D infinite-medium checks.

### Single-group baselines
Re-captured 2026-08-01 after the Dirichlet-row fix to the matrix-free scatter (see below).
Before that, 2026-07-31 after the negative-angle upwind sign fix, and before *that* they
came from the pre-refactor code (the single-file `UBOLTk.kokkos.cxx`); Phase 1a and Phase
1b each reproduced all 16 bitwise, which is what those phases were verified against. The
`-ubolt_coo_two_call` assembly fallback reproduces them bitwise.

- Full matrix: {default pc, `-precon_stream -ksp_pc_side right`} x {`slab_st0.json`,
  `slab_st2.json`} x {np 1, 2} x {`-diag_scale` off, on}. File naming:
  `slab1d_<default|stream>_st<0|2>_np<1|2>[_ds].log`.
- Capture is capped at `-ksp_max_it 200`: all four st=2 diag_scale configs are pathological
  (see table) and a 200-iteration residual history is a strong enough fingerprint.
  Non-converged baselines are still valid fingerprints for a refactor diff. None of those
  four converges, so all four capture lines carry `|| true`.
- Re-capture with `make baselines` — only do this deliberately (i.e. when the reference
  behavior itself is being intentionally changed), never to make a failing test pass.
- Environment matters for exact reproduction: these were captured with a debug PETSc main
  build (`arch-linux-c-debug`), gcc 13, OpenMPI, np as named. Different
  compilers/optimization may shift trailing digits; iteration counts should still match.
- np=1 and np=2 logs differ in the trailing digits (parallel reduction order); compare a
  log against the baseline for the *same* np, never across np.

## Single-group baseline iteration counts
Captured 2026-08-01, capped at `-ksp_max_it 200`. The two older columns are kept because
earlier phases were verified against them — "sign fix" is the 2026-07-31 capture, before
the scatter's Dirichlet rows were fixed, and "pre-refactor" is what the original single
file did:

| config | np=1 | np=2 | sign fix (np=1, np=2) | pre-refactor |
|---|---|---|---|---|
| default pc, st=0 | 1 | 1 | 1, 1 | 1, 1 |
| default pc, st=2 | 6 | 6 | 5, 5 | 10, 10 |
| default pc, st=0, diag_scale | 1 | 1 | 1, 1 | 1, 1 |
| default pc, st=2, diag_scale | DIVERGED_ITS (200) | DIVERGED_ITS (200) | 175, 173 | DIVERGED_ITS |
| precon_stream, st=0 | 1 | 1 | 1, 1 | 1, 1 |
| precon_stream, st=2 | 9 | 9 | 10, 10 | 16, 16 |
| precon_stream, st=0, diag_scale | 3 | 3 | 3, 3 | 3, 3 |
| precon_stream, st=2, diag_scale | DIVERGED_BREAKDOWN (150) | DIVERGED_BREAKDOWN (150) | DIVERGED_BREAKDOWN (90) | DIVERGED_ITS |

The st=0 rows are identical across all three captures, and must be: `sigma_s = 0` there,
so neither the scatter nor its Dirichlet rows exist.

History of these baselines:
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

## Multigroup baselines (Phase 2)
`slab1dmg_<default|stream>_st<0|2>_g4_t<0|05>_np<1|2>.log` — 4 groups, `t0`/`t05` being
downscatter 0.0/0.5 into the next group down (the `slab_mg4_t0.json` /
`slab_mg4_t05.json` / `slab_mg4_t05_st0.json` problem files; before the driver
unification these were `slab_1d_mgk -sigma_transfer` recipes, whose derived per-group
values are exactly what the files now tabulate). Each log is the whole group sweep: one
KSP history after another, in group order. That is what pins the per-group iteration
counts and the downscatter source arithmetic, neither of which a single `-ksp_max_it`
can reach. The six t05/stream logs were re-captured 2026-08-02 for the isotropic-source
change (below); the two t0 logs deliberately did not move.

The `t0` logs are exactly four byte-for-byte copies of the matching single-group
`slab1d_default_st2_np<n>.log`. That is the Phase 2 uncoupled-groups verification, and it
is worth re-checking as a structural property rather than only diffing the file — a change
that broke both files the same way would still diff clean. It was re-checked that way on
the 2026-08-01 re-capture, when both files moved, and again on the driver unification.
`slab_mg4_t0.json` carries `Source: [2, 2, 2, 2]`: the isotropic strength that puts 1.0
on each 1D ordinate, which is exactly the rhs the single-group problems carry — that is
what keeps the identity byte-for-byte.

| multigroup config | per-group iterations (np=1 and np=2) |
|---|---|
| default pc, st=0, transfer 0.5 | 1, 1, 1, 1 |
| default pc, st=2, transfer 0.0 | 6, 6, 6, 6 (= single group, four times; Source 2.0) |
| default pc, st=2, transfer 0.5 | 6, 6, 6, 6 |
| default pc, st=2, transfer 2.0 | 6, 6, 5, 5 (recipe only, not captured) |
| precon_stream, st=2, transfer 0.5 | 11, 11, 11, 9 |

Group 0 matches the single-group count because its rhs is just the external source (a
count match at Source 1.0, a history match at Source 2.0, see above); the later groups
carry a downscatter source as well. The last group has no outgoing transfer, so its
`sigma_t` is lower than the others' — that is why the counts are not uniform.

With `-precon_stream` the streaming-only pmat does not depend on the group, so PCAIR is
set up once for the whole sweep; with the default pc the assembled matrix is refilled per
group and PCAIR re-runs its setup each time. Both are exercised by the recipes.

## 2D verification (Phase 4)
There are **no 2D baselines**, and that is deliberate. A residual-history baseline is an
indirect fingerprint of the numerics; in 2D `tests/verify_2dk` compares the operator
itself, so it is a strictly sharper oracle and there is nothing left for a baseline to
catch. `make baselines` is still 1D + multigroup only.

`verify_2dk` runs two checks, both S2 (4 angles) and S4 (12), and exits non-zero on either:

1. **Pure-streaming closed form.** `psi = x + y` is linear, and first-order upwind
   differencing is exact on a linear function, so the *discrete* operator has the PDE's
   solution: `mu dpsi/dx + eta dpsi/dy = mu + eta`, with `psi = x + y` prescribed on the
   inflow boundaries. Two halves — the residual `||A psi_exact - b||_inf` (no solver
   tolerance in the way; ~4e-15 against a 1e-12 tolerance) and then the solve, which says
   the system really does have that solution and is nonsingular. Run on a 16x12 grid over
   a 1x2 box, so nothing about it is symmetric in x and y and a mix-up cannot hide.
2. **Shell operator vs a reference matrix.** `MatComputeOperator` on the *shell*, so the
   matrix-free scatter is included, against a matrix built entry by entry in the driver on
   a 4x3 grid — for three BC configurations apiece: all-vacuum, all-reflect, and mixed
   (left + bottom reflect), so both reflection maps and the vacuum-wins corner rule are in
   the checked matrix. Currently agrees to 0.0, i.e. bitwise, in all six runs. Serial
   only: the reference is written in the natural ordering, which is what a 2D DMDA uses on
   one rank. The reference deliberately reimplements the boundary-row classification —
   including its own search for the mirrored ordinate — rather than reading the
   discretisation's BC mask, reflect slots or reflection maps: a check that asked the code
   under test which rows it thought were boundary rows would agree with it by
   construction. It says a Dirichlet row is the identity and nothing else, and a
   reflective row is the identity plus the -1 on its mirrored angle and nothing else; the
   former is how it caught the matrix-free scatter writing to those rows, see the Phase 4
   postscript in `TODO.md`.

**Parallel.** A 2D DMDA at `-n 2` splits in one direction only, so `-n 4` is the first
decomposition with a genuinely 2D processor grid and the one where the patch-lexicographic
global numbering is least like the natural one. Both are in `run_tests_short_parallel`.

## 2D iteration counts
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
8 -> 9 and 9 -> 10.

| config | np=1 | np=2 |
|---|---|---|
| 50x50, st=2 (ratio 1) | 6 | 6 |
| 50x50, st=2, ratio 0.5 | 5 | 5 |
| 40x20 (dx != dy), st=2 | 6 | 6 |
| 80x40 (same shape, 4x finer), st=2 | 6 | 6 |
| 80x20 in a 1x0.25 box (dx == dy), st=2 | 5 | 5 |
| 50x50, st=0 (pure streaming) | 3 | 3 |
| 60x60, st=0 | — | 3 |
| 60x60, st=2 | 7 | 7 |
| 50x50, streaming-only pmat, st=2 | 9 | 9 |
| 80x40, streaming-only pmat, st=2 | 10 | — |
| 30x30 S4, st=2 | 6 | 6 |
| 60x60 S4, st=2 | 6 | 6 |

Mesh independent on every shape: 40x20 to 80x40 holds at 6, and 200x50 in the 1x0.25
box (not a recipe) is 5, the same as 80x20.

These sizes were cut on 2026-08-02 to bring CI down (see "Recipe cost" below): the
"larger grid" family went 100x100 -> 60x60, the non-square mesh-independence pair
60x30/120x60 -> 40x20/80x40 with its 2x ratio intact, and the S4 reflective run joined
the plain S4 recipe at 30x30. The counts above are the re-measured references.

A pin is the max over the reference build and the two sensitive CI arches (64-bit
indices and the OpenMP Kokkos backend), re-swept 2026-08-02 in both CI images after the
isotropic-source change. Which configs sit one above the reference count changed with
it: the two streaming-only pmat serial recipes now measure the same everywhere (so
their pins equal the table), and instead the 50x50 ratio-1 SERIAL solve takes 7 under
64-bit — pinned 7 on all three recipes that run it (plain, `-ubolt_coo_two_call`, and
the painting identity, which is the same history by construction) — while under OpenMP
the np2 streaming-only pmat takes 10 (pinned 10) and the serial large-grid S4 takes 7
(pinned 7 — that slack is kept across the 2026-08-02 resize, since the OpenMP image has
not been re-swept at 60x60). Every other recipe measures its reference count in both
images, and the resized recipes' pins are debug-arch measurements pending a sweep. The
cost is
one iteration of slack on the reference build for those five recipes — the 1D
streaming-pmat baselines still pin their counts exactly.

## Reflective boundary conditions
A problem file's `boundary_conditions` object takes `vacuum` (the default for an unset
face) or `reflect` per face. The default is bitwise the pre-reflective behaviour —
verified by re-capturing all 24 baselines when the plumbing landed — so every recipe and
baseline above is untouched, and the reflective recipes below are NEW pins, not
adjustments. Reflective rows couple the angle blocks at the boundary (the streaming-only
pmat carries the coupling too), so these counts have no reason to match their vacuum
counterparts.

The sharp oracles are the verify_2dk reference configs above and the **infinite-medium
check**: with every face reflective, a uniform source and uniform xsections, the exact
discrete solution is constant in every cell and angle — no discretisation error, so
`-check_inf_medium` compares against 1e-9 under `-ksp_rtol 1e-12` (lands at ~1e-13).
The constant is per group, by forward substitution down the sweep:
`psi_g = (Source_g / sum_weights + sum_{g'<g} Sigma_s[g'][g] psi_g') /
(Sigma_t[g] - Sigma_s[g][g])` — which is the retired `slab_1dk`'s
`1 / (sigma_t - sigma_s)` for the Source 2.0 slab files (its VecSet source was per-angle
by design; 2.0 shared over sum_weights 2 restores it) and the retired `box_2dk`'s
`1 / (sum_weights (sigma_t - sigma_s))` for the Source 1.0 box files. It is
decomposition independent, which is what makes it the parallel oracle for the reflect
partner columns.

Singularity constraint: all-reflect with a scattering ratio of exactly 1 has the
constants in the operator's kernel. Hence the all-reflect problem files carry absorption
(within-group `Sigma_s` 1.0 under `Sigma_t` 2.0 — `slab_inf_medium.json`,
`box_50_inf_medium.json`), and the multigroup reflective file keeps the right face
vacuum — the last group always has ratio 1 (no downscatter out of it).

| reflective config | np=1 | np=2 |
|---|---|---|
| 1D, left reflect, st=2 (ratio 1) | 8 | — |
| 1D, left reflect, st=2, two-call assembly | 8 | — |
| 1D all-reflect infinite medium (rtol 1e-12) | 10 | 11 |
| 1D multigroup 4 groups t05, left reflect | 8 (max over groups) | — |
| 2D 50x50, left+bottom reflect, st=2 ratio 0.5 | 6 | 6 |
| 2D 50x50 S4, left+bottom reflect, ratio 0.5 | 6 | — |
| 2D all-reflect infinite medium (rtol 1e-12) | 10 | 10 |

## Painted regions (MaterialSpec)
A problem file's `regions.paint` list paints shapes — boxes in 2D, intervals in 1D —
over the background material (see `MaterialSpec` and `docs/problem_files.md`). No paint
is the uniform problem bit-for-bit, which was verified the strong way when the plumbing
landed: everything builds its xsections and rhs through the MaterialSpec path, and all
24 baselines reproduce bitwise.

The sharp oracle is the **painting identity**: regions whose values equal the
background's must land exactly on the uniform recipe's residual history, because
painting and expansion change nothing but which table entry a cell reads. The recipe
pins the count; the bitwise history match was checked at np=1 and np=4 when the pins
were captured, and re-checked on the 2026-08-02 isotropic-source re-measure (where the
np=4 count moved 6 -> 7, uniform and painted together). It runs at `-n 4` in the
parallel suite deliberately — painting maps cell centres through the
patch-lexicographic local ordering, and the 2D processor grid is where that ordering is
least like the natural one. The heterogeneous recipes are ordinary pins (and the
absorbing block was eyeballed via `-flux_vtk`: a clear flux depression over the block,
~1.9 against ~7-10 in the surrounding medium).

The **overlap** recipes sharpen the same oracle onto paint ORDER. An extreme sourceless
material (sigma_t 100, sigma_s 0, q 0) is painted first and a copy of the background
over a box that contains it, so later-paint-wins makes the result uniform and the run
has to land on the uniform ratio-0.5 history. Both were checked bitwise against a
uniform twin of the same mesh when the pins were captured. Reversing the paint list
(letting the earlier box win) moves 2D 30x30 from 5 to 4 iterations, so the pin alone
catches it; in 3D 10^3 the count does not move at the default rtol — the history does,
and that is what the bitwise check covers. The covering box stays strictly inside the
domain on purpose: a whole-domain cover would leave the background material unused.

| painted config | np=1 | np=2 | np=4 |
|---|---|---|---|
| 2D 50x50 identity: 2 regions = background, st=2 | 6 | — | 7 |
| 2D 30x30 overlap: masked box covered by a background copy, st=2 ratio 0.5 | 5 | — | — |
| 2D 50x50 absorbing sourceless block (sigma_t 10, sigma_s 1, q 0) over st=2 ratio 0.5 | 5 | 5 | — |
| 2D 60x60 cold box: `inflow 0.0`, zero source outside a central 0.1x0.1 region | 5 | 5 | — |
| 1D multigroup 4 groups t05, double-density region 0.4-0.6 | 6, 6, 6, 6 | 6, 6, 6, 6 | — |

The cold box could not move: zero inflow leaves the painted source as the only rhs, so
the isotropic normalisation scales b uniformly, which a relative residual never sees.

## 3D verification
The 2D story, one axis wider, and the same deliberate absence: there are **no 3D
baselines** — `tests/verify_3dk` compares the operator itself, `make baselines` stays
1D + multigroup only.

`verify_3dk` runs the two checks of `verify_2dk`, both S2 (8 angles) and S4 (24 — a 3D
level-symmetric set has twice the ordinates of the same-order 2D set):

1. **Pure-streaming closed form**, `psi = x + y + z` on an 8x6x4 grid over a 1x2x3 box —
   all three extents distinct, so no axis mix-up can hide. Residual ~8e-15 against the
   1e-12 tolerance, then the solve.
2. **Shell operator vs a reference matrix** on 4x3x2, natural ordering, serial only, with
   its own independent boundary classification as in 2D. Three BC configurations apiece:
   all-vacuum, all-reflect, and mixed with reflect on **left + front + bottom** — three
   reflective faces meeting at one corner, so the checked matrix carries the single,
   double AND triple cosine flips (the triple is new behaviour 2D cannot exercise) plus
   vacuum-wins edges and corners against the other three faces. Agrees to 0.0, i.e.
   bitwise, in all six runs.

Mind the face names in 3D: `bottom`/`top` are the **z** faces (PETSc's box-mesh "Face
Sets" convention), the y faces are `front`/`back` — see `docs/problem_files.md`.

**Parallel.** A 3D DMDA splits one direction at `-n 2` and two at `-n 4`; the recipes run
the closed form and the painting identity at both, and `-n 8` (the first genuinely 3D
processor grid) was checked by hand when the backend landed but stays out of the recipes —
CI runners have few cores. The infinite-medium check is the decomposition-independent
parallel oracle, exactly as in 1D/2D (`cube_10_inf_medium.json`, ~1e-13 against 1e-9).

## 3D iteration counts
Measured 2026-08-02 on the reference build at capture. **These pins have not yet been
swept over the CI arches** (64-bit indices, OpenMP Kokkos) — a pin is the max over those,
so expect the first CI run to move a few by one, and re-sweep both images rather than
carrying slack (see the 2D section for the precedent).

| config | np=1 | np=2 |
|---|---|---|
| 10^3, st=2 (ratio 1) | 5 | 5 |
| 10^3, st=2, ratio 0.5 | 4 | 4 |
| 10x5x5 (dx, dy, dz all differ), st=2 | 6 | 6 |
| 15^3 (same shape, 1.5x finer), st=2 | 6 | 6 |
| 10^3, streaming-only pmat, st=2 | 6 | 6 |
| 10^3 S4, st=2 | 6 | 6 |
| 10^3, left+front+bottom reflect, ratio 0.5 | 5 | 5 |
| 10^3 all-reflect infinite medium (rtol 1e-12) | 10 | 10 |
| 10^3 identity: 2 regions = background, st=2 | 5 | 5 (np=4) |
| 10^3 overlap: masked box covered by a background copy, st=2 ratio 0.5 | 4 | — |
| 10^3 absorbing sourceless block over ratio 0.5 | 4 | 4 |
| 10^3 cold cube: `inflow 0.0`, zero source outside a central 0.2^3 region | 4 | 4 |
| 10^3 multigroup 4 groups t05 | 5, 5, 5, 5 | 5, 5, 5, 5 |

Every cube dropped from 20^3 to 10^3 on 2026-08-02 to bring CI down (see "Recipe cost"
below), and the mesh-independence twin from 30^3 to 15^3 to keep its 1.5x ratio; the
counts above are the re-measured references, one lower than the 20^3 ones almost
throughout. Mesh independence is now a weaker statement than it was: 10^3 to 15^3 moves
5 to 6, where 20^3 to 30^3 held flat at 6. A one-iteration move over a 1.5x refinement is
the same behaviour 2D shows across its own pair, but if the flat span is wanted back,
that costs the 27000-cell run.

The cold cube's source region had to widen from 0.1^3 to 0.2^3: at 10 cells the old
0.45/0.55 bounds land exactly on cell centres, where membership is a floating-point coin
toss. Any resize has to re-check that — a bound `b` is fragile at `n` cells whenever
`b*n` is a half-integer.

The painting identity was re-checked bitwise
against the uniform 10^3 history at np=1 when the pins were re-captured. The absorbing
block was eyeballed via `-flux_vtk` (sigma_t 10 painted mid-cube, flux ~3.3 over the
block against ~10 in the surrounding medium) — those numbers are from the 20^3 file,
before the 2026-08-02 resize, and have not been re-measured at 10^3.

## DSA (the diffusion correction, `-precon_dsa`)

`DSAPrecon` adds a third stage to the composite preconditioner: a cell-centred
diffusion operator on the same grid, restricted from and prolonged back onto the
ordinates (`include/ubolt/dsa.hpp`). It is off by default, so every recipe,
count and baseline above is untouched — the DSA recipes below are NEW pins.
**1D and 2D so far**; the 3D `create` overload lands with its own stage.

The regime it exists for is the one nothing else in the preconditioner
addresses, and which no test problem demonstrated until now. The diffusive
problem files are deliberately the same physics in every dimension — `Sigma_t
100` over cells of width 0.1, so ten mean free paths per cell, at a scattering
ratio of 0.99, every face vacuum so the diffusion operator is nonsingular
whatever the ratio: `slab_diffusive.json` is 100 cells over a length of 10,
`box_diffusive.json` 50x50 over 5x5 (S4, so 12 ordinates). Only the dimension
changes between them, which is what makes the counts comparable.

| config | np=1 | np=2 |
|---|---|---|
| 1D diffusive slab, no DSA (the reference) | 20 | 20 |
| 1D diffusive slab, `-precon_dsa` | 11 | 10 |
| 1D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 10 | 10 |
| 1D slab st=2, `-precon_dsa` | 5 | 5 |
| 2D diffusive box, no DSA (the reference) | 29 | 29 |
| 2D diffusive box, `-precon_dsa` | 11 | 11 |
| 2D diffusive box, `-precon_dsa -pc_composite_type additive` | 23 | 23 |
| 2D diffusive box, `-precon_dsa -dsa_ksp_type cg -dsa_ksp_max_it 5` | 11 | 11 |
| 2D all-reflect infinite medium, `-precon_dsa` (rtol 1e-12) | 10 | 10 |
| 2D box st=2, `-precon_dsa` | 5 | 5 |

Read each dimension's first two rows together: that pair IS the test, and
halving the count (20 to 11 in 1D, 29 to 11 in 2D) is what the correction buys.
The 2D reference count is the higher of the two and the corrected count is the
same, so what the correction removes is exactly the part that got worse with
dimension — which is the claim DSA makes.

The remaining rows say DSA does not break what already worked — 1D st=2 goes
6 to 5 and 2D st=2 7 to 5, and the infinite media still land on the exact
constant (1D 2.3e-13 serial and 7.1e-12 at np=2, 2D 3.4e-14 and 8.3e-14,
against the 1e-9 tolerance). Those runs are also the reflective-branch check:
every face reflective means every face is a zero-Neumann one in the diffusion
operator, and the singularity guard is satisfied by the files' absorption. The
2D solution is unchanged by the correction, as it must be — `box_50_st2.json`
at rtol 1e-12 agrees to 1.3e-12 relative with and without `-precon_dsa`.

The two 2D generality rows pin the CLI rather than a regime. The shell is added
to the composite BEFORE `KSPSetFromOptions`, so the paper's additive
combination is just `-pc_composite_type additive` (slower here — 23 against 11
— which is why multiplicative is the default order, but it must keep working),
and the inner diffusion solve is reachable under its own prefix, so
`-dsa_ksp_type cg -dsa_ksp_max_it 5` replaces the single PCGAMG application
with five CG iterations. That buys nothing on this problem (11 either way), and
that is the point: the default inexact solve is already enough.

**These pins have not been swept over the CI arches yet**, and unlike the 3D
pins they carry one iteration of deliberate slack (measured count + 1) rather
than sitting on the measured number: PCGAMG is new to this test matrix and its
aggregation is the most arch-sensitive thing in the suite. The table above is
the reference measurement, so a later sweep can tighten the pins onto it.

`-precon_stream` + `-precon_dsa` on the diffusive problems does not converge in
300 iterations — but neither does `-precon_stream` alone there, in 1D or 2D. A
streaming-only
pmat against `Sigma_t 100` is the pre-existing strong-removal problem (the
Phase 5 open question in `TODO.md`), not something DSA made worse or was
expected to fix; the correction is added to a preconditioner that is already
failing on the hyperbolic part. No recipe combines them.

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

### Checks that are not recipes
A recipe passes on exit 0, so the two guards are checked BY HAND:

- **Void guard**: `./transportk -problem problems/slab_st0.json -precon_dsa`
  must fail with the `Sigma_t > 0 in every cell` message —
  `D = 1/(3 Sigma_t)` is undefined in a void, and the fix is a region-masked
  diffusion operator (future work, as in the paper) rather than a fudged
  coefficient.
- **Singularity guard**: an all-reflective problem whose within-group `Sigma_s`
  equals its `Sigma_t` must fail with the pure-Neumann message. This is the same
  constraint the transport operator carries (see the reflective section above),
  which is why no shipped problem file is in that state.

## Source and inflow conventions
A material's `Source` array is an **isotropic, angle-integrated strength**:
`UboltFillSource` writes `source / sum_weights` on each ordinate — `/2` in 1D, `/4 pi`
in 2D and 3D — so the same value means the same physics in every dimension. Until Aug
2026 the value was
the literal per-angle rhs entry; the 2026-08-02 baseline re-capture (the six mg
t05/stream logs) and the pin re-measure above are that change landing.

A problem file's `inflow` prescribes the vacuum faces' incoming flux — the value the
Dirichlet rows of b carry, historically hard-coded at 1.0, which stays the default so
every single-group recipe and baseline above is untouched. It is deliberately still
per-angle: it prescribes the incoming ANGULAR flux, which is per-ordinate by nature.
The retired `slab_1dk` had neither knob — its source and inflow were one per-angle
`VecSet(b, 1.0)` by design — and its baselines survive it because `Source 2.0` with
`inflow 1.0` rebuilds that rhs exactly (the 1D single-group problem files all carry
that pair), which was verified byte-for-byte on the unification.

Two old recipes died with the option surface, both strictly subsumed: the "runtime
phase space sizes" slab run (every size is runtime-from-file on every run now) and the
"multigroup with 1 group" run (a 1-group file IS the single-group path in the unified
driver — there is no separate code path left to compare).

## Recipe cost
CI runs the whole suite on four arches in parallel (opt/debug/64-bit/OpenMP), so the wall
time is the slowest job — the debug one. The suite is what dominates it, and within the
suite 3D dominated everything else: on the reference debug build a single 20^3 cube run
cost 21-34 s against ~2 s for a 50x50 2D run, because cost grows faster than cell count
(20x10x10, at a quarter of 20^3's cells, took 3.5 s to its 21 s).

That is what the 2026-08-02 resize addressed. Measured over the 43 serial recipes of
`run_check` + `run_tests_short_serial`, on the reference debug build:

| | before | after |
|---|---|---|
| the 9 resized recipes in that set | 145 s | 19 s |
| all 43 (the rest are 1D, unchanged) | 200 s | 77 s |

`make check && make tests` end to end after the resize is **272 s** on that build. The
before-figure for the whole suite was never captured cleanly (it ran well past 20
minutes), so the 43-recipe subset above is the like-for-like measurement; the full suite
gains more than the 2.6x it shows, because it also carries the parallel duplicates of
every 3D recipe and the 30^3 mesh-independence run, the single most expensive line in it.

Keep new recipes cheap by default. A 3D recipe wants a reason to be bigger than 10^3, and
the discretisation checks (`verify_2dk`/`verify_3dk`) make their point on 4x3x2 grids —
resolution is not what a pinned iteration count or an operator comparison is testing.

### The OpenMP job and rank/thread oversubscription
The runners have 2 vCPUs and the OpenMP job sets `OMP_NUM_THREADS=2`, which is right for
a serial recipe and wrong for every `mpiexec -n 2` one: 2 ranks x 2 threads is 4 threads
on 2 cores, and the four `-n 4` recipes make it 8. Left alone that cost the job **862 s
of test time against the opt job's 53 s** — the same build, the same problems, a 16x
gap that lived entirely in the parallel recipes (the serial ones were, if anything,
faster than opt).

Two defaults compounded to produce it. OpenMPI confines each rank to a single core at
small rank counts, so a rank's two threads shared one core; and libgomp spins at
parallel-region barriers, so the descheduled thread burned that core rather than
yielding. The result is a per-kernel-launch penalty, not a per-element one, which is why
a 1D slab converging in one iteration took 36 s there against 0.2 s under opt, and why
shrinking meshes barely moved that job while it cut the debug job nearly in half.

`dockerfiles/Dockerfile_kokkos` now sets `OMPI_MCA_hwloc_base_binding_policy=none`,
`OMP_PROC_BIND=false` and `OMP_WAIT_POLICY=PASSIVE`. The job still runs 2 threads per
rank, so it still tests threaded execution under MPI; it just no longer pins them onto
one core or spins while waiting. If a future recipe set makes that job the long pole
again, the next lever is `OMP_NUM_THREADS=1`, which trades the threading coverage away.

## Adding a test problem
1. Write a problem file in `tests/problems/` (schema: `docs/problem_files.md`,
   walkthrough: `docs/problem_setup.md`), with a `"_comment"` saying what it pins.
   Values that are not exact dyadics get 17 significant digits so the file round-trips
   to the intended doubles.
2. Add invocation lines to the appropriate `run_*` recipe in `tests/Makefile`: an
   `@echo` label plus the literal `./transportk -problem problems/<file> -options`
   line, serial and `-n 2` variants, with `-ksp_max_it` pinned to the observed
   converged count.
3. No output files from any recipe: `output.flux_vtk` (and the `-flux_vtk` override)
   are fine on a problem file you run by hand but must not appear in anything a `run_*`
   recipe names — a test run leaves nothing behind.

## Adding a test driver
Should a second driver ever be needed (the next `verify_*`, say):
1. Add the executable name to `TEST_TARGETS` (and `CHECK_TARGETS` if it belongs in
   `make check`) in the top `Makefile`. The two lists are identical today; the split
   exists for the day a driver is too slow for `check`.
2. Nothing to do for `.gitignore` — `tests/*k` covers every driver binary, since every
   translation unit is a Kokkos one and the executables all end in `k`.
3. Driver rules: exit non-zero unless `KSPGetConvergedReason() > 0`; diagnostics to
   stderr so `-ksp_monitor` stdout stays capture-clean; problem definition through
   `ProblemSpec`, not new physics options.

## DMDA layout (Phases 3 and 4)
Every discretisation backend builds its layout from a DMDA, which is also what decides the
parallel decomposition. There is no second layout path and no option to select one: Phase 3a carried
a hand-rolled twin behind `-ubolt_use_dm` purely to verify the DM extraction, and 3b deleted
it. What replaced the twin as the check is `tests/baselines/` — all 24 logs reproduce
bitwise, which is what 3a and 3b were each verified against.

The DM chooses its own distribution. PETSc's default 1D DMDA split is
`M/size + ((M % size) > rank)` (`src/dm/impls/da/da1.c`), the same formula
`PetscSplitOwnership` uses, so this is the same decomposition UBOLT had when `PhaseSpace`
computed it — 3a confirmed that empirically at np=1,2,3 (including the uneven 334/333/333
split) before 3b handed the decision over.

`CheckDALayout` in `src/structured_fd_1dk.kokkos.cxx` asserts at setup that the DM's sizes
match the phase space and that its global numbering really is the angle-fastest,
contiguous one every COO index is written against. That property holds for 1D DMDA (PETSc
ordering coincides with natural ordering) but it is load-bearing, and a violation would
otherwise surface as a wrong answer rather than an error.

**2D is where that stops being free.** A 2D DMDA numbers each rank's patch contiguously
and *lexicographically within the patch*, so the global natural ordering `j * n_cells_x + i`
is wrong as soon as more than one rank splits a direction. `StructuredFD2D` therefore takes
its COO **columns** from the DM's local-to-global map (which covers the ghost nodes, so a
column can point into a neighbour's patch) rather than computing them, and its own
`CheckDALayout` asserts the weaker property that actually holds: sizes; angle-fastest
contiguous dof; and this rank's owned nodes numbered contiguously from its `rstart` in the
patch's own lexicographic order. That last one is what makes
`local cell index = (j - ys) * xm + (i - xs)` true, which is the indexing the per-cell
xsection views and `RemovalTerm`'s `r / n_angles` both depend on.

**Ordering constraint (Phase 3b).** `PhaseSpace::create` no longer decides the
decomposition — it leaves `local_cells` as `PETSC_DECIDE` and the backend's `create()`
fills it in. So the discretisation must be created *before* anything sized off the phase
space. Everything that reads `local_cells`/`local_rows()` calls `ps.check_decomposed()`
first, which fails with `PETSC_ERR_ARG_WRONGSTATE` and a pointed message rather than
letting a Kokkos view be allocated with a negative extent further downstream.

## np=3 caveat
From Phase 1a the parallel decomposition is decided in cells (rows = local_cells *
n_angles) — by `PetscSplitOwnership` then, by the DMDA since Phase 3, which is the same
split. The pre-refactor code used PETSC_DECIDE over rows, which can split mid-cell
(e.g. 1000 cells x 4 angles on 3 ranks) — so np=3 results legitimately differ from the
pre-refactor binary (np=3, st=2 converges in the same 5 iterations as serial).
Baselines and tests use np=1,2 only, where the decompositions coincide.
