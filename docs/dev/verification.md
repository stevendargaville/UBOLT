# Verification in UBOLT

What each check that is NOT a pinned iteration count proves: the `verify_*` drivers, the in-driver oracles (`-check_inf_medium`, `-check_matfree`, `-check_ref_shift`), the identity recipes, and the guards checked by hand. Pass/fail policy is in `docs/dev/testing.md`; the pins are in `tests/Makefile` (tabulated in `docs/dev/iteration_counts.md`).
This text moved here from `docs/dev/testing.md` (Sep 2026) and keeps its cross-references: a count table or a dated story ("see the DSA section", "the table below", "Unstructured iteration counts", "Switching the default", "Ghost-flux reflective faces", "DSA with voids", "Voids") resolves in `docs/dev/history.md`, Testing history; "DMDA layout" is in `docs/dev/testing.md`; "above" / "below" may point into either.

Where they run: every `verify_*` driver runs serially in `make check` (`run_check`); in `run_tests_short_parallel` `verify_quadraturek` runs at `-n 2`, `verify_2dk` / `verify_3dk` at `-n 2` and `-n 4` with `-skip_reference` (the reference matrices are serial-only), and `verify_plexk` / `verify_cgk` at `-n 2` and `-n 4`; `run_tests_short_serial` repeats `verify_quadraturek`, `verify_2dk` and `verify_3dk`.

## Quadrature verification
`tests/verify_quadraturek` checks the angular quadratures in isolation, and it exists
because since 2026-08-04 they are **generated rather than tabulated**: 1D is a
Gauss-Legendre rule built by Newton in `SNQuadrature::create`, and 2D/3D read the
level-symmetric table `src/sn_lqn_table.py` emits into `src/sn_lqn_table.hpp`. Neither is
anything a reader can eyeball, and nothing downstream can tell a subtly wrong set from a
right one — a wrong quadrature makes a plausible, converging, wrong answer.

Tolerances, not iteration counts, decide its exit code, and it prints every measurement
against its tolerance. It is in both `make check` and `make tests_short`, serial and at
`-n 2`; the quadrature is replicated on every rank, so the parallel run says only that the
driver is clean under MPI. It costs well under a second.

What it pins:
- **1D**, at S2, S4, S6, S8, S12, S16 and S32: exactness on every polynomial of degree
  up to 2N−1, which is the whole of what an N point Gauss rule promises (measures ~1e-15
  against 1e-13); the weights summing to 2; the nodes strictly ascending inside (−1, 1);
  and the ± pairing, weight symmetry and reflection map, all as **exact** equality — the
  reflection search compares cosines with `==`, so the mirrored node has to be the bitwise
  negation of its partner, which is why the generator mirrors by negating a shared value
  rather than recomputing. S2 and S4 are additionally checked against their published
  nodes and weights to 1e-15, which is the only outside reference in the file.
- **2D and 3D**, at every order the table carries (even 2 to 18): the even axis moments
  through order N — the conditions that *define* a level-symmetric set, so this is the
  real test of the generated table (measures ~1e-15 against 1e-12); the ordinate counts
  N(N+2)/2 and N(N+2); weights summing to 4π; strictly positive weights; unit direction
  vectors in 3D and mu²+eta² < 1 strictly in 2D; and all three reflection maps being
  weight-preserving involutions that flip exactly their own cosine. 3D also checks
  invariance under all six permutations of the axes, by search — that is what "level
  symmetric" means, and it is what a class weight attached to the wrong permutation would
  break.
- **The fold**: the 2D set is exactly the ξ > 0 half of the 3D set with doubled weights,
  matched by (mu, eta) with `==`. The two classes build their ordinates from the same
  table but by separate loops, so nothing else would catch them diverging.
- **The two closed-form cosines**: S2's 1/√3 and S4's √((5−√10)/15), to 1e-15. Everything
  above S4 rests on the generator's own assertions, so these are what say the generator is
  solving the right problem.
- **The error paths**: an odd order in any dimension, and S20 in 2D and 3D, all have to
  fail. Run under `PetscReturnErrorHandler` so an expected error neither aborts the run
  (CI puts `-on_error_abort` in `PETSC_OPTIONS`) nor prints a stack trace.

The driver hard-codes 2 and 18 as the range rather than reading it out of the library, so
a silently shortened table fails it — and the S20 rejection is what says the table has not
silently grown either.

The generator carries its own assertions and is checked separately: `python3
src/sn_lqn_table.py` re-derives everything and asserts the moment residuals, positivity,
the cosine ordering, the point counts, and agreement with the published mu_1 values; the
output is deterministic, so `python3 src/sn_lqn_table.py --check` fails if the checked-in
header is not what the script generates. That is not a `make` target — the table changes
about as often as the definition of a level-symmetric set does.

**Why the table stops at S18.** From S14 up, the axis moments do not determine the weights
uniquely — a family of dimension 1 at S14 and S16, 2 at S18, 3 at S20 — and the generator
picks the member minimising the error in the MIXED even moments, one least-squares problem
and the whole rule. At S20 that member has a weight of −1.8e-5, so the generator stops
there; the cap is a computed result, printed in the generated header with the offending
number, not a constant anyone typed. That is the level-symmetric family running out, which
is the well-known reason SN codes cap LQn around here; a higher order needs a different
family (product, Gauss-Chebyshev), not a different choice within this one.

## 2D verification (Phase 4)
There are **no 2D baselines**, and that is deliberate. A residual-history baseline is an
indirect fingerprint of the numerics; in 2D `tests/verify_2dk` compares the operator
itself, so it is a strictly sharper oracle and there is nothing left for a baseline to
catch. `make baselines` is still 1D + multigroup only.

`verify_2dk` runs these checks, both S2 (4 angles) and S4 (12), and exits non-zero on any.
Checks 1 and 2 run under both vacuum treatments; the ghost-flux runs and checks 3-4
are described under "Ghost-flux vacuum treatment" below:

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

## Unequal quadrature weights
Up to and including S4 a level-symmetric set shares 4π equally between its ordinates; from
S6 up it does not — the weights are constant on the permutation classes of a direction's
level indices, two distinct values at S6, three at S8, ten at S18. Nothing in the
solver ever assumed otherwise (the discretisation backends read only the sign of a cosine,
`w_d()` is always consumed through `UboltAngularIntegral`'s gemm, and DSA's
`/sum_weights` prolongation is the isotropic projection rather than an equal-weight
assumption), but until 2026-08-04 no order above S4 existed, so nothing exercised it.

`box_s8_reflect_coarse.json` is what does: 2D, S8 (40 ordinates, three distinct weights),
coarse 20x20 because none of this is about resolution, with reflective left and bottom
faces so the reflection maps and the reflect partner columns see the unequal weights too.
It runs serial and at `-n 2`, 6 iterations both ways.

Note that `verify_2dk` and `verify_3dk` build their reference matrices with
`sum_weights / n_angles` as the weight, which is only correct at S2 and S4 — the orders
they run. A future reference-matrix run at a higher order has to read the weights out of
the quadrature instead.

## Reflective boundary conditions
A problem file's `boundary_conditions` object takes `vacuum` (the default for an unset
face) or `reflect` per face. The default is bitwise the pre-reflective behaviour —
verified by re-capturing all 24 baselines when the plumbing landed — so every recipe and
baseline above is untouched, and the reflective recipes below are NEW pins, not
adjustments. Reflective rows couple the angle blocks at the boundary (the streaming-only
pmat carries the coupling too), so these counts have no reason to match their vacuum
counterparts.

The sharp oracles are the verify_2dk reference configs above and the **infinite-medium
check**: with every face reflective (or a vacuum face feeding in exactly the
infinite-medium flux, see the ghost-flux section), a uniform source and uniform xsections, the exact
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

## Ghost-flux vacuum treatment
`"vacuum_treatment": "ghost_flux"` (see `docs/problem_files.md`) changes only the
boundary rows. It landed opt-in, with its checks being the existing ones run a second
time in that mode plus two that only it needs, and became the DEFAULT on 2026-09-25 -
see "Switching the default" at the end of this section for what that moved.

- **Closed form** (`verify_2dk`/`verify_3dk` check 1, ghost mode). A ghost row is a
  stencil row, so its rhs is the streamed source plus `|cosine| / h` times `psi` at the
  GHOST node one cell further upwind, per outside axis. That cell against the Dirichlet
  row's boundary node is exactly the O(h) the two treatments differ by.
- **Reference matrix** (check 2, ghost mode). A ghost row is the interior row with each
  outside-pointing upwind entry dropped (vacuum face) or moved onto the mirrored angle
  in the same cell (reflective face). The mixed configs pin the ghost-mode corner rule:
  there is none - a direction entering through both a vacuum and a reflective face
  takes each face's own ghost value (see "Ghost-flux reflective faces" below).
- **Constant inflow** (check 3, both modes). Inflow `psi_in` on every face and a source
  `sigma_t psi_in` make `psi = psi_in` exact, and the rhs is built by `UboltFillInflow` +
  `UboltFillSource` exactly as `transportk` builds it. That pins the per-face
  `|cosine| / h` weights in `BoundaryInfo::ghost_inflow_d` and that `UboltFillSource`
  adds rather than overwrites. Checks 1 and 2 build their rhs by hand and cannot see
  either.
- **Opposite-ordinate identity** (check 4, ghost mode only, parallel too).
  `||A^T - P A P||_F / ||A||_F` on streaming + removal with heterogeneous sigma_t, where P
  swaps `(cell, Omega)` with `(cell, -Omega)` (found by the driver's own cosine search).
  It must be below 1e-14 and is 0.0, on all-vacuum, mixed and all-reflect boxes (the 3D
  mixed config has the triple-reflect corner). There is no 1D verify driver; the 1D
  identity was checked by hand on 2026-09-26 by dumping the assembled operator
  (`transportk -ksp_view_pmat`) of a 6-cell S6 slab with the left face reflective and
  the right vacuum, and with both reflective: 0.0 in both. This is
  why the mode exists: it lets one hierarchy built on half the ordinates precondition
  the other half through its transpose. Under Dirichlet-cell it fails on the boundary
  rows (residue ~0.1 relative in 2D), so it is not run there.
- **Solves** against the infinite-medium oracle. `*_inf_medium_ghost.json` replace some
  or all reflective faces with ghost-flux vacuum faces whose inflow IS the
  infinite-medium flux, so the constant stays exact. The 2D/3D files keep reflective
  faces on the low sides, which puts the mixed corners in the solve. `-check_inf_medium`
  accepts such a face (whole-face, and inflow / sum_weights equal to the expected
  constant in every group, otherwise it errors).
- `DSAPrecon` refused ghost mode until the switch; it now takes either treatment (see
  below).

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

## 3D verification
The 2D story, one axis wider, and the same deliberate absence: there are **no 3D
baselines** — `tests/verify_3dk` compares the operator itself, `make baselines` stays
1D + multigroup only.

`verify_3dk` runs the checks of `verify_2dk`, both S2 (8 angles) and S4 (24 — a 3D
level-symmetric set has twice the ordinates of the same-order 2D set), each under both
vacuum treatments (see "Ghost-flux vacuum treatment" below):

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

## Unstructured DG verification
The unstructured backend (`UnstructuredDG` + `StreamingTermDG0`, Phase 6a; at order 1
`StreamingTermDG1`, checks 8 and 9; `ElementBlockInverse`, check 10; the plex
`DSAPrecon`, check 11) has no
baselines either, for the 2D/3D reason: `tests/verify_plexk` compares the operator
itself. It runs serially in `run_check` and at `-n 2` and `-n 4` in
`run_tests_short_parallel`, and every comparison in it is parallel-safe — nothing assumes
rank 0 owns anything, or that either backend numbers its rows naturally.

1. **FD twin, 2D quads.** On a uniform quad box DG0 upwind IS the structured upwind
   stencil (`|mu| dy / (dx dy) = |mu| / dx`), so each case is built on `StructuredFD2D`
   and on the plex, and compared: (a) the assembled matrices, brought onto one ordering
   by an explicit permutation matrix P (not `MatPermute`, which wants matching local
   sizes and the two decompositions differ), `||A_plex - P^T A_fd P||_inf <= 1e-12
   ||A_fd||_inf` (measured ~1e-15 to 3e-15); (b) the BC row mask and reflect flags **exactly** and the Dirichlet values to
   1e-14; (c) the rhs (`UboltFillInflow` + `UboltFillSource` + `UboltZeroReflectRows`)
   to 1e-14; (d) the matrix-free scatter on the same permuted random vector, 1e-14;
   (e) the composed `TransportOperator::diagonal()` against the plex's own
   `MatGetDiagonal`, **bitwise** (0.0); (f) a full solve at rtol 1e-13, the permuted
   solutions to 1e-9 (two different Krylov histories, so no tighter); (g) the DSA
   correction (check 11). Cases: 4x3 S2 and
   S4 vacuum, 5x4 S2 reflect left + bottom, 5x4 S4 mixed (reflect left + top, vacuum
   right + bottom), 6x4 S2 with inflow 1.0 on the left windowed to y in [0.5, 1.5] and
   0.3 on the bottom (the corner winning-face rule and the window), and a painted 4x3 S4
   — painted by box on both sides, and again with the plex side painted through a
   "Cell Sets" label written onto the same cells (the path a Gmsh physical group takes,
   so `paint_cell_sets` is held to the same twin).
2. **FD twin, 3D hexes**, the same (a)-(f) against `StructuredFD3D`: 3x2x2 S2 vacuum,
   3x3x2 S4 reflect left + front + bottom (the three-face corner), 4x2x3 S2 with inflow
   on the left windowed in (y, z).
3. **Simplex meshes** (triangles 6^2 = 72 cells, tets 3^3 = 162 cells — a simplex box
   has 2 triangles or 6 tets per box cell), which have no twin: (a) the infinite medium,
   every face reflective, flat to 1e-10 at `1 / (sum_weights (sigma_t - sigma_s))`
   (measured ~1e-14) — which pins reflection on the axis-aligned faces of triangles and
   tets and the scatter reading BC rows; (b) vacuum, a pure absorber and a source only in
   the central box [1/4, 3/4]^dim: converged, inside the exact discrete bounds
   `0 <= psi <= q / (sum_weights sigma_t)` to 1e-12 (DG0 upwind is monotone and the
   infinite medium is an upper solution), and the scalar flux integrated per unit source
   within 25% of the quad/hex mesh's at the same nominal h (measured 3% in 2D, 5% in
   3D). A sign or scaling error fails it; it is not a convergence claim.
4. **Layout and geometry invariants** on every mesh: the volumes sum to the box's to
   1e-12 relative, every cell's outward area-weighted normals close to 1e-12 (a normal
   that failed to flip outward shows up as 2 nA), the cell count is the box's on
   quads/hexes, and on the simplex boxes every "Face Sets" id sits on the box face the
   structured `FACE_*` constant of that id names, exactly (simplex and tensor boxes get
   the label from different PETSc code paths).
5. **Error paths**, through `PetscReturnErrorHandler`: a reflection partner that is
   itself a REFLECTIVE row (reflect on both x faces of a one-cell-wide box), `cell_sets`
   on a mesh with no "Cell Sets" label, a 2D quadrature on a 3D mesh, a `.vts` name on
   the plex, reflect on the slanted hypotenuse of `tests/meshes/tri_slanted.msh` (a
   4-triangle right triangle), a boundary condition on a "Face Sets" value no face
   carries, DG1 under `"dirichlet_cell"`, and `StreamingTermDG0` handed a DG1 backend.
   All eight must be rejected. The POSITIVE twin of the slanted case is checked
   too: reflect on that mesh's axis-aligned bottom with the hypotenuse a vacuum face
   must be accepted (some reflection partners there are Dirichlet rows, the
   symmetry-reduced geometry a file mesh exists for), converge, and stay inside the
   discrete bounds - found by the debug-arch sweep of 23 Sep 2026, which the first cut
   rejected with a misleading "single-cell-wide" message.
6. **VTU output**: writes a `.vtu` into the current directory, reads it back on rank 0
   after a barrier and requires the `<Piece NumberOfCells>` entries to sum to the global
   cell count — PETSc's writer writes every cell of a rank's local mesh unless a "vtk"
   label says otherwise, so an overlap cell written by two ranks would push it over —
   then deletes it. The file is `verify_plexk_tmp_<size>.vtu`, so nothing is left behind.
7. **Ghost-flux vacuum treatment.** Every FD twin case of 1 and 2 bar the "Cell Sets" one
   again under `"ghost_flux"`, the same (a)-(f): the rhs comparison now carries the ghost
   inflow (the windowed 2D case has both corner faces feeding the origin cell, each on
   its own window), and the mixed configs pin the reflective faces' mirror couplings and
   the corners where they meet a vacuum face's inflow. Then, with no twin: (a) on streaming + removal with
   sigma_t varying cell to cell, `||(VA)^T - P(VA)P||_F / ||VA||_F <= 1e-13` (measured
   ~1e-16), P the opposite-ordinate swap found by the driver's own cosine search and V
   the cell volumes per row. This is the property a half-quadrature preconditioner
   applied transposed rests on. It is not bitwise: the diagonal's outflow sum equals the
   opposite direction's inflow sum only because the cell's `nA_f` close, which is to
   rounding. It runs on 6^2 triangles S2, on 3^3 tets S4, and on
   `tests/meshes/square_irregular_tri.msh` S6, each all-vacuum and again with the low
   face of every axis reflective (6^2 triangles at S4 there). That file is a 4x4 triangulation with the
   interior nodes perturbed, so the areas vary by +-17%. There the UNWEIGHTED
   `||A^T - PAP|| / ||A||` is required to be above 1e-3 (it is ~0.07), because the
   generated simplex boxes have equal volumes and would not tell the two identities
   apart. (b) inflow `psi_in` on every face and source `sigma_t psi_in` against
   `psi = psi_in` through `UboltFillInflow` + `UboltFillSource`, residual below
   `1e-12 ||b||` (measured ~1e-15), which pins the `|Omega . nA_f| / V_c` weights on
   faces of every orientation, both runs. (c) `meshes/tri_slanted.msh` in ghost mode:
   no BC rows at all, the solve converged and in bounds.
8. **DG1** (order 1), which has no FD twin, on 5x6 quads S4, `square_irregular_tri.msh`
   S6, 4^3 hexes S2 and — where PETSc can mesh them — 5x6 triangles S4 and 4^3 tets S4,
   each with the faces around one corner reflective (left + bottom; + front in 3D) and
   the rest driven vacuum: (a) `n_basis == dimension + 1` and ZERO BC rows; (b)
   `||(VA)^T - P(VA)P||_F / ||VA||_F <= 1e-12` on streaming + heterogeneous removal
   (measured 4e-16 to 1e-15) — at DG1 the reflective couplings are inside this identity
   rather than excluded BC rows, and the own-cell block only closes by the divergence
   theorem on the cell, so it also pins the fan face moments against the fan cell
   moments; (c) the constant `psi_in` on basis 0 (zero slope) through `UboltFillInflow`
   + `UboltFillSource` with source `sigma_t psi_in`, residual below `1e-12 ||b||`
   (measured ~1e-15) — the ghost weights `phi_i(x_f)` and the reflective couplings; (d)
   the projection of a LINEAR field `alpha + b . x`, the same in every direction, through
   the streaming matrix alone, against `Omega . b` on basis 0 and zero on the slopes in
   every cell with no boundary face, to `1e-11 max |Omega . b|` (measured 3e-15 to
   4e-14) — the volume term and both face matrices, across neighbours whose bases
   differ, on the irregular mesh included.
9. **DG1 is second order**: a pure absorber (sigma_t 1) on [0, 1]^2 with inflow 1 on the
   left, reflective top and bottom (the solution does not depend on y, so reflecting it
   is exact) and a cold right face, whose exact discrete-ordinates solution is
   `psi_in exp(-sigma x / mu)`. The cell averages against the exact scalar flux at the
   centroids, volume-weighted RMS, at n = 4, 8, 16, S4: the observed order between the
   last two must be at least 1.8. Measured 2.03 on quads (3.5e-3, 8.6e-4, 2.1e-4) and
   2.01 on triangles (2.4e-3, 6.1e-4, 1.5e-4). Kept this small because the check runs
   three times in the suite, on the debug arch too. The measure itself is O(h^2) even for
   the exact solution (average against centroid value), so this cannot see anything
   better than second order - it is there to catch a first-order DG1.
10. **The element-block inverse** (`ElementBlockInverse`, what `-precon_block_scale`
   scales PCAIR's pmat by), on every operator 7 (DG0, 1 x 1 blocks) and 8 (DG1,
   (dim + 1)^2 blocks strided by n_angles, reflective couplings included) assemble: the
   element blocks of `D^{-1} A` read back on the host against the identity; `D^{-1} A`
   unchanged by a random left row scaling of A (a pointwise scaling is block-diagonal,
   so it cancels - this also drives the `MAT_REUSE_MATRIX` path with changed values);
   and `(D^{-1} A) x` against `apply(A x)`. Tolerance 1e-12, measured 1e-16 to 1e-15,
   serial and -n 2. And the blocks the removal stage (composite index 0, the same class
   on the operator) takes under `-matfree_removal` - the streaming-only matrix with the
   operator's composed diagonal in place of its own - against the assembled streaming +
   removal operator's: EXACTLY equal (0, not a tolerance: the composed diagonal is
   bitwise the assembled one and the off-diagonals are the same numbers), and genuinely
   different from the bare streaming-only blocks, so a dropped diagonal would show.
11. **DSA on the plex** (`DSAPrecon::create` on an `UnstructuredDG`). In every FD twin
   case of 1, 2 and 7, as their (g): the plex's two-point-flux diffusion matrix against
   the structured star times the (constant) cell volume, permuted by a cell-level twin
   of the same centroid join, to `1e-12 ||D||_inf` (measured 1e-16 to 2e-15); and the
   whole correction `P D^-1 R` on the permuted random vector of (d), both inner solves
   switched to CG at rtol 1e-14, to `1e-9 ||y||_inf` (measured ~1e-16 to 3e-15 absolute
   on a norm of ~0.3). The volume factor is the plex's deliberate volume weighting (it
   keeps the matrix SPD where volumes vary); the twin catches the harmonic face D, the
   Marshak faces, the zero-Neumann reflective faces, and the Dirichlet-cell mask in the
   restriction and prolongation. At DG1, with no twin, on `square_irregular_tri.msh` S4
   (reflect left + bottom) and 4^3 hexes S2 (reflect left), one material: the interior
   penalty matrix is symmetric to `1e-12 ||D||_inf` (measured exactly 0); a LINEAR field
   written in each cell's modal basis maps to `V sigma_a u` on every row of every cell
   with no boundary face, to `1e-12 ||Du||_inf` (measured 2e-16 and 2e-15) - no jumps
   and a constant flux, so the face terms cancel the volume diffusion term by the
   divergence theorem, which exercises the face matrices, both cells' gradients and the
   normals at once (a 10% error in one face coefficient measures ~1e-2); and the round
   trip through `apply()` with a CG inner solve at rtol 1e-14 - an x whose ordinates at
   node (c, i) are `(Du)(c, i) / (V_c sum_w)` must come back as `u / sum_w` on every
   ordinate, to `1e-9 ||u / sum_w||` (measured ~1e-15).

**Why the twin comparison is to rounding and not bitwise.** The two backends reach the
same coefficient through different arithmetic — the FD stencil writes `|mu| / dx`, DG0
writes `(Omega . nA_f) / V_c` with the area-weighted normal and the volume coming out of
`DMPlexComputeCellGeometryFVM` — so `|mu| dy / (dx dy)` and `|mu| / dx` agree to the last
bit or two, not exactly. The structure (which rows are BC rows, which slots are live) is
exact, and is checked exactly.

**The permutation, by centroid.** Neither backend's global numbering is assumed. The
join key is each cell's natural lexicographic index — on the FD side from its (i, j[, k]),
on the plex side `floor(x / dx)` etc. from its centroid — and it is used ONLY as a key:
each side's row of its local cell c is its OWN `rstart + c * n_angles + a`, the property
each backend's layout check asserts (`CheckDALayout`, `CheckPlexStratumLayout`). The DMDA's row
must come from its own layout rather than from the natural index, because a 2D/3D DMDA
numbers patch-lexicographically in parallel (see "DMDA layout" below); the plex's global
numbering comes from its global section and the simple partitioner, and is unrelated to
either. The key goes through a Vec indexed by key — the plex ranks write their row base
into it, the FD ranks read theirs back — and out of it come the permutation matrix and a
`VecScatter` for the vectors, whatever the two decompositions are.

**Parallel.** `-n 2` splits both twins, `-n 4` is the first run where the DMDA splits in
two directions and the plex's simple partition cuts the cell points into four
contiguous ranges; both pass with the same measured differences as serial.

**Checks 12 and 13: the DSA's voids** (`-dsa_void_bridge 0` masks, the default bridges; the design and the counts are in `docs/dsa.md` and the history's "DSA with voids").

`verify_plexk` check 12 holds the masking itself: the FD twin with its painted
box made a void (Dirichlet-cell and ghost-flux with a reflective corner — the
structured and plex operators agree to rounding, masked AND bridged, the chord
included), and under `-dsa_void_bridge 0` at DG0 and DG1 a
quad (and at DG1 a hex) box whose high-x cells are void against the box of only
its low-x cells: every non-void row identical (so the face into the void IS the
cut box's vacuum face), void rows V I, and `P D^-1 R` equal on the non-void
nodes and exactly zero on the void ones, serial and -n 2/4.

`verify_plexk` check 13 holds the bridging, at DG0 and DG1 on a 7x4 quad box
and a 5x3x3 hex box split in two by a void slab (reflective faces included):
bridged, the void counted, the chord exactly `4 V / S` of the slab, the
diffusion matrix symmetric and positive definite (a dense Cholesky), a
constant mapped to `V sigma_a` on every node off the vacuum boundary (zero in
the void - it is a conductor, not a hole), and a residual on the cells left of
the void corrected on the cells right of it (~1/3 of the peak), where the mask
gives zero (to 1e-12: the inner GAMG's aggregates can straddle the void).

## CG-SUPG verification
The CG-SUPG backend (`UnstructuredCG` + `terms_cg.hpp`, Phase 6b) is checked by
`tests/verify_cgk`, serially in `run_check` and at `-n 2` and `-n 4` in
`run_tests_short_parallel`. Every measured value below is identical at 1, 2 and 4
ranks (28 Sep 2026, opt arch).

1. **Element tables** on quads, triangles, hexes, tets and the irregular Gmsh
   triangulation: `sum_ij M = V`, `sum_j G_ij = 0`, `sum_j K_ij = 0` and the linear
   exactness `sum_j G^d_ij x^d'_j = delta_dd' int phi_i`, tol 1e-12 (measured
   <= 1.2e-15). The last one is also what catches a basis function matched to the
   wrong vertex.
2. **Layout**: rows = global vertices x angles, lumped masses of the owned vertices
   summing to the domain volume (<= 4.8e-16).
3. **Consistency**: an infinite medium (Sigma_t 1.5, Sigma_s 0.7) with a painted
   VOID box, reflective faces on one corner and vacuum faces fed the medium's flux,
   through the library's operator and rhs: `|A c - b| / |b| <= 1e-12` (measured
   <= 1.3e-15) at zeta 0.5 and 2 on all four shapes, and `SUPGTermCG::add_diagonal`
   against `MatGetDiagonal` **bitwise** (0.0).
   **3b, the global balance**: pure streaming, a LINEAR psi, vacuum faces with an
   inflow; the lumped-mass sum of `A psi - b` per ordinate equals the outflow of psi
   minus the prescribed inflow through the box faces, exactly (measured <= 1.4e-15,
   tol 1e-12). This is the check that fixes the SCALE of the weak boundary terms,
   which a constant cannot see (it makes `psi - psi_in` vanish whatever the weight):
   the first cut divided the face coefficient by the face area twice, and every
   other check here passed with it.
4. **Order**: the pure absorber with left inflow and reflective y faces (DG1 check
   9's problem), nodal RMS error against the exact SN solution at n = 8/16/32:
   quads 1.5e-3 / 3.8e-4 / 9.8e-5 (order 1.95), triangles 1.6e-3 / 4.0e-4 / 1.0e-4
   (1.96); min 1.8.
5. **SAAF-LS void slab** (arXiv 1605.05388 s4.3: source | void | absorber,
   reflective at x = 0), a 1-cell-high strip with reflective y faces, S8, against
   the exact SN solution of the same quadrature (per ordinate, region by region in
   closed form). Relative RMS error at 40/80/160/320 cells, zeta 0.5: 3.0e-3 /
   7.4e-4 / 1.8e-4 / 4.3e-5 (order 2.05); zeta 2: 3.5e-3 / 8.1e-4 / 1.9e-4 / 4.4e-5
   (2.08). Asserted: order >= 1.8, finest <= 1e-4. Iterations at rtol 1e-13: 8-10 at
   every level, flat to 640 cells, with or without the element-block scaling -
   where that paper's GMRES + BoomerAMG on SAAF-tau took 801 to 8120.
6. **Thin/thick slab** (Hammer, Morel & Wang, arXiv 1902.08729 s III.B: Sigma 0.1 |
   Sigma 10, a source everywhere, vacuum both ends), S8, 16/32/64/128 cells: zeta
   0.5 7.8e-2 / 2.9e-2 / 8.6e-3 / 2.3e-3 (order 1.93), zeta 2 9.6e-2 / 3.7e-2 /
   1.1e-2 / 2.7e-3 (2.04); asserted order >= 1.5, finest <= 5e-3. PRINTED, not
   asserted: at their 8 cells per region the flux at the interface vertex sits 33%
   (zeta 0.5) / 35% (zeta 2) below the exact one - a continuous flux cannot bend
   into the thick side's boundary layer within one element, the dip they report for
   SAAF-tau. Zeta 0.5 (Rattlesnake's) is slightly the better of the two on this
   problem's error at every mesh, so it is the default.
7. **Error paths**: `dirichlet_cell`, zeta 0 and -1, an unknown "Face Sets" id and a
   reflective slanted face (`meshes/tri_slanted.msh`, set 12) all rejected.
8. **The DSA** (`DSAPrecon`'s CG overload): the diffusion matrix against the
   transport restricted onto an isotropic flux, `D phi = m R A P phi` for a random
   phi, in a thick material (Sigma_t 50, so tau = 1/sigma_t everywhere) with an
   UNBRIDGED painted void (the SUPG tensor there) and reflective faces on one
   corner: <= 4.6e-16 on all four shapes at 1, 2 and 4 ranks (tol 1e-12). The
   bridged voids' chord `4 V / S` against closed forms, to 1e-12: an interior block
   of quads, the same block on the reflective left face (that side not in S), and a
   hex block on the vacuum top (in S). All-reflective with no absorption refused.

## `-check_matfree` (the matrix-free removal oracle)

**`-check_matfree` is the sharp oracle**, and it is what makes the mode trustworthy
rather than merely green. It builds BOTH operators on whatever problem file is being
run — the same discretisation and the same streaming term, with only which half of
`RemovalTerm` is live differing — and per group applies both shells to three random
vectors, then compares the composed diagonal against `MatGetDiagonal` of the fully
assembled matrix. The assembled operator is refilled per group and the matrix-free one
is never touched after its single assembly, so the per-group behaviour is in the check
and not just the operator at one group. Two tolerances, deliberately different:

- the matvec, `max|y_f - y_a| / max|y_a|` against **1e-13**. Measured 2.5e-16 to
  4.2e-16 in 1D, 2D and 3D, vacuum and reflective, serial and at `-n 2` — rounding and
  nothing else, and a mistake in the apply is many orders larger.
- the diagonal, `max|d_f - d_a|` against **0.0**, i.e. bitwise. Measured 0.0 everywhere,
  and it has to be: each term's `add_diagonal` writes the same expression its
  `assemble_add` writes into the diagonal slot, the composition sums them in the order
  the COO values are summed in, and the BC rows get the same 1.0. That identity is the
  whole reason the removal shell PC may compose its diagonal instead of reading it off a
  matrix, so it is asserted rather than assumed.

It is independent of `-matfree_removal` — it says the same thing whichever mode the
solve around it runs in — and it is in `make check`, on the 1D multigroup file.

## `-check_ref_shift` (the reference-shifted pmat oracle)

`-check_ref_shift` (driver) is the matrix-level form of the exact-coverage
identity: every group's pmat against its own streaming + removal operator
assembled separately, `max |P - A| / max Sigma_t(g)` against 1e-12. It holds on a
density-scaled problem at worst mismatch 1: 9e-14 on `box_/plex_decades4_void`,
1.6e-14 on `slab_decades4_void -precon_ref_k 4`, 0 on the single-group files; the
single-reference variant above measured 0.2 and FAILS.

## Checks that are not recipes

### DSA
A recipe passes on exit 0, so the guard is checked BY HAND:

- **Singularity guard**: an all-reflective, void-free problem whose within-group `Sigma_s`
  equals its `Sigma_t` must fail with the pure-Neumann message. This is the same
  constraint the transport operator carries (see the reflective section above),
  which is why no shipped problem file is in that state.

### The reference-shifted pmat
Three guards, all of which exit non-zero, so all of which are checked BY HAND:

- `./transportk -problem problems/slab_mg4_t05.json -precon_ref_shift` must fail
  with the "run the two together" message: the shift repairs the pmat
  `-matfree_removal` leaves behind, and every other mode already has its removal.
- `./transportk -problem problems/slab_mg4_t05.json -matfree_removal -precon_ref_k 2`
  must fail with the "it needs `-precon_ref_shift`" message rather than silently
  ignoring the number.
- `-check_ref_shift` must be able to fail: `slab_decades4_void.json` with
  `-matfree_removal -precon_ref_shift -check_ref_shift` at the default k (a
  mismatch of 3.16, so not the full operator) must print FAIL (2.16) and exit
  non-zero; and `-check_ref_shift` without `-precon_ref_shift` must fail with
  the "it needs `-precon_ref_shift`" message. (Until 27 Sep 2026 the third guard
  here was the mixed-void refusal - a group zero in only some cells - which is
  gone: see "Voids" above.)
