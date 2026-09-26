# Problem-definition files

A solve is described by ONE JSON problem file: everything physical lives in
the file, and the command line keeps everything about how the problem is
solved. `transportk` is the driver that reads them:

```
./transportk -problem problems/slab_st2.json -ksp_monitor
```

The split is deliberate. The file carries the problem - dimension, mesh,
angles, materials, painted regions, boundary conditions, output - so
the same file solves identically under any solver configuration, and the
solver is configured the way PETSc always is: `-ksp_*`, `-pc_*`,
`-sub_1_pc_air_*` and friends on the command line, plus the driver's own
strategy and verification knobs (`-precon_stream`, `-precon_block_scale`, `-diag_scale`,
`-check_inf_medium`, and `-flux_vtk` as an output override). Nothing physical
can be set from the command line, so a problem file names a reproducible
problem, full stop.

Parsing lives in `ProblemSpec` (`include/ubolt/problem_spec.hpp`), which any
future driver can reuse. The vendored parser is `src/external/nlohmann/json.hpp`
and must never be included from a public header.

This file is the schema REFERENCE. For the hands-on walkthrough — the
decisions in order, the pitfalls, checking a new problem and turning it into
a test — see `docs/problem_setup.md`.

## The problem file

The problem file is STRICT: an unknown key anywhere in it is an error, because
an unknown key is a typo, not an extension. The exception is `"_comment"`,
ignored everywhere (JSON has no comments), holding provenance prose.

| key | type | required | meaning |
|---|---|---|---|
| `dimension` | int, 1, 2 or 3 | yes | with `mesh.type`, picks the backend: `StructuredFD1D`, `StructuredFD2D` or `StructuredFD3D` on a structured mesh, `UnstructuredDG` (2 or 3 only) on an unstructured one |
| `mesh.type` | `"structured"` or `"unstructured"` | no, default `"structured"` | the backend family: the DMDA finite-difference backends, or the upwind DG backend on a DMPlex - see "Unstructured meshes" below |
| `mesh.n_cells` | int[dimension] | yes, except with `mesh.file` | `[nx]`, `[nx, ny]` or `[nx, ny, nz]`, all positive. On an unstructured mesh, the cells per axis of the box PETSc builds |
| `mesh.lengths` | number[dimension] | yes, except with `mesh.file` | `[lx]`, `[lx, ly]` or `[lx, ly, lz]`, all positive; the box runs from the origin |
| `mesh.order` | int, 0 or 1 | no, default 0; unstructured only | the DG order: 0 is DG0 (one flux per cell and ordinate), 1 is linear DG (dimension + 1 per cell) - see "Linear DG" below |
| `mesh.simplex` | bool | no, default `false`; unstructured box only | triangles (2D) / tetrahedra (3D) instead of quads / hexes |
| `mesh.file` | string | no; unstructured only | a mesh file PETSc reads (Gmsh `.msh`, ...), resolved relative to the problem file's own directory like a `materials` path. The file decides the mesh, so `n_cells`, `lengths` and `simplex` are errors alongside it; `dimension` is still required and must match the file |
| `sn_order` | int, positive and even | yes | the SN order N, NOT the ordinate count - how many ordinates that is, is the quadrature's business and differs by dimension (N in 1D, N(N+2)/2 in 2D, N(N+2) in 3D, so S4 is 4, 12 and 24 ordinates; a 3D set has twice the ordinates of the same-order 2D set, because there is no xi > 0 half to fold over). 1D takes ANY even order - it is a Gauss-Legendre rule, generated at run time; 2D and 3D take the even orders 2 to 18, the level-symmetric (LQn) sets, which is as far as that family goes with all-positive weights |
| `materials` | string or object | yes | a path to a materials file, resolved relative to the problem file's own directory, or the same schema inline |
| `regions` | object | no | which cells are which material - see below; absent = uniform background. An unstructured mesh may also paint by `"Cell Sets"` label value, `regions.cell_sets` |
| `boundary_conditions` | object | no | per-face `"vacuum"`/`"reflect"`, or an object `{"type", "inflow", "window"}` - see below; unset faces are vacuum with inflow 0. An unstructured mesh may also key faces by `"Face Sets"` label value (`"13": "reflect"`) |
| `vacuum_treatment` | string | no | how every VACUUM face is discretised: `"ghost_flux"` (default) or `"dirichlet_cell"` - see below. Reflective faces are unaffected |
| `output.flux_vtk` | string | no | output path: `.vts` or `.vtr` on a structured mesh, `.vtu` on an unstructured one (the parser checks the extension against `mesh.type`); `-flux_vtk` on the command line overrides it. A single-group problem writes the filename as given, multigroup writes one file per group (`flux.vts` becomes `flux_g0.vts`, ...). Each file carries three per-cell fields for its group: `scalar_flux`, `sigma_t` and `source` (the isotropic strength as written here, not the per-ordinate share) |

### Vacuum treatment

Two discretisations of the same physics - a prescribed incoming flux on a
vacuum face - differing in whether the boundary cell stays an unknown.

`"ghost_flux"` (the default since Sep 2026): the boundary cell stays an
ordinary unknown. Its row carries the full upwind stencil - the same diagonal
`sum_a |Omega_a| / h_a` an interior row has - and the one off-diagonal per axis
whose upwind neighbour lies outside the domain is simply absent, its
coefficient `|Omega_a| / h_a` times the face's per-angle inflow moved to the
rhs. That is the usual upwind flux with a ghost cell holding the prescribed
inflow, and what a DG face flux does. A corner cell fed through two vacuum
faces gets a contribution from each.

A reflective face is treated the same way under `"ghost_flux"`: the ghost cell
behind it holds the MIRRORED direction's flux in the same boundary cell, so the
row keeps its full stencil and the outside-pointing entry couples to that
mirrored angle. Each incoming face supplies its own ghost value, so a cell at a
mixed corner or edge - a direction entering through a vacuum AND a reflective
face - gets the vacuum face's inflow and the reflective face's mirror both, and
there is no precedence rule. A reflective box is then the exact image of the
full box it is a symmetric part of: the quarter box with reflective low faces
reproduces the full box's quadrant to rounding.

`"dirichlet_cell"` (opt-in, and the default until Sep 2026): for a direction
that enters through the face, the boundary cell's row is REPLACED by the
identity and the rhs there carries the incoming flux. The cell is not an
unknown for that direction, so the boundary effectively sits at the boundary
cell's centre. Per-face inflows and tangential windows work exactly as they do
under ghost-flux, but a corner cell fed through two vacuum faces takes only
the first vacuum face in axis order x, y, z. A reflective face's row is
replaced too, by `psi(a) - psi(mirror a) = 0` in the boundary cell, and where
a direction enters through a vacuum and a reflective face the vacuum face wins.

The two differ at the boundary cell by O(h) and converge to the same solution
under mesh refinement. `"ghost_flux"` leaves no boundary rows in the operator,
which is what makes the upwind operator for `-Omega` the exact transpose of the
one for `+Omega` on every row rather than only the interior ones - for ANY mix
of vacuum and reflective faces, since the inflow values and windows only touch
the rhs. That holds for the streaming + removal operator (the part a
preconditioner inverts), with `Sigma_t` varying freely; on an unstructured mesh
it is the cell-volume-weighted version (below). `"dirichlet_cell"` breaks it on
every boundary row. A problem
file written before the switch that wants its old numbers back adds
`"vacuum_treatment": "dirichlet_cell"`.

Restrictions:

- The key switches reflective faces too (above). Until Sep 2026 ghost-flux
  kept the reflective row `psi(a) - psi(mirror a) = 0` and let it win a mixed
  corner, mirrored over every axis the direction entered through - so the
  corner cell never saw the vacuum face's inflow, an O(1) error that did not
  shrink with h. A problem with no reflective face is unaffected by that
  change.
- The DSA correction (`-precon_dsa`) works under either: its Marshak face sits
  on the domain boundary, which is where ghost-flux puts the transport
  boundary too.
- On an unstructured mesh the same key works (see "Unstructured meshes"): the
  coefficient is `|Omega . nA_f| / V_c` for each incoming vacuum face, whatever
  its orientation, and the window is tested on the face centroid.

### Regions

`regions.background` (material id or name, default 0) fills every cell;
`regions.paint` is an ordered list painted over it, later entries winning,
membership decided by the cell CENTRE - exactly `paint_intervals` /
`paint_boxes` semantics. Each entry names its material by id or by name and
its shape by the dimension's key:

```json
"regions": {
  "background": 0,
  "paint": [
    {"material": "absorber", "box": [0.4, 0.6, 0.4, 0.6]}
  ]
}
```

In 1D the shape is `"interval": [x0, x1]` instead, and in 3D the box carries
its z extents too: `"box": [x0, x1, y0, y1, z0, z1]`.

### Boundary conditions

Face names are `left`/`right` in 1D, `left`/`right`/`bottom`/`top` in 2D and
`left`/`right`/`front`/`back`/`bottom`/`top` in 3D, mapped onto the backends'
`FACE_*` label ids (PETSc's box-mesh "Face Sets" convention). CAREFUL:
following that convention, `bottom`/`top` change axis between dimensions -
they are the y faces in 2D but the **z** faces in 3D, where the y faces are
`front` (y-min) and `back` (y-max). `left`/`right` are the x faces
everywhere. Keep at least one face vacuum when the scattering ratio is
exactly 1 anywhere: an all-reflective group with no absorption is singular
(see `docs/dev/testing.md`).

A face is either the bare family string or an object:

| key | type | required | meaning |
|---|---|---|---|
| `type` | `"vacuum"` or `"reflect"` | yes | the BC family - the same thing the bare string says |
| `inflow` | number | no, default 0 | vacuum faces only: the incoming flux prescribed on that face's Dirichlet rows |
| `window` | number[2*(dimension-1)] | no, default the whole face | vacuum-with-`inflow` faces only: where on the face the inflow enters |

`inflow` is an **isotropic, angle-integrated strength**, exactly like a
material's `Source`: the backend shares it over the ordinates as
`inflow / sum_weights` (`/2` in 1D, `/4 pi` in 2D and 3D), so the same value
means the same physics in every dimension. A bare `"vacuum"` string is a cold
face, inflow 0, and so is an unlisted face - a problem with no
`boundary_conditions` block at all is a cold box driven by its source regions
alone.

`window` is one `[lo, hi]` pair per TANGENTIAL axis of the face, in ascending
global-axis order: 2 numbers on a 2D face, 4 on a 3D one (`[lo1, hi1, lo2,
hi2]`). A 1D face is a point and takes no window. A boundary cell is inside
the window when its CENTRE is, inclusive at both ends - the same membership
test `regions.paint` uses - and the rest of the face is cold. So the tangents
are y for a 2D `left`/`right` face and x for `bottom`/`top`; in 3D they are
(y, z) for `left`/`right`, (x, z) for `front`/`back` and (x, y) for
`bottom`/`top`.

A direction incoming through more than one vacuum face (a corner or an edge)
takes the **first vacuum incoming face in axis order x, y, z** - its inflow,
and its window test.

```json
"boundary_conditions": {
 "left": {"type": "vacuum", "inflow": 1.0, "window": [2.0, 3.0]},
 "right": "vacuum",
 "bottom": "vacuum",
 "top": "reflect"
}
```

That is `box_crooked_pipe.json`'s left face: unit isotropic inflow driven
through the pipe mouth only, the strip of the left boundary with `2 <= y <= 3`.

Errors: an `inflow` or a `window` on a `"reflect"` face, a `window` on a 1D
face, a `window` of the wrong length or with `lo > hi`, a `window` without an
`inflow` (it would restrict nothing), and an unknown key inside the face
object. The **top-level** `"inflow"` key of older files was removed in Aug 2026
- it was one global per-angle value - and a file still carrying it errors with
a migration message rather than being silently reinterpreted.

### Unstructured meshes

`"type": "unstructured"` in `mesh` selects the upwind DG backend on a DMPlex,
at DG0 by default (`"order": 1` is linear DG, below): one flux per cell and
ordinate, upwind on every face - first order in space,
like the structured finite differences, and on a uniform quad/hex box exactly
the structured stencil (to rounding). The rows are still `cell * n_angles +
angle`, so everything above the discretisation - scattering, removal, the
group sweep, `-matfree_removal`, `-precon_stream`, `-precon_ref_shift`,
`-diag_scale`, `-check_matfree`, `-check_inf_medium` - works unchanged. Only
2D and 3D: a 1D unstructured mesh is an error (the structured slab IS the 1D
backend). Not yet: `-precon_dsa` (the diffusion correction is a DMDA
operator) errors on an unstructured mesh. One solver default differs: the
streaming stage's PCAIR is built on the element-block-scaled pmat
(`-precon_block_scale`, ON by default here at both orders and off on the
structured backends; `-precon_block_scale 0` turns it off). At DG0 that is a
diagonal scaling and changes little; DG1 needs it for PCAIR to coarsen at its
default strong threshold.

The mesh comes one of two ways:
- **A box built in code**: `n_cells` and `lengths` exactly as on a
  structured mesh, plus `"simplex": true` for triangles/tetrahedra. PETSc's box
  mesher labels the boundary with the same "Face Sets" ids the structured face
  names map onto (2D: 1 bottom, 2 right, 3 top, 4 left; 3D: 1 bottom (z-), 2
  top (z+), 3 front (y-), 4 back (y+), 5 right (x+), 6 left (x-)), so **a
  structured problem becomes its unstructured twin by adding `"type":
  "unstructured"` and nothing else** - `tests/problems/plex_box_50_st2.json`
  is `box_50_st2.json` with that one line.
- **A file**: `"file": "../meshes/square_2x2_tri.msh"`, relative to the
  problem file's directory. Gmsh physical groups arrive as two labels: the
  physical surfaces (3D: volumes) as **"Cell Sets"**, the physical lines (3D:
  surfaces) as **"Face Sets"**. `n_cells`/`lengths`/`simplex` are errors next
  to a file; `dimension` must match it.

Materials: `regions.cell_sets` maps "Cell Sets" values - object keys, written
as decimal strings - to a material id or name:

```json
"regions": {"cell_sets": {"1": "scatterer", "2": "absorber"}}
```

Painting is **background, then `cell_sets`, then `paint`**, later winning: a
labelled cell takes its set's material, an unlisted value or an unlabelled cell
keeps the background, and `paint` boxes then go over the top by cell
**centroid** (the unstructured stand-in for the cell centre), so a box can
still carve a region out of a labelled one. `cell_sets` on a structured mesh
is an error.

Boundary conditions: the face names work on an unstructured mesh too - they
ARE the box's "Face Sets" ids - and so does any key that is a non-negative
integer, taken as a "Face Sets" value as given; that is how a file mesh's own
ids are named. The value is the same bare string or `{"type", "inflow",
"window"}` object. A name and an integer reaching the same id (`"left"` and
`"4"` in 2D) is an error, and an integer key on a structured mesh is an error.
Boundary faces with no "Face Sets" value, or a value the file does not list,
are cold vacuum - the same default as an unset face name. The other way round
is an error: a boundary condition on a "Face Sets" value that NO boundary face
of the mesh carries (a mistyped id) is rejected when the backend classifies
the faces, rather than leaving the face you meant silently cold.

Per face, what the rows do is the structured rule transplanted:
- **reflect needs an axis-aligned face.** The mirrored direction of an
  arbitrary face is not one of the quadrature's ordinates, so a reflective
  face whose outward normal is not along x, y or z is an error when the
  backend classifies it. Boxes of either cell shape have only axis-aligned
  boundary faces; a file mesh can reflect on the straight axis-aligned parts
  of its boundary, INCLUDING where such a plane meets a slanted or curved
  vacuum boundary (the symmetry-reduced quarter geometry). Under
  `"dirichlet_cell"` the reflection partner of a direction there may itself be
  a Dirichlet row, which is fine; what is rejected is a partner that is itself
  reflective - a single-cell-wide direction between two reflective faces.
  Under ghost-flux a reflective face is a face coupling, so neither case needs
  a rule.
- **`"vacuum_treatment": "ghost_flux"`** (the default) is the natural DG0 vacuum condition:
  a direction coming in only through vacuum faces keeps its physical row and
  the face flux `|Omega . nA_f| / V_c` times the inflow goes on the rhs, for
  every incoming vacuum face (slanted ones included). A reflective face is a
  face flux too, its slot coupled to the mirrored angle in the same cell (the
  DG1 rule, and the structured one, so a box still matches its structured
  twin), and a direction coming in through both kinds takes both. There are no
  BC rows. The upwind operator for `-Omega` is then the
  transpose of the one for `+Omega` after weighting the rows by cell volume
  (see `unstructured_dg.hpp`) - exactly the transpose only where the cells
  have equal volumes.
- **a direction with nowhere to come from** (`"dirichlet_cell"` only - under
  the default ghost-flux those rows stay unknowns). With the Dirichlet-cell BC
  convention every row whose direction enters through a vacuum face is
  prescribed, so a mesh that is one cell wide in a direction with vacuum on
  both sides prescribes EVERY row of EVERY angle that has a component along
  it; a 1 x N box with vacuum everywhere "solves" to its inflow (zero on cold
  faces) in zero iterations. That is the convention, not a bug, and the
  structured backends do the same - keep at least two cells per direction.
- **a `window` is tested at the face centroid**: its coordinates along the
  face's tangential axes - the axes other than the dominant axis of the
  outward normal, ascending axis order - inclusive at both ends. On a quad/hex
  box that is the structured boundary-cell-centre test exactly.
- **a direction incoming through several vacuum faces** of one cell takes an
  inflow from each of them under the default ghost-flux; under
  `"dirichlet_cell"` it takes the face whose outward normal's dominant axis
  comes first in x, y, z order (ties by the faces' mesh point numbers) - on a
  box, the structured "first vacuum incoming face in axis order" rule.

Output is `.vtu` (an unstructured grid), with the same `scalar_flux`,
`sigma_t` and `source` cell fields as the structured files, plus a `Rank`
cell field PETSc's writer always adds (which rank owned the cell).

#### Linear DG

`"order": 1` in `mesh` makes the unstructured backend linear DG: in each cell
the flux of every ordinate is a linear function, `dimension + 1` unknowns per
cell and ordinate, second order in space (DG0 is first order). The basis is
modal - the cell average plus orthonormal slopes - so the cross sections and
the isotropic source act exactly as they do at DG0, and everything above the
discretisation (the group sweep, `-matfree_removal`, `-precon_stream`,
`-precon_ref_shift`, `-check_matfree`, `-check_inf_medium`) works unchanged.
What differs from DG0:
- **the vacuum treatment is ghost-flux only.** A face's inflow enters through
  the face integral, which IS the ghost-flux condition; there is no single
  row per cell to replace, so `"vacuum_treatment": "dirichlet_cell"` is an
  error at order 1.
- **no boundary-condition rows at all, reflective faces included.** A
  reflective face feeds each direction coming in through it the mirrored
  direction's flux on the face, from the same cell - mirrored over THAT face's
  axis only, so the DG0 caveats about composed partners at corners and a
  single-cell-wide direction between two reflective faces do not arise.
  Reflective faces must still be axis-aligned.
- **`scalar_flux` in the output is the cell average**, and the slope rides
  along as `scalar_flux_grad_x`, `_y` (and `_z`) cell fields: the flux at a
  point x of a cell is `scalar_flux + grad . (x - x_c)`, x_c the cell's
  centroid. A multigroup file writes them per group like the rest.
- the mesh's faces must be planar (every box, simplex and Gmsh mesh here is);
  a cell whose faces are not is an error when the backend builds the basis.

Parallel: the mesh is distributed by PETSc's **`simple`** partitioner by
default - deterministic on every machine and CI image, so iteration counts
reproduce. `-petscpartitioner_type parmetis` (or any other PETSc partitioner)
on the command line overrides it; the file does not name one, because the
decomposition is how a problem is solved, not what it is.

## The materials schema

The `materials` entry - standalone file or inline object - is a multigroup
materials table designed to interoperate with materials JSON files authored
by other transport codes, which load directly. Unlike the problem file this
schema is TOLERANT: unknown keys are accepted and ignored, because externally
authored files carry fields UBOLT has no use for yet.

| key | handling |
|---|---|
| `n_groups` | required, positive; becomes the phase space's group count |
| `materials` | required, non-empty array - see below |
| `representation` | optional; only `"multigroup"` is supported (absent means multigroup) |
| `schema_version`, `length_unit`, `outside_id`, `energy_edges_ev` | accepted, ignored |

Per material:

| key | handling |
|---|---|
| `id` | required; ids must be DENSE `0..n-1` - they are the `MaterialSpec` indices, because the tables reach device kernels |
| `name` | optional string; lets regions reference the material by name |
| `Sigma_t` | required, `[n_groups]`: the total xsection, including scattering |
| `Sigma_s` | required, `[n_groups][n_groups]` indexed `[from][to]`: row `g` is everything scattering OUT of group `g`; the diagonal is the within-group scatter that stays on the lhs. Groups are ordered high energy (0) to low. The driver's forward sweep consumes the upper triangle only - upscatter entries (below the diagonal) are silently unused until an outer iteration arrives |
| `Source` | optional, `[n_groups]`: UBOLT's one extension - the external source as an isotropic, angle-integrated strength, shared over the ordinates as `Source / sum_weights` by `UboltFillSource`. Absent = zero (void) |
| `Sigma_f`, `Nu`, `Chi` | accepted, ignored - UBOLT is a fixed-source solver today; these come back with a fission phase |

The interoperability asymmetry, documented so nobody trips on it: UBOLT is
unitless and ignores `length_unit` (the mesh lengths are in whatever unit the
xsections are per), while an external consumer of the same schema may require
it. So an externally authored file loads in UBOLT, but a UBOLT-authored
materials file that omits `length_unit` (and the fission arrays) may not load
elsewhere - add those fields when a file is meant to travel.

## Writing values that survive the round trip

A problem file is read back as C doubles. Every value in the migrated test
files is an exact dyadic (0.25, 0.5, 2.5, ...) so what is written is what is
computed with; when a future file needs a value that is not, write it with 17
significant digits (`%.17g`) - that is the shortest form guaranteed to parse
back to the identical double, which is what keeps residual-history baselines
byte-for-byte reproducible from the file alone.

## Worked example

`tests/problems/box_50_absorber.json` - an absorbing sourceless block painted
mid-box in a scattering background, materials inline:

```json
{
 "dimension": 2,
 "mesh": {"n_cells": [50, 50], "lengths": [1.0, 1.0]},
 "sn_order": 2,
 "materials": {
  "n_groups": 1,
  "materials": [
   {"id": 0, "name": "scatterer", "Sigma_t": [2.0], "Sigma_s": [[1.0]], "Source": [1.0]},
   {"id": 1, "name": "absorber", "Sigma_t": [10.0], "Sigma_s": [[1.0]], "Source": [0.0]}
  ]
 },
 "regions": {"paint": [{"material": "absorber", "box": [0.4, 0.6, 0.4, 0.6]}]}
}
```

The same problem with materials by path: `"materials": "materials/box.json"`,
where the path is relative to the problem file's directory and the named file
holds the `n_groups`/`materials` object above.

### An unstructured file mesh

`tests/problems/plex_square_msh.json` - the unit square as eight Gmsh
triangles (`tests/meshes/square_2x2_tri.msh`), the left half scattering and
the right half absorbing by "Cell Sets", driven through the left face and
reflective along the top, every face named by its "Face Sets" value:

```json
{
 "dimension": 2,
 "mesh": {"type": "unstructured", "file": "../meshes/square_2x2_tri.msh"},
 "sn_order": 4,
 "materials": {
  "n_groups": 1,
  "materials": [
   {"id": 0, "name": "scatterer", "Sigma_t": [2.0], "Sigma_s": [[1.0]], "Source": [1.0]},
   {"id": 1, "name": "absorber", "Sigma_t": [10.0], "Sigma_s": [[1.0]], "Source": [0.0]}
  ]
 },
 "regions": {"cell_sets": {"1": "scatterer", "2": "absorber"}},
 "boundary_conditions": {
  "13": {"type": "vacuum", "inflow": 1.0},
  "12": "reflect",
  "10": "vacuum",
  "11": "vacuum"
 }
}
```

The mesh file's physical groups are what the integers mean: surfaces 1 (`x <
0.5`) and 2 (`x >= 0.5`), lines 10 bottom, 11 right, 12 top, 13 left. The
mesh path is relative to `tests/problems/`, where the problem file lives.
