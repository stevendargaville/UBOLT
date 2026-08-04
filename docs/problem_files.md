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
strategy and verification knobs (`-precon_stream`, `-diag_scale`,
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
| `dimension` | int, 1, 2 or 3 | yes | picks the backend: `StructuredFD1D`, `StructuredFD2D` or `StructuredFD3D` |
| `mesh.n_cells` | int[dimension] | yes | `[nx]`, `[nx, ny]` or `[nx, ny, nz]`, all positive |
| `mesh.lengths` | number[dimension] | yes | `[lx]`, `[lx, ly]` or `[lx, ly, lz]`, all positive |
| `sn_order` | int, positive and even | yes | the SN order N, NOT the ordinate count - how many ordinates that is, is the quadrature's business and differs by dimension (N in 1D, N(N+2)/2 in 2D, N(N+2) in 3D, so S4 is 4, 12 and 24 ordinates; a 3D set has twice the ordinates of the same-order 2D set, because there is no xi > 0 half to fold over). Orders 2 and 4 are implemented today |
| `materials` | string or object | yes | a path to a materials file, resolved relative to the problem file's own directory, or the same schema inline |
| `regions` | object | no | which cells are which material - see below; absent = uniform background |
| `boundary_conditions` | object | no | per-face `"vacuum"`/`"reflect"`, or an object `{"type", "inflow", "window"}` - see below; unset faces are vacuum with inflow 0 |
| `output.flux_vtk` | string | no | output path, `.vts` or `.vtr`; `-flux_vtk` on the command line overrides it. A single-group problem writes the filename as given, multigroup writes one file per group (`flux.vts` becomes `flux_g0.vts`, ...). Each file carries three per-cell fields for its group: `scalar_flux`, `sigma_t` and `source` (the isotropic strength as written here, not the per-ordinate share) |

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
