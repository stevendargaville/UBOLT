# Problem-definition files

A solve is described by ONE JSON problem file: everything physical lives in
the file, and the command line keeps everything about how the problem is
solved. `transportk` is the driver that reads them:

```
./transportk -problem problems/slab_st2.json -ksp_monitor
```

The split is deliberate. The file carries the problem - dimension, mesh,
angles, materials, painted regions, boundary conditions, inflow, output - so
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
| `dimension` | int, 1 or 2 | yes | picks the backend: `StructuredFD1D` or `StructuredFD2D` |
| `mesh.n_cells` | int[dimension] | yes | `[nx]` or `[nx, ny]`, all positive |
| `mesh.lengths` | number[dimension] | yes | `[lx]` or `[lx, ly]`, all positive |
| `n_angles` | int | yes | SN ordinates; what is valid is the quadrature's business ({2, 4} in 1D, {4, 12} in 2D today) |
| `materials` | string or object | yes | a path to a materials file, resolved relative to the problem file's own directory, or the same schema inline |
| `regions` | object | no | which cells are which material - see below; absent = uniform background |
| `boundary_conditions` | object | no | per-face `"vacuum"` or `"reflect"`; unset faces are vacuum |
| `inflow` | number | no, default 1.0 | the per-angle incoming flux prescribed on the Dirichlet (vacuum-face) rows of the rhs; 0.0 with a painted source region is a cold problem driven by that region alone |
| `output.flux_vtk` | string | no | scalar flux output path, `.vts` or `.vtr`; `-flux_vtk` on the command line overrides it. A single-group problem writes the filename as given, multigroup writes one file per group (`flux.vts` becomes `flux_g0.vts`, ...) |

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

In 1D the shape is `"interval": [x0, x1]` instead.

### Boundary conditions

Face names are `left`/`right` in 1D and `left`/`right`/`bottom`/`top` in 2D,
mapped onto the backends' `FACE_*` label ids (PETSc's box-mesh "Face Sets"
convention). Keep at least one face vacuum when the scattering ratio is
exactly 1 anywhere: an all-reflective group with no absorption is singular
(see `docs/dev/testing.md`).

```json
"boundary_conditions": {"left": "reflect", "bottom": "reflect"}
```

## The materials schema

The `materials` entry - standalone file or inline object - uses the multigroup
materials schema of the upstream code
(`Neural_Physics_RT/sn_solver/data/materials_c5g7.json` is the reference
example), so an existing upstream materials file loads directly. Unlike the
problem file this schema is TOLERANT: unknown keys are accepted and ignored,
because it is a foreign schema that carries fields UBOLT has no use for yet.

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

The asymmetry with the upstream code, documented so nobody trips on it: the upstream code
REQUIRES `length_unit` and UBOLT ignores it (UBOLT is unitless - the mesh
lengths are in whatever unit the xsections are per). So every upstream file
loads in UBOLT, but a UBOLT-authored materials file without `length_unit`
(and without the fission arrays) will not load in the upstream code.

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
 "n_angles": 4,
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
