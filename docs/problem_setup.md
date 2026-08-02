# Setting up a new problem

The practical walkthrough: how to go from "I want to solve X" to a problem
file that runs, checks out, and (optionally) becomes a test. The field-by-field
schema reference is `docs/problem_files.md` — this document is about the
decisions and the pitfalls, in the order you meet them.

## Quick start

The smallest valid problem — a uniform 1D slab, one group, unit inflow:

```json
{
 "dimension": 1,
 "mesh": {"n_cells": [1000], "lengths": [1.0]},
 "n_angles": 4,
 "materials": {
  "n_groups": 1,
  "materials": [
   {"id": 0, "name": "slab", "Sigma_t": [2.0], "Sigma_s": [[1.0]], "Source": [1.0]}
  ]
 }
}
```

```
make build_tests
cd tests && ./transportk -problem my_problem.json -ksp_monitor -ksp_converged_reason
```

Everything omitted has a sane default: all faces vacuum, `inflow` 1.0, no
painted regions, no output. Start from the nearest file in `tests/problems/`
rather than from scratch — the table at the bottom maps features to exemplars.

## The decisions, in order

### 1. Dimension and mesh
`dimension` picks the backend; `mesh.n_cells`/`mesh.lengths` take one entry
per dimension. UBOLT is unitless — the lengths are in whatever unit your cross
sections are per (an upstream file states `length_unit`; UBOLT does not
convert, it just trusts you to be consistent).

### 2. Angles
`n_angles` is the SN ordinate count: {2, 4} in 1D, {4 (S2), 12 (S4)} in 2D
today. An invalid count fails in the quadrature's `create` with the valid set
in the message.

### 3. Materials
Three ways to fill `"materials"`:
- **Inline** (`{"n_groups": ..., "materials": [...]}`): right for small,
  self-contained problems — one file tells the whole story.
- **By path** (`"materials": "materials/my.json"`, relative to the problem
  file's directory): right when several problems share a material set — the
  values live once.
- **An existing upstream materials file by path**: loads as-is; the fission fields
  and bookkeeping are ignored (see the reference for the exact list). Add a
  `Source` array per material if you need an external source — the upstream code's
  k-eigenvalue format has none, so a pristine upstream file is sourceless
  and the problem is driven by `inflow` alone.

Rules the loader enforces: ids dense `0..n-1` (they ARE the device-table
indices), every array sized by `n_groups`. Conventions it cannot check for
you:
- `Sigma_t` INCLUDES the scattering — the removal the operator assembles is
  `Sigma_t`, and the within-group scatter `Sigma_s[g][g]` comes back on the
  lhs matrix-free. A material with `Sigma_s[g][g] > Sigma_t[g]` is
  net-multiplying and probably a typo.
- Groups are ordered high energy (0) to low, and `Sigma_s` is `[from][to]`:
  row `g` is everything scattering OUT of `g`. The sweep is downscatter-only
  today — anything below the diagonal is silently unused.
- A group whose `Sigma_t` equals its `Sigma_s[g][g]` has scattering ratio
  exactly 1 (no absorption). Fine with a vacuum face to leak through; singular
  if every face is reflective (see step 5).
- `Source` is isotropic and angle-integrated: each ordinate receives
  `Source / sum_weights`. To reproduce a per-angle rhs value of `q` on every
  ordinate, set `Source = q * sum_weights` — 2q in 1D, 4*pi*q in 2D (the 1D
  single-group test files carry `Source 2.0` for exactly this reason).

### 4. Regions
No `regions` = the background material everywhere. Otherwise paint shapes
(`interval` in 1D, `box` in 2D) over `regions.background` in list order,
later entries winning, membership by cell CENTRE — so a region boundary that
falls exactly on a cell face owns the cells whose centres it covers, and a
region thinner than a cell can own nothing. Reference materials by `id` or by
`name`; names read better in files with more than two materials.

The whole-box paint trick is legitimate: painting `[0, 1, 0, 1]` with a
sourceless copy of the background and then a small source region over it
(see `box_100_cold.json`) is how "zero source except HERE" is said without a
negative-space shape.

### 5. Boundary conditions and inflow
Unset faces are vacuum: the rhs carries `inflow` on their incoming
directions. `reflect` mirrors the outgoing flux back in.

The one trap: **all faces reflective + scattering ratio 1 anywhere is
singular** (the constants are in the operator's kernel). Keep a vacuum face,
or give every group absorption. The multigroup corollary: the LAST group
always has ratio 1 in a pure-downscatter recipe with `Sigma_s[g][g] =
Sigma_t[g]`, so keep a vacuum face in that case regardless of the other
groups.

`inflow: 0.0` plus a painted source region is the cold-problem pattern —
everything is driven by the region.

### 6. Output
`output.flux_vtk` writes the scalar flux (`.vts`/`.vtr`) when the solve
converges; `-flux_vtk` on the command line overrides it, which is the better
habit — keep files that are test recipes output-free (a test run leaves
nothing behind) and opt in from the command line when you want to look:

```
./transportk -problem problems/box_50_absorber.json -flux_vtk flux.vts
```

Multigroup writes `flux_g0.vts`, `flux_g1.vts`, ...

## Values that must survive the round trip

The file is read back as C doubles. Powers of two and their short sums (0.25,
0.5, 2.5, ...) are exact as written; anything else, write with 17 significant
digits (`%.17g`) so it parses back to the identical double. This only matters
when bitwise reproducibility matters — pinned residual histories, baseline
captures — but that is exactly when it matters completely.

## Checking a new problem

- **Typos**: the problem file is strict — any unknown key errors with the
  file and key named. So a misspelled optional key cannot silently default.
- **Infinite medium**: if the problem is (or can be reduced to) uniform
  all-reflect with absorption, run it with `-check_inf_medium -ksp_rtol 1e-12`
  — the solution is checked against the exact per-group constant to 1e-9,
  which validates the whole chain from file to solve with no discretisation
  error in the way.
- **Painting identity**: to check a region setup, duplicate the background
  material under new ids, paint the regions with the copies, and diff the
  `-ksp_monitor` history against the unpainted file — it must be bitwise
  identical (`box_50_identity.json` is this as a permanent test).
- **Eyeball**: `-flux_vtk` and look. An absorber should dent the flux; a
  source region should glow.

## Turning it into a test

See "Adding a test problem" in `docs/dev/testing.md`: commit the file under
`tests/problems/` with a `"_comment"` saying what it pins, add the recipe
line with `-ksp_max_it` pinned to the observed count, and keep output flags
out of the recipe. Remember a pin is the max over the CI arches, not just
your machine.

## Exemplars in tests/problems/

| you want | start from |
|---|---|
| 1D single group, uniform | `slab_st2.json` (+ `materials/slab_st2.json` for the path form) |
| 1D multigroup with downscatter | `slab_mg4_t05.json` |
| Painted 1D region | `slab_mg4_t05_dense.json` |
| 2D uniform box | `box_50_st2.json` |
| 2D with reflective faces | `box_50_reflect_lb.json` |
| Infinite-medium check setup | `slab_inf_medium.json`, `box_50_inf_medium.json` |
| Heterogeneous 2D (absorber block, by-name reference) | `box_50_absorber.json` |
| Cold problem driven by a source region | `box_100_cold.json` |
| S4 angular resolution | `box_30_s4_st2.json` |
