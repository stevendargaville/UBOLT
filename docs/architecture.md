# Discretisation backends

The long form behind the backend headers (`include/ubolt/discretisation.hpp`,
`structured_fd_*.hpp`, `plex_discretisation.hpp`, `unstructured_dg.hpp`,
`unstructured_cg.hpp`). The headers keep the contract; the reasoning and the
corner-case rules live here.

## The base and its contract

`Discretisation` owns the DM (and through it the mesh, the layout and the parallel
decomposition) and hands out what every dimension-independent piece of the library
needs: the COO slot maps (`CooPattern`), the boundary rows (`BoundaryInfo`), matrices
preallocated for that sparsity (`create_matrix`, deliberately NOT `DMCreateMatrix` -
a DMDA preallocates its whole stencil and PCAIR would see a different matrix), and
`comm()` / `phase_space()` / `n_material_entries()`. Geometry stays on the concrete
classes, because only the streaming term reads it and that term has a sibling per
backend anyway.

A backend's `create()`:

1. sets `comm_` and builds `dm_` - the DM decides the decomposition;
2. writes `ps.local_cells` (and `ps.n_basis` if not 1) and copies `ps` into `ps_`;
3. fills `oor_`/`ooc_` in slot order, `-1/-1` for a nulled slot (the slot stays in the
   values array, terms may write it, PETSc drops it - this keeps the value fills
   branch-free and the pattern the same for every group);
4. fills a `BoundaryRows` (`reset()` gives the no-BC defaults) and calls
   `set_uniform_pattern` (fixed slots per row, diagonal last) or `set_pattern`
   (CSR-shaped, any slot the diagonal);
5. overrides `destroy()` if it owns a PETSc handle beyond `dm_`, and
   `n_material_entries()` if materials are not per local cell (CG: per local element).

`set_pattern`'s device allocations are frozen in set and order: a moved allocation
shifts later buffers, changes the chunking a vectorised reduction picks and so the last
bits of a residual norm, and `tests/baselines/` is compared bit for bit.
`ghost_inflow_d` is allocated last and only under ghost-flux for that reason.
`UboltFillInflow` / `UboltZeroReflectRows` launch a kernel only if it can write
something (host flags `has_dirichlet_rows`, `has_reflect_rows`, `ghost_flux_vacuum`).

## Vacuum treatments

`VacuumTreatment::GHOST_FLUX` (the default since Sep 2026; `DIRICHLET_CELL` was the
only treatment before and is now opt-in, kept so its recipes reproduce the pre-switch
logs):

- **Ghost-flux.** An inflow boundary cell stays an ordinary unknown with its full
  diagonal. Each slot pointing out of the domain takes its own face's ghost value: on
  a vacuum face it is nulled and its coefficient times the face's per-angle inflow
  (windowed) is summed into `BoundaryInfo::ghost_inflow_d`, which is ADDED to the rhs;
  on a reflective face it points at the mirrored angle in the same cell. There are no
  BC rows at all, and a mixed corner takes each face's value with no precedence rule.
  (The first ghost-flux cut let "reflect win" at a mixed corner, mirroring the row over
  the vacuum axis too; the corner cell never saw that face's inflow, an O(1) error that
  did not shrink with h - fixed Sep 2026, PR #8.)
- **Dirichlet-cell.** An inflow boundary row is REPLACED: the identity with the inflow
  on the rhs (vacuum), or `psi(a) - psi(partner) = 0` (reflective). The boundary sits at
  the cell centre rather than the face, an O(h) difference.

### Dirichlet-cell corner rules

Structured backends (`src/structured_fd_commonk.hpp`, and the 1D loops): a row is a BC
row iff its upwind neighbour on some axis is outside the box. If ANY face it comes in
through is vacuum the row is Dirichlet - vacuum wins at mixed corners and edges - and
the value and window are the first vacuum inflow face's in axis order x, y, z. Else the
partner composes the per-axis reflection maps over every inflow axis (a triple flip at
a 3D corner between three reflective faces), still one column, in slot 0. The partner
must itself be interior: a reflective face on a single-cell-wide direction is
PETSC_ERR_SUP.

DG0 (`UnstructuredDG`): a row is a BC row iff some BOUNDARY face (support size 1) of its
cell has `s_f = Omega . nA_f < 0`. If any such face is vacuum the row is Dirichlet, the
winner being the incoming vacuum face with the lowest DOMINANT normal axis (x < y < z),
ties by face point number - on a box, the structured "first vacuum inflow face in axis
order". Else the row is identity minus the partner, the -1.0 in the slot of the first
incoming reflective face in cone order, the partner composing the reflection maps over
the axes of the incoming reflective faces. A reflective face must be AXIS-ALIGNED (the
mirror of an ordinate in a general plane is not an ordinate). The partner's own row may
be Dirichlet - a reflective plane meeting a slanted vacuum face, the usual
symmetry-reduced geometry, where `psi(a) = psi(partner) = inflow` is a good pair of
equations - but not reflective (a single-cell-wide direction between two reflective
faces): PETSC_ERR_SUP. Windows are tested on the face CENTROID along the non-dominant
axes, ascending, inclusive - the boundary cell's centre on a quad/hex box.

Known limitation: on a non-box mesh where two incoming vacuum faces of a cell share a
dominant axis, the tie-break by rank-local point number can pick a different winner in
serial and parallel (Dirichlet-cell with windowed or unequal inflows only).

## The opposite-ordinate identity

Under ghost-flux, with P swapping (cell, Omega) and (cell, -Omega), the assembled
streaming + removal block satisfies `A^T = P A P` on the structured backends for ANY mix
of vacuum and reflective faces (inflow values and windows touch only the rhs; sigma_t
may vary cell to cell). It is what a half-quadrature preconditioner applied transposed
on the other half rests on. It does not hold under Dirichlet-cell, and the matrix-free
isotropic scatter commutes with P but is not symmetric once the quadrature weights
differ (level-symmetric beyond S4).

On the plex DG backend the rows are per unit volume, so with V the cell volumes expanded
to rows the VOLUME-WEIGHTED operator `S = V A` satisfies `S^T = P S P` - exactly off the
diagonal, to rounding on it (a cell's outward nA_f sum to zero only to rounding). A
itself satisfies only the similarity `A^T = V (P A P) V^{-1}`: on a mesh with unequal
volumes the transposed apply of a hierarchy built for one half is wrapped in the V
scalings, `y = V^{-1} M^{-T} (V x)` (V commutes with P). On a uniform box V is a multiple
of the identity. At DG1 the identity holds to rounding too, reflective faces included:
the face couplings pair up across the face, and the own-cell block is antisymmetric in
Omega up to the divergence theorem on the cell. CG-SUPG is not adjoint-consistent and
has no such identity.

## Structured backends

`StructuredFD2D`/`3D` share `src/structured_fd_commonk.hpp`, templated on the dimension:
the DMDA patch and its global numbering (a 2D/3D DMDA numbers each rank's patch
lexicographically WITHIN the patch, so the local-to-global map is the authority), the
layout check, the row classification, the COO build and the box painting. The only
per-dimension inputs are the DMDA and each axis's (lo, hi) face ids - PETSc's 3D box
puts bottom/top on Z and front/back on Y, stated in one table per backend. The ghost
inflow is summed in ascending axis order from 0.0 and the centres are
`(idx + 0.5) * h`, so the refactor onto the template was bit-identical.
`StructuredFD1D` keeps its own loops (its mirror left/right boundary loops were kept
separate on purpose, Aug 2026). A BCSpec id outside a structured box's `1 .. 2 dim`
faces is an error, as an unseen "Face Sets" value is on the plex backends.

## Plex backends

`PlexDiscretisation` holds what `UnstructuredDG` and `UnstructuredCG` share: the mesh
(`create_plex_mesh` - built or read, distributed with a one-cell overlap under the
backend's adjacency, "simple" partitioner, no `DMSetFromOptions`) and the painting,
over the backend's material unit (`paint_point_` / `paint_centroid_d_`: the owned cells
on DG, every local element on CG). The rank-local helpers - layout check per stratum,
outward area normal, BC validation, window coordinates, the collective-failure idiom -
are in `src/plex_commonk.hpp`. Both backends take the quadrature as the
`AngularQuadrature` base and read its host directions (`omega_host()`,
`reflect_host(axis)`), so a new direction set is a new quadrature subclass.

A configuration `create()` cannot build is usually found by one rank (a bad cell, a
non-axis-aligned reflective face, a window of the wrong shape, a mixed-cell-type CG
mesh); the first failure is recorded and every rank errors together at the next
`CollectiveFailure`, so no rank is left waiting in a collective.
