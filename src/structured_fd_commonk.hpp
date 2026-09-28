#ifndef UBOLT_STRUCTURED_FD_COMMON_HPP
#define UBOLT_STRUCTURED_FD_COMMON_HPP

// The DMDA plumbing StructuredFD2D and StructuredFD3D share, templated on the
// dimension: the patch and its global numbering, the layout check, the
// per-(node, angle) boundary classification, the COO build and the box
// painting. The only per-dimension inputs are the DMDA itself and each axis's
// (lo, hi) face ids, so PETSc's 3D naming (bottom/top are the Z faces) lives in
// one table per backend. StructuredFD1D keeps its own loops (only the face id
// check is shared)
//
// INTERNAL, like plex_commonk.hpp: under src/ so it never enters the public
// include tree, everything static so each TU gets its own copy; the Makefile
// declares the objects' dependency on it by hand

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/material_regions.hpp"
#include "petsc_kokkos.hpp"
#include <petscdmda.h>
#include <vector>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A structured backend's face ids are 1 .. n_faces (PETSc's box "Face Sets"),
// and a boundary condition on any other id is a typo that would otherwise leave
// the intended face silently cold - the plex backends error the same way
static inline PetscErrorCode CheckStructuredFaceIds(MPI_Comm comm, const BCSpec &bcs, PetscInt dim)
{
   PetscFunctionBeginUser;

   for (const auto &entry : bcs.faces()) {
      PetscCheck(entry.first >= 1 && entry.first <= 2 * dim, comm, PETSC_ERR_ARG_WRONG, "a boundary condition " \
         "was given for face id %" PetscInt_FMT " but a %" PetscInt_FMT "D box has faces 1 to %" PetscInt_FMT, \
         entry.first, dim, 2 * dim);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This rank's patch of the DMDA, in cells: the owned corner and extent, and
// the ghosted ones the local-to-global map is indexed over. With
// DM_BOUNDARY_NONE the ghosted patch is clipped at the physical boundary -
// fine, since a column is only ever a face neighbour or an owned node
template <int DIM> struct DAPatch {
   PetscInt n[DIM] = {};                      // global cells per axis
   PetscInt start[DIM] = {}, m[DIM] = {};     // owned
   PetscInt gstart[DIM] = {}, gm[DIM] = {};   // ghosted

   PetscInt local_cells() const
   {
      PetscInt count = 1;
      for (PetscInt d = 0; d < DIM; d++) count *= m[d];
      return count;
   }
   // Local cell c -> its global (i, j[, k]): the patch's own lexicographic
   // order, x fastest - the order every per-cell view is indexed in
   void cell_index(PetscInt c, PetscInt idx[DIM]) const
   {
      for (PetscInt d = 0; d < DIM; d++) {
         idx[d] = start[d] + c % m[d];
         c /= m[d];
      }
   }
};

template <int DIM> static PetscErrorCode GetDAPatch(DM da, const PetscInt n[DIM], DAPatch<DIM> *patch)
{
   PetscInt s[3] = {0, 0, 0}, m[3] = {1, 1, 1}, gs[3] = {0, 0, 0}, gm[3] = {1, 1, 1};

   PetscFunctionBeginUser;

   // DMDAGetCorners divides the dof back out, so these come out in cells
   PetscCall(DMDAGetCorners(da, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2]));
   PetscCall(DMDAGetGhostCorners(da, &gs[0], &gs[1], &gs[2], &gm[0], &gm[1], &gm[2]));
   for (PetscInt d = 0; d < DIM; d++) {
      patch->n[d] = n[d];
      patch->start[d] = s[d];
      patch->m[d] = m[d];
      patch->gstart[d] = gs[d];
      patch->gm[d] = gm[d];
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Where a ghosted local node's dof sits in the DM's GLOBAL numbering
//
// A 1D DMDA numbers globally in the natural order; a 2D/3D one does not: it
// numbers each rank's patch contiguously, lexicographically WITHIN the patch,
// so the natural ordering is wrong the moment there is more than one rank in a
// direction. The local-to-global map is the authority, and it covers the ghost
// nodes too, which is what lets a column point into a neighbouring patch.
// idx are GLOBAL grid coordinates inside the ghosted patch
template <int DIM> static inline PetscInt GlobalDof(const PetscInt *ltog, const DAPatch<DIM> &patch, \
   const PetscInt idx[DIM], PetscInt a, PetscInt dof)
{
   PetscInt off = 0;
   for (PetscInt d = DIM - 1; d >= 0; d--) off = off * patch.gm[d] + (idx[d] - patch.gstart[d]);
   return ltog[off * dof + a];
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check the DM's layout is the one everything downstream is written against:
//   (i)   the sizes agree with the phase space,
//   (ii)  the dof of a node are contiguous and angle-fastest,
//   (iii) this rank's owned nodes are numbered contiguously from its rstart in
//         the PATCH's OWN lexicographic order.
// (iii) is what makes DAPatch::cell_index the local cell order, which the
// per-cell views and RemovalTerm's r / n_angles rely on. It is NOT the global
// natural ordering, and asserting it is what stops that difference from
// surfacing as a wrong answer
template <int DIM> static PetscErrorCode CheckDALayout(DM da, const PhaseSpace &ps, const PetscInt *ltog, \
   const DAPatch<DIM> &patch)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)da);
   const PetscInt dof = ps.n_angles;
   const PetscInt local_cells = patch.local_cells();
   PetscInt dm_n[3] = {1, 1, 1}, dm_dof = 0, rstart = 0, rend = 0;
   Vec gv;

   PetscFunctionBeginUser;

   PetscCall(DMDAGetInfo(da, NULL, &dm_n[0], &dm_n[1], &dm_n[2], NULL, NULL, NULL, &dm_dof, \
      NULL, NULL, NULL, NULL, NULL));
   PetscCheck(dm_n[0] * dm_n[1] * dm_n[2] == ps.n_cells && dm_dof == ps.n_angles, comm, PETSC_ERR_PLIB, \
      "DMDA is %" PetscInt_FMT " cells x %" PetscInt_FMT " dof, phase space is %" PetscInt_FMT " cells x %" \
      PetscInt_FMT, dm_n[0] * dm_n[1] * dm_n[2], dm_dof, ps.n_cells, ps.n_angles);

   PetscCall(DMCreateGlobalVector(da, &gv));
   PetscCall(VecGetOwnershipRange(gv, &rstart, &rend));
   PetscCall(VecDestroy(&gv));
   PetscCheck(rend - rstart == local_cells * dof, comm, PETSC_ERR_PLIB, "DMDA owns %" PetscInt_FMT \
      " rows but its corners say %" PetscInt_FMT " nodes x %" PetscInt_FMT " dof", rend - rstart, local_cells, dof);

   for (PetscInt c = 0; c < local_cells; c++) {
      PetscInt idx[DIM];
      patch.cell_index(c, idx);
      for (PetscInt a = 0; a < dof; a++) {
         const PetscInt got = GlobalDof<DIM>(ltog, patch, idx, a, dof);
         PetscCheck(got == rstart + c * dof + a, comm, PETSC_ERR_PLIB, "DMDA numbers local cell %" PetscInt_FMT \
            " angle %" PetscInt_FMT " globally as %" PetscInt_FMT ", not the patch-lexicographic angle-fastest %" \
            PetscInt_FMT, c, a, got, rstart + c * dof + a);
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The two faces of one axis, [0] lo and [1] hi: their ids, their BCs and their
// per-angle inflow - the angle-integrated inflow shared over the ordinates,
// exactly as UboltFillSource shares a material's Source
struct AxisBC {
   PetscInt id[2] = {-1, -1};
   BCFace face[2];
   PetscScalar value[2] = {0.0, 0.0};
};

// Fill the per-axis BC table from the backend's face ids, checking the ids the
// spec names and the window shapes: a DIM-dimensional face takes DIM - 1
// tangential [lo, hi] pairs
template <int DIM> static PetscErrorCode BuildAxisBC(MPI_Comm comm, const BCSpec &bcs, \
   const PetscInt face_id[DIM][2], PetscScalar sum_weights, AxisBC axis_bc[DIM])
{
   PetscFunctionBeginUser;

   PetscCall(CheckStructuredFaceIds(comm, bcs, DIM));
   for (PetscInt d = 0; d < DIM; d++) {
      for (PetscInt side = 0; side < 2; side++) {
         const BCFace face = bcs.face(face_id[d][side]);
         PetscCheck(face.n_window_pairs == 0 || face.n_window_pairs == DIM - 1, comm, PETSC_ERR_ARG_WRONG, \
            "a %dD face takes %d tangential [lo, hi] window pair(s), face %" PetscInt_FMT " was given %" \
            PetscInt_FMT, DIM, DIM - 1, face_id[d][side], face.n_window_pairs);
         axis_bc[d].id[side] = face_id[d][side];
         axis_bc[d].face[side] = face;
         axis_bc[d].value[side] = (PetscScalar)face.inflow / sum_weights;
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Does a boundary cell whose centre sits at these tangential coordinates
// (the axes other than the face's, ascending) let the face's inflow in? An
// unwindowed face lets it in everywhere; a window is inclusive at both ends,
// the same membership test the painting uses
static inline bool InWindow(const BCFace &face, const PetscScalar *t, PetscInt n_t)
{
   if (face.n_window_pairs == 0) return true;
   for (PetscInt p = 0; p < n_t; p++) {
      if (!(PetscRealPart(t[p]) >= face.window[2 * p] && PetscRealPart(t[p]) <= face.window[2 * p + 1])) return false;
   }
   return true;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// What one (node, angle) row is. INTERIOR: no upwind neighbour outside the
// box. Under DIRICHLET_CELL an inflow row is DIRICHLET if any face it comes in
// through is vacuum (vacuum wins a corner, the FIRST vacuum axis in x, y, z
// order supplying the value and window), else REFLECT: identity minus the
// partner, mirrored over every axis it came in through - still one column.
// Under GHOST_FLUX it is GHOST, an ordinary unknown whose outside-pointing
// slots each take their own face's ghost value (a vacuum inflow moved to the
// rhs, or the mirrored angle in this cell), so a mixed corner needs no
// precedence rule
enum class RowKind { INTERIOR, DIRICHLET, REFLECT, GHOST };

template <int DIM> struct RowClass {
   RowKind kind = RowKind::INTERIOR;
   PetscInt upwind[DIM];       // the upwind neighbour's index on each axis
   bool has[DIM];              // the cosine on this axis is nonzero
   bool out[DIM];              // ... and the upwind neighbour is outside the box
   PetscInt side[DIM];         // the face it comes in through: 0 lo (cosine > 0), 1 hi
   PetscInt partner = -1;      // REFLECT
   PetscInt dirichlet_axis = -1;
   PetscInt ghost_mirror[DIM]; // GHOST: the mirrored angle through a reflective face, else -1
   bool ghost_vacuum[DIM];     // GHOST: a vacuum face on this axis supplies an inflow
};

// The upwind neighbour is behind the direction: below for a positive cosine,
// above for a negative one, and none for a zero cosine. The label decides only
// which BC family each face belongs to; whether a row is a boundary row at all
// is the physics - the sign of the direction against the face
template <int DIM> static RowClass<DIM> ClassifyRow(const PetscInt idx[DIM], PetscInt a, \
   const PetscScalar *const cosine[DIM], const PetscInt *const reflect[DIM], const PetscInt n[DIM], \
   const AxisBC axis_bc[DIM], bool ghost)
{
   RowClass<DIM> rc;
   bool any_out = false;
   for (PetscInt d = 0; d < DIM; d++) {
      const PetscScalar c = cosine[d][a];
      rc.upwind[d] = (c > 0) ? idx[d] - 1 : ((c < 0) ? idx[d] + 1 : idx[d]);
      rc.has[d] = (c != 0.0);
      rc.out[d] = rc.has[d] && (rc.upwind[d] < 0 || rc.upwind[d] >= n[d]);
      rc.side[d] = (c > 0) ? 0 : 1;
      rc.ghost_mirror[d] = -1;
      rc.ghost_vacuum[d] = false;
      any_out = any_out || rc.out[d];
   }
   if (!any_out) return rc;

   if (ghost) {
      rc.kind = RowKind::GHOST;
      for (PetscInt d = 0; d < DIM; d++) {
         if (!rc.out[d]) continue;
         if (axis_bc[d].face[rc.side[d]].type == BCType::REFLECT) rc.ghost_mirror[d] = reflect[d][a];
         else rc.ghost_vacuum[d] = true;
      }
      return rc;
   }

   for (PetscInt d = 0; d < DIM; d++) {
      if (rc.out[d] && axis_bc[d].face[rc.side[d]].type == BCType::VACUUM) {
         rc.kind = RowKind::DIRICHLET;
         rc.dirichlet_axis = d;
         return rc;
      }
   }

   PetscInt ap = a;
   for (PetscInt d = 0; d < DIM; d++) {
      if (rc.out[d]) ap = reflect[d][ap];
   }
   rc.kind = RowKind::REFLECT;
   rc.partner = ap;
   return rc;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The COO coordinates and the boundary rows: DIM + 1 slots per row, upwind on
// each axis in axis order then the diagonal LAST. A slot a row should not have
// (a zero cosine, a Dirichlet row, a ghost-flux vacuum inflow) keeps its
// place with a -1 row/col, so the value fills never branch; a slot pointing
// out through a ghost-flux reflective face is repointed at the mirrored angle
// in the same cell; a Dirichlet-cell reflective row puts its partner in slot 0
// and nulls the rest. Host, once. h[d] is the cell width, cosine/reflect the
// quadrature's per-axis cosines and reflection maps
template <int DIM> static PetscErrorCode BuildStructuredCOO(MPI_Comm comm, const DAPatch<DIM> &patch, \
   const PetscInt *ltog, PetscInt n_angles, const PetscScalar h[DIM], const PetscScalar *const cosine[DIM], \
   const PetscInt *const reflect[DIM], const AxisBC axis_bc[DIM], PetscBool ghost, std::vector<PetscInt> &oor, \
   std::vector<PetscInt> &ooc, BoundaryRows &rows)
{
   const PetscInt local_cells = patch.local_cells();
   const PetscInt local_rows = local_cells * n_angles;
   const PetscInt slots = DIM + 1;

   PetscFunctionBeginUser;

   oor.assign(slots * local_rows, 0);
   ooc.assign(slots * local_rows, 0);
   rows.reset(local_rows, ghost);

   for (PetscInt c = 0; c < local_cells; c++) {

      PetscInt idx[DIM];
      patch.cell_index(c, idx);
      // Node idx sits at idx * h, so a cell centre is (idx + 1/2) h
      PetscScalar centre[DIM];
      for (PetscInt d = 0; d < DIM; d++) centre[d] = ((PetscScalar)idx[d] + 0.5) * h[d];

      for (PetscInt a = 0; a < n_angles; a++) {

         const PetscInt r = c * n_angles + a;
         const PetscInt row = GlobalDof<DIM>(ltog, patch, idx, a, n_angles);
         const PetscInt s0 = slots * r;
         const RowClass<DIM> rc = ClassifyRow<DIM>(idx, a, cosine, reflect, patch.n, axis_bc, ghost);
         if (rc.kind == RowKind::DIRICHLET || rc.kind == RowKind::REFLECT) rows.is_bc_row[r] = 1;

         // The tangential centre coordinates of this cell on the face normal
         // to `axis`: the other axes, ascending
         auto tangential = [&](PetscInt axis, PetscScalar t[3]) {
            PetscInt n_t = 0;
            for (PetscInt d = 0; d < DIM; d++) {
               if (d != axis) t[n_t++] = centre[d];
            }
         };

         if (rc.kind == RowKind::GHOST) {
            // One rhs contribution per vacuum face the direction enters
            // through - a corner cell gets several - each |cosine| / h times
            // that face's per-angle inflow, windowed; summed in axis order
            for (PetscInt d = 0; d < DIM; d++) {
               if (!rc.ghost_vacuum[d]) continue;
               PetscScalar t[3];
               tangential(d, t);
               const PetscScalar coef = PetscAbsScalar(cosine[d][a]) / h[d];
               if (InWindow(axis_bc[d].face[rc.side[d]], t, DIM - 1)) \
                  rows.ghost_inflow[r] += coef * axis_bc[d].value[rc.side[d]];
            }
         }

         if (rc.kind == RowKind::DIRICHLET) {
            const PetscInt d = rc.dirichlet_axis;
            PetscScalar t[3];
            tangential(d, t);
            rows.dirichlet_value[r] = InWindow(axis_bc[d].face[rc.side[d]], t, DIM - 1) ? \
               axis_bc[d].value[rc.side[d]] : (PetscScalar)0.0;
         }

         if (rc.kind == RowKind::REFLECT) {
            // The partner is outgoing through every face this direction came
            // in through, so its row is a real unknown - unless the grid is a
            // single cell wide on an axis and the opposite face catches it,
            // which there is no sensible matrix for
            const RowClass<DIM> rc2 = ClassifyRow<DIM>(idx, rc.partner, cosine, reflect, patch.n, axis_bc, false);
            PetscCheck(rc2.kind == RowKind::INTERIOR, comm, PETSC_ERR_SUP, "the reflection partner of local cell %" \
               PetscInt_FMT " angle %" PetscInt_FMT " is itself a boundary row - a reflective face on a " \
               "single-cell-wide direction is not supported", c, a);

            // Same cell, mirrored angle - an owned node, so always rank-local
            oor[s0] = row;
            ooc[s0] = GlobalDof<DIM>(ltog, patch, idx, rc.partner, n_angles);
            rows.reflect_slot[r] = s0;
            for (PetscInt d = 1; d < DIM; d++) {
               oor[s0 + d] = -1;
               ooc[s0 + d] = -1;
            }
         } else {
            for (PetscInt d = 0; d < DIM; d++) {
               const bool null_slot = rc.kind == RowKind::DIRICHLET || !rc.has[d] || \
                  (rc.kind == RowKind::GHOST && rc.out[d] && rc.ghost_mirror[d] < 0);
               PetscInt col = -1;
               if (!null_slot && rc.out[d]) col = GlobalDof<DIM>(ltog, patch, idx, rc.ghost_mirror[d], n_angles);
               else if (!null_slot) {
                  PetscInt up[DIM];
                  for (PetscInt e = 0; e < DIM; e++) up[e] = idx[e];
                  up[d] = rc.upwind[d];
                  col = GlobalDof<DIM>(ltog, patch, up, a, n_angles);
               }
               oor[s0 + d] = null_slot ? -1 : row;
               ooc[s0 + d] = col;
            }
         }

         oor[s0 + DIM] = row;
         ooc[s0 + DIM] = row;

         // A stray -1 column would silently drop a coefficient rather than
         // fail. A deliberately nulled slot has its ROW at -1 too, which is
         // what separates it from a lookup that failed
         for (PetscInt d = 0; d < DIM; d++) {
            PetscCheck(ooc[s0 + d] >= 0 || oor[s0 + d] == -1, comm, PETSC_ERR_PLIB, "no global index for the " \
               "axis-%" PetscInt_FMT " upwind neighbour of local cell %" PetscInt_FMT, d, c);
         }
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda. The boxes ride along flat (UboltFlattenBoxes)
template <int DIM> static void PaintStructuredBoxesKernel(PetscIntKokkosView mat_id_d, PetscInt background_material, \
   PetscScalarKokkosView box_lohi_d, PetscIntKokkosView box_material_d, PetscInt n_boxes, PetscInt start_x, \
   PetscInt start_y, PetscInt start_z, PetscInt m_x, PetscInt m_y, PetscScalar h_x, PetscScalar h_y, \
   PetscScalar h_z, PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

         // The centre of local cell c, in the patch's lexicographic order
         PetscScalar centre[3] = {0.0, 0.0, 0.0};
         centre[0] = ((PetscScalar)(start_x + c % m_x) + 0.5) * h_x;
         if (DIM == 2) centre[1] = ((PetscScalar)(start_y + c / m_x) + 0.5) * h_y;
         if (DIM == 3) {
            centre[1] = ((PetscScalar)(start_y + (c / m_x) % m_y) + 0.5) * h_y;
            centre[2] = ((PetscScalar)(start_z + c / (m_x * m_y)) + 0.5) * h_z;
         }

         PetscInt m = background_material;
         for (PetscInt b = 0; b < n_boxes; b++) {
            bool inside = true;
            for (PetscInt d = 0; d < DIM; d++) {
               inside = inside && centre[d] >= box_lohi_d(2 * DIM * b + 2 * d) && \
                  centre[d] <= box_lohi_d(2 * DIM * b + 2 * d + 1);
            }
            if (inside) m = box_material_d(b);
         }
         mat_id_d(c) = m;
      });
}

// Paint material indices onto this rank's cells: the background everywhere,
// then the boxes in order with the later ones winning, by cell centre
template <class Box> static PetscErrorCode PaintStructuredBoxes(MPI_Comm comm, PetscInt background_material, \
   const std::vector<Box> &boxes, const PetscInt start[3], const PetscInt m[3], const PetscScalar h[3], \
   PetscInt local_cells, PetscIntKokkosView &mat_id_d)
{
   constexpr PetscInt DIM = Box::dim;
   const PetscInt n_boxes = (PetscInt)boxes.size();
   std::vector<PetscScalar> box_lohi;
   std::vector<PetscInt> box_material;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(background_material >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);
   UboltFlattenBoxes(boxes, box_lohi, box_material);
   for (PetscInt b = 0; b < n_boxes; b++) {
      PetscCheck(box_material[b] >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "box %" PetscInt_FMT "'s material index %" PetscInt_FMT " is negative", b, box_material[b]);
      for (PetscInt d = 0; d < DIM; d++) {
         PetscCheck(PetscRealPart(box_lohi[2 * DIM * b + 2 * d]) <= PetscRealPart(box_lohi[2 * DIM * b + 2 * d + 1]), \
            comm, PETSC_ERR_ARG_OUTOFRANGE, "box %" PetscInt_FMT " is inside out on axis %" PetscInt_FMT \
            " - give each axis as lo, hi with lo <= hi", b, d);
      }
   }

   PetscScalarKokkosView box_lohi_d("box_lohi_d", 2 * DIM * n_boxes);
   PetscIntKokkosView box_material_d("box_material_d", n_boxes);
   if (n_boxes > 0) {
      PetscScalarKokkosViewHostUnmanaged box_lohi_h(box_lohi.data(), 2 * DIM * n_boxes);
      PetscIntKokkosViewHostUnmanaged box_material_h(box_material.data(), n_boxes);
      Kokkos::deep_copy(box_lohi_d, box_lohi_h);
      Kokkos::deep_copy(box_material_d, box_material_h);
   }

   mat_id_d = PetscIntKokkosView("mat_id_d", local_cells);
   PaintStructuredBoxesKernel<DIM>(mat_id_d, background_material, box_lohi_d, box_material_d, n_boxes, \
      start[0], start[1], start[2], m[0], m[1], h[0], h[1], h[2], local_cells);

   PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
