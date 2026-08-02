#include "ubolt/structured_fd_3d.hpp"
#include "petsc_kokkos.hpp"
#include <petscdmda.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build a 3D DMDA over the cells: one dof per angle, star stencil of width 1
// because the upwind operator reaches one neighbour per axis and never an edge
// or corner
//
// As in 1D and 2D we deliberately do NOT call DMSetFromOptions - it would
// expose -da_grid_x/-da_grid_y/-da_grid_z, which could resize the mesh out
// from under the PhaseSpace that every other object has already been sized from
//
// The DM chooses the 3D processor decomposition (m = n = p = PETSC_DECIDE) as
// well as the split within it, and the PhaseSpace is told what it decided
static PetscErrorCode CreateDA(MPI_Comm comm, const PhaseSpace &ps, PetscInt n_cells_x, \
   PetscInt n_cells_y, PetscInt n_cells_z, DM *da)
{
   PetscFunctionBeginUser;

   PetscCall(DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, \
      DMDA_STENCIL_STAR, n_cells_x, n_cells_y, n_cells_z, PETSC_DECIDE, PETSC_DECIDE, \
      PETSC_DECIDE, ps.n_angles, 1, NULL, NULL, NULL, da));
   PetscCall(DMSetUp(*da));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Where a ghosted local node's dof sits in the DM's GLOBAL numbering
//
// Same story as 2D: the DMDA numbers each rank's patch contiguously,
// lexicographically WITHIN the patch, so the natural ((k * ny + j) * nx + i)
// ordering is wrong the moment there is more than one rank in a direction. The
// local-to-global map is the authority, and it covers the ghost nodes too,
// which is what lets a column point into a neighbouring rank's patch
//
// (i, j, k) are GLOBAL grid coordinates and must lie in the ghosted patch
static inline PetscInt GlobalDof(const PetscInt *ltog, PetscInt i, PetscInt j, PetscInt k, \
   PetscInt a, PetscInt gxs, PetscInt gys, PetscInt gzs, PetscInt gxm, PetscInt gym, PetscInt dof)
{
   return ltog[(((k - gzs) * gym + (j - gys)) * gxm + (i - gxs)) * dof + a];
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check the DM's layout is the one everything downstream is written against
//
// The same three properties the 2D check asserts, one axis wider:
//   (i)  the sizes agree with the phase space,
//   (ii) the dof of a node are contiguous and angle-fastest,
//   (iii) this rank's owned nodes are numbered contiguously from its rstart in
//         the PATCH's OWN lexicographic order.
// (iii) is what makes "local cell index = ((k - zs) * ym + (j - ys)) * xm +
// (i - xs)" true, which is the indexing the per-cell xsection views and
// RemovalTerm's r / n_angles both rely on. It is NOT the global natural
// ordering, and asserting it here is what stops that difference from surfacing
// as a wrong answer
static PetscErrorCode CheckDALayout(DM da, const PhaseSpace &ps, const PetscInt *ltog, \
   PetscInt xs, PetscInt ys, PetscInt zs, PetscInt xm, PetscInt ym, PetscInt zm, \
   PetscInt gxs, PetscInt gys, PetscInt gzs, PetscInt gxm, PetscInt gym)
{
   const PetscInt dof = ps.n_angles;
   PetscInt dm_x = 0, dm_y = 0, dm_z = 0, dm_dof = 0, rstart = 0, rend = 0;
   Vec gv;

   PetscFunctionBeginUser;

   PetscCall(DMDAGetInfo(da, NULL, &dm_x, &dm_y, &dm_z, NULL, NULL, NULL, &dm_dof, \
      NULL, NULL, NULL, NULL, NULL));
   PetscCheck(dm_x * dm_y * dm_z == ps.n_cells && dm_dof == ps.n_angles, \
      PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "DMDA is %" PetscInt_FMT " x %" \
      PetscInt_FMT " x %" PetscInt_FMT " cells x %" PetscInt_FMT " dof, phase space is %" \
      PetscInt_FMT " cells x %" PetscInt_FMT, dm_x, dm_y, dm_z, dm_dof, ps.n_cells, ps.n_angles);

   PetscCall(DMCreateGlobalVector(da, &gv));
   PetscCall(VecGetOwnershipRange(gv, &rstart, &rend));
   PetscCall(VecDestroy(&gv));
   PetscCheck(rend - rstart == xm * ym * zm * dof, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, \
      "DMDA owns %" PetscInt_FMT " rows but its corners say %" PetscInt_FMT " nodes x %" \
      PetscInt_FMT " dof", rend - rstart, xm * ym * zm, dof);

   // Walk the owned patch in the order we are about to call local cell order
   for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
         for (PetscInt i = xs; i < xs + xm; i++) {
            const PetscInt local_cell = ((k - zs) * ym + (j - ys)) * xm + (i - xs);
            for (PetscInt a = 0; a < dof; a++) {
               const PetscInt expected = rstart + local_cell * dof + a;
               const PetscInt got = GlobalDof(ltog, i, j, k, a, gxs, gys, gzs, gxm, gym, dof);
               PetscCheck(got == expected, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, \
                  "DMDA numbers node (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT \
                  ") angle %" PetscInt_FMT " globally as %" PetscInt_FMT ", not the " \
                  "patch-lexicographic angle-fastest %" PetscInt_FMT, i, j, k, a, got, expected);
            }
         }
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum class RowKind { INTERIOR, DIRICHLET, REFLECT };

// The upwind neighbour is behind the direction on each axis: to the
// left/front/below for a positive cosine, to the right/back/above for a
// negative one. A zero cosine has no upwind neighbour on that axis at all
struct Upwind {
   PetscInt upwind_i = 0;
   PetscInt upwind_j = 0;
   PetscInt upwind_k = 0;
   PetscBool has_x = PETSC_FALSE;
   PetscBool has_y = PETSC_FALSE;
   PetscBool has_z = PETSC_FALSE;
};

// Classify one (node, angle) row - filling in its upwind neighbours as the
// side product, so the sign convention above is stated once - and for a
// reflective row say which angle it couples to
//
// The label decides only which BC family each face belongs to; whether a row
// is a boundary row at all is the physics - the existing upwind test, which is
// the sign of the direction against the face. A direction incoming through no
// face is an interior unknown. If ANY incoming face is vacuum the row is
// Dirichlet - vacuum wins at mixed edges and corners - else the direction
// reflects in every incoming axis: an edge between two reflective faces flips
// two cosines and a corner between three flips all three, which is still one
// partner angle and so one column
static RowKind ClassifyRow(PetscInt i, PetscInt j, PetscInt k, PetscInt a, \
   const PetscScalar *mu, const PetscScalar *eta, const PetscScalar *xi, \
   PetscInt n_cells_x, PetscInt n_cells_y, PetscInt n_cells_z, \
   BCType left, BCType right, BCType front, BCType back, BCType bottom, BCType top, \
   const PetscInt *reflect_mu, const PetscInt *reflect_eta, const PetscInt *reflect_xi, \
   Upwind *upwind, PetscInt *partner)
{
   upwind->upwind_i = (mu[a] > 0) ? i - 1 : ((mu[a] < 0) ? i + 1 : i);
   upwind->upwind_j = (eta[a] > 0) ? j - 1 : ((eta[a] < 0) ? j + 1 : j);
   upwind->upwind_k = (xi[a] > 0) ? k - 1 : ((xi[a] < 0) ? k + 1 : k);
   upwind->has_x = (PetscBool)(mu[a] != 0.0);
   upwind->has_y = (PetscBool)(eta[a] != 0.0);
   upwind->has_z = (PetscBool)(xi[a] != 0.0);
   const PetscBool x_outside = (PetscBool)(upwind->has_x && \
      (upwind->upwind_i < 0 || upwind->upwind_i >= n_cells_x));
   const PetscBool y_outside = (PetscBool)(upwind->has_y && \
      (upwind->upwind_j < 0 || upwind->upwind_j >= n_cells_y));
   const PetscBool z_outside = (PetscBool)(upwind->has_z && \
      (upwind->upwind_k < 0 || upwind->upwind_k >= n_cells_z));

   *partner = -1;
   if (!x_outside && !y_outside && !z_outside) return RowKind::INTERIOR;

   // Which face the direction comes in through on each outside axis. The
   // bottom/top faces are the Z ones - see the FACE_* comment in the header
   const BCType x_bc = (mu[a] > 0) ? left : right;
   const BCType y_bc = (eta[a] > 0) ? front : back;
   const BCType z_bc = (xi[a] > 0) ? bottom : top;
   if ((x_outside && x_bc == BCType::VACUUM) || (y_outside && y_bc == BCType::VACUUM) || \
       (z_outside && z_bc == BCType::VACUUM)) return RowKind::DIRICHLET;

   PetscInt ap = a;
   if (x_outside) ap = reflect_mu[ap];
   if (y_outside) ap = reflect_eta[ap];
   if (z_outside) ap = reflect_xi[ap];
   *partner = ap;
   return RowKind::REFLECT;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build the COO sparsity and the slot maps
// This happens on the host but we only need to do it once
PetscErrorCode StructuredFD3D::create(MPI_Comm comm, PhaseSpace &ps, PetscInt n_cells_x, \
   PetscInt n_cells_y, PetscInt n_cells_z, PetscReal length_x, PetscReal length_y, \
   PetscReal length_z, const SNQuadrature3D &quad, const BCSpec &bcs)
{
   const PetscInt *ltog = nullptr;
   ISLocalToGlobalMapping ltog_map = NULL;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck(n_cells_x > 0 && n_cells_y > 0 && n_cells_z > 0 && \
      n_cells_x * n_cells_y * n_cells_z == ps.n_cells, comm, PETSC_ERR_ARG_INCOMP, \
      "a %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT " grid is not the phase " \
      "space's %" PetscInt_FMT " cells", n_cells_x, n_cells_y, n_cells_z, ps.n_cells);

   comm_ = comm;
   n_cells_x_ = n_cells_x;
   n_cells_y_ = n_cells_y;
   n_cells_z_ = n_cells_z;
   // Uniform grid so every cell has the same width
   dx_ = length_x / n_cells_x;
   dy_ = length_y / n_cells_y;
   dz_ = length_z / n_cells_z;

   const PetscInt n_angles = ps.n_angles;
   const PetscScalar *mu = quad.mu_host();
   const PetscScalar *eta = quad.eta_host();
   const PetscScalar *xi = quad.xi_host();

   // The DM decides the parallel decomposition, so building it has to come
   // before anything that reads local_cells - starting with the PhaseSpace,
   // which we fill in rather than read
   PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
   PetscInt gxs = 0, gys = 0, gzs = 0, gxm = 0, gym = 0;
   PetscCall(CreateDA(comm, ps, n_cells_x, n_cells_y, n_cells_z, &dm_));
   // DMDAGetCorners divides the dof back out, so these come out in cells
   PetscCall(DMDAGetCorners(dm_, &xs, &ys, &zs, &xm, &ym, &zm));
   // The ghosted patch is what the local-to-global map is indexed over. With
   // DM_BOUNDARY_NONE it is clipped at the physical boundary, so a node just
   // outside the box is not in it at all - which is fine: a Dirichlet row
   // never asks for a column, and the column a reflective row asks for is its
   // own cell's mirrored angle, an owned node
   PetscCall(DMDAGetGhostCorners(dm_, &gxs, &gys, &gzs, &gxm, &gym, NULL));
   cell_start_x_ = xs;
   cell_start_y_ = ys;
   cell_start_z_ = zs;
   local_cells_x_ = xm;
   local_cells_y_ = ym;
   local_cells_z_ = zm;
   ps.local_cells = xm * ym * zm;

   // The DM owns the mesh, so it carries the node positions too: node
   // (i, j, k) sits at (i dx, j dy, k dz), the same convention the closed-form
   // check in tests/verify_3dk.kokkos.cxx uses. Nothing in the solve reads
   // these - they are for output (the scalar flux VTK writer picks them up). A
   // direction with a single node has no spacing to define and PETSc would
   // divide by n - 1, so that degenerate grid keeps no coordinates
   if (n_cells_x > 1 && n_cells_y > 1 && n_cells_z > 1) PetscCall(DMDASetUniformCoordinates(dm_, \
      0.0, length_x - length_x / n_cells_x, 0.0, length_y - length_y / n_cells_y, \
      0.0, length_z - length_z / n_cells_z));

   PetscCall(DMGetLocalToGlobalMapping(dm_, &ltog_map));
   PetscCall(ISLocalToGlobalMappingGetIndices(ltog_map, &ltog));
   PetscCall(CheckDALayout(dm_, ps, ltog, xs, ys, zs, xm, ym, zm, gxs, gys, gzs, gxm, gym));

   ps_ = ps;
   const PetscInt local_rows = ps.local_rows();

   // ~~~~~~~~~~
   // COO coordinates - four entries per row, in slot order
   // upwind-x, upwind-y, upwind-z, diagonal
   // ~~~~~~~~~~
   // A -1 row/col index in the COO format says just ignore this entry. Rows
   // that should not have one of these entries - a Dirichlet boundary row, or
   // a direction with no component along that axis - keep the slot and hand
   // the preallocation a -1, so the value fills never branch. A reflective
   // boundary row instead repurposes its first slot for the coupling to the
   // mirrored angle
   oor_.assign(4 * local_rows, 0);
   ooc_.assign(4 * local_rows, 0);
   std::vector<PetscInt> is_bc_row(local_rows, 0);
   std::vector<PetscInt> reflect_slot(local_rows, -1);

   const BCType left   = bcs.type(FACE_LEFT);
   const BCType right  = bcs.type(FACE_RIGHT);
   const BCType front  = bcs.type(FACE_FRONT);
   const BCType back   = bcs.type(FACE_BACK);
   const BCType bottom = bcs.type(FACE_BOTTOM);
   const BCType top    = bcs.type(FACE_TOP);
   const PetscInt *reflect_mu = quad.reflect_mu_host();
   const PetscInt *reflect_eta = quad.reflect_eta_host();
   const PetscInt *reflect_xi = quad.reflect_xi_host();

   for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
         for (PetscInt i = xs; i < xs + xm; i++) {

            const PetscInt local_cell = ((k - zs) * ym + (j - ys)) * xm + (i - xs);

            for (PetscInt a = 0; a < n_angles; a++) {

               const PetscInt r = local_cell * n_angles + a;
               const PetscInt row = GlobalDof(ltog, i, j, k, a, gxs, gys, gzs, gxm, gym, n_angles);

               // An inflow row - this direction enters the box through a face
               // this node is on - has its equation replaced by the boundary
               // condition: the identity on a Dirichlet (vacuum) row, the
               // reflection condition psi(a) - psi(partner) = 0 on a
               // reflective one. Edge and corner nodes classify through any of
               // their axes
               Upwind upwind;
               PetscInt partner = -1;
               const RowKind kind = ClassifyRow(i, j, k, a, mu, eta, xi, n_cells_x, n_cells_y, \
                  n_cells_z, left, right, front, back, bottom, top, reflect_mu, reflect_eta, \
                  reflect_xi, &upwind, &partner);
               if (kind != RowKind::INTERIOR) is_bc_row[r] = 1;

               if (kind == RowKind::REFLECT)
               {
                  // The partner is outgoing through every face this direction
                  // came in through, so its row is a real unknown - unless a
                  // direction of the grid is a single cell wide and the
                  // opposite face catches it, which there is no sensible
                  // matrix for
                  Upwind upwind2;
                  PetscInt partner2 = -1;
                  PetscCheck(ClassifyRow(i, j, k, partner, mu, eta, xi, n_cells_x, n_cells_y, \
                     n_cells_z, left, right, front, back, bottom, top, reflect_mu, reflect_eta, \
                     reflect_xi, &upwind2, &partner2) == RowKind::INTERIOR, \
                     comm, PETSC_ERR_SUP, "the reflection partner of node (%" PetscInt_FMT ", %" \
                     PetscInt_FMT ", %" PetscInt_FMT ") angle %" PetscInt_FMT " is itself a " \
                     "boundary row - a reflective face on a single-cell-wide direction is not " \
                     "supported", i, j, k, a);

                  // Same cell, mirrored angle - an owned node, so always in
                  // the ltog map and always rank-local
                  oor_[4 * r] = row;
                  ooc_[4 * r] = GlobalDof(ltog, i, j, k, partner, gxs, gys, gzs, gxm, gym, n_angles);
                  reflect_slot[r] = 4 * r;

                  oor_[4 * r + 1] = -1;
                  ooc_[4 * r + 1] = -1;
                  oor_[4 * r + 2] = -1;
                  ooc_[4 * r + 2] = -1;
               }
               else
               {
                  const PetscBool null_x = (PetscBool)(kind == RowKind::DIRICHLET || !upwind.has_x);
                  const PetscBool null_y = (PetscBool)(kind == RowKind::DIRICHLET || !upwind.has_y);
                  const PetscBool null_z = (PetscBool)(kind == RowKind::DIRICHLET || !upwind.has_z);

                  oor_[4 * r]     = null_x ? -1 : row;
                  ooc_[4 * r]     = null_x ? -1 : \
                     GlobalDof(ltog, upwind.upwind_i, j, k, a, gxs, gys, gzs, gxm, gym, n_angles);

                  oor_[4 * r + 1] = null_y ? -1 : row;
                  ooc_[4 * r + 1] = null_y ? -1 : \
                     GlobalDof(ltog, i, upwind.upwind_j, k, a, gxs, gys, gzs, gxm, gym, n_angles);

                  oor_[4 * r + 2] = null_z ? -1 : row;
                  ooc_[4 * r + 2] = null_z ? -1 : \
                     GlobalDof(ltog, i, j, upwind.upwind_k, a, gxs, gys, gzs, gxm, gym, n_angles);
               }

               oor_[4 * r + 3] = row;
               ooc_[4 * r + 3] = row;

               // A ghost node the star stencil does not communicate carries a
               // -1 in the map. We only ever ask for face neighbours or an
               // owned node, so this cannot fire - but a stray -1 column would
               // silently drop a coefficient rather than fail, so say it out
               // loud. A slot that was deliberately nulled has its ROW index
               // at -1 too, which is what separates it from a lookup that
               // failed
               PetscCheck(ooc_[4 * r] >= 0 || oor_[4 * r] == -1, comm, PETSC_ERR_PLIB, \
                  "no global index for the x-upwind neighbour of node (%" PetscInt_FMT ", %" \
                  PetscInt_FMT ", %" PetscInt_FMT ")", i, j, k);
               PetscCheck(ooc_[4 * r + 1] >= 0 || oor_[4 * r + 1] == -1, comm, PETSC_ERR_PLIB, \
                  "no global index for the y-upwind neighbour of node (%" PetscInt_FMT ", %" \
                  PetscInt_FMT ", %" PetscInt_FMT ")", i, j, k);
               PetscCheck(ooc_[4 * r + 2] >= 0 || oor_[4 * r + 2] == -1, comm, PETSC_ERR_PLIB, \
                  "no global index for the z-upwind neighbour of node (%" PetscInt_FMT ", %" \
                  PetscInt_FMT ", %" PetscInt_FMT ")", i, j, k);
            }
         }
      }
   }

   PetscCall(ISLocalToGlobalMappingRestoreIndices(ltog_map, &ltog));

   // Slot maps - row r owns COO slots 4r (upwind-x), 4r + 1 (upwind-y),
   // 4r + 2 (upwind-z) and 4r + 3 (diagonal)
   PetscCall(set_uniform_pattern(4, is_bc_row, reflect_slot));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda. The box list rides along as flat views -
// coordinates as (x0, x1, y0, y1, z0, z1) sextuples - because a view of
// structs is not one of the typedefs types.hpp allows
static void PaintBoxesKernel(PetscIntKokkosView mat_id_d, PetscInt background_material, \
   PetscScalarKokkosView box_xyz_d, PetscIntKokkosView box_material_d, PetscInt n_boxes, \
   PetscInt cell_start_x, PetscInt cell_start_y, PetscInt cell_start_z, \
   PetscInt local_cells_x, PetscInt local_cells_y, \
   PetscScalar dx, PetscScalar dy, PetscScalar dz, PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

         // The centre of local cell c - the local ordering is the patch's own
         // lexicographic one, and node (i, j, k) sits at (i dx, j dy, k dz)
         const PetscInt i = c % local_cells_x;
         const PetscInt j = (c / local_cells_x) % local_cells_y;
         const PetscInt k = c / (local_cells_x * local_cells_y);
         const PetscScalar xc = ((PetscScalar)(cell_start_x + i) + 0.5) * dx;
         const PetscScalar yc = ((PetscScalar)(cell_start_y + j) + 0.5) * dy;
         const PetscScalar zc = ((PetscScalar)(cell_start_z + k) + 0.5) * dz;

         PetscInt m = background_material;
         for (PetscInt b = 0; b < n_boxes; b++) {
            if (xc >= box_xyz_d(6 * b)     && xc <= box_xyz_d(6 * b + 1) && \
                yc >= box_xyz_d(6 * b + 2) && yc <= box_xyz_d(6 * b + 3) && \
                zc >= box_xyz_d(6 * b + 4) && zc <= box_xyz_d(6 * b + 5)) m = box_material_d(b);
         }
         mat_id_d(c) = m;
      });
}

// See the declaration for the painting rules (background, later boxes win,
// membership by cell centre)
PetscErrorCode StructuredFD3D::paint_boxes(PetscInt background_material, \
   const std::vector<MaterialBox3D> &boxes, PetscIntKokkosView &mat_id_d) const
{
   const PetscInt n_boxes = (PetscInt)boxes.size();

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps_.check_decomposed());

   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);
   for (PetscInt b = 0; b < n_boxes; b++) {
      PetscCheck(boxes[b].material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
         "box %" PetscInt_FMT "'s material index %" PetscInt_FMT " is negative", b, boxes[b].material);
      PetscCheck(boxes[b].x0 <= boxes[b].x1 && boxes[b].y0 <= boxes[b].y1 && \
         boxes[b].z0 <= boxes[b].z1, comm_, PETSC_ERR_ARG_OUTOFRANGE, "box %" PetscInt_FMT \
         " is inside out - give it as x0,x1,y0,y1,z0,z1 with x0 <= x1, y0 <= y1 and z0 <= z1", b);
   }

   // Flatten the boxes for the device
   std::vector<PetscScalar> box_xyz(6 * n_boxes);
   std::vector<PetscInt> box_material(n_boxes);
   for (PetscInt b = 0; b < n_boxes; b++) {
      box_xyz[6 * b]     = boxes[b].x0;
      box_xyz[6 * b + 1] = boxes[b].x1;
      box_xyz[6 * b + 2] = boxes[b].y0;
      box_xyz[6 * b + 3] = boxes[b].y1;
      box_xyz[6 * b + 4] = boxes[b].z0;
      box_xyz[6 * b + 5] = boxes[b].z1;
      box_material[b] = boxes[b].material;
   }

   PetscScalarKokkosView box_xyz_d("box_xyz_d", 6 * n_boxes);
   PetscIntKokkosView box_material_d("box_material_d", n_boxes);
   if (n_boxes > 0) {
      PetscScalarKokkosViewHostUnmanaged box_xyz_h(box_xyz.data(), 6 * n_boxes);
      PetscIntKokkosViewHostUnmanaged box_material_h(box_material.data(), n_boxes);
      Kokkos::deep_copy(box_xyz_d, box_xyz_h);
      Kokkos::deep_copy(box_material_d, box_material_h);
   }

   mat_id_d = PetscIntKokkosView("mat_id_d", ps_.local_cells);
   PaintBoxesKernel(mat_id_d, background_material, box_xyz_d, box_material_d, n_boxes, \
      cell_start_x_, cell_start_y_, cell_start_z_, local_cells_x_, local_cells_y_, \
      dx_, dy_, dz_, ps_.local_cells);

   PetscFunctionReturn(PETSC_SUCCESS);
}
