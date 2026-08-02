#include "ubolt/structured_fd_2d.hpp"
#include <petscdmda.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build a 2D DMDA over the cells: one dof per angle, star stencil of width 1
// because the upwind operator reaches one neighbour per axis and never a corner
//
// As in 1D we deliberately do NOT call DMSetFromOptions - it would expose
// -da_grid_x/-da_grid_y, which could resize the mesh out from under the
// PhaseSpace that every other object has already been sized from
//
// The DM chooses the 2D processor decomposition (m = n = PETSC_DECIDE) as well
// as the split within it, and the PhaseSpace is told what it decided
static PetscErrorCode CreateDA(MPI_Comm comm, const PhaseSpace &ps, PetscInt n_cells_x, PetscInt n_cells_y, DM *da)
{
   PetscFunctionBeginUser;

   PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, \
      n_cells_x, n_cells_y, PETSC_DECIDE, PETSC_DECIDE, ps.n_angles, 1, NULL, NULL, da));
   PetscCall(DMSetUp(*da));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Where a ghosted local node's dof sits in the DM's GLOBAL numbering
//
// This is the piece 1D did not need. A 1D DMDA numbers globally in the natural
// order, so the row of (cell, angle) was arithmetic. A 2D DMDA does not: it
// numbers each rank's patch contiguously, lexicographically WITHIN the patch,
// so the natural (j * n_cells_x + i) ordering is wrong the moment there is more
// than one rank in a direction. The local-to-global map is the authority, and
// it covers the ghost nodes too, which is what lets a column point into a
// neighbouring rank's patch
//
// (i, j) are GLOBAL grid coordinates and must lie in the ghosted patch
static inline PetscInt GlobalDof(const PetscInt *ltog, PetscInt i, PetscInt j, PetscInt a, \
   PetscInt gxs, PetscInt gys, PetscInt gxm, PetscInt dof)
{
   return ltog[((j - gys) * gxm + (i - gxs)) * dof + a];
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check the DM's layout is the one everything downstream is written against
//
// Not a copy of the 1D check: the property that holds in 2D is weaker. What we
// need, and what this asserts, is
//   (i)  the sizes agree with the phase space,
//   (ii) the dof of a node are contiguous and angle-fastest,
//   (iii) this rank's owned nodes are numbered contiguously from its rstart in
//         the PATCH's OWN lexicographic order.
// (iii) is what makes "local cell index = (j - ys) * xm + (i - xs)" true, which
// is the indexing the per-cell xsection views and RemovalTerm's r / n_angles
// both rely on. It is NOT the global natural ordering, and asserting it here is
// what stops that difference from surfacing as a wrong answer
static PetscErrorCode CheckDALayout(DM da, const PhaseSpace &ps, const PetscInt *ltog, \
   PetscInt xs, PetscInt ys, PetscInt xm, PetscInt ym, PetscInt gxs, PetscInt gys, PetscInt gxm)
{
   const PetscInt dof = ps.n_angles;
   PetscInt dm_x = 0, dm_y = 0, dm_dof = 0, rstart = 0, rend = 0;
   Vec gv;

   PetscFunctionBeginUser;

   PetscCall(DMDAGetInfo(da, NULL, &dm_x, &dm_y, NULL, NULL, NULL, NULL, &dm_dof, \
      NULL, NULL, NULL, NULL, NULL));
   PetscCheck(dm_x * dm_y == ps.n_cells && dm_dof == ps.n_angles, PetscObjectComm((PetscObject)da), \
      PETSC_ERR_PLIB, "DMDA is %" PetscInt_FMT " x %" PetscInt_FMT " cells x %" PetscInt_FMT \
      " dof, phase space is %" PetscInt_FMT " cells x %" PetscInt_FMT, dm_x, dm_y, dm_dof, \
      ps.n_cells, ps.n_angles);

   PetscCall(DMCreateGlobalVector(da, &gv));
   PetscCall(VecGetOwnershipRange(gv, &rstart, &rend));
   PetscCall(VecDestroy(&gv));
   PetscCheck(rend - rstart == xm * ym * dof, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, \
      "DMDA owns %" PetscInt_FMT " rows but its corners say %" PetscInt_FMT " nodes x %" \
      PetscInt_FMT " dof", rend - rstart, xm * ym, dof);

   // Walk the owned patch in the order we are about to call local cell order
   for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {
         const PetscInt local_cell = (j - ys) * xm + (i - xs);
         for (PetscInt a = 0; a < dof; a++) {
            const PetscInt expected = rstart + local_cell * dof + a;
            const PetscInt got = GlobalDof(ltog, i, j, a, gxs, gys, gxm, dof);
            PetscCheck(got == expected, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, \
               "DMDA numbers node (%" PetscInt_FMT ", %" PetscInt_FMT ") angle %" PetscInt_FMT \
               " globally as %" PetscInt_FMT ", not the patch-lexicographic angle-fastest %" \
               PetscInt_FMT, i, j, a, got, expected);
         }
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum class RowKind { INTERIOR, DIRICHLET, REFLECT };

// Classify one (node, angle) row, and for a reflective row say which angle it
// couples to
//
// The label decides only which BC family each face belongs to; whether a row
// is a boundary row at all is the physics - the existing upwind test, which is
// the sign of the direction against the face. A direction incoming through no
// face is an interior unknown. If ANY incoming face is vacuum the row is
// Dirichlet - vacuum wins at mixed corners - else the direction reflects in
// every incoming axis: a corner between two reflective faces flips both
// cosines, which is still one partner angle and so one column
static RowKind ClassifyRow(PetscInt i, PetscInt j, PetscInt a, \
   const PetscScalar *mu, const PetscScalar *eta, PetscInt n_cells_x, PetscInt n_cells_y, \
   BCType left, BCType right, BCType bottom, BCType top, \
   const PetscInt *reflect_mu, const PetscInt *reflect_eta, PetscInt *partner)
{
   const PetscInt upwind_i = (mu[a] > 0) ? i - 1 : ((mu[a] < 0) ? i + 1 : i);
   const PetscInt upwind_j = (eta[a] > 0) ? j - 1 : ((eta[a] < 0) ? j + 1 : j);
   const PetscBool has_x = (PetscBool)(mu[a] != 0.0);
   const PetscBool has_y = (PetscBool)(eta[a] != 0.0);
   const PetscBool x_outside = (PetscBool)(has_x && (upwind_i < 0 || upwind_i >= n_cells_x));
   const PetscBool y_outside = (PetscBool)(has_y && (upwind_j < 0 || upwind_j >= n_cells_y));

   *partner = -1;
   if (!x_outside && !y_outside) return RowKind::INTERIOR;

   // Which face the direction comes in through on each outside axis
   const BCType x_bc = (mu[a] > 0) ? left : right;
   const BCType y_bc = (eta[a] > 0) ? bottom : top;
   if ((x_outside && x_bc == BCType::VACUUM) || (y_outside && y_bc == BCType::VACUUM)) \
      return RowKind::DIRICHLET;

   PetscInt ap = a;
   if (x_outside) ap = reflect_mu[ap];
   if (y_outside) ap = reflect_eta[ap];
   *partner = ap;
   return RowKind::REFLECT;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build the COO sparsity and the slot maps
// This happens on the host but we only need to do it once
PetscErrorCode StructuredFD2D::create(MPI_Comm comm, PhaseSpace &ps, PetscInt n_cells_x, PetscInt n_cells_y, \
   PetscReal length_x, PetscReal length_y, const SNQuadrature2D &quad, const BCSpec &bcs)
{
   const PetscInt *ltog = NULL;
   ISLocalToGlobalMapping ltog_map = NULL;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck(n_cells_x > 0 && n_cells_y > 0 && n_cells_x * n_cells_y == ps.n_cells, comm, \
      PETSC_ERR_ARG_INCOMP, "a %" PetscInt_FMT " x %" PetscInt_FMT " grid is not the phase space's %" \
      PetscInt_FMT " cells", n_cells_x, n_cells_y, ps.n_cells);

   comm_ = comm;
   n_cells_x_ = n_cells_x;
   n_cells_y_ = n_cells_y;
   // Uniform grid so every cell has the same width
   dx_ = length_x / n_cells_x;
   dy_ = length_y / n_cells_y;

   const PetscInt n_angles = ps.n_angles;
   const PetscScalar *mu = quad.mu_host();
   const PetscScalar *eta = quad.eta_host();

   // The DM decides the parallel decomposition, so building it has to come
   // before anything that reads local_cells - starting with the PhaseSpace,
   // which we fill in rather than read
   PetscInt xs = 0, ys = 0, xm = 0, ym = 0;
   PetscInt gxs = 0, gys = 0, gxm = 0;
   PetscCall(CreateDA(comm, ps, n_cells_x, n_cells_y, &dm_));
   // DMDAGetCorners divides the dof back out, so these come out in cells
   PetscCall(DMDAGetCorners(dm_, &xs, &ys, NULL, &xm, &ym, NULL));
   // The ghosted patch is what the local-to-global map is indexed over. With
   // DM_BOUNDARY_NONE it is clipped at the physical boundary, so a node just
   // outside the box is not in it at all - which is fine: a Dirichlet row
   // never asks for a column, and the column a reflective row asks for is its
   // own cell's mirrored angle, an owned node
   PetscCall(DMDAGetGhostCorners(dm_, &gxs, &gys, NULL, &gxm, NULL, NULL));
   cell_start_x_ = xs;
   cell_start_y_ = ys;
   local_cells_x_ = xm;
   local_cells_y_ = ym;
   ps.local_cells = xm * ym;

   // The DM owns the mesh, so it carries the node positions too: node (i, j)
   // sits at (i dx, j dy), the same convention the closed-form check in
   // tests/verify_2dk.kokkos.cxx uses. Nothing in the solve reads these - they
   // are for output (the scalar flux VTK writer picks them up). A direction
   // with a single node has no spacing to define and PETSc would divide by
   // n - 1, so that degenerate grid keeps no coordinates
   if (n_cells_x > 1 && n_cells_y > 1) PetscCall(DMDASetUniformCoordinates(dm_, \
      0.0, length_x - length_x / n_cells_x, 0.0, length_y - length_y / n_cells_y, 0.0, 0.0));

   PetscCall(DMGetLocalToGlobalMapping(dm_, &ltog_map));
   PetscCall(ISLocalToGlobalMappingGetIndices(ltog_map, &ltog));
   PetscCall(CheckDALayout(dm_, ps, ltog, xs, ys, xm, ym, gxs, gys, gxm));

   ps_ = ps;
   const PetscInt local_rows = ps.local_rows();

   // ~~~~~~~~~~
   // COO coordinates - three entries per row, in slot order
   // upwind-x, upwind-y, diagonal
   // ~~~~~~~~~~
   // A -1 row/col index in the COO format says just ignore this entry. Rows
   // that should not have one of these entries - a Dirichlet boundary row, or
   // a direction with no component along that axis - keep the slot and hand
   // the preallocation a -1, so the value fills never branch. A reflective
   // boundary row instead repurposes its first slot for the coupling to the
   // mirrored angle
   oor_.assign(3 * local_rows, 0);
   ooc_.assign(3 * local_rows, 0);
   std::vector<PetscInt> is_bc_row(local_rows, 0);
   std::vector<PetscInt> reflect_slot(local_rows, -1);

   const BCType left   = bcs.type(FACE_LEFT);
   const BCType right  = bcs.type(FACE_RIGHT);
   const BCType bottom = bcs.type(FACE_BOTTOM);
   const BCType top    = bcs.type(FACE_TOP);
   const PetscInt *reflect_mu = quad.reflect_mu_host();
   const PetscInt *reflect_eta = quad.reflect_eta_host();

   for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {

         const PetscInt local_cell = (j - ys) * xm + (i - xs);

         for (PetscInt a = 0; a < n_angles; a++) {

            const PetscInt r = local_cell * n_angles + a;
            const PetscInt row = GlobalDof(ltog, i, j, a, gxs, gys, gxm, n_angles);

            // The upwind neighbour is behind the direction on each axis: to the
            // left/below for a positive cosine, to the right/above for a
            // negative one. A zero cosine has no upwind neighbour at all
            const PetscInt upwind_i = (mu[a] > 0) ? i - 1 : ((mu[a] < 0) ? i + 1 : i);
            const PetscInt upwind_j = (eta[a] > 0) ? j - 1 : ((eta[a] < 0) ? j + 1 : j);
            const PetscBool has_x = (PetscBool)(mu[a] != 0.0);
            const PetscBool has_y = (PetscBool)(eta[a] != 0.0);

            // An inflow row - this direction enters the box through a face
            // this node is on - has its equation replaced by the boundary
            // condition: the identity on a Dirichlet (vacuum) row, the
            // reflection condition psi(a) - psi(partner) = 0 on a reflective
            // one. Corner nodes classify through either axis
            PetscInt partner = -1;
            const RowKind kind = ClassifyRow(i, j, a, mu, eta, n_cells_x, n_cells_y, \
               left, right, bottom, top, reflect_mu, reflect_eta, &partner);
            if (kind != RowKind::INTERIOR) is_bc_row[r] = 1;

            if (kind == RowKind::REFLECT)
            {
               // The partner is outgoing through every face this direction
               // came in through, so its row is a real unknown - unless a
               // direction of the grid is a single cell wide and the opposite
               // face catches it, which there is no sensible matrix for
               PetscInt partner2 = -1;
               PetscCheck(ClassifyRow(i, j, partner, mu, eta, n_cells_x, n_cells_y, \
                  left, right, bottom, top, reflect_mu, reflect_eta, &partner2) == RowKind::INTERIOR, \
                  comm, PETSC_ERR_SUP, "the reflection partner of node (%" PetscInt_FMT ", %" \
                  PetscInt_FMT ") angle %" PetscInt_FMT " is itself a boundary row - a reflective " \
                  "face on a single-cell-wide direction is not supported", i, j, a);

               // Same cell, mirrored angle - an owned node, so always in the
               // ltog map and always rank-local
               oor_[3 * r] = row;
               ooc_[3 * r] = GlobalDof(ltog, i, j, partner, gxs, gys, gxm, n_angles);
               reflect_slot[r] = 3 * r;

               oor_[3 * r + 1] = -1;
               ooc_[3 * r + 1] = -1;
            }
            else
            {
               const PetscBool null_x = (PetscBool)(kind == RowKind::DIRICHLET || !has_x);
               const PetscBool null_y = (PetscBool)(kind == RowKind::DIRICHLET || !has_y);

               oor_[3 * r]     = null_x ? -1 : row;
               ooc_[3 * r]     = null_x ? -1 : \
                  GlobalDof(ltog, upwind_i, j, a, gxs, gys, gxm, n_angles);

               oor_[3 * r + 1] = null_y ? -1 : row;
               ooc_[3 * r + 1] = null_y ? -1 : \
                  GlobalDof(ltog, i, upwind_j, a, gxs, gys, gxm, n_angles);
            }

            oor_[3 * r + 2] = row;
            ooc_[3 * r + 2] = row;

            // A ghost node the star stencil does not communicate carries a -1
            // in the map. We only ever ask for face neighbours or an owned
            // node, so this cannot fire - but a stray -1 column would silently
            // drop a coefficient rather than fail, so say it out loud. A slot
            // that was deliberately nulled has its ROW index at -1 too, which
            // is what separates it from a lookup that failed
            PetscCheck(ooc_[3 * r] >= 0 || oor_[3 * r] == -1, comm, PETSC_ERR_PLIB, \
               "no global index for the x-upwind neighbour of node (%" PetscInt_FMT ", %" PetscInt_FMT ")", i, j);
            PetscCheck(ooc_[3 * r + 1] >= 0 || oor_[3 * r + 1] == -1, comm, PETSC_ERR_PLIB, \
               "no global index for the y-upwind neighbour of node (%" PetscInt_FMT ", %" PetscInt_FMT ")", i, j);
         }
      }
   }

   PetscCall(ISLocalToGlobalMappingRestoreIndices(ltog_map, &ltog));

   // Slot maps - row r owns COO slots 3r (upwind-x), 3r + 1 (upwind-y) and
   // 3r + 2 (diagonal)
   PetscCall(set_uniform_pattern(3, is_bc_row, reflect_slot));

   PetscFunctionReturn(PETSC_SUCCESS);
}
