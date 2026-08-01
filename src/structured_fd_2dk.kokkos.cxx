#include "ubolt/structured_fd_2d.hpp"

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

// Build the COO sparsity and the slot maps
// This happens on the host but we only need to do it once
PetscErrorCode StructuredFD2D::create(MPI_Comm comm, PhaseSpace &ps, PetscInt n_cells_x, PetscInt n_cells_y, \
   PetscReal length_x, PetscReal length_y, const SNQuadrature2D &quad)
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
   // outside the box is not in it at all - which is fine, those rows are the
   // Dirichlet ones and never ask for a column
   PetscCall(DMDAGetGhostCorners(dm_, &gxs, &gys, NULL, &gxm, NULL, NULL));
   cell_start_x_ = xs;
   cell_start_y_ = ys;
   local_cells_x_ = xm;
   local_cells_y_ = ym;
   ps.local_cells = xm * ym;

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
   // that should not have one of these entries - an inflow boundary row, or a
   // direction with no component along that axis - keep the slot and hand the
   // preallocation a -1, so the value fills never branch
   oor_.assign(3 * local_rows, 0);
   ooc_.assign(3 * local_rows, 0);
   std::vector<PetscInt> is_dirichlet_row(local_rows, 0);

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
            const PetscBool x_outside = (PetscBool)(has_x && (upwind_i < 0 || upwind_i >= n_cells_x));
            const PetscBool y_outside = (PetscBool)(has_y && (upwind_j < 0 || upwind_j >= n_cells_y));

            // An inflow row: this direction enters the box through a face this
            // node is on, so the boundary condition gives the node its value
            // outright and the equation is replaced by the identity. Corner
            // nodes reach here through either test
            if (x_outside || y_outside) is_dirichlet_row[r] = 1;

            oor_[3 * r]     = (is_dirichlet_row[r] || !has_x) ? -1 : row;
            ooc_[3 * r]     = (is_dirichlet_row[r] || !has_x) ? -1 : \
               GlobalDof(ltog, upwind_i, j, a, gxs, gys, gxm, n_angles);

            oor_[3 * r + 1] = (is_dirichlet_row[r] || !has_y) ? -1 : row;
            ooc_[3 * r + 1] = (is_dirichlet_row[r] || !has_y) ? -1 : \
               GlobalDof(ltog, i, upwind_j, a, gxs, gys, gxm, n_angles);

            oor_[3 * r + 2] = row;
            ooc_[3 * r + 2] = row;

            // A ghost node the star stencil does not communicate carries a -1
            // in the map. We only ever ask for face neighbours, so this cannot
            // fire - but a stray -1 column would silently drop a coefficient
            // rather than fail, so say it out loud
            PetscCheck(ooc_[3 * r] >= 0 || is_dirichlet_row[r] || !has_x, comm, PETSC_ERR_PLIB, \
               "no global index for the x-upwind neighbour of node (%" PetscInt_FMT ", %" PetscInt_FMT ")", i, j);
            PetscCheck(ooc_[3 * r + 1] >= 0 || is_dirichlet_row[r] || !has_y, comm, PETSC_ERR_PLIB, \
               "no global index for the y-upwind neighbour of node (%" PetscInt_FMT ", %" PetscInt_FMT ")", i, j);
         }
      }
   }

   PetscCall(ISLocalToGlobalMappingRestoreIndices(ltog_map, &ltog));

   // Slot maps - row r owns COO slots 3r (upwind-x), 3r + 1 (upwind-y) and
   // 3r + 2 (diagonal)
   PetscCall(set_uniform_pattern(3, is_dirichlet_row));

   PetscFunctionReturn(PETSC_SUCCESS);
}
