#include "ubolt/structured_fd_1d.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build a 1D DMDA over the cells: one dof per angle, stencil width 1 because
// the upwind operator reaches exactly one neighbour
//
// We deliberately do NOT call DMSetFromOptions - it would expose -da_grid_x,
// which could resize the mesh out from under the PhaseSpace that every other
// object has already been sized from. The sizes come from the PhaseSpace, which
// the drivers already expose as -n_cells/-n_angles
//
// We also deliberately let the DM choose its own distribution rather than
// handing it PhaseSpace's: PETSc's default 1D DMDA split is
// M/size + ((M % size) > rank), which is the same formula PetscSplitOwnership
// uses, so the two agree by construction. CheckDALayout is what holds us to
// that instead of assuming it
static PetscErrorCode CreateDA(MPI_Comm comm, const PhaseSpace &ps, DM *da)
{
   PetscFunctionBeginUser;

   PetscCall(DMDACreate1d(comm, DM_BOUNDARY_NONE, ps.n_cells, ps.n_angles, 1, NULL, da));
   PetscCall(DMSetUp(*da));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check the DM's layout is the one every COO index below is written against:
// angle-fastest dof ordering (row = cell * n_angles + angle), contiguous
// ownership, and a decomposition matching the PhaseSpace's
//
// 1D DMDA ordering coincides with the natural ordering, so this passes - but it
// is load-bearing for every index we write, and a silent disagreement would
// show up as a wrong answer rather than an error, so check it once at setup
static PetscErrorCode CheckDALayout(DM da, const PhaseSpace &ps, PetscInt cell_start, PetscInt local_cells)
{
   PetscInt dm_cells = 0, dm_dof = 0, rstart = 0, rend = 0;
   Vec gv;

   PetscFunctionBeginUser;

   PetscCall(DMDAGetInfo(da, NULL, &dm_cells, NULL, NULL, NULL, NULL, NULL, &dm_dof, \
      NULL, NULL, NULL, NULL, NULL));
   PetscCheck(dm_cells == ps.n_cells && dm_dof == ps.n_angles, PetscObjectComm((PetscObject)da), \
      PETSC_ERR_PLIB, "DMDA is %" PetscInt_FMT " cells x %" PetscInt_FMT " dof, phase space is %" \
      PetscInt_FMT " x %" PetscInt_FMT, dm_cells, dm_dof, ps.n_cells, ps.n_angles);

   // The DM picked its own distribution - this is where we find out it is the
   // same one PhaseSpace computed with PetscSplitOwnership
   PetscCheck(local_cells == ps.local_cells, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, \
      "DMDA owns %" PetscInt_FMT " cells but the phase space was split into %" PetscInt_FMT, \
      local_cells, ps.local_cells);

   // Angle-fastest and contiguous: the rows the DM's global vector owns must be
   // exactly [cell_start * n_angles, (cell_start + local_cells) * n_angles)
   PetscCall(DMCreateGlobalVector(da, &gv));
   PetscCall(VecGetOwnershipRange(gv, &rstart, &rend));
   PetscCheck(rstart == cell_start * ps.n_angles && rend - rstart == local_cells * ps.n_angles, \
      PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, \
      "DMDA global rows [%" PetscInt_FMT ", %" PetscInt_FMT ") are not the angle-fastest rows [%" \
      PetscInt_FMT ", %" PetscInt_FMT ")", rstart, rend, cell_start * ps.n_angles, \
      (cell_start + local_cells) * ps.n_angles);
   PetscCall(VecDestroy(&gv));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build the COO sparsity and the slot maps
// This happens on the host but we only need to do it once
PetscErrorCode StructuredFD1D::create(MPI_Comm comm, const PhaseSpace &ps, PetscReal length, const SNQuadrature &quad)
{
   PetscBool use_dm = PETSC_FALSE;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);

   comm_ = comm;
   ps_ = ps;
   // Uniform grid so every cell has the same width
   dx_ = length / ps.n_cells;

   const PetscInt n_angles   = ps.n_angles;
   const PetscInt local_cells = ps.local_cells;
   const PetscInt local_rows  = ps.local_rows();
   const PetscScalar *mu = quad.mu_host();

   // Where our cells start globally - the decomposition is in cells
   //
   // This is the ONLY thing the hand-rolled and DMDA layouts disagree about:
   // everything below is shared, so -ubolt_use_dm is a tight test of the DM
   // extraction rather than of a second copy of the sparsity logic
   PetscInt cell_start = 0;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-ubolt_use_dm", &use_dm, NULL));
   if (use_dm) {
      PetscInt dm_local_cells = 0;
      PetscCall(CreateDA(comm, ps_, &da_));
      // DMDAGetCorners divides the dof back out, so these are in cells
      PetscCall(DMDAGetCorners(da_, &cell_start, NULL, NULL, &dm_local_cells, NULL, NULL));
      PetscCall(CheckDALayout(da_, ps_, cell_start, dm_local_cells));
   } else {
      PetscCallMPI(MPI_Exscan(&ps_.local_cells, &cell_start, 1, MPIU_INT, MPI_SUM, comm));
   }
   global_row_start_ = cell_start * n_angles;
   const PetscBool on_left_boundary  = (PetscBool)(cell_start == 0);
   const PetscBool on_right_boundary = (PetscBool)(cell_start + local_cells == ps.n_cells);

   // ~~~~~~~~~~
   // COO coordinates - two entries per row
   // ~~~~~~~~~~
   oor_.assign(2 * local_rows, 0);
   ooc_.assign(2 * local_rows, 0);

   // Loop over local cells, which have n_angles unknowns on them
   // We add two non-zeros per unknown. On boundary rows we still include two,
   // and make the preallocation ignore one of them, so that the value fills
   // don't have to change depending on the boundary condition
   PetscCount counter = 0;
   for (PetscInt i = 0; i < local_cells; i++)
   {
      for (PetscInt a = 0; a < n_angles; a++) {

         // Upwinded uniform grid finite difference operator
         // The upwind neighbour is on the left for positive velocities and on
         // the right for negative ones. The diagonal is always written last so
         // the value fills don't have to detect the direction
         const PetscInt upwind_cell = (mu[a] > 0) ? i - 1 : i + 1;

         oor_[counter] = i * n_angles + a + global_row_start_;
         ooc_[counter] = upwind_cell * n_angles + a + global_row_start_;

         oor_[counter + 1] = i * n_angles + a + global_row_start_;
         ooc_[counter + 1] = i * n_angles + a + global_row_start_;

         counter = counter + 2;
      }
   }

   // ~~~~~~~~~~
   // Dirichlet conditions on the left and right boundaries
   // ~~~~~~~~~~
   // A -1 row/col index in the COO format says just ignore this entry, so the
   // boundary rows keep only their diagonal
   std::vector<PetscInt> is_dirichlet_row(local_rows, 0);

   // All positive angles are incoming on the left boundary
   if (on_left_boundary)
   {
      for (PetscInt a = 0; a < n_angles; a++) {
         if (mu[a] > 0)
         {
            oor_[a * 2] = -1;
            ooc_[a * 2] = -1;
            is_dirichlet_row[a] = 1;
         }
      }
   }
   // All negative angles are incoming on the right boundary
   if (on_right_boundary)
   {
      for (PetscInt a = 0; a < n_angles; a++) {
         if (mu[a] < 0)
         {
            oor_[(local_cells - 1) * n_angles * 2 + a * 2] = -1;
            ooc_[(local_cells - 1) * n_angles * 2 + a * 2] = -1;
            is_dirichlet_row[(local_cells - 1) * n_angles + a] = 1;
         }
      }
   }

   // ~~~~~~~~~~
   // Slot maps - row r owns COO slots 2r (upwind neighbour) and 2r + 1 (diagonal)
   // ~~~~~~~~~~
   std::vector<PetscInt> row_slot_offset(local_rows + 1);
   std::vector<PetscInt> diag_slot(local_rows);
   for (PetscInt r = 0; r < local_rows; r++)
   {
      row_slot_offset[r] = 2 * r;
      diag_slot[r] = 2 * r + 1;
   }
   row_slot_offset[local_rows] = 2 * local_rows;

   pattern_.n_slots = 2 * local_rows;
   pattern_.row_slot_offset_d = PetscIntKokkosView("row_slot_offset_d", local_rows + 1);
   pattern_.diag_slot_d = PetscIntKokkosView("diag_slot_d", local_rows);
   boundary_.is_dirichlet_row_d = PetscIntKokkosView("is_dirichlet_row_d", local_rows);

   PetscIntKokkosViewHostUnmanaged row_slot_offset_h(row_slot_offset.data(), local_rows + 1);
   PetscIntKokkosViewHostUnmanaged diag_slot_h(diag_slot.data(), local_rows);
   PetscIntKokkosViewHostUnmanaged is_dirichlet_row_h(is_dirichlet_row.data(), local_rows);
   Kokkos::deep_copy(pattern_.row_slot_offset_d, row_slot_offset_h);
   Kokkos::deep_copy(pattern_.diag_slot_d, diag_slot_h);
   Kokkos::deep_copy(boundary_.is_dirichlet_row_d, is_dirichlet_row_h);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DM is the only PETSc handle we own, and only on the -ubolt_use_dm path.
// DMDestroy on a NULL handle is a no-op, so this is safe either way
PetscErrorCode StructuredFD1D::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(DMDestroy(&da_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A matrix with our sparsity preallocated for COO assembly
PetscErrorCode StructuredFD1D::create_matrix(Mat *mat) const
{
   PetscFunctionBeginUser;

   const PetscInt local_rows = ps_.local_rows();
   const PetscInt global_rows = ps_.global_rows();

   PetscCall(MatCreate(comm_, mat));
   PetscCall(MatSetSizes(*mat, local_rows, local_rows, global_rows, global_rows));
   // pflare only takes its Kokkos paths when handed MATAIJKOKKOS
   PetscCall(MatSetType(*mat, MATAIJKOKKOS));
   PetscCall(MatSetFromOptions(*mat));
   PetscCall(MatSetUp(*mat));

   // MatSetPreallocationCOO takes non-const arrays, so hand it a copy and keep
   // ours for the next matrix built from this discretisation
   std::vector<PetscInt> oor(oor_);
   std::vector<PetscInt> ooc(ooc_);
   PetscCall(MatSetPreallocationCOO(*mat, pattern_.n_slots, oor.data(), ooc.data()));

   PetscFunctionReturn(PETSC_SUCCESS);
}
