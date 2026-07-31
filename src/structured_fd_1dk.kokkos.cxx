#include "ubolt/structured_fd_1d.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build the COO sparsity and the slot maps
// This happens on the host but we only need to do it once
PetscErrorCode StructuredFD1D::create(MPI_Comm comm, const PhaseSpace &ps, PetscReal length, const SNQuadrature &quad)
{
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
   PetscInt cell_start = 0;
   PetscCallMPI(MPI_Exscan(&ps_.local_cells, &cell_start, 1, MPIU_INT, MPI_SUM, comm));
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
