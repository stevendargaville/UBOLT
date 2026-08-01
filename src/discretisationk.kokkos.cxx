#include "ubolt/discretisation.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A matrix with our sparsity preallocated for COO assembly
PetscErrorCode Discretisation::create_matrix(Mat *mat) const
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DM is the only PETSc handle a backend owns. DMDestroy on a NULL handle is
// a no-op, so this is safe on a never-created discretisation too
PetscErrorCode Discretisation::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(DMDestroy(&dm_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Slot maps for a fixed number of entries per row with the diagonal last:
// row r owns slots [slots_per_row * r, slots_per_row * (r + 1)) and its
// diagonal is the last of them
//
// The value fills address entries through these, so a backend that lays its COO
// coordinates out this way never has to hand a term a raw index. Building them
// here rather than per backend also keeps "the diagonal is last" stated once
PetscErrorCode Discretisation::set_uniform_pattern(PetscInt slots_per_row, const std::vector<PetscInt> &is_dirichlet_row)
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());

   const PetscInt local_rows = ps_.local_rows();
   PetscCheck((PetscInt)is_dirichlet_row.size() == local_rows, comm_, PETSC_ERR_ARG_INCOMP, \
      "Dirichlet mask has %" PetscInt_FMT " entries but there are %" PetscInt_FMT " local rows", \
      (PetscInt)is_dirichlet_row.size(), local_rows);
   PetscCheck((PetscInt)oor_.size() == slots_per_row * local_rows && oor_.size() == ooc_.size(), \
      comm_, PETSC_ERR_ARG_INCOMP, "COO coordinates are %" PetscInt_FMT " long, not %" PetscInt_FMT \
      " slots x %" PetscInt_FMT " local rows", (PetscInt)oor_.size(), slots_per_row, local_rows);

   std::vector<PetscInt> row_slot_offset(local_rows + 1);
   std::vector<PetscInt> diag_slot(local_rows);
   for (PetscInt r = 0; r < local_rows; r++)
   {
      row_slot_offset[r] = slots_per_row * r;
      diag_slot[r] = slots_per_row * r + slots_per_row - 1;
   }
   row_slot_offset[local_rows] = slots_per_row * local_rows;

   pattern_.n_slots = slots_per_row * local_rows;
   pattern_.row_slot_offset_d = PetscIntKokkosView("row_slot_offset_d", local_rows + 1);
   pattern_.diag_slot_d = PetscIntKokkosView("diag_slot_d", local_rows);
   boundary_.is_dirichlet_row_d = PetscIntKokkosView("is_dirichlet_row_d", local_rows);

   PetscIntKokkosViewHostUnmanaged row_slot_offset_h(row_slot_offset.data(), local_rows + 1);
   PetscIntKokkosViewHostUnmanaged diag_slot_h(diag_slot.data(), local_rows);
   // const_cast only because the unmanaged host view type is non-const; the
   // deep_copy below reads it
   PetscIntKokkosViewHostUnmanaged is_dirichlet_row_h(const_cast<PetscInt *>(is_dirichlet_row.data()), local_rows);
   Kokkos::deep_copy(pattern_.row_slot_offset_d, row_slot_offset_h);
   Kokkos::deep_copy(pattern_.diag_slot_d, diag_slot_h);
   Kokkos::deep_copy(boundary_.is_dirichlet_row_d, is_dirichlet_row_h);

   PetscFunctionReturn(PETSC_SUCCESS);
}
