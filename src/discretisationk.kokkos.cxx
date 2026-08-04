#include "ubolt/discretisation.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A matrix with our sparsity preallocated for COO assembly
PetscErrorCode Discretisation::create_matrix(Mat *mat) const
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());

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
PetscErrorCode Discretisation::set_uniform_pattern(PetscInt slots_per_row, const std::vector<PetscInt> &is_bc_row, \
   const std::vector<PetscInt> &reflect_slot, const std::vector<PetscScalar> &dirichlet_value)
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());

   const PetscInt local_rows = ps_.local_rows();
   PetscCheck((PetscInt)is_bc_row.size() == local_rows, comm_, PETSC_ERR_ARG_INCOMP, \
      "BC row mask has %" PetscInt_FMT " entries but there are %" PetscInt_FMT " local rows", \
      (PetscInt)is_bc_row.size(), local_rows);
   PetscCheck((PetscInt)reflect_slot.size() == local_rows, comm_, PETSC_ERR_ARG_INCOMP, \
      "reflect slots have %" PetscInt_FMT " entries but there are %" PetscInt_FMT " local rows", \
      (PetscInt)reflect_slot.size(), local_rows);
   PetscCheck((PetscInt)dirichlet_value.size() == local_rows, comm_, PETSC_ERR_ARG_INCOMP, \
      "Dirichlet values have %" PetscInt_FMT " entries but there are %" PetscInt_FMT " local rows", \
      (PetscInt)dirichlet_value.size(), local_rows);
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
   boundary_.is_bc_row_d = PetscIntKokkosView("is_bc_row_d", local_rows);
   boundary_.reflect_slot_d = PetscIntKokkosView("reflect_slot_d", local_rows);
   boundary_.dirichlet_value_d = PetscScalarKokkosView("dirichlet_value_d", local_rows);

   PetscIntKokkosViewHostUnmanaged row_slot_offset_h(row_slot_offset.data(), local_rows + 1);
   PetscIntKokkosViewHostUnmanaged diag_slot_h(diag_slot.data(), local_rows);
   // const_cast only because the unmanaged host view type is non-const; the
   // deep_copy below reads it
   PetscIntKokkosViewHostUnmanaged is_bc_row_h(const_cast<PetscInt *>(is_bc_row.data()), local_rows);
   PetscIntKokkosViewHostUnmanaged reflect_slot_h(const_cast<PetscInt *>(reflect_slot.data()), local_rows);
   PetscScalarKokkosViewHostUnmanaged dirichlet_value_h(const_cast<PetscScalar *>(dirichlet_value.data()), local_rows);
   Kokkos::deep_copy(pattern_.row_slot_offset_d, row_slot_offset_h);
   Kokkos::deep_copy(pattern_.diag_slot_d, diag_slot_h);
   Kokkos::deep_copy(boundary_.is_bc_row_d, is_bc_row_h);
   Kokkos::deep_copy(boundary_.reflect_slot_d, reflect_slot_h);
   Kokkos::deep_copy(boundary_.dirichlet_value_d, dirichlet_value_h);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda
static void FillInflowKernel(PetscScalarKokkosView b_d, PetscIntKokkosView is_bc_row_d, \
   PetscIntKokkosView reflect_slot_d, PetscScalarKokkosView dirichlet_value_d, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         // Dirichlet = a BC row that is not reflective, the existing idiom
         if (is_bc_row_d(r) && reflect_slot_d(r) < 0) b_d(r) = dirichlet_value_d(r);
      });
}

// The rhs of a Dirichlet row IS the boundary value: the per-angle inflow the
// backend computed at create time - the winning face's angle-integrated
// strength divided by the quadrature's measure, zeroed outside that face's
// window. Every other row is left alone, so this goes onto a zeroed b before
// the external source is filled in
PetscErrorCode UboltFillInflow(const BoundaryInfo &boundary, Vec b)
{
   PetscFunctionBeginUser;

   PetscInt local_rows;
   PetscCall(VecGetLocalSize(b, &local_rows));
   PetscCheck(local_rows == (PetscInt)boundary.dirichlet_value_d.extent(0), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "b has %" PetscInt_FMT " local rows but the boundary info covers %" \
      PetscInt_FMT, local_rows, (PetscInt)boundary.dirichlet_value_d.extent(0));

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));
   FillInflowKernel(b_d, boundary.is_bc_row_d, boundary.reflect_slot_d, boundary.dirichlet_value_d, \
      local_rows);
   PetscCall(VecRestoreKokkosView(b, &b_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda
static void ZeroReflectRowsKernel(PetscScalarKokkosView b_d, PetscIntKokkosView reflect_slot_d, \
   PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         if (reflect_slot_d(r) >= 0) b_d(r) = 0.0;
      });
}

// The rhs of a reflective row is zero - psi_r - psi_partner = 0 - so whatever
// the driver filled b with (the external source, the inflow value) has to come
// off those rows again. The Dirichlet rows are left alone: their rhs IS the
// boundary value
PetscErrorCode UboltZeroReflectRows(const BoundaryInfo &boundary, Vec b)
{
   PetscFunctionBeginUser;

   PetscInt local_rows;
   PetscCall(VecGetLocalSize(b, &local_rows));
   PetscCheck(local_rows == (PetscInt)boundary.reflect_slot_d.extent(0), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "b has %" PetscInt_FMT " local rows but the boundary info covers %" \
      PetscInt_FMT, local_rows, (PetscInt)boundary.reflect_slot_d.extent(0));

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));
   ZeroReflectRowsKernel(b_d, boundary.reflect_slot_d, local_rows);
   PetscCall(VecRestoreKokkosView(b, &b_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
