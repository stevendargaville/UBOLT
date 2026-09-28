#include "ubolt/discretisation.hpp"
#include "petsc_kokkos.hpp"

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

// DMDestroy on a NULL handle is a no-op, so this is safe on a never-created
// discretisation too
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
PetscErrorCode Discretisation::set_uniform_pattern(PetscInt slots_per_row, const BoundaryRows &rows)
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());

   const PetscInt local_rows = ps_.local_rows();
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

   PetscCall(set_pattern(row_slot_offset, diag_slot, rows));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The general slot maps: whatever CSR layout the backend chose, as long as
// every slot belongs to exactly one row and the diagonal is one of that row's
// slots
PetscErrorCode Discretisation::set_pattern(const std::vector<PetscInt> &row_slot_offset, \
   const std::vector<PetscInt> &diag_slot, const BoundaryRows &rows)
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());

   const PetscInt local_rows = ps_.local_rows();
   PetscCheck((PetscInt)rows.is_bc_row.size() == local_rows && (PetscInt)rows.reflect_slot.size() == local_rows && \
      (PetscInt)rows.dirichlet_value.size() == local_rows && (PetscInt)rows.ghost_inflow.size() == local_rows, \
      comm_, PETSC_ERR_ARG_INCOMP, "boundary rows are sized %" PetscInt_FMT " / %" PetscInt_FMT " / %" \
      PetscInt_FMT " / %" PetscInt_FMT " but there are %" PetscInt_FMT " local rows (BoundaryRows::reset)", \
      (PetscInt)rows.is_bc_row.size(), (PetscInt)rows.reflect_slot.size(), (PetscInt)rows.dirichlet_value.size(), \
      (PetscInt)rows.ghost_inflow.size(), local_rows);
   PetscCheck((PetscInt)row_slot_offset.size() == local_rows + 1 && (PetscInt)diag_slot.size() == local_rows, \
      comm_, PETSC_ERR_ARG_INCOMP, "slot maps are sized %" PetscInt_FMT " / %" PetscInt_FMT \
      " but there are %" PetscInt_FMT " local rows", (PetscInt)row_slot_offset.size(), \
      (PetscInt)diag_slot.size(), local_rows);
   PetscCheck(oor_.size() == ooc_.size() && row_slot_offset[0] == 0 && \
      row_slot_offset[local_rows] == (PetscInt)oor_.size(), comm_, PETSC_ERR_ARG_INCOMP, \
      "COO coordinates are %" PetscInt_FMT " long but the slot maps cover [%" PetscInt_FMT ", %" \
      PetscInt_FMT ")", (PetscInt)oor_.size(), row_slot_offset[0], row_slot_offset[local_rows]);
   for (PetscInt r = 0; r < local_rows; r++) {
      PetscCheck(row_slot_offset[r] < row_slot_offset[r + 1] && diag_slot[r] >= row_slot_offset[r] && \
         diag_slot[r] < row_slot_offset[r + 1], comm_, PETSC_ERR_ARG_INCOMP, "row %" PetscInt_FMT \
         " owns slots [%" PetscInt_FMT ", %" PetscInt_FMT ") but its diagonal is slot %" PetscInt_FMT, \
         r, row_slot_offset[r], row_slot_offset[r + 1], diag_slot[r]);
   }

   boundary_.ghost_flux_vacuum = rows.ghost_flux;
   boundary_.has_dirichlet_rows = PETSC_FALSE;
   boundary_.has_reflect_rows = PETSC_FALSE;
   for (PetscInt r = 0; r < local_rows; r++) {
      if (rows.is_bc_row[r] && rows.reflect_slot[r] < 0) boundary_.has_dirichlet_rows = PETSC_TRUE;
      if (rows.reflect_slot[r] >= 0) boundary_.has_reflect_rows = PETSC_TRUE;
   }

   // The set and ORDER of the device allocations here is frozen: an extra or
   // moved allocation shifts every later buffer, which changes the chunking a
   // vectorised reduction picks and so the last bits of a residual norm - and
   // tests/baselines is compared bit for bit. Hence ghost_inflow_d last, and
   // only under ghost-flux
   pattern_.n_slots = (PetscCount)oor_.size();
   pattern_.row_slot_offset_d = PetscIntKokkosView("row_slot_offset_d", local_rows + 1);
   pattern_.diag_slot_d = PetscIntKokkosView("diag_slot_d", local_rows);
   boundary_.is_bc_row_d = PetscIntKokkosView("is_bc_row_d", local_rows);
   boundary_.reflect_slot_d = PetscIntKokkosView("reflect_slot_d", local_rows);
   boundary_.dirichlet_value_d = PetscScalarKokkosView("dirichlet_value_d", local_rows);
   if (rows.ghost_flux) boundary_.ghost_inflow_d = PetscScalarKokkosView("ghost_inflow_d", local_rows);

   // const_cast only because the unmanaged host view type is non-const; the
   // deep_copy below reads it
   PetscIntKokkosViewHostUnmanaged row_slot_offset_h(const_cast<PetscInt *>(row_slot_offset.data()), local_rows + 1);
   PetscIntKokkosViewHostUnmanaged diag_slot_h(const_cast<PetscInt *>(diag_slot.data()), local_rows);
   PetscIntKokkosViewHostUnmanaged is_bc_row_h(const_cast<PetscInt *>(rows.is_bc_row.data()), local_rows);
   PetscIntKokkosViewHostUnmanaged reflect_slot_h(const_cast<PetscInt *>(rows.reflect_slot.data()), local_rows);
   PetscScalarKokkosViewHostUnmanaged dirichlet_value_h(const_cast<PetscScalar *>(rows.dirichlet_value.data()), \
      local_rows);
   auto exec = PetscGetKokkosExecutionSpace();
   Kokkos::deep_copy(exec, pattern_.row_slot_offset_d, row_slot_offset_h);
   Kokkos::deep_copy(exec, pattern_.diag_slot_d, diag_slot_h);
   Kokkos::deep_copy(exec, boundary_.is_bc_row_d, is_bc_row_h);
   Kokkos::deep_copy(exec, boundary_.reflect_slot_d, reflect_slot_h);
   Kokkos::deep_copy(exec, boundary_.dirichlet_value_d, dirichlet_value_h);
   if (rows.ghost_flux) {
      PetscScalarKokkosViewHostUnmanaged ghost_inflow_h( \
         const_cast<PetscScalar *>(rows.ghost_inflow.data()), local_rows);
      Kokkos::deep_copy(exec, boundary_.ghost_inflow_d, ghost_inflow_h);
   }
   // The copies are asynchronous and the host arrays are the caller's
   exec.fence();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda
static void FillInflowKernel(PetscScalarKokkosView b_d, PetscIntKokkosView is_bc_row_d, \
   PetscIntKokkosView reflect_slot_d, PetscScalarKokkosView dirichlet_value_d, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         // Dirichlet = a BC row that is not reflective, the existing idiom
         if (is_bc_row_d(r) && reflect_slot_d(r) < 0) b_d(r) = dirichlet_value_d(r);
      });
}

// A ghost-flux row is an ordinary unknown, so its boundary value is ADDED to
// the rhs rather than written over it - the external source lands on the same
// row afterwards. A free function for the same reason FillInflowKernel is one
static void FillGhostInflowKernel(PetscScalarKokkosView b_d, \
   PetscScalarKokkosView ghost_inflow_d, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         b_d(r) += ghost_inflow_d(r);
      });
}

// The rhs of a Dirichlet row IS the boundary value the backend computed at
// create time, and every other row is left alone, so this goes onto a zeroed
// b before the external source is filled in. Under GHOST_FLUX the inflow is
// ADDED instead. A kernel is launched only if it can write something
PetscErrorCode UboltFillInflow(const BoundaryInfo &boundary, Vec b)
{
   PetscFunctionBeginUser;

   PetscInt local_rows;
   PetscCall(VecGetLocalSize(b, &local_rows));
   PetscCheck(local_rows == (PetscInt)boundary.dirichlet_value_d.extent(0), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "b has %" PetscInt_FMT " local rows but the boundary info covers %" \
      PetscInt_FMT, local_rows, (PetscInt)boundary.dirichlet_value_d.extent(0));
   if (!boundary.has_dirichlet_rows && !boundary.ghost_flux_vacuum) PetscFunctionReturn(PETSC_SUCCESS);

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));
   if (boundary.has_dirichlet_rows) FillInflowKernel(b_d, boundary.is_bc_row_d, boundary.reflect_slot_d, \
      boundary.dirichlet_value_d, local_rows);
   if (boundary.ghost_flux_vacuum) FillGhostInflowKernel(b_d, boundary.ghost_inflow_d, local_rows);
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
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         if (reflect_slot_d(r) >= 0) b_d(r) = 0.0;
      });
}

// The rhs of a reflective row is zero - psi_r - psi_partner = 0 - so whatever
// the driver filled b with (the external source, the inflow value) has to come
// off those rows again. The Dirichlet rows are left alone: their rhs IS the
// boundary value. Nothing to do on a rank with no reflective row, which is
// every rank under ghost-flux
PetscErrorCode UboltZeroReflectRows(const BoundaryInfo &boundary, Vec b)
{
   PetscFunctionBeginUser;

   PetscInt local_rows;
   PetscCall(VecGetLocalSize(b, &local_rows));
   PetscCheck(local_rows == (PetscInt)boundary.reflect_slot_d.extent(0), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "b has %" PetscInt_FMT " local rows but the boundary info covers %" \
      PetscInt_FMT, local_rows, (PetscInt)boundary.reflect_slot_d.extent(0));
   if (!boundary.has_reflect_rows) PetscFunctionReturn(PETSC_SUCCESS);

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));
   ZeroReflectRowsKernel(b_d, boundary.reflect_slot_d, local_rows);
   PetscCall(VecRestoreKokkosView(b, &b_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
