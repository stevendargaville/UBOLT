#include "ubolt/transport_operator.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Put the identity on the Dirichlet rows. Terms contribute nothing there, so
// this is the whole of those rows
// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda
static void SetDirichletIdentity(PetscScalarKokkosView coo_v_d, \
   PetscIntKokkosView row_slot_offset_d, \
   PetscIntKokkosView diag_slot_d, \
   PetscIntKokkosView is_dirichlet_row_d, \
   PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         if (!is_dirichlet_row_d(r)) return;

         for (PetscInt s = row_slot_offset_d(r); s < row_slot_offset_d(r + 1); s++) coo_v_d(s) = 0.0;
         coo_v_d(diag_slot_d(r)) = 1.0;
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the assembled terms, then every matrix-free term on top
static PetscErrorCode ShellMatMultApply(Mat A, Vec x, Vec y)
{
   TransportOperator *op;

   PetscFunctionBeginUser;

   PetscCall(MatShellGetContext(A, &op));

   PetscCall(MatMult(op->assembled_mat(), x, y));
   for (const OperatorTerm *term : op->matrix_free_terms()) PetscCall(term->apply_add(x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode TransportOperator::create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD1D &disc)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());

   comm_ = comm;
   ps_ = ps;
   disc_ = &disc;

   // The shared COO values every assembled term adds into
   coo_v_d_ = PetscScalarKokkosView("coo_v_d", disc.coo_pattern().n_slots);
   PetscCall(disc.create_matrix(&assembled_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode TransportOperator::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(MatDestroy(&shell_));
   PetscCall(MatDestroy(&assembled_));
   // The device views have to go before PetscFinalize takes Kokkos down
   coo_v_d_ = PetscScalarKokkosView();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode TransportOperator::add_term(const OperatorTerm *term)
{
   PetscFunctionBeginUser;

   terms_.push_back(term);
   if (term->matrix_free()) matrix_free_.push_back(term);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Fill the assembled matrix from every assembled term and build the shell
PetscErrorCode TransportOperator::assemble()
{
   std::vector<const OperatorTerm *> assembled_terms;

   PetscFunctionBeginUser;

   for (const OperatorTerm *term : terms_) {
      if (term->assembled()) assembled_terms.push_back(term);
   }
   PetscCall(assemble_into(assembled_, (PetscInt)assembled_terms.size(), assembled_terms.data()));

   // Nothing to do on a values-only refill
   if (shell_) PetscFunctionReturn(PETSC_SUCCESS);

   const PetscInt local_rows = ps_.local_rows();
   const PetscInt global_rows = ps_.global_rows();
   PetscCall(MatCreateShell(comm_, local_rows, local_rows, global_rows, global_rows, this, &shell_));
   PetscCall(MatShellSetOperation(shell_, MATOP_MULT, (void (*)(void))ShellMatMultApply));
   PetscCall(MatAssemblyBegin(shell_, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(shell_, MAT_FINAL_ASSEMBLY));
   // Have to make sure to set the type of vectors the shell creates - without
   // it the shell hands out host vectors and every Kokkos view access silently
   // round-trips through host
   VecType vtype;
   PetscCall(MatGetVecType(assembled_, &vtype));
   PetscCall(MatShellSetVecType(shell_, vtype));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Assemble a subset of the terms into a matrix of their own
PetscErrorCode TransportOperator::assemble_subset(PetscInt n_terms, const OperatorTerm *const terms[], Mat *mat) const
{
   PetscFunctionBeginUser;

   PetscCall(disc_->create_matrix(mat));
   PetscCall(assemble_into(*mat, n_terms, terms));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Values-only COO fill: every term adds into the shared values array and the
// whole thing goes in with a single MatSetValuesCOO
//
// -ubolt_coo_two_call keeps the old behaviour of one MatSetValuesCOO per term
// (the first INSERTs, the rest ADD) as a debug fallback. The two paths do the
// same per-entry arithmetic in the same order, so they agree bit for bit
PetscErrorCode TransportOperator::assemble_into(Mat mat, PetscInt n_terms, const OperatorTerm *const terms[]) const
{
   // Shallow view copies so the kernels don't capture `this`
   PetscScalarKokkosView coo_v_d = coo_v_d_;
   const PetscIntKokkosView row_slot_offset_d = disc_->coo_pattern().row_slot_offset_d;
   const PetscIntKokkosView diag_slot_d = disc_->coo_pattern().diag_slot_d;
   const PetscIntKokkosView is_dirichlet_row_d = disc_->boundary_info().is_dirichlet_row_d;
   const PetscInt local_rows = ps_.local_rows();

   PetscBool two_call = PETSC_FALSE;

   PetscFunctionBeginUser;

   PetscCall(PetscOptionsGetBool(NULL, NULL, "-ubolt_coo_two_call", &two_call, NULL));

   // With no terms there is nothing to split over two calls
   if (n_terms == 0) two_call = PETSC_FALSE;

   if (!two_call) {

      Kokkos::deep_copy(coo_v_d, 0.0);
      for (PetscInt t = 0; t < n_terms; t++) PetscCall(terms[t]->assemble_add(coo_v_d));
      SetDirichletIdentity(coo_v_d, row_slot_offset_d, diag_slot_d, is_dirichlet_row_d, local_rows);
      // This should all happen on the gpu
      PetscCall(MatSetValuesCOO(mat, coo_v_d.data(), INSERT_VALUES));

   } else {

      for (PetscInt t = 0; t < n_terms; t++) {

         Kokkos::deep_copy(coo_v_d, 0.0);
         PetscCall(terms[t]->assemble_add(coo_v_d));
         // The identity rides along with the first call and the later ADDs
         // leave it alone, since terms contribute nothing to those rows
         if (t == 0) SetDirichletIdentity(coo_v_d, row_slot_offset_d, diag_slot_d, is_dirichlet_row_d, local_rows);
         PetscCall(MatSetValuesCOO(mat, coo_v_d.data(), t == 0 ? INSERT_VALUES : ADD_VALUES));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}
