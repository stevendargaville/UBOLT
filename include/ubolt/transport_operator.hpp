#ifndef UBOLT_TRANSPORT_OPERATOR_HPP
#define UBOLT_TRANSPORT_OPERATOR_HPP

#include "ubolt/operator_term.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include <vector>

// The transport operator: the assembled terms in one MATAIJKOKKOS matrix plus
// the matrix-free terms, presented to the KSP as a single MatShell
//
// Terms are not owned - the caller keeps them alive for as long as the operator
class PETSC_VISIBILITY_PUBLIC TransportOperator {
public:
   // Any discretisation backend: everything used here (the slot maps, the
   // Dirichlet mask, the preallocated matrices) is dimension-independent
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc);
   PetscErrorCode destroy();

   PetscErrorCode add_term(const OperatorTerm *term);

   // Fill the assembled matrix from every assembled term and build the shell.
   // Values-only: the sparsity is preallocated once by the discretisation, so
   // this is what a per-group refill will call in Phase 2
   PetscErrorCode assemble();

   // Assemble a subset of the terms into a matrix of their own - used to build a
   // streaming-only preconditioning matrix. The caller destroys it
   PetscErrorCode assemble_subset(PetscInt n_terms, const OperatorTerm *const terms[], Mat *mat) const;

   // What the KSP solves with, and the assembled part of it
   Mat mat() const { return shell_; }
   Mat assembled_mat() const { return assembled_; }

   const std::vector<const OperatorTerm *> &matrix_free_terms() const { return matrix_free_; }
   PetscInt n_angles() const { return ps_.n_angles; }

private:
   PetscErrorCode assemble_into(Mat mat, PetscInt n_terms, const OperatorTerm *const terms[]) const;

   MPI_Comm comm_ = MPI_COMM_NULL;
   PhaseSpace ps_;
   const Discretisation *disc_ = nullptr;
   std::vector<const OperatorTerm *> terms_;
   std::vector<const OperatorTerm *> matrix_free_;
   // Shared COO values - every assembled term adds into this one array
   PetscScalarKokkosView coo_v_d_;
   Mat assembled_ = NULL;
   Mat shell_ = NULL;
};

#endif
