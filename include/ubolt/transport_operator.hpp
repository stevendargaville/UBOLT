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

   // A term's assembled()/matrix_free() answer is NOT read here - assemble()
   // partitions the terms every time it runs. So a term that can be either
   // (RemovalTerm::set_matrix_free) may be switched before or after it is
   // added, as long as it is before the first assemble()
   PetscErrorCode add_term(const OperatorTerm *term);

   // Fill the assembled matrix from every assembled term and build the shell.
   // Values-only: the sparsity is preallocated once by the discretisation, so
   // this is what the multigroup sweep's per-group refill calls
   PetscErrorCode assemble();

   // Assemble a subset of the terms into a matrix of their own - used to build a
   // streaming-only preconditioning matrix. The caller destroys it
   PetscErrorCode assemble_subset(PetscInt n_terms, const OperatorTerm *const terms[], Mat *mat) const;

   // The operator's diagonal, composed from the terms rather than read off the
   // assembled matrix: every term's add_diagonal in the order they were added -
   // the order the COO values are summed in, so the two agree bitwise - then
   // 1.0 on the BC rows, which is what the assembly writes there
   //
   // d must be a vector over the operator's rows (MatCreateVecs off either
   // matrix). Only the local part is touched, as everywhere else
   PetscErrorCode diagonal(Vec d) const;

   // Is that composition the ONLY way to get the diagonal right - i.e. is some
   // term carrying a diagonal being applied matrix-free, so the assembled
   // matrix is missing it? Then MatGetDiagonal is silently incomplete and the
   // Jacobi stage of the preconditioner has to compose instead (see
   // TransportSolver). FALSE for the assembled default, where MatGetDiagonal is
   // both correct and what every existing baseline was captured with
   PetscBool diagonal_is_composed() const;

   // What the KSP solves with, and the assembled part of it
   Mat mat() const { return shell_; }
   Mat assembled_mat() const { return assembled_; }

   const std::vector<const OperatorTerm *> &matrix_free_terms() const { return matrix_free_; }

private:
   PetscErrorCode assemble_into(Mat mat, PetscInt n_terms, const OperatorTerm *const terms[]) const;

   MPI_Comm comm_ = MPI_COMM_NULL;
   PhaseSpace ps_;
   const Discretisation *disc_ = nullptr;
   std::vector<const OperatorTerm *> terms_;
   // Rebuilt by every assemble(), never by add_term - see above
   std::vector<const OperatorTerm *> matrix_free_;
   // Shared COO values - every assembled term adds into this one array
   PetscScalarKokkosView coo_v_d_;
   Mat assembled_ = NULL;
   Mat shell_ = NULL;
};

#endif
