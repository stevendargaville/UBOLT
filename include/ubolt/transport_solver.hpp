#ifndef UBOLT_TRANSPORT_SOLVER_HPP
#define UBOLT_TRANSPORT_SOLVER_HPP

#include "ubolt/transport_operator.hpp"
#include <petscksp.h>

// KSP + the transport preconditioner: a multiplicative composite of a shell
// preconditioner for the removal term (index 0) and PCAIR for the streaming
// term (index 1, so its options take the -sub_1_pc_air_ prefix)
class PETSC_VISIBILITY_PUBLIC TransportSolver {
public:
   // pmat is what the preconditioner is built from - either the operator's own
   // assembled matrix or a streaming-only one
   PetscErrorCode create(MPI_Comm comm, const TransportOperator &op, Mat pmat);
   PetscErrorCode destroy();

   PetscErrorCode solve(Vec b, Vec x);

   KSPConvergedReason converged_reason() const { return reason_; }
   KSP ksp() const { return ksp_; }

private:
   KSP ksp_ = NULL;
   KSPConvergedReason reason_ = KSP_CONVERGED_ITERATING;
};

#endif
