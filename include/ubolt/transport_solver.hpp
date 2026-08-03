#ifndef UBOLT_TRANSPORT_SOLVER_HPP
#define UBOLT_TRANSPORT_SOLVER_HPP

#include "ubolt/transport_operator.hpp"
#include "ubolt/dsa.hpp"
#include <petscksp.h>

// KSP + the transport preconditioner: a multiplicative composite of a shell
// preconditioner for the removal term (index 0), PCAIR for the streaming term
// (index 1, so its options take the -sub_1_pc_air_ prefix) and, if the caller
// hands one over, a shell for the DSA diffusion correction (index 2, whose own
// inner solve takes the -dsa_ prefix - see DSAPrecon)
//
// Removal, then streaming, then diffusion is the classic
// solve-then-diffusion-correction order. The composite type is not hardwired:
// KSPSetFromOptions runs last in create(), so -pc_composite_type additive
// selects the paper's additive combination from the command line
//
// With a DSA the composite is also switched to updating its intermediate
// residuals from the AMAT rather than pmat, since pmat carries no scattering -
// see the comment at that call. Without one, nothing about the preconditioner
// changes
class PETSC_VISIBILITY_PUBLIC TransportSolver {
public:
   // pmat is what the preconditioner is built from - either the operator's own
   // assembled matrix or a streaming-only one
   //
   // dsa is optional and CALLER-OWNED: it must outlive the solver, and passing
   // nullptr (the default) is exactly the preconditioner UBOLT had before DSA
   // existed. Whoever passes one also drives it - DSAPrecon::set_group() is a
   // per-group call the solver has no context to make, see refresh() below
   PetscErrorCode create(MPI_Comm comm, const TransportOperator &op, Mat pmat, \
      DSAPrecon *dsa = nullptr);
   PetscErrorCode destroy();

   // Re-read whatever the preconditioner cached off the assembled matrix.
   // Must be called after a values-only refill of that matrix: the removal
   // shell PC holds the inverse diagonal, and the multigroup sweep changes
   // sigma_t under it every group. PCAIR needs no help - it tracks pmat's
   // state itself (and with a streaming-only pmat there is nothing to redo,
   // since streaming does not depend on the group). Neither does the DSA
   // shell: its per-group refill is DSAPrecon::set_group(), which needs the
   // group's xsections and so belongs where the terms are re-pointed
   PetscErrorCode refresh();

   PetscErrorCode solve(Vec b, Vec x);

   KSPConvergedReason converged_reason() const { return reason_; }
   KSP ksp() const { return ksp_; }

private:
   KSP ksp_ = NULL;
   // Not owned - the operator's assembled matrix, which the removal PC reads
   Mat assembled_ = NULL;
   KSPConvergedReason reason_ = KSP_CONVERGED_ITERATING;
};

#endif
