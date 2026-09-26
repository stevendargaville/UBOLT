#ifndef UBOLT_TRANSPORT_SOLVER_HPP
#define UBOLT_TRANSPORT_SOLVER_HPP

#include "ubolt/transport_operator.hpp"
#include "ubolt/dsa.hpp"
#include <petscksp.h>

// KSP + the transport preconditioner: a multiplicative composite of a shell
// preconditioner for the removal term (index 0), PCAIR for the streaming term
// (index 1, so its options take the -sub_1_pc_air_ prefix - under block_scale a
// shell whose inner PCAIR keeps that prefix, see create) and, if the caller
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
   // op is not owned and must outlive the solver: the removal shell PC takes
   // its diagonal from the operator, not from a matrix handle, so that a term
   // applied matrix-free (and therefore absent from the assembled matrix) is
   // still in the diagonal being inverted - see RemovalPCFillContext
   //
   // dsa is optional and CALLER-OWNED: it must outlive the solver, and passing
   // nullptr (the default) is exactly the preconditioner UBOLT had before DSA
   // existed. Whoever passes one also drives it - DSAPrecon::set_group() is a
   // per-group call the solver has no context to make, see refresh() below
   //
   // block_scale wraps the streaming stage (index 1) in the element-block
   // inverse: PCAIR is built on D^{-1} pmat and applied to D^{-1} x, D the
   // n_basis x n_basis (cell, angle) blocks of pmat (see ElementBlockInverse).
   // It is a left scaling of what the multigrid sees and nothing else - the
   // operator, the rhs, the residuals the KSP monitors and the other composite
   // stages are all unchanged - so it is purely a preconditioner choice. What
   // it buys: at DG1 the cell-local block has off-diagonals as large as the
   // diagonal (the volume term), which PCAIR's strength of connection reads
   // as strong couplings, and its coarsening stalls; once the blocks are the
   // identity, the only couplings left are the upwind ones, as at DG0. At
   // n_basis 1 D is pmat's diagonal, which makes PCAIR's view of pmat
   // independent of any per-row scaling - a DG0 row's 1 / V_c included - and
   // is the same code, so both DG orders take it. The inner PC keeps the
   // sub_1_ prefix
   PetscErrorCode create(MPI_Comm comm, const TransportOperator &op, Mat pmat, \
      DSAPrecon *dsa = nullptr, PetscBool block_scale = PETSC_FALSE);
   PetscErrorCode destroy();

   // Re-take whatever the preconditioner cached off the operator's diagonal.
   // Must be called once the group's xsections have changed under it: the
   // removal shell PC holds the inverse diagonal, and the multigroup sweep
   // changes sigma_t every group - whether that lands in the assembled matrix
   // (the default, so this follows the values-only refill) or straight in the
   // term (matrix-free removal, where there is no refill to follow). PCAIR
   // needs no help - it tracks pmat's state itself (and with a streaming-only
   // pmat there is nothing to redo, since streaming does not depend on the
   // group). Neither does the DSA shell: its per-group refill is
   // DSAPrecon::set_group(), which needs the group's xsections and so belongs
   // where the terms are re-pointed
   PetscErrorCode refresh();

   PetscErrorCode solve(Vec b, Vec x);

   KSPConvergedReason converged_reason() const { return reason_; }
   KSP ksp() const { return ksp_; }

private:
   KSP ksp_ = NULL;
   // Not owned, and it must outlive the solver: the operator the removal PC
   // takes its diagonal from, per group
   const TransportOperator *op_ = nullptr;
   KSPConvergedReason reason_ = KSP_CONVERGED_ITERATING;
};

#endif
