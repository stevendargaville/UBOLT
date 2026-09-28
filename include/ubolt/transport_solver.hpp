#ifndef UBOLT_TRANSPORT_SOLVER_HPP
#define UBOLT_TRANSPORT_SOLVER_HPP

#include "ubolt/transport_operator.hpp"
#include "ubolt/dsa.hpp"
#include <petscksp.h>

struct RemovalPCCtx;

// KSP + the transport preconditioner: a multiplicative composite of a shell
// preconditioner for the removal term (index 0: the inverse element blocks of
// the operator - the point diagonal at n_basis 1, the cell-local block at
// DG1, see ElementBlockInverse), PCAIR for the streaming term
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
   // its blocks from the operator, not from a matrix handle, so that a term
   // applied matrix-free (and therefore absent from the assembled matrix) is
   // still in the blocks being inverted - see RemovalPCFillContext
   //
   // dsa is optional and CALLER-OWNED (nullptr: no DSA stage): it must outlive
   // the solver, and whoever passes one also drives it - DSAPrecon::set_group()
   // is a per-group call the solver has no context to make
   //
   // block_scale wraps the streaming stage (index 1) in the element-block
   // inverse: PCAIR is built on D^{-1} pmat and applied to D^{-1} x, D the
   // n_basis x n_basis (cell, angle) blocks of pmat (see ElementBlockInverse).
   // A left scaling of what the multigrid sees only - operator, rhs, monitored
   // residuals and the other stages are unchanged. At DG1 the cell-local block
   // has off-diagonals as large as its diagonal, which PCAIR's strength of
   // connection reads as strong and its coarsening stalls on; scaled, only the
   // upwind couplings are left, as at DG0. At n_basis 1 D is the diagonal. The
   // inner PC keeps the sub_1_ prefix
   PetscErrorCode create(MPI_Comm comm, const TransportOperator &op, Mat pmat, \
      DSAPrecon *dsa = nullptr, PetscBool block_scale = PETSC_FALSE);
   PetscErrorCode destroy();

   // Re-take whatever the preconditioner cached off the operator's blocks.
   // Must be called once the group's xsections have changed under it: the
   // removal shell PC holds the inverse element blocks, and the multigroup sweep
   // changes sigma_t every group - whether that lands in the assembled matrix
   // (the default, so this follows the values-only refill) or straight in the
   // term (matrix-free removal, where there is no refill to follow). PCAIR
   // needs no help - it tracks pmat's state itself (and with a streaming-only
   // pmat there is nothing to redo, since streaming does not depend on the
   // group). Neither does the DSA shell: its per-group refill is
   // DSAPrecon::set_group(), which needs the group's xsections and so belongs
   // where the terms are re-pointed. With one solver per pmat (ref-shift bins)
   // only the solver about to solve needs it
   PetscErrorCode refresh();

   PetscErrorCode solve(Vec b, Vec x);

   KSPConvergedReason converged_reason() const { return reason_; }
   KSP ksp() const { return ksp_; }

private:
   KSP ksp_ = NULL;
   // Not owned, and it must outlive the solver: the operator the removal PC
   // takes its blocks from, per group
   const TransportOperator *op_ = nullptr;
   // The removal stage's context, kept so refresh() need not find it in the
   // composite. Owned (and freed) by that stage's PC
   RemovalPCCtx *removal_ = nullptr;
   KSPConvergedReason reason_ = KSP_CONVERGED_ITERATING;
};

#endif
