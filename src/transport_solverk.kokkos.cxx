#include "ubolt/transport_solver.hpp"
#include "ubolt/block_inverse.hpp"
#include "pflare.h"

// The removal stage: y = D_op^{-1} x, D_op the element blocks of the
// OPERATOR (not of pmat - whatever pmat the streaming stage was handed, this
// stage inverts the true local operator). At n_basis 1 - the structured
// backends and DG0 - a block is the diagonal entry, and this is the point
// Jacobi stage it always was, to the bit; at DG1 it is the n_basis x n_basis
// upwind cell-local block, the same thing the block-scaled streaming stage
// inverts, which the point diagonal was a crude stand-in for. new/delete
// rather than PetscNew: the blocks hold device views, which need their
// constructors and destructors run
struct RemovalPCCtx {
   ElementBlockInverse blocks;
   // The composed diagonal, when the assembled matrix is missing part of it
   Vec diag = NULL;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode RemovalPCDestroy(PC pc)
{
   RemovalPCCtx *shell = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &shell));
   PetscCall(VecDestroy(&shell->diag));
   delete shell;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The removal stage's blocks
//
// Read off the assembled matrix, which is where the diagonal has always come
// from - unless a term carrying a diagonal is being applied matrix-free, in
// which case the assembled matrix no longer holds the whole diagonal and
// reading it alone would silently precondition with whatever part of it
// happens to be assembled (streaming alone, under -matfree_removal). Then the
// operator composes the diagonal from the terms instead - the same number,
// bitwise, transportk's -check_matfree pins exactly that - and it replaces
// the blocks' diagonal. The off-diagonals are right either way: the only
// diagonal-carrying term that goes matrix-free is the removal, which has none
// (the DG mass matrix is the identity), so the assembled ones are all there
// are - verify_plexk check 10 pins those blocks against the assembled
// operator's, bitwise
static PetscErrorCode RemovalPCFillContext(RemovalPCCtx *shell, const TransportOperator *op)
{
   PetscFunctionBeginUser;

   if (op->diagonal_is_composed()) {
      PetscCall(op->diagonal(shell->diag));
      PetscCall(shell->blocks.setup(op->assembled_mat(), shell->diag));
   }
   else PetscCall(shell->blocks.setup(op->assembled_mat()));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode RemovalPCCreateContext(RemovalPCCtx **shell, const TransportOperator *op)
{
   RemovalPCCtx *newctx = new RemovalPCCtx;

   PetscFunctionBeginUser;

   PetscCall(newctx->blocks.create(op->phase_space()));
   PetscCall(MatCreateVecs(op->assembled_mat(), &newctx->diag, NULL));
   PetscCall(RemovalPCFillContext(newctx, op));

   *shell = newctx;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the preconditioner for the removal term
static PetscErrorCode RemovalPCApply(PC pc, Vec x, Vec y)
{
   RemovalPCCtx *shell = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &shell));
   PetscCall(shell->blocks.apply(x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the DSA diffusion correction. The context is the caller's DSAPrecon,
// so there is no PCShellSetDestroy to match this: the object follows the
// library's create/destroy idiom and is destroyed by whoever created it,
// unlike the PetscNew'd removal context above
static PetscErrorCode DSAPCApply(PC pc, Vec x, Vec y)
{
   DSAPrecon *dsa = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &dsa));
   PetscCall(dsa->apply(x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The element-block-scaled streaming stage: y = AIR(D^{-1} pmat)^{-1} D^{-1} x,
// D the element blocks of pmat. The inner PC is built on the SCALED pmat,
// which is what the multigrid sees; the composite around this shell still
// forms its residuals with the unscaled one, so only this stage knows the
// scaling exists. new/delete rather than PetscNew: the context holds device
// views, which need their constructors and destructors run
struct BlockScaledPCCtx {
   ElementBlockInverse blocks;
   PC inner = NULL;
   Mat scaled = NULL;
   Vec work = NULL;
};

static PetscErrorCode BlockScaledPCDestroy(PC pc)
{
   BlockScaledPCCtx *ctx = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &ctx));
   PetscCall(PCDestroy(&ctx->inner));
   PetscCall(MatDestroy(&ctx->scaled));
   PetscCall(VecDestroy(&ctx->work));
   delete ctx;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Runs whenever pmat's state has changed since the last setup - every group
// refill in the default mode, once for the whole sweep when pmat is
// streaming-only. The inner PC then sees its own pmat's state change and
// rebuilds (or reuses, under its own options) exactly as it would unscaled
static PetscErrorCode BlockScaledPCSetUp(PC pc)
{
   BlockScaledPCCtx *ctx = nullptr;
   Mat pmat = NULL;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &ctx));
   PetscCall(PCGetOperators(pc, NULL, &pmat));
   PetscCall(ctx->blocks.setup(pmat));
   PetscCall(ctx->blocks.scale(pmat, ctx->scaled ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, &ctx->scaled));
   PetscCall(PCSetOperators(ctx->inner, ctx->scaled, ctx->scaled));
   if (!ctx->work) PetscCall(MatCreateVecs(pmat, &ctx->work, NULL));

   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BlockScaledPCApply(PC pc, Vec x, Vec y)
{
   BlockScaledPCCtx *ctx = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &ctx));
   PetscCall(ctx->blocks.apply(x, ctx->work));
   PetscCall(PCApply(ctx->inner, ctx->work, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BlockScaledPCView(PC pc, PetscViewer viewer)
{
   BlockScaledPCCtx *ctx = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &ctx));
   PetscCall(PetscViewerASCIIPrintf(viewer, "left-scaled by the inverse element blocks (%" PetscInt_FMT \
      " x %" PetscInt_FMT "), then:\n", ctx->blocks.block_size(), ctx->blocks.block_size()));
   PetscCall(PetscViewerASCIIPushTab(viewer));
   PetscCall(PCView(ctx->inner, viewer));
   PetscCall(PetscViewerASCIIPopTab(viewer));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode TransportSolver::create(MPI_Comm comm, const TransportOperator &op, Mat pmat, \
   DSAPrecon *dsa, PetscBool block_scale)
{
   PC pc, pc_removal;
   PC block_inner = NULL;
   RemovalPCCtx *shell = nullptr;

   PetscFunctionBeginUser;

   op_ = &op;

   PetscCall(KSPCreate(comm, &ksp_));
   PetscCall(KSPSetOperators(ksp_, op.mat(), pmat));
   PetscCall(KSPGetPC(ksp_, &pc));

   // Set our PC type to be a multiplicative combination of preconditioners
   PetscCall(PCSetType(pc, PCCOMPOSITE));
   PetscCall(PCCompositeSetType(pc, PC_COMPOSITE_MULTIPLICATIVE));

   // ~~~~~~~~~~~~~
   // Preconditioner for removal term
   // ~~~~~~~~~~~~~
   PetscCall(PCCompositeAddPCType(pc, PCSHELL));
   PetscCall(PCCompositeGetPC(pc, 0, &pc_removal));

   // The removal preconditioner always works off the operator's own blocks,
   // whatever pmat the streaming preconditioner was handed
   PetscCall(RemovalPCCreateContext(&shell, op_));

   PetscCall(PCShellSetApply(pc_removal, RemovalPCApply));
   PetscCall(PCShellSetContext(pc_removal, shell));
   PetscCall(PCShellSetDestroy(pc_removal, RemovalPCDestroy));
   PetscCall(PCShellSetName(pc_removal, "RemovalPCShell"));

   // ~~~~~~~~~~~~~
   // Preconditioner for streaming term
   // ~~~~~~~~~~~~~
   // AIR preconditioner for the streaming operator - this operates on pmat
   //
   // Or, under block_scale, on D^{-1} pmat applied to D^{-1} x, through a
   // shell. The inner PC takes the prefix the composite gave index 1, so every
   // -sub_1_pc_air_* option (and -sub_1_pc_type) means what it did before;
   // the shell itself moves out of the way to sub_1_block_
   if (!block_scale) PetscCall(PCCompositeAddPCType(pc, PCAIR));
   else {
      PC pc_stream;
      const char *prefix = nullptr;
      BlockScaledPCCtx *ctx = new BlockScaledPCCtx;

      PetscCall(ctx->blocks.create(op.phase_space()));
      PetscCall(PCCompositeAddPCType(pc, PCSHELL));
      PetscCall(PCCompositeGetPC(pc, 1, &pc_stream));
      PetscCall(PCCreate(comm, &ctx->inner));
      PetscCall(PCSetType(ctx->inner, PCAIR));
      PetscCall(PCGetOptionsPrefix(pc_stream, &prefix));
      PetscCall(PCSetOptionsPrefix(ctx->inner, prefix));
      PetscCall(PCAppendOptionsPrefix(pc_stream, "block_"));
      block_inner = ctx->inner;

      PetscCall(PCShellSetContext(pc_stream, ctx));
      PetscCall(PCShellSetSetUp(pc_stream, BlockScaledPCSetUp));
      PetscCall(PCShellSetApply(pc_stream, BlockScaledPCApply));
      PetscCall(PCShellSetView(pc_stream, BlockScaledPCView));
      PetscCall(PCShellSetDestroy(pc_stream, BlockScaledPCDestroy));
      PetscCall(PCShellSetName(pc_stream, "BlockScaledStreamingPCShell"));
   }

   // ~~~~~~~~~~~~~
   // Preconditioner for the scattering term - optional
   // ~~~~~~~~~~~~~
   // The DSA diffusion correction, last in the multiplicative order. Nothing
   // happens at all without one, so every configuration that predates it is
   // untouched
   if (dsa)
   {
      PC pc_dsa;

      PetscCall(PCCompositeAddPCType(pc, PCSHELL));
      PetscCall(PCCompositeGetPC(pc, 2, &pc_dsa));

      PetscCall(PCShellSetApply(pc_dsa, DSAPCApply));
      PetscCall(PCShellSetContext(pc_dsa, dsa));
      PetscCall(PCShellSetName(pc_dsa, "DSAPCShell"));

      // A multiplicative composite hands each stage the residual left by the
      // ones before it, and by default it forms that residual with PMAT. Pmat
      // has no scattering in it - it is the assembled streaming/removal matrix,
      // or a streaming-only one - so by the time PCAIR has inverted it the
      // residual reaching this shell is tiny AND carries no trace of the
      // scattering, which is the one thing DSA exists to correct. Measured on
      // slab_diffusive.json: the moment reaching the diffusion solve is ~1e-6
      // of the residual and the count does not move at all. Updating with AMAT
      // - the shell, matrix-free scatter included - is what makes the composite
      // a real residual correction: 19 iterations to 11 on that problem
      //
      // Only when there is a DSA to feed, so the preconditioner without one is
      // untouched, and before KSPSetFromOptions so -pc_use_amat false still wins
      PetscCall(PCSetUseAmat(pc, PETSC_TRUE));
   }

   // Last, so the command line still wins - including -pc_composite_type
   PetscCall(KSPSetFromOptions(ksp_));
   // The composite does not know the block-scaled stage's inner PC exists
   if (block_inner) PetscCall(PCSetFromOptions(block_inner));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Re-read (or re-compose) the removal PC's inverse blocks after the group's
// xsections have changed under it
PetscErrorCode TransportSolver::refresh()
{
   PC pc, pc_removal;
   RemovalPCCtx *shell = nullptr;

   PetscFunctionBeginUser;

   PetscCall(KSPGetPC(ksp_, &pc));
   // Index 0 is the removal shell, added first in create()
   PetscCall(PCCompositeGetPC(pc, 0, &pc_removal));
   PetscCall(PCShellGetContext(pc_removal, &shell));
   PetscCall(RemovalPCFillContext(shell, op_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode TransportSolver::solve(Vec b, Vec x)
{
   PetscFunctionBeginUser;

   PetscCall(KSPSolve(ksp_, b, x));
   PetscCall(KSPGetConvergedReason(ksp_, &reason_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode TransportSolver::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(KSPDestroy(&ksp_));
   op_ = nullptr;

   PetscFunctionReturn(PETSC_SUCCESS);
}
