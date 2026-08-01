#include "ubolt/transport_solver.hpp"
#include "pflare.h"

// Define context for the removal pcshell
typedef struct {
   Vec inverse_sigma_t_vec;
} removal_pc_ctx;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode RemovalPCDestroy(PC pc)
{
   removal_pc_ctx *shell = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &shell));
   PetscCall(VecDestroy(&shell->inverse_sigma_t_vec));
   PetscCall(PetscFree(shell));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Read the inverse diagonal off the assembled matrix
// Phase 5 composes this analytically from the terms instead
static PetscErrorCode RemovalPCFillContext(removal_pc_ctx *shell, Mat assembled)
{
   PetscFunctionBeginUser;

   PetscCall(MatGetDiagonal(assembled, shell->inverse_sigma_t_vec));
   PetscCall(VecReciprocal(shell->inverse_sigma_t_vec));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode RemovalPCCreateContext(removal_pc_ctx **shell, Mat assembled)
{
   removal_pc_ctx *newctx;

   PetscFunctionBeginUser;

   PetscCall(PetscNew(&newctx));
   // Create vector for removal preconditioning
   PetscCall(MatCreateVecs(assembled, &newctx->inverse_sigma_t_vec, NULL));
   PetscCall(RemovalPCFillContext(newctx, assembled));

   *shell = newctx;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the preconditioner for the removal term
static PetscErrorCode RemovalPCApply(PC pc, Vec x, Vec y)
{
   removal_pc_ctx *shell = nullptr;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &shell));
   PetscCall(VecPointwiseMult(y, x, shell->inverse_sigma_t_vec));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode TransportSolver::create(MPI_Comm comm, const TransportOperator &op, Mat pmat)
{
   PC pc, pc_removal;
   removal_pc_ctx *shell = nullptr;

   PetscFunctionBeginUser;

   assembled_ = op.assembled_mat();

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

   // The removal preconditioner always works off the operator's own assembled
   // matrix, whatever pmat the streaming preconditioner was handed
   PetscCall(RemovalPCCreateContext(&shell, op.assembled_mat()));

   PetscCall(PCShellSetApply(pc_removal, RemovalPCApply));
   PetscCall(PCShellSetContext(pc_removal, shell));
   PetscCall(PCShellSetDestroy(pc_removal, RemovalPCDestroy));
   PetscCall(PCShellSetName(pc_removal, "RemovalPCShell"));

   // ~~~~~~~~~~~~~
   // Preconditioner for streaming term
   // ~~~~~~~~~~~~~
   // AIR preconditioner for the streaming operator - this operates on pmat
   PetscCall(PCCompositeAddPCType(pc, PCAIR));

   PetscCall(KSPSetFromOptions(ksp_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Re-read the removal PC's inverse diagonal after a values-only refill
PetscErrorCode TransportSolver::refresh()
{
   PC pc, pc_removal;
   removal_pc_ctx *shell = nullptr;

   PetscFunctionBeginUser;

   PetscCall(KSPGetPC(ksp_, &pc));
   // Index 0 is the removal shell, added first in create()
   PetscCall(PCCompositeGetPC(pc, 0, &pc_removal));
   PetscCall(PCShellGetContext(pc_removal, &shell));
   PetscCall(RemovalPCFillContext(shell, assembled_));

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
   assembled_ = NULL;

   PetscFunctionReturn(PETSC_SUCCESS);
}
