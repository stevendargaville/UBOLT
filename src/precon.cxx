#include "ubolt/precon.hpp"
#include "pflare.h"

// Define context for pcshell
typedef struct {
   Vec inverse_sigma_t_vec;
} transport_ctx;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode destroy_transport_ctx(PC pc)
{
   transport_ctx *shell;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &shell));
   PetscCall(VecDestroy(&shell->inverse_sigma_t_vec));
   PetscCall(PetscFree(shell));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode create_transport_ctx(transport_ctx **shell, Mat A_stream_removal)
{
   transport_ctx *newctx;

   PetscFunctionBeginUser;

   PetscCall(PetscNew(&newctx));
   // Create vector for removal preconditioning
   PetscCall(MatCreateVecs(A_stream_removal, &newctx->inverse_sigma_t_vec, NULL));
   PetscCall(MatGetDiagonal(A_stream_removal, newctx->inverse_sigma_t_vec));
   PetscCall(VecReciprocal(newctx->inverse_sigma_t_vec));

   *shell = newctx;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the preconditioner for the removal term
static PetscErrorCode ShellPCApply(PC pc, Vec x, Vec y)
{
   transport_ctx *shell;

   PetscFunctionBeginUser;

   PetscCall(PCShellGetContext(pc, &shell));
   PetscCall(VecPointwiseMult(y, x, shell->inverse_sigma_t_vec));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Create the PETSc preconditioner
PetscErrorCode create_pc(PC pc, Mat A_stream_removal) {

   PC pc_removal;
   // Context for the removal shell preconditioner
   transport_ctx *shell;

   PetscFunctionBeginUser;

   // Set our PC type to be an multiplicative combination of preconditioners
   PetscCall(PCSetType(pc, PCCOMPOSITE));
   PetscCall(PCCompositeSetType(pc, PC_COMPOSITE_MULTIPLICATIVE));

   // ~~~~~~~~~~~~~
   // Preconditioner for removal term
   // ~~~~~~~~~~~~~
   // Shell preconditioner for removal term
   PetscCall(PCCompositeAddPCType(pc, PCSHELL));
   // Get the 1st PC object out (ie the removal one)
   PetscCall(PCCompositeGetPC(pc, 0, &pc_removal));

   // Create the context for the preconditioner
   PetscCall(create_transport_ctx(&shell, A_stream_removal));

   // Set the routine that does the apply
   PetscCall(PCShellSetApply(pc_removal, ShellPCApply));
   PetscCall(PCShellSetContext(pc_removal, shell));

   // Set the routine that destroys
   PetscCall(PCShellSetDestroy(pc_removal, destroy_transport_ctx));

   // Set the name
   PetscCall(PCShellSetName(pc_removal, "RemovalPCShell"));

   // ~~~~~~~~~~~~~
   // Preconditioner for streaming term
   // ~~~~~~~~~~~~~
   // AIR preconditioner for streaming operator
   // This will operate on pmat
   PetscCall(PCCompositeAddPCType(pc, PCAIR));

   PetscFunctionReturn(PETSC_SUCCESS);
}
