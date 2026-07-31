#include "ubolt/matshell.hpp"
#include <KokkosBlas.hpp>
#include "petsc_kokkos.hpp"

// Define context for matshell
typedef struct {
   Mat A_stream_removal;
   PetscInt n_angles;
   PetscScalarKokkosView *sigma_s_d = NULL;
   PetscScalar2DKokkosView *scalar_flux_d = NULL;
   PetscScalar2DKokkosView *w_d = NULL;
   PetscScalar sum_weights;
} matshell_ctx;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the streaming/removal + scattering terms
static PetscErrorCode ShellMatMultApply(Mat A, Vec x, Vec y)
{
   matshell_ctx *ctx;
   PetscFunctionBeginUser;
   PetscCall(MatShellGetContext(A, &ctx));

   const PetscInt n_angles = ctx->n_angles;

   // All the scatter kernels below operate on the local part of the
   // vectors only - this is how many local cells we have
   PetscInt local_rows;
   PetscCall(VecGetLocalSize(x, &local_rows));
   const PetscInt local_cells = local_rows / n_angles;

   // ~~~~~~~~~~~~~
   // Apply the streaming/removal term
   // ~~~~~~~~~~~~~
   PetscCall(MatMult(ctx->A_stream_removal, x, y));

   // ~~~~~~~~~~~~~
   // Apply the scattering term matrix-free
   // ~~~~~~~~~~~~~

   // Get the device views
   // Making sure to use a const view for the input
   PetscScalarConstKokkosView x_d;
   PetscCall(VecGetKokkosView(x, &x_d));
   PetscScalarKokkosView y_d;
   PetscCall(VecGetKokkosView(y, &y_d));

   // Let's reshape (without copying) the 1D input view into a 2D view
   // Get the raw pointer from the 1D view
   const PetscScalar* x_d_1d = x_d.data();

   // Create the 2D view using the pointer and new dimensions
   // The 2D view uses LayoutRight so we have the correct ordering of contiguous
   // in angle
   PetscScalar2DConstKokkosView x_d_2d(x_d_1d, local_cells, n_angles);

   // Now let's integrate the angular flux to get the scalar flux
   // This is just a dgemm (on the device)
   const double alpha = double(1.0);
   const double beta = double(0.0);
   KokkosBlas::gemm("N","N",alpha,x_d_2d,*ctx->w_d,beta,*ctx->scalar_flux_d);

   // Now let's multiply by the scattering xsection
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(int i) {

         // We have to divide by the sum of weights to get the amount going
         // into each angle
         (*ctx->scalar_flux_d)(i, 0) *= (*ctx->sigma_s_d)(i)/ctx->sum_weights;
      });

   // Now let's put the scatter back into y
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_cells, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // cell
         PetscInt i = t.league_rank();

         // For all the angles
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt j) {

               // Minus the scattering term to the output vector as it's on the lhs
               y_d(i * n_angles + j) -= (*ctx->scalar_flux_d)(i, 0);
         });
   });

   // Restore the views
   PetscCall(VecRestoreKokkosView(x, &x_d));
   PetscCall(VecRestoreKokkosView(y, &y_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Create the matshell which we use to apply streaming/removal + scattering
PetscErrorCode create_matshell(PetscInt n_angles, Mat A_stream_removal, Mat *A, \
   PetscScalarKokkosView *sigma_s_d, \
   PetscScalar2DKokkosView *scalar_flux_d, \
   PetscScalar2DKokkosView *w_d, \
   PetscScalar sum_weights) {

   // Create the context for the matshell
   matshell_ctx *newctx;

   PetscFunctionBeginUser;

   PetscCall(PetscNew(&newctx));

   // How many angles we have
   newctx->n_angles = n_angles;
   // Copy pointer to scattering xsections on the device
   newctx->sigma_s_d = sigma_s_d;
   // Copy pointer to space for scalar flux on the device
   newctx->scalar_flux_d = scalar_flux_d;
   // Copy pointer to Sn weights on the device
   newctx->w_d = w_d;
   // What is the sum of the Sn weights
   newctx->sum_weights = sum_weights;
   // Copy pointer to the streaming/removal operator
   newctx->A_stream_removal = A_stream_removal;

   PetscInt global_rows, global_cols, local_rows, local_cols;
   PetscCall(MatGetSize(A_stream_removal, &global_rows, &global_cols));
   PetscCall(MatGetLocalSize(A_stream_removal, &local_rows, &local_cols));

   PetscCall(MatCreateShell(PETSC_COMM_WORLD, local_rows, local_cols, global_rows, global_cols, newctx, A));
   // The subroutine ShellMatMultApply applies the matrix
   PetscCall(MatShellSetOperation(*A, MATOP_MULT, (void (*)(void))ShellMatMultApply));

   PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
   // Have to make sure to set the type of vectors the shell creates
   VecType vtype;
   PetscCall(MatGetVecType(A_stream_removal, &vtype));
   PetscCall(MatShellSetVecType(*A, vtype));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode destroy_matshell_ctx(Mat A)
{
   matshell_ctx *ctx;

   PetscFunctionBeginUser;

   PetscCall(MatShellGetContext(A, &ctx));
   PetscCall(PetscFree(ctx));

   PetscFunctionReturn(PETSC_SUCCESS);
}
