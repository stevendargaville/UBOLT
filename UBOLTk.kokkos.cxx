// Currently run with:
// make && ./UBOLTk -ksp_monitor -sub_1_pc_air_print_stats_timings -ksp_pc_side right -precon_stream -max_exponent 2

#include <petscvec_kokkos.hpp>
#include <Kokkos_DualView.hpp>
#include <KokkosBlas.hpp>
#include <petscksp.h>
#include "petsc_kokkos.hpp"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "pflare.h"

using DefaultMemorySpace         = Kokkos::DefaultExecutionSpace::memory_space;
using PetscScalarKokkosView      = Kokkos::View<PetscScalar *, DefaultMemorySpace>;
using PetscScalar2DKokkosView    = Kokkos::View<PetscScalar **, DefaultMemorySpace>;
using PetscScalarConstKokkosView = Kokkos::View<const PetscScalar *, DefaultMemorySpace>;
using PetscScalar2DConstKokkosView = Kokkos::View<const PetscScalar **, Kokkos::LayoutRight, DefaultMemorySpace>;
using HostMirrorMemorySpace      = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;
using PetscScalarKokkosViewHost  = Kokkos::View<PetscScalar *, HostMirrorMemorySpace>;
using PetscScalarKokkosViewHostUnmanaged = Kokkos::View<PetscScalar *, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using PetscScalar2DKokkosViewHostUnmanaged = Kokkos::View<PetscScalar **, Kokkos::LayoutRight, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using KokkosTeamMemberType = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;

#define N_CELLS 1000
#define N_ANGLES 4
#define LENGTH 1.0

// Define context for pcshell
typedef struct {
   Vec inverse_sigma_t_vec;
} transport_ctx;

// Define context for matshell
typedef struct {
   Mat A_stream_removal;
   PetscScalarKokkosView *sigma_s_d = NULL;
   PetscScalar2DKokkosView *scalar_flux_d = NULL;
   PetscScalar2DKokkosView *w_d = NULL;
   PetscScalar sum_weights;
} matshell_ctx;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// SN quadrature (Gauss-Legendre for N_ANGLES=2 or 4)
void create_sn_quadrature(int no_angles, PetscScalar *mu, PetscScalar *w, PetscScalar *sum_weights) {
    if (no_angles == 2) {
        mu[0] = -0.5773502692; mu[1] = 0.5773502692;
        w[0] = 1.0;            w[1] = 1.0;
    } else if (no_angles == 4) {
        mu[0] = -0.8611363116; mu[1] = -0.3399810436;
        mu[2] =  0.3399810436; mu[3] =  0.8611363116;
        w[0] = 0.3478548451;   w[1] = 0.6521451549;
        w[2] = 0.6521451549;   w[3] = 0.3478548451;
    }
    // In 1D we integrate over [-1, 1] so the sum of weights is 2
    *sum_weights = 2.0; 
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Create the preallocation structure for the
// assembled streaming/removal operator for a single group
// This happens on the host but we only need to call this once
void preallocate_streaming_removal(PetscScalar *mu, Mat A_stream_removal) {

   PetscInt i, global_row_start, global_row_end_plus_one;
   PetscInt global_rows, global_cols, local_rows, local_cols;
   MatGetSize(A_stream_removal, &global_rows, &global_cols);
   MatGetLocalSize(A_stream_removal, &local_rows, &local_cols);
   MatGetOwnershipRange(A_stream_removal, &global_row_start, &global_row_end_plus_one);

    // Row and column indices for COO assembly
    PetscInt *oor, *ooc;    
    // Going to do assembly in the COO interface so assembly happens on the gpu when needed
    // Allocate memory for the coordinates
    PetscMalloc2(2 * local_rows, &oor, 2 * local_rows, &ooc);    

    // This is how many local spatal nodes we have
    PetscInt local_nodes = local_rows / N_ANGLES;

   // ~~~~~~~~~~
   // MatSetPreallocationCOO - happens on the host
   // ~~~~~~~~~~
   PetscCount counter = 0;

   // Loop over the local spatial nodes which have N_ANGLES on them
   // Ordering of our unknowns doesn't matter
   // Ordering in fill_streaming_removal when calling MatSetValuesCOO
   // has to match the preallocation below

   // Loop over local nodes 
   // We add two non-zeros per unknown
   // On boundary nodes we also include two non-zeros
   // We make it ignore one of those entries given dirichlet conditions
   // That way the setvalues code doesn't have to change
   // depending on the boundary condition
   for (i = 0; i < local_nodes; i++)
   {
      // Loop over angles
      for (int a = 0; a < N_ANGLES; a++) {

         // Upwinded uniform grid finite difference operator

         // Positive velocity, we have -> -1 1
         if (mu[a] > 0)
         {
            oor[counter] = i * N_ANGLES + a + global_row_start;
            ooc[counter] = (i - 1) * N_ANGLES + a + global_row_start;

            oor[counter + 1] = i * N_ANGLES + a + global_row_start;
            ooc[counter + 1] = i * N_ANGLES + a + global_row_start;

         }
         // Negative velocity, we have -> 1 -1     
         // but we always add the diagonal last 
         // This makes the setvalues very simple as this matches the ordering
         // of the positive velocity and hence we don't have to detect
         // what direction    
         else
         {
            oor[counter] = i * N_ANGLES + a + global_row_start;
            ooc[counter] = (i + 1) * N_ANGLES + a + global_row_start;

            oor[counter + 1] = i * N_ANGLES + a + global_row_start;
            ooc[counter + 1] = i * N_ANGLES + a + global_row_start;
         }
         counter = counter + 2;         
      }
   }

   // Now let's go and fix up the boundary conditions
   // Let's impose Dirichlet conditions on the left and right boundaries
   // Putting a -1 in the COO format says just ignore this entry
   // ie we are keeping only the diagonal entry   

   // Left boundary
   if (global_row_start == 0)
   {
      // All positive angles just have a diagonal entry on the left boundary
      for (int a = 0; a < N_ANGLES; a++) {

         // Ignore left entry
         if (mu[a] > 0)
         {
            oor[a * 2] = -1;
            ooc[a * 2] = -1;                   
         }           
      }
   }
   // Right boundary
   if (global_row_end_plus_one == global_rows)
   {
      // All negative angles just have a diagonal entry on the right boundary
      for (int a = 0; a < N_ANGLES; a++) {

         // Ignore right entry
         if (mu[a] < 0)
         {
            oor[(local_nodes - 1) * N_ANGLES * 2 + a * 2] = -1;
            ooc[(local_nodes - 1) * N_ANGLES * 2 + a * 2] = -1;
         }           
      }
   }

   // Set the indices
   counter = 2 * local_rows;
   MatSetPreallocationCOO(A_stream_removal, counter, oor, ooc); 
   // Can delete oor and ooc now
   PetscFree2(oor, ooc);    
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Insert the streaming term for a single group
// This happens entirely on the device
void insert_streaming(PetscScalarKokkosView &mu_d, PetscScalarKokkosView &coo_v_d, Mat A_stream_removal) {

   const double dx = LENGTH / N_CELLS;
   
   PetscInt global_rows, global_cols, local_rows, local_cols;
   MatGetSize(A_stream_removal, &global_rows, &global_cols);   
   MatGetLocalSize(A_stream_removal, &local_rows, &local_cols);

   PetscInt global_row_start, global_row_end_plus_one;
   MatGetOwnershipRange(A_stream_removal, &global_row_start, &global_row_end_plus_one);   

   // This is how many local spatal nodes we have
   PetscInt local_nodes = local_rows / N_ANGLES;   

   // ~~~~~~~~~~
   // MatSetValuesCOO - happens on the device
   // ~~~~~~~~~~  

   // Set the values of the device pointer - has to match the ordering
   // of oor and ooc in preallocate_streaming_removal
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

         // Diagonal is always the second entry regardless of angle
         // Non-diagonal boundary conditions are ignored
         coo_v_d(i*2) = -mu_d[i % N_ANGLES]/dx;
         coo_v_d(i*2 + 1) = fabs(mu_d[i % N_ANGLES])/dx;
      });

   // Left boundary
   if (global_row_start == 0)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, N_ANGLES), KOKKOS_LAMBDA(int a) {      

         // Set diagonal to 1
         if (mu_d[a] > 0)
         {
            coo_v_d[a * 2 + 1] = 1;
         }           
      });
   }
   // Right boundary
   if (global_row_end_plus_one == global_rows)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, N_ANGLES), KOKKOS_LAMBDA(int a) { 

         // Set diagonal to 1
         if (mu_d[a] < 0)
         {
            coo_v_d[(local_nodes - 1) * N_ANGLES * 2 + a * 2 + 1] = 1;
         }           
      });
   }      

   // This should all happen on the gpu
   MatSetValuesCOO(A_stream_removal, coo_v_d.data(), INSERT_VALUES);      

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Add the removal term for a single group
// This happens entirely on the device
void add_removal(PetscScalarKokkosView &mu_d, \
   PetscScalarKokkosView &sigma_t_d, \
   PetscScalarKokkosView &coo_v_d, \
   Mat A_stream_removal) {

   //const double dx = LENGTH / N_CELLS;
   
   PetscInt global_rows, global_cols, local_rows, local_cols;
   MatGetSize(A_stream_removal, &global_rows, &global_cols);   
   MatGetLocalSize(A_stream_removal, &local_rows, &local_cols);

   PetscInt global_row_start, global_row_end_plus_one;
   MatGetOwnershipRange(A_stream_removal, &global_row_start, &global_row_end_plus_one);   

   // This is how many local spatal nodes we have
   PetscInt local_nodes = local_rows / N_ANGLES;   

   // ~~~~~~~~~~
   // MatSetValuesCOO - happens on the device
   // ~~~~~~~~~~  

   // Set to zero first
   Kokkos::deep_copy(coo_v_d, 0.0);

   // Add in the removal term
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

         // Diagonal is always the second entry regardless of angle
         coo_v_d(i*2 + 1) = sigma_t_d[i / N_ANGLES];
   });  

   // Add nothing to the bcs

   // Left boundary
   if (global_row_start == 0)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, N_ANGLES), KOKKOS_LAMBDA(int a) {      

         if (mu_d[a] > 0)
         {
            coo_v_d[a * 2 + 1] = 0;
         }           
      });
   }
   // Right boundary
   if (global_row_end_plus_one == global_rows)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, N_ANGLES), KOKKOS_LAMBDA(int a) { 

         if (mu_d[a] < 0)
         {
            coo_v_d[(local_nodes - 1) * N_ANGLES * 2 + a * 2 + 1] = 0;
         }           
      });
   }      

   // This should all happen on the gpu
   MatSetValuesCOO(A_stream_removal, coo_v_d.data(), ADD_VALUES);      

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode destroy_transport_ctx(PC pc)
{
   transport_ctx *shell;

   PetscFunctionBeginUser;
   
   PCShellGetContext(pc, &shell);
   VecDestroy(&shell->inverse_sigma_t_vec);
   PetscFree(shell);
   
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode create_transport_ctx(transport_ctx **shell, Mat A_stream_removal, PetscScalar *sigma_t)
{
   transport_ctx *newctx;

   PetscFunctionBeginUser;

   PetscNew(&newctx);
   // Create vector for removal preconditioning
   MatCreateVecs(A_stream_removal, &newctx->inverse_sigma_t_vec, NULL);
   // VecSet(newctx->inverse_sigma_t_vec, 1.0);
   // for (int i = 0; i < N_CELLS; i++)
   // {
   //    // Copy the inverse of the total xsection to the vector
   //    if (sigma_t[i] > 0.1){
   //       VecSetValue(newctx->inverse_sigma_t_vec, i, 1.0/sigma_t[i], INSERT_VALUES);
   //    }
   // }   
   MatGetDiagonal(A_stream_removal, newctx->inverse_sigma_t_vec);
   VecReciprocal(newctx->inverse_sigma_t_vec);

   *shell = newctx;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the preconditioner for the removal term
PetscErrorCode ShellPCApply(PC pc, Vec x, Vec y)
{
   transport_ctx *shell;
   PetscFunctionBeginUser;
   
   PCShellGetContext(pc, &shell);
   VecPointwiseMult(y, x, shell->inverse_sigma_t_vec);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Apply the streaming/removal + scattering terms
PetscErrorCode ShellMatMultApply(Mat A, Vec x, Vec y)
{
   matshell_ctx *ctx;
   PetscFunctionBeginUser;
   MatShellGetContext(A, &ctx);

   // ~~~~~~~~~~~~~
   // Apply the streaming/removal term
   // ~~~~~~~~~~~~~ 
   MatMult(ctx->A_stream_removal, x, y);

   // ~~~~~~~~~~~~~
   // Apply the scattering term matrix-free
   // ~~~~~~~~~~~~~ 

   // Get the device views
   // Making sure to use a const view for the input
   PetscScalarConstKokkosView x_d;
   VecGetKokkosView(x, &x_d);
   PetscScalarKokkosView y_d;
   VecGetKokkosView(y, &y_d);   

   // Let's reshape (without copying) the 1D input view into a 2D view
   // Get the raw pointer from the 1D view
   const PetscScalar* x_d_1d = x_d.data();

   // Create the 2D view using the pointer and new dimensions
   // The 2D view uses LayoutRight so we have the correct ordering of contiguous 
   // in angle
   PetscScalar2DConstKokkosView x_d_2d(x_d_1d, N_CELLS, N_ANGLES);   

   // Now let's integrate the angular flux to get the scalar flux
   // This is just a dgemm (on the device)
   const double alpha = double(1.0);
   const double beta = double(0.0);   
   KokkosBlas::gemm("N","N",alpha,x_d_2d,*ctx->w_d,beta,*ctx->scalar_flux_d);

   // Now let's multiply by the scattering xsection
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, N_CELLS), KOKKOS_LAMBDA(int i) {

         // We have to divide by the sum of weights to get the amount going
         // into each angle
         (*ctx->scalar_flux_d)(i, 0) *= (*ctx->sigma_s_d)(i)/ctx->sum_weights;
      });   
   
   // Now let's put the scatter back into y
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), N_CELLS, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // cell
         PetscInt i = t.league_rank();

         // For all the angles
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, N_ANGLES), [&](const PetscInt j) {   

               // Minus the scattering term to the output vector as it's on the lhs
               y_d(i * N_ANGLES + j) -= (*ctx->scalar_flux_d)(i, 0);
         });
   });

   // Restore the views
   VecRestoreKokkosView(x, &x_d);
   VecRestoreKokkosView(y, &y_d);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Create the PETSc preconditioner
void create_matshell(Mat A_stream_removal, Mat *A, PetscScalarKokkosView *sigma_s_d, PetscScalar2DKokkosView *scalar_flux_d, PetscScalar2DKokkosView *w_d, PetscScalar sum_weights) {

   // ~~~~~~~~~~~~~
   // Create the matshell which we use to apply streaming/removal + scattering
   // ~~~~~~~~~~~~~   

   // Create the context for the preconditioner   
   matshell_ctx *newctx;
   PetscNew(&newctx);

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
   MatGetSize(A_stream_removal, &global_rows, &global_cols);
   MatGetLocalSize(A_stream_removal, &local_rows, &local_cols);

   MatCreateShell(PETSC_COMM_WORLD, local_rows, local_cols, global_rows, global_cols, newctx, A);
   // The subroutine ShellMatMultApply applies the matrix
   MatShellSetOperation(*A, MATOP_MULT, (void (*)(void))ShellMatMultApply);

   MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
   // Have to make sure to set the type of vectors the shell creates
   VecType vtype;
   MatGetVecType(A_stream_removal, &vtype);
   MatShellSetVecType(*A, vtype);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode destroy_matshell_ctx(Mat A)
{
   matshell_ctx *ctx;

   PetscFunctionBeginUser;
   
   MatShellGetContext(A, &ctx);
   PetscFree(ctx);
   
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Create the PETSc preconditioner
void create_pc(PC pc, Mat A_stream_removal, PetscScalar *sigma_t) {

   // Set our PC type to be an multiplicative combination of preconditioners
   PCSetType(pc, PCCOMPOSITE);
   PCCompositeSetType(pc, PC_COMPOSITE_MULTIPLICATIVE);

   // ~~~~~~~~~~~~~
   // Preconditioner for removal term
   // ~~~~~~~~~~~~~
   // Shell preconditioner for removal term
   PCCompositeAddPCType(pc, PCSHELL);
   PC pc_removal;
   // Get the 2nd PC object out (ie the removal one)
   PCCompositeGetPC(pc, 0, &pc_removal);

   // Create the context for the preconditioner   
   transport_ctx *shell;
   create_transport_ctx(&shell, A_stream_removal, sigma_t);
   
   // Set the routine that does the apply
   PCShellSetApply(pc_removal, ShellPCApply);
   PCShellSetContext(pc_removal, shell);
   
   // Set the routine that destroys
   PCShellSetDestroy(pc_removal, destroy_transport_ctx);
   
   // Set the name
   PCShellSetName(pc_removal, "RemovalPCShell");

   // ~~~~~~~~~~~~~
   // Preconditioner for streaming term
   // ~~~~~~~~~~~~~
   // AIR preconditioner for streaming operator
   // This will operate on pmat
   PCCompositeAddPCType(pc, PCAIR);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   // ~~~~~~~~~~~~~
   // Initialise petsc and pflare
   // ~~~~~~~~~~~~~

   PetscInitialize(&argc, &args, (char*)0, NULL);
   // Register the pflare types
   PCRegister_PFLARE();    

   // ~~~~~~~~~~~~~
   // Get options from the command line
   // ~~~~~~~~~~~~~

   // Do we diagonally scale our matrix before solving
   // Defaults to false
   PetscBool diag_scale = PETSC_FALSE;
   PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL); 
   // Do we precondition with streaming or streaming/removal
   // Defaults to false
   PetscBool precon_stream = PETSC_FALSE;
   PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL);        
   // Total xsections range between 10^0 and 10^max_exponent
   PetscInt max_exponent = 1;
   PetscOptionsGetInt(NULL, NULL, "-max_exponent", &max_exponent, NULL);

   // ~~~~~~~~~~~~~
   // Compute the Sn quadrature
   // ~~~~~~~~~~~~~  
   PetscScalar mu[N_ANGLES], w[N_ANGLES];
   PetscScalar sum_weights;
   create_sn_quadrature(N_ANGLES, mu, w, &sum_weights);   

   // ~~~~~~~~~~~~~
   // Preallocate the sparsity of the streaming/removal operator
   // ~~~~~~~~~~~~~   

   const int total_unknowns = N_ANGLES * N_CELLS;
   PetscInt local_rows, local_cols;

   Mat A_stream, A_stream_removal, A;
   Vec x, b, diag_vec;
   KSP ksp;
   PC pc;

   // Create the streaming operator
   MatCreate(PETSC_COMM_WORLD, &A_stream);
   MatSetSizes(A_stream, PETSC_DECIDE, PETSC_DECIDE, total_unknowns, total_unknowns);
   MatSetType(A_stream, MATAIJKOKKOS);
   MatSetFromOptions(A_stream);
   MatSetUp(A_stream);
   MatGetLocalSize(A_stream, &local_rows, &local_cols);
   // Preallocate the streaming operator
   preallocate_streaming_removal(mu, A_stream);

   // Vectors
   MatCreateVecs(A_stream, &x, &b);
   VecDuplicate(x, &diag_vec);   
   
   // This is how many local spatal nodes we have
   PetscInt local_nodes = local_rows / N_ANGLES;     

   // ~~~~~~~~~~~~~
   // Define the cross sections
   // ~~~~~~~~~~~~~

   // Seed the random number generator
   srand(0);   

   // Define the range for the random exponent
   int min_exponent = 0;
   int exponent_range = max_exponent - min_exponent + 1;   

   // Total cross-section on each local cell  
   // Random between 0 and 10^max_exponent
   PetscScalar *sigma_t, *sigma_s;
   PetscMalloc2(local_nodes, &sigma_t, local_nodes, &sigma_s);
   for (int i = 0; i < local_nodes; i++)
   {
      // Generate a random mantissa between 0.1 and 1.0
      PetscScalar mantissa = (PetscScalar)rand() / (PetscScalar)RAND_MAX;
      // Generate a random integer exponent in the desired range
      int exponent = min_exponent + (rand() % exponent_range);
      // Combine mantissa and exponent
      //sigma_t[i] = mantissa * pow(10.0, (PetscScalar)exponent);
      sigma_t[i] = max_exponent;
      sigma_s[i] = max_exponent;
   }

   // for (int i = 0; i < local_nodes; i++)
   // {
   //    std::cout << "sigma_t[" << i << "] = " << sigma_t[i] << " sigma_s[" << i << "] = " << sigma_s[i] << std::endl;
   // }

   // ~~~~~~~~~~~~~
   // Copy data to the device where needed
   // ~~~~~~~~~~~~~
   
   // Be careful about the scope of device memory
   {
   // Create device memory used in assembly of
   // the streaming/removal operator
   // This is how many local nnzs we have
   // We have 2 non-zeros per unknown
   PetscScalarKokkosView coo_v_d("coo_v_d", 2 * local_rows);   

   // Device memory for Sn quadrature
   PetscScalarKokkosView mu_d("mu_d", N_ANGLES);      
   PetscScalar2DKokkosView w_d("w_d", N_ANGLES, 1);      
   PetscScalarKokkosViewHostUnmanaged mu_h(mu, N_ANGLES); 
   PetscScalar2DKokkosViewHostUnmanaged w_h(w, N_ANGLES, 1); 
   // Copy the Sn quadrature to device memory     
   Kokkos::deep_copy(mu_d, mu_h);
   Kokkos::deep_copy(w_d, w_h);

   // Device memory for xsections
   PetscScalarKokkosView sigma_t_d("sigma_t_d", N_CELLS);      
   PetscScalarKokkosView sigma_s_d("sigma_s_d", N_CELLS);      
   PetscScalarKokkosViewHostUnmanaged sigma_t_h(sigma_t, N_CELLS); 
   PetscScalarKokkosViewHostUnmanaged sigma_s_h(sigma_s, N_CELLS);    
   // Copy the xsections to device memory     
   Kokkos::deep_copy(sigma_t_d, sigma_t_h);   
   Kokkos::deep_copy(sigma_s_d, sigma_s_h);    

   // Device memory for scalar flux
   PetscScalar2DKokkosView scalar_flux_d("scalar_flux_d", N_CELLS, 1);

   // Fill the streaming operator
   insert_streaming(mu_d, coo_v_d, A_stream);

   // // Diagonally scale the streaming operator 
   // if (diag_scale) {

   //    MatGetDiagonal(A_stream, diag_vec);
   //    VecReciprocal(diag_vec);
   //    MatDiagonalScale(A_stream, diag_vec, PETSC_NULLPTR);
   // }        

   // Duplicate the sparsity of the streaming operator
   MatDuplicate(A_stream, MAT_DO_NOT_COPY_VALUES, &A_stream_removal);      

   // ~~~~~~~~~~~~~~
   // Group loop
   // ~~~~~~~~~~~~~~

   MatCopy(A_stream, A_stream_removal, SAME_NONZERO_PATTERN);
   
   // Add in the removal operator
   add_removal(mu_d, sigma_t_d, coo_v_d, A_stream_removal); 

   // RHS: external source
   VecSet(b, 1.0);

   // Diagonally scale the streaming/removal operator 
   if (diag_scale) {

      MatGetDiagonal(A_stream_removal, diag_vec);
      VecReciprocal(diag_vec);
      MatDiagonalScale(A_stream_removal, diag_vec, PETSC_NULLPTR);
      VecPointwiseMult(b, diag_vec, b);
   }    

   create_matshell(A_stream_removal, &A, &sigma_s_d, &scalar_flux_d, &w_d, sum_weights);

   // Solve
   KSPCreate(PETSC_COMM_WORLD, &ksp);
   // Do we precondition with the streaming operator
   if (precon_stream)
   {
      KSPSetOperators(ksp, A, A_stream);
   }
   else
   {
      KSPSetOperators(ksp, A, A_stream_removal);
   }
   KSPGetPC(ksp, &pc);

   // Set the preconditioners
   create_pc(pc, A_stream_removal, sigma_t);

   KSPSetFromOptions(ksp);
   KSPSolve(ksp, b, x);

   }

   // ~~~~~~~~~~~~~~
   // Cleanup
   // ~~~~~~~~~~~~~~   

   // Clean up
   KSPDestroy(&ksp);
   VecDestroy(&x);
   VecDestroy(&b);
   VecDestroy(&diag_vec);
   MatDestroy(&A_stream_removal);
   MatDestroy(&A_stream);
   destroy_matshell_ctx(A);
   MatDestroy(&A);
   PetscFree2(sigma_t, sigma_s);
   PetscFinalize();
   return 0;
}
