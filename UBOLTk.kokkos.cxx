#include <petscvec_kokkos.hpp>
#include <petscksp.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "pflare.h"

using DefaultMemorySpace       = Kokkos::DefaultExecutionSpace::memory_space;
using PetscScalarKokkosView    = Kokkos::View<PetscScalar *, DefaultMemorySpace>;

#define N_CELLS 10
#define N_ANGLES 4
#define N_GROUPS 3
#define LENGTH 1.0

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// SN quadrature (Gauss-Legendre for N_ANGLES=2 or 4)
void get_sn_quadrature(int no_angles, double *mu, double *w) {
    if (no_angles == 2) {
        mu[0] = -0.5773502692; mu[1] = 0.5773502692;
        w[0] = 1.0;            w[1] = 1.0;
    } else if (no_angles == 4) {
        mu[0] = -0.8611363116; mu[1] = -0.3399810436;
        mu[2] =  0.3399810436; mu[3] =  0.8611363116;
        w[0] = 0.3478548451;   w[1] = 0.6521451549;
        w[2] = 0.6521451549;   w[3] = 0.3478548451;
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Create the preallocation structure for the
// assembled streaming/removal operator for a single group
// This happens on the host but we only need to call this once
void preallocate_streaming_removal(double *mu, Mat A) {

   PetscInt i, global_row_start, global_row_end_plus_one;
   PetscInt global_rows, global_cols, local_rows, local_cols;
   MatGetSize(A, &global_rows, &global_cols);
   MatGetLocalSize(A, &local_rows, &local_cols);
   MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one);

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
   // That way the fill code doesn't have to change
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
         // This makes the fill very simple as this matches the ordering
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
   MatSetPreallocationCOO(A, counter, oor, ooc); 
   // Can delete oor and ooc now
   PetscFree2(oor, ooc);    
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Fill the assembled streaming/removal operator for a single group
// This happens entirely on the device
void fill_streaming_removal(double *mu, double sigma_t, PetscScalarKokkosView &coo_v, Mat A) {

   const double dx = LENGTH / N_CELLS;
   
   PetscInt global_rows, global_cols, local_rows, local_cols;
   MatGetSize(A, &global_rows, &global_cols);   
   MatGetLocalSize(A, &local_rows, &local_cols);

   PetscInt global_row_start, global_row_end_plus_one;
   MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one);   

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
         coo_v(i*2) = -mu[i % N_ANGLES]/dx;
         coo_v(i*2 + 1) = fabs(mu[i % N_ANGLES])/dx + sigma_t;
      });

   // Left boundary
   if (global_row_start == 0)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, N_ANGLES), KOKKOS_LAMBDA(int a) {      

         // Set diagonal to 1
         if (mu[a] > 0)
         {
            coo_v[a * 2 + 1] = 1;
         }           
      });
   }
   // Right boundary
   if (global_row_end_plus_one == global_rows)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, N_ANGLES), KOKKOS_LAMBDA(int a) { 

         // Set diagonal to 1
         if (mu[a] < 0)
         {
            coo_v[(local_nodes - 1) * N_ANGLES * 2 + a * 2 + 1] = 1;
         }           
      });
   }      

   // This should all happen on the gpu
   MatSetValuesCOO(A, coo_v.data(), INSERT_VALUES);      

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

    PetscInitialize(&argc, &args, (char*)0, NULL);
    // Register the pflare types
    PCRegister_PFLARE();    

    // Do we diagonally scale our matrix before solving
    // Defaults to false
    PetscBool diag_scale = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL); 
    // Do we precondition with streaming or streaming/removal
    // Defaults to false
    PetscBool precon_stream = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL);        

    const int total_unknowns = N_ANGLES * N_CELLS;
    PetscInt local_rows, local_cols;

    // Cross sections
    double sigma_t[N_GROUPS] = {1.0, 0.9, 0.8};  // Total cross section per group
    double sigma_s[N_GROUPS][N_GROUPS] = {
        {0.5, 0.2, 0.1}, // scattering from g'=0 to g
        {0.1, 0.6, 0.2},
        {0.0, 0.1, 0.5}
    };

    // Quadrature
    double mu[N_ANGLES], w[N_ANGLES];
    get_sn_quadrature(N_ANGLES, mu, w);

    Mat A_stream, A;
    Vec x, b;
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
    
    // Be careful about the scope of coo_v
    {
      // Create device memory to fill the streaming/removal operator
      PetscScalarKokkosView coo_v("coo_v", 2 * local_rows);    

      // Fill the streaming operator
      fill_streaming_removal(mu, 0.0, coo_v, A_stream);

      // Diagonally scale the streaming operator 
      if (diag_scale) {

         Vec diag_vec;
         VecDuplicate(x, &diag_vec);
         MatGetDiagonal(A_stream, diag_vec);
         VecReciprocal(diag_vec);
         MatDiagonalScale(A_stream, diag_vec, PETSC_NULLPTR);
         VecPointwiseMult(b, diag_vec, b);
         VecDestroy(&diag_vec);
      }        

      // This will be the streaming/removal operator
      MatDuplicate(A_stream, MAT_DO_NOT_COPY_VALUES, &A);      

      // ~~~~~~~~~~~~
      // This is where our group loop would start
      // ~~~~~~~~~~~~
      
      // Fill the streaming/removal operator
      fill_streaming_removal(mu, sigma_t[0], coo_v, A); 

      // RHS: external source
      VecSet(b, 1.0);

      // Diagonally scale the streaming/removal operator 
      if (diag_scale) {

         Vec diag_vec;
         VecDuplicate(x, &diag_vec);
         MatGetDiagonal(A, diag_vec);
         VecReciprocal(diag_vec);
         MatDiagonalScale(A, diag_vec, PETSC_NULLPTR);
         VecPointwiseMult(b, diag_vec, b);
         VecDestroy(&diag_vec);
      }      

      MatView(A_stream, PETSC_VIEWER_STDOUT_WORLD);
      MatView(A, PETSC_VIEWER_STDOUT_WORLD);      

      // Solve
      KSPCreate(PETSC_COMM_WORLD, &ksp);
      // Do we precondition with the streaming operator
      if (precon_stream)
      {
         KSPSetOperators(ksp, A, A_stream);
      }
      else
      {
         KSPSetOperators(ksp, A, A);
      }
      KSPGetPC(ksp, &pc);
      PCSetType(pc, PCAIR);
      KSPSetFromOptions(ksp);
      KSPSolve(ksp, b, x);

    }

    // Clean up
    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}
