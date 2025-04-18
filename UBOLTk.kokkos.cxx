#include <petscvec_kokkos.hpp>
#include <Kokkos_DualView.hpp>
#include <petscksp.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "pflare.h"

using DefaultMemorySpace        = Kokkos::DefaultExecutionSpace::memory_space;
using PetscScalarKokkosView     = Kokkos::View<PetscScalar *, DefaultMemorySpace>;
using HostMirrorMemorySpace     = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;
using PetscScalarKokkosViewHost = Kokkos::View<PetscScalar *, HostMirrorMemorySpace>;
using PetscScalarKokkosViewHostUnmanaged = Kokkos::View<PetscScalar *, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

#define N_CELLS 1000
#define N_ANGLES 4
#define LENGTH 1.0

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// SN quadrature (Gauss-Legendre for N_ANGLES=2 or 4)
void get_sn_quadrature(int no_angles, PetscScalar *mu, PetscScalar *w) {
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
void preallocate_streaming_removal(PetscScalar *mu, Mat A) {

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
   MatSetPreallocationCOO(A, counter, oor, ooc); 
   // Can delete oor and ooc now
   PetscFree2(oor, ooc);    
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Insert the streaming term for a single group
// This happens entirely on the device
void insert_streaming(PetscScalarKokkosView &mu_d, PetscScalarKokkosView &coo_v_d, Mat A) {

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
   MatSetValuesCOO(A, coo_v_d.data(), INSERT_VALUES);      

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Add the removal term for a single group
// This happens entirely on the device
void add_removal(PetscScalarKokkosView &mu_d, \
   PetscScalarKokkosView &sigma_t_d, \
   PetscScalarKokkosView &coo_v_d, \
   Mat A) {

   //const double dx = LENGTH / N_CELLS;
   
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
   MatSetValuesCOO(A, coo_v_d.data(), ADD_VALUES);      

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   PetscInitialize(&argc, &args, (char*)0, NULL);
   // Register the pflare types
   PCRegister_PFLARE();    

   // Do we diagonally scale our matrix before solving
   // Defaults to false
   PetscBool diag_scale = PETSC_TRUE;
   PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL); 
   // Do we precondition with streaming or streaming/removal
   // Defaults to false
   PetscBool precon_stream = PETSC_FALSE;
   PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL);        

   const int total_unknowns = N_ANGLES * N_CELLS;
   PetscInt local_rows, local_cols;

   // Sn quadrature
   PetscScalar mu[N_ANGLES], w[N_ANGLES];
   get_sn_quadrature(N_ANGLES, mu, w);

   Mat A_stream, A;
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

   // Seed the random number generator
   srand(0);   

   // ~~~~~~~~~~~
   // Cross sections
   // ~~~~~~~~~~~

   // Define the range for the random exponent
   int min_exponent = -2;
   int max_exponent = 5;
   int exponent_range = max_exponent - min_exponent + 1;   

   // Total cross-section on each local cell  
   // Random between 0 and 10^max_exponent
   PetscScalar *sigma_t;
   PetscMalloc1(local_nodes, &sigma_t);
   for (int i = 0; i < local_nodes; i++)
   {
      // Generate a random mantissa between 0.1 and 1.0
      PetscScalar mantissa = (PetscScalar)rand() / (PetscScalar)RAND_MAX;
      // Generate a random integer exponent in the desired range
      int exponent = min_exponent + (rand() % exponent_range);
      // Combine mantissa and exponent
      sigma_t[i] = mantissa * pow(10.0, (PetscScalar)exponent);
   }

   // ~~~~~~~~~~~
   // ~~~~~~~~~~~
   
   // Be careful about the scope of device memory
   {
   // Create device memory used in assembly of
   // the streaming/removal operator
   // This is how many local nnzs we have
   // We have 2 non-zeros per unknown
   PetscScalarKokkosView coo_v_d("coo_v_d", 2 * local_rows);   

   // Device memory for Sn quadrature
   PetscScalarKokkosView mu_d("mu_d", N_ANGLES);      
   PetscScalarKokkosViewHostUnmanaged mu_h(mu, N_ANGLES); 
   // Copy the Sn quadrature to device memory     
   Kokkos::deep_copy(mu_d, mu_h);

   // Device memory for total xsection
   PetscScalarKokkosView sigma_t_d("sigma_t_d", N_CELLS);      
   PetscScalarKokkosViewHostUnmanaged sigma_t_h(sigma_t, N_CELLS); 
   // Copy the total xsection to device memory     
   Kokkos::deep_copy(sigma_t_d, sigma_t_h);   

   // Fill the streaming operator
   insert_streaming(mu_d, coo_v_d, A_stream);

   // // Diagonally scale the streaming operator 
   // if (diag_scale) {

   //    MatGetDiagonal(A_stream, diag_vec);
   //    VecReciprocal(diag_vec);
   //    MatDiagonalScale(A_stream, diag_vec, PETSC_NULLPTR);
   // }        

   // Duplicate the sparsity of the streaming operator
   MatDuplicate(A_stream, MAT_DO_NOT_COPY_VALUES, &A);      

   // ~~~~~~~~~~~~
   // This is where our group loop would start
   // ~~~~~~~~~~~~

   MatCopy(A_stream, A, SAME_NONZERO_PATTERN);
   
   // Add in the removal operator
   add_removal(mu_d, sigma_t_d, coo_v_d, A); 

   // RHS: external source
   VecSet(b, 1.0);

   // Diagonally scale the streaming/removal operator 
   if (diag_scale) {

      MatGetDiagonal(A, diag_vec);
      VecReciprocal(diag_vec);
      MatDiagonalScale(A, diag_vec, PETSC_NULLPTR);
      VecPointwiseMult(b, diag_vec, b);
   }    

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
   VecDestroy(&diag_vec);
   MatDestroy(&A);
   PetscFree(sigma_t);
   PetscFinalize();
   return 0;
}
