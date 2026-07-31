#include "ubolt/streaming.hpp"
#include <math.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Create the preallocation structure for the
// assembled streaming/removal operator for a single group
// This happens on the host but we only need to call this once
PetscErrorCode preallocate_streaming_removal(PetscInt n_angles, const PetscScalar *mu, Mat A_stream_removal) {

   PetscInt i, global_row_start, global_row_end_plus_one;
   PetscInt global_rows, global_cols, local_rows, local_cols;

   PetscFunctionBeginUser;

   PetscCall(MatGetSize(A_stream_removal, &global_rows, &global_cols));
   PetscCall(MatGetLocalSize(A_stream_removal, &local_rows, &local_cols));
   PetscCall(MatGetOwnershipRange(A_stream_removal, &global_row_start, &global_row_end_plus_one));

    // Row and column indices for COO assembly
    PetscInt *oor, *ooc;
    // Going to do assembly in the COO interface so assembly happens on the gpu when needed
    // Allocate memory for the coordinates
    PetscCall(PetscMalloc2(2 * local_rows, &oor, 2 * local_rows, &ooc));

    // This is how many local spatal nodes we have
    PetscInt local_nodes = local_rows / n_angles;

   // ~~~~~~~~~~
   // MatSetPreallocationCOO - happens on the host
   // ~~~~~~~~~~
   PetscCount counter = 0;

   // Loop over the local spatial nodes which have n_angles on them
   // Ordering of our unknowns doesn't matter
   // Ordering in insert_streaming when calling MatSetValuesCOO
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
      for (PetscInt a = 0; a < n_angles; a++) {

         // Upwinded uniform grid finite difference operator

         // Positive velocity, we have -> -1 1
         if (mu[a] > 0)
         {
            oor[counter] = i * n_angles + a + global_row_start;
            ooc[counter] = (i - 1) * n_angles + a + global_row_start;

            oor[counter + 1] = i * n_angles + a + global_row_start;
            ooc[counter + 1] = i * n_angles + a + global_row_start;

         }
         // Negative velocity, we have -> 1 -1
         // but we always add the diagonal last
         // This makes the setvalues very simple as this matches the ordering
         // of the positive velocity and hence we don't have to detect
         // what direction
         else
         {
            oor[counter] = i * n_angles + a + global_row_start;
            ooc[counter] = (i + 1) * n_angles + a + global_row_start;

            oor[counter + 1] = i * n_angles + a + global_row_start;
            ooc[counter + 1] = i * n_angles + a + global_row_start;
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
      for (PetscInt a = 0; a < n_angles; a++) {

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
      for (PetscInt a = 0; a < n_angles; a++) {

         // Ignore right entry
         if (mu[a] < 0)
         {
            oor[(local_nodes - 1) * n_angles * 2 + a * 2] = -1;
            ooc[(local_nodes - 1) * n_angles * 2 + a * 2] = -1;
         }
      }
   }

   // Set the indices
   counter = 2 * local_rows;
   PetscCall(MatSetPreallocationCOO(A_stream_removal, counter, oor, ooc));
   // Can delete oor and ooc now
   PetscCall(PetscFree2(oor, ooc));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Insert the streaming term for a single group
// This happens entirely on the device
PetscErrorCode insert_streaming(PetscInt n_angles, PetscScalar dx, PetscScalarKokkosView &mu_d, \
   PetscScalarKokkosView &coo_v_d, \
   Mat A_stream_removal) {

   PetscInt global_rows, global_cols, local_rows, local_cols;
   PetscInt global_row_start, global_row_end_plus_one;

   PetscFunctionBeginUser;

   PetscCall(MatGetSize(A_stream_removal, &global_rows, &global_cols));
   PetscCall(MatGetLocalSize(A_stream_removal, &local_rows, &local_cols));
   PetscCall(MatGetOwnershipRange(A_stream_removal, &global_row_start, &global_row_end_plus_one));

   // This is how many local spatal nodes we have
   PetscInt local_nodes = local_rows / n_angles;

   // ~~~~~~~~~~
   // MatSetValuesCOO - happens on the device
   // ~~~~~~~~~~

   // Set the values of the device pointer - has to match the ordering
   // of oor and ooc in preallocate_streaming_removal
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

         // Diagonal is always the second entry regardless of angle
         // Non-diagonal boundary conditions are ignored
         coo_v_d(i*2) = -mu_d[i % n_angles]/dx;
         coo_v_d(i*2 + 1) = fabs(mu_d[i % n_angles])/dx;
      });

   // Left boundary
   if (global_row_start == 0)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, n_angles), KOKKOS_LAMBDA(int a) {

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
         Kokkos::RangePolicy<>(0, n_angles), KOKKOS_LAMBDA(int a) {

         // Set diagonal to 1
         if (mu_d[a] < 0)
         {
            coo_v_d[(local_nodes - 1) * n_angles * 2 + a * 2 + 1] = 1;
         }
      });
   }

   // This should all happen on the gpu
   PetscCall(MatSetValuesCOO(A_stream_removal, coo_v_d.data(), INSERT_VALUES));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Add the removal term for a single group
// This happens entirely on the device
PetscErrorCode add_removal(PetscInt n_angles, PetscScalarKokkosView &mu_d, \
   PetscScalarKokkosView &sigma_t_d, \
   PetscScalarKokkosView &coo_v_d, \
   Mat A_stream_removal) {

   PetscInt global_rows, global_cols, local_rows, local_cols;
   PetscInt global_row_start, global_row_end_plus_one;

   PetscFunctionBeginUser;

   PetscCall(MatGetSize(A_stream_removal, &global_rows, &global_cols));
   PetscCall(MatGetLocalSize(A_stream_removal, &local_rows, &local_cols));
   PetscCall(MatGetOwnershipRange(A_stream_removal, &global_row_start, &global_row_end_plus_one));

   // This is how many local spatal nodes we have
   PetscInt local_nodes = local_rows / n_angles;

   // ~~~~~~~~~~
   // MatSetValuesCOO - happens on the device
   // ~~~~~~~~~~

   // Set to zero first
   Kokkos::deep_copy(coo_v_d, 0.0);

   // Add in the removal term
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

         // Diagonal is always the second entry regardless of angle
         coo_v_d(i*2 + 1) = sigma_t_d[i / n_angles];
   });

   // Add nothing to the bcs

   // Left boundary
   if (global_row_start == 0)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, n_angles), KOKKOS_LAMBDA(int a) {

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
         Kokkos::RangePolicy<>(0, n_angles), KOKKOS_LAMBDA(int a) {

         if (mu_d[a] < 0)
         {
            coo_v_d[(local_nodes - 1) * n_angles * 2 + a * 2 + 1] = 0;
         }
      });
   }

   // This should all happen on the gpu
   PetscCall(MatSetValuesCOO(A_stream_removal, coo_v_d.data(), ADD_VALUES));

   PetscFunctionReturn(PETSC_SUCCESS);
}
