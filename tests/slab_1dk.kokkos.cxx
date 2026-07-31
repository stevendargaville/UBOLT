// 1D slab single-group SN driver
//
// Currently run with:
// make build_tests && ./slab_1dk -ksp_monitor -sub_1_pc_air_print_stats_timings -ksp_pc_side right -precon_stream -max_exponent 2

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include <petscksp.h>
#include <stdlib.h>
#include "pflare.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   // ~~~~~~~~~~~~~
   // Initialise petsc and pflare
   // ~~~~~~~~~~~~~

   PetscFunctionBeginUser;

   PetscCall(PetscInitialize(&argc, &args, (char*)0, NULL));
   // Register the pflare types
   PCRegister_PFLARE();

   // ~~~~~~~~~~~~~
   // Get options from the command line
   // ~~~~~~~~~~~~~

   // Size of the phase space and the slab
   PetscInt n_cells = 1000;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_cells", &n_cells, NULL));
   PetscInt n_angles = 4;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_angles", &n_angles, NULL));
   PetscReal length = 1.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-length", &length, NULL));
   // Do we diagonally scale our matrix before solving
   // Defaults to false
   PetscBool diag_scale = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));
   // Do we precondition with streaming or streaming/removal
   // Defaults to false
   PetscBool precon_stream = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL));
   // Total xsections range between 10^0 and 10^max_exponent
   PetscInt max_exponent = 1;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_exponent", &max_exponent, NULL));

   // ~~~~~~~~~~~~~
   // Compute the Sn quadrature
   // ~~~~~~~~~~~~~
   PetscScalar *mu, *w;
   PetscScalar sum_weights;
   PetscCall(PetscMalloc2(n_angles, &mu, n_angles, &w));
   PetscCall(create_sn_quadrature(n_angles, mu, w, &sum_weights));

   // ~~~~~~~~~~~~~
   // Preallocate the sparsity of the streaming/removal operator
   // ~~~~~~~~~~~~~

   const PetscInt total_unknowns = n_angles * n_cells;
   PetscInt local_rows, local_cols;

   Mat A_stream, A_stream_removal, A;
   Vec x, b, diag_vec;
   KSP ksp;
   PC pc;
   KSPConvergedReason reason;

   // The parallel decomposition is decided in cells - all the angles on a cell
   // live on the same rank, so a row split can never land mid-cell
   PetscInt local_cells = PETSC_DECIDE;
   PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &local_cells, &n_cells));

   // Create the streaming operator
   PetscCall(MatCreate(PETSC_COMM_WORLD, &A_stream));
   PetscCall(MatSetSizes(A_stream, local_cells * n_angles, local_cells * n_angles, total_unknowns, total_unknowns));
   PetscCall(MatSetType(A_stream, MATAIJKOKKOS));
   PetscCall(MatSetFromOptions(A_stream));
   PetscCall(MatSetUp(A_stream));
   PetscCall(MatGetLocalSize(A_stream, &local_rows, &local_cols));
   // Preallocate the streaming operator
   PetscCall(preallocate_streaming_removal(n_angles, mu, A_stream));

   // Vectors
   PetscCall(MatCreateVecs(A_stream, &x, &b));
   PetscCall(VecDuplicate(x, &diag_vec));

   // This is how many local spatal nodes we have
   PetscInt local_nodes = local_rows / n_angles;

   // ~~~~~~~~~~~~~
   // Define the cross sections
   // ~~~~~~~~~~~~~

   // Total and scattering cross-section on each local cell
   // These are currently constant across the slab - the commented out lines
   // give each cell a random xsection between 10^0 and 10^max_exponent
   //srand(0);
   //int min_exponent = 0;
   //int exponent_range = max_exponent - min_exponent + 1;
   PetscScalar *sigma_t, *sigma_s;
   PetscCall(PetscMalloc2(local_nodes, &sigma_t, local_nodes, &sigma_s));
   for (PetscInt i = 0; i < local_nodes; i++)
   {
      // Generate a random mantissa between 0.1 and 1.0
      //PetscScalar mantissa = (PetscScalar)rand() / (PetscScalar)RAND_MAX;
      // Generate a random integer exponent in the desired range
      //int exponent = min_exponent + (rand() % exponent_range);
      // Combine mantissa and exponent
      //sigma_t[i] = mantissa * pow(10.0, (PetscScalar)exponent);
      sigma_t[i] = max_exponent;
      sigma_s[i] = max_exponent;
   }

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
   PetscScalarKokkosView mu_d("mu_d", n_angles);
   PetscScalar2DKokkosView w_d("w_d", n_angles, 1);
   PetscScalarKokkosViewHostUnmanaged mu_h(mu, n_angles);
   PetscScalar2DKokkosViewHostUnmanaged w_h(w, n_angles, 1);
   // Copy the Sn quadrature to device memory
   Kokkos::deep_copy(mu_d, mu_h);
   Kokkos::deep_copy(w_d, w_h);

   // Device memory for xsections
   // These are all sized by the local number of cells - the scatter is
   // applied to the local part of the vectors only
   PetscScalarKokkosView sigma_t_d("sigma_t_d", local_nodes);
   PetscScalarKokkosView sigma_s_d("sigma_s_d", local_nodes);
   PetscScalarKokkosViewHostUnmanaged sigma_t_h(sigma_t, local_nodes);
   PetscScalarKokkosViewHostUnmanaged sigma_s_h(sigma_s, local_nodes);
   // Copy the xsections to device memory
   Kokkos::deep_copy(sigma_t_d, sigma_t_h);
   Kokkos::deep_copy(sigma_s_d, sigma_s_h);

   // Device memory for scalar flux
   PetscScalar2DKokkosView scalar_flux_d("scalar_flux_d", local_nodes, 1);

   // Fill the streaming operator
   // Uniform grid so every cell has the same width
   const PetscScalar dx = length / n_cells;
   PetscCall(insert_streaming(n_angles, dx, mu_d, coo_v_d, A_stream));

   // Duplicate the sparsity of the streaming operator
   PetscCall(MatDuplicate(A_stream, MAT_DO_NOT_COPY_VALUES, &A_stream_removal));

   // ~~~~~~~~~~~~~~
   // Group loop
   // ~~~~~~~~~~~~~~

   PetscCall(MatCopy(A_stream, A_stream_removal, SAME_NONZERO_PATTERN));

   // Add in the removal operator
   PetscCall(add_removal(n_angles, mu_d, sigma_t_d, coo_v_d, A_stream_removal));

   // RHS: external source
   PetscCall(VecSet(b, 1.0));

   // Diagonally scale the streaming/removal operator
   if (diag_scale) {

      PetscCall(MatGetDiagonal(A_stream_removal, diag_vec));
      PetscCall(VecReciprocal(diag_vec));
      PetscCall(MatDiagonalScale(A_stream_removal, diag_vec, PETSC_NULLPTR));
      PetscCall(VecPointwiseMult(b, diag_vec, b));
   }

   PetscCall(create_matshell(n_angles, A_stream_removal, &A, &sigma_s_d, &scalar_flux_d, &w_d, sum_weights));

   // Solve
   PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
   // Do we precondition with the streaming operator
   if (precon_stream)
   {
      PetscCall(KSPSetOperators(ksp, A, A_stream));
   }
   else
   {
      PetscCall(KSPSetOperators(ksp, A, A_stream_removal));
   }
   PetscCall(KSPGetPC(ksp, &pc));

   // Set the preconditioners
   PetscCall(create_pc(pc, A_stream_removal));

   PetscCall(KSPSetFromOptions(ksp));
   PetscCall(KSPSolve(ksp, b, x));

   }

   // ~~~~~~~~~~~~~~
   // Did we converge - this is the pass/fail of the test
   // ~~~~~~~~~~~~~~

   PetscCall(KSPGetConvergedReason(ksp, &reason));
   if (reason <= 0)
   {
      // Diagnostics go to stderr so they don't pollute the -ksp_monitor
      // output that the baselines in tests/baselines are captured from
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "KSP did not converge: %s\n", KSPConvergedReasons[reason]));
   }

   // ~~~~~~~~~~~~~~
   // Cleanup
   // ~~~~~~~~~~~~~~

   // Clean up
   PetscCall(KSPDestroy(&ksp));
   PetscCall(VecDestroy(&x));
   PetscCall(VecDestroy(&b));
   PetscCall(VecDestroy(&diag_vec));
   PetscCall(MatDestroy(&A_stream_removal));
   PetscCall(MatDestroy(&A_stream));
   PetscCall(destroy_matshell_ctx(A));
   PetscCall(MatDestroy(&A));
   PetscCall(PetscFree2(sigma_t, sigma_s));
   PetscCall(PetscFree2(mu, w));
   PetscCall(PetscFinalize());

   // Non-zero exit code unless the solve converged
   return (reason > 0) ? 0 : 1;
}
