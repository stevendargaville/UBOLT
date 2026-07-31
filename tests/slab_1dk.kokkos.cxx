// 1D slab single-group SN driver
//
// Currently run with:
// make build_tests && ./slab_1dk -ksp_monitor -sub_1_pc_air_print_stats_timings -ksp_pc_side right -precon_stream -max_exponent 2

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include <petscksp.h>
#include "pflare.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   PetscFunctionBeginUser;

   PetscCall(PetscInitialize(&argc, &args, (char*)0, NULL));
   // Register the pflare types
   PCRegister_PFLARE();

   // ~~~~~~~~~~~~~
   // Options
   // ~~~~~~~~~~~~~
   // Size of the phase space and the slab
   PetscInt n_cells = 1000;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_cells", &n_cells, NULL));
   PetscInt n_angles = 4;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_angles", &n_angles, NULL));
   PetscReal length = 1.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-length", &length, NULL));
   // Do we diagonally scale our matrix before solving
   PetscBool diag_scale = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));
   // Do we precondition with streaming or streaming/removal
   PetscBool precon_stream = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL));
   // Total and scattering xsections, constant across the slab
   PetscInt max_exponent = 1;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_exponent", &max_exponent, NULL));

   KSPConvergedReason reason;

   // Device memory has to be gone before PetscFinalize takes Kokkos down
   {
      PhaseSpace ps;
      PetscCall(ps.create(PETSC_COMM_WORLD, n_cells, n_angles));
      SNQuadrature quad;
      PetscCall(quad.create(n_angles));
      StructuredFD1D disc;
      PetscCall(disc.create(PETSC_COMM_WORLD, ps, length, quad));

      // ~~~~~~~~~~~~~
      // Cross sections, one per local cell
      // ~~~~~~~~~~~~~
      PetscScalarKokkosView sigma_t_d("sigma_t_d", ps.local_cells);
      PetscScalarKokkosView sigma_s_d("sigma_s_d", ps.local_cells);
      Kokkos::deep_copy(sigma_t_d, (PetscScalar)max_exponent);
      Kokkos::deep_copy(sigma_s_d, (PetscScalar)max_exponent);

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free
      // ~~~~~~~~~~~~~
      StreamingTerm streaming;
      PetscCall(streaming.create(ps, disc, quad));
      RemovalTerm removal;
      PetscCall(removal.create(ps, disc, sigma_t_d));
      ScatteringTerm scattering;
      PetscCall(scattering.create(ps, quad, sigma_s_d));

      TransportOperator op;
      PetscCall(op.create(PETSC_COMM_WORLD, ps, disc));
      PetscCall(op.add_term(&streaming));
      PetscCall(op.add_term(&removal));
      PetscCall(op.add_term(&scattering));
      PetscCall(op.assemble());

      // ~~~~~~~~~~~~~
      // RHS: external source, and the incoming flux on the Dirichlet rows
      // ~~~~~~~~~~~~~
      Vec x, b, diag_vec;
      PetscCall(MatCreateVecs(op.assembled_mat(), &x, &b));
      PetscCall(VecDuplicate(x, &diag_vec));
      PetscCall(VecSet(b, 1.0));

      // Diagonally scale the streaming/removal operator
      if (diag_scale) {

         PetscCall(MatGetDiagonal(op.assembled_mat(), diag_vec));
         PetscCall(VecReciprocal(diag_vec));
         PetscCall(MatDiagonalScale(op.assembled_mat(), diag_vec, PETSC_NULLPTR));
         PetscCall(VecPointwiseMult(b, diag_vec, b));
      }

      // Do we precondition with the streaming operator only
      Mat pmat = op.assembled_mat();
      Mat streaming_mat = NULL;
      if (precon_stream)
      {
         const OperatorTerm *streaming_only[] = {&streaming};
         PetscCall(op.assemble_subset(1, streaming_only, &streaming_mat));
         pmat = streaming_mat;
      }

      // ~~~~~~~~~~~~~
      // Solve
      // ~~~~~~~~~~~~~
      TransportSolver solver;
      PetscCall(solver.create(PETSC_COMM_WORLD, op, pmat));
      PetscCall(solver.solve(b, x));
      reason = solver.converged_reason();

      PetscCall(solver.destroy());
      PetscCall(op.destroy());
      PetscCall(MatDestroy(&streaming_mat));
      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&b));
      PetscCall(VecDestroy(&diag_vec));
   }

   // Did we converge - this is the pass/fail of the test. Diagnostics go to
   // stderr so they don't pollute the -ksp_monitor output the baselines in
   // tests/baselines are captured from
   if (reason <= 0) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "KSP did not converge: %s\n", KSPConvergedReasons[reason]));

   PetscCall(PetscFinalize());
   return (reason > 0) ? 0 : 1;
}
