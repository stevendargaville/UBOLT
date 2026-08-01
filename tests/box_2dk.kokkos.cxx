// 2D box single-group SN driver
//
// The 2D sibling of slab_1dk: streaming and removal assembled into one
// MATAIJKOKKOS matrix, scattering applied matrix-free, PCComposite of the
// removal shell PC and PCAIR
//
// Currently run with:
// make build_tests && ./box_2dk -ksp_monitor -precon_stream -ksp_pc_side right -max_exponent 2

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
   // Size of the phase space and the box
   PetscInt n_cells_x = 100;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_cells_x", &n_cells_x, NULL));
   PetscInt n_cells_y = 100;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_cells_y", &n_cells_y, NULL));
   // 4 is S2 and 12 is S4 - a 2D level-symmetric set has four quadrants
   PetscInt n_angles = 4;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_angles", &n_angles, NULL));
   PetscReal length_x = 1.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-length_x", &length_x, NULL));
   PetscReal length_y = 1.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-length_y", &length_y, NULL));
   // Do we precondition with streaming or streaming/removal
   PetscBool precon_stream = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL));
   // Total xsection, constant across the box
   PetscInt max_exponent = 1;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_exponent", &max_exponent, NULL));
   // Scattering xsection, defaulting to the total one as in the 1D drivers -
   // that is a scattering ratio of exactly 1, which is the hardest case for a
   // preconditioner that does nothing about the scattering. Set it lower to get
   // some absorption
   PetscReal sigma_scatter = (PetscReal)max_exponent;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-sigma_scatter", &sigma_scatter, NULL));

   KSPConvergedReason reason;

   // Device memory has to be gone before PetscFinalize takes Kokkos down
   {
      PhaseSpace ps;
      PetscCall(ps.create(PETSC_COMM_WORLD, n_cells_x * n_cells_y, n_angles));
      SNQuadrature2D quad;
      PetscCall(quad.create(n_angles));
      StructuredFD2D disc;
      PetscCall(disc.create(PETSC_COMM_WORLD, ps, n_cells_x, n_cells_y, length_x, length_y, quad));

      // ~~~~~~~~~~~~~
      // Cross sections, one per local cell
      // ~~~~~~~~~~~~~
      PetscScalarKokkosView sigma_t_d("sigma_t_d", ps.local_cells);
      PetscScalarKokkosView sigma_s_d("sigma_s_d", ps.local_cells);
      Kokkos::deep_copy(sigma_t_d, (PetscScalar)max_exponent);
      Kokkos::deep_copy(sigma_s_d, (PetscScalar)sigma_scatter);

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free
      // ~~~~~~~~~~~~~
      StreamingTerm2D streaming;
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
      Vec x, b;
      PetscCall(MatCreateVecs(op.assembled_mat(), &x, &b));
      PetscCall(VecSet(b, 1.0));

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
      PetscCall(disc.destroy());
      PetscCall(MatDestroy(&streaming_mat));
      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&b));
   }

   // Did we converge - this is the pass/fail of the test. Diagnostics go to
   // stderr so they don't pollute the -ksp_monitor output
   if (reason <= 0) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "KSP did not converge: %s\n", KSPConvergedReasons[reason]));

   PetscCall(PetscFinalize());
   return (reason > 0) ? 0 : 1;
}
