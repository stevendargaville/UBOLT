// 1D slab multigroup SN driver
//
// Group Gauss-Seidel with downscatter only: groups are ordered high energy to
// low, the within-group scatter stays on the lhs (matrix-free) and everything
// scattering out of an already-solved group goes to the rhs. With no upscatter
// the group system is block lower triangular, so one forward sweep is exact and
// there is no outer iteration - that arrives with upscatter
//
// The sparsity is preallocated once and every group refills the same matrix
// through TransportOperator::assemble()
//
// Currently run with:
// make build_tests && ./slab_1d_mgk -ksp_monitor -n_groups 4 -sigma_transfer 0.5 -max_exponent 2

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include <petscksp.h>
#include "pflare.h"
#include <vector>

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
   PetscInt n_groups = 1;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_groups", &n_groups, NULL));
   PetscReal length = 1.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-length", &length, NULL));
   // Do we precondition with streaming or streaming/removal
   PetscBool precon_stream = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL));
   // Within-group total and scattering xsections, constant across the slab
   PetscInt max_exponent = 1;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_exponent", &max_exponent, NULL));
   // Downscatter into the next group down. Zero leaves the groups uncoupled,
   // which is what makes -n_groups 1 the single-group problem exactly
   PetscReal sigma_transfer = 0.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-sigma_transfer", &sigma_transfer, NULL));

   KSPConvergedReason reason = KSP_CONVERGED_ITERATING;

   // Device memory has to be gone before PetscFinalize takes Kokkos down
   {
      PhaseSpace ps;
      PetscCall(ps.create(PETSC_COMM_WORLD, n_cells, n_angles, n_groups));
      SNQuadrature quad;
      PetscCall(quad.create(n_angles));
      StructuredFD1D disc;
      PetscCall(disc.create(PETSC_COMM_WORLD, ps, length, quad));

      // ~~~~~~~~~~~~~
      // Cross sections, per group and local cell
      //
      // Absorption is the same in every group: sigma_t carries the within-group
      // scatter plus whatever leaves the group by downscatter, so adding
      // transfer moves reaction rate between groups instead of inventing it.
      // The last group has nowhere to scatter down to
      // ~~~~~~~~~~~~~
      GroupXSections xs;
      PetscCall(xs.create(ps));
      for (PetscInt g = 0; g < n_groups; g++) {

         const PetscBool has_downscatter = (g + 1 < n_groups) ? PETSC_TRUE : PETSC_FALSE;
         const PetscScalar out = has_downscatter ? (PetscScalar)sigma_transfer : 0.0;

         PetscCall(xs.set_sigma_t(g, (PetscScalar)max_exponent + out));
         PetscCall(xs.set_sigma_s(g, g, (PetscScalar)max_exponent));
         if (has_downscatter) PetscCall(xs.set_sigma_s(g, g + 1, out));
      }

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free.
      // Built once for group 0 and re-pointed at each group's xsections below
      // ~~~~~~~~~~~~~
      StreamingTerm streaming;
      PetscCall(streaming.create(ps, disc, quad));
      RemovalTerm removal;
      PetscCall(removal.create(ps, disc, xs.sigma_t(0)));
      ScatteringTerm scattering;
      PetscCall(scattering.create(ps, quad, xs.sigma_s(0, 0)));

      TransportOperator op;
      PetscCall(op.create(PETSC_COMM_WORLD, ps, disc));
      PetscCall(op.add_term(&streaming));
      PetscCall(op.add_term(&removal));
      PetscCall(op.add_term(&scattering));

      GroupTransfer transfer;
      PetscCall(transfer.create(ps, quad, xs, disc.boundary_info()));

      // ~~~~~~~~~~~~~
      // One angular flux Vec per group, plus a single rhs we rebuild per group
      // ~~~~~~~~~~~~~
      Vec b;
      std::vector<Vec> psi(n_groups, NULL);
      PetscCall(MatCreateVecs(op.assembled_mat(), &psi[0], &b));
      for (PetscInt g = 1; g < n_groups; g++) PetscCall(VecDuplicate(psi[0], &psi[g]));

      // The streaming-only pmat does not depend on the group, so it is built
      // once here and PCAIR keeps its setup across the whole sweep
      Mat streaming_mat = NULL;
      if (precon_stream)
      {
         const OperatorTerm *streaming_only[] = {&streaming};
         PetscCall(op.assemble_subset(1, streaming_only, &streaming_mat));
      }

      // ~~~~~~~~~~~~~
      // Sweep the groups
      // ~~~~~~~~~~~~~
      TransportSolver solver;
      for (PetscInt g = 0; g < n_groups; g++) {

         // Point the terms at this group and refill the values. No
         // re-preallocation - the sparsity is the discretisation's, not the
         // group's
         removal.set_sigma_t(xs.sigma_t(g));
         scattering.set_sigma_s(xs.sigma_s(g, g));
         PetscCall(op.assemble());

         // The shell only exists once something has been assembled
         if (g == 0) PetscCall(solver.create(PETSC_COMM_WORLD, op, \
            precon_stream ? streaming_mat : op.assembled_mat()));
         // sigma_t changed under the removal preconditioner
         else PetscCall(solver.refresh());

         // Rhs: the external source, the incoming flux on the Dirichlet rows,
         // and everything downscattered out of the groups already solved
         PetscCall(VecSet(b, 1.0));
         for (PetscInt g_from = 0; g_from < g; g_from++) {
            PetscCall(transfer.add_source(g_from, g, b));
         }

         PetscCall(solver.solve(b, psi[g]));
         reason = solver.converged_reason();
         if (reason <= 0) break;

         // This group's scalar flux is fixed now, and every group below it
         // scatters from it. Integrate once here rather than once per target
         PetscCall(transfer.set_scalar_flux(g, psi[g]));
      }

      PetscCall(solver.destroy());
      PetscCall(transfer.destroy());
      PetscCall(op.destroy());
      PetscCall(xs.destroy());
      PetscCall(disc.destroy());
      PetscCall(MatDestroy(&streaming_mat));
      PetscCall(VecDestroy(&b));
      for (PetscInt g = 0; g < n_groups; g++) PetscCall(VecDestroy(&psi[g]));
   }

   // Did we converge - this is the pass/fail of the test. Diagnostics go to
   // stderr so they don't pollute the -ksp_monitor output the baselines in
   // tests/baselines are captured from
   if (reason <= 0) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "KSP did not converge: %s\n", KSPConvergedReasons[reason]));

   PetscCall(PetscFinalize());
   return (reason > 0) ? 0 : 1;
}
