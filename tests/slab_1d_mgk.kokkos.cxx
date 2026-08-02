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
#include <cstring>

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
   // Heterogeneous regions: -n_regions N, then per region r (1-based) an
   // interval -region_<r>_interval x0,x1 painted over the background in order,
   // later regions winning. A region's material is the background group recipe
   // scaled by -region_<r>_density (sigma_t, sigma_s and the transfer
   // together, i.e. a denser or lighter slab of the same stuff - one knob
   // rather than one per group and pair), plus its own -region_<r>_source
   // (the same value in every group, default 1.0). Region r is material index
   // r; material 0 is the background. -n_regions 0 is the uniform problem
   PetscInt n_regions = 0;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_regions", &n_regions, NULL));
   PetscCheck(n_regions >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, \
      "-n_regions must not be negative, was given %" PetscInt_FMT, n_regions);

   std::vector<MaterialInterval1D> intervals(n_regions);
   std::vector<PetscReal> region_density(n_regions, 1.0);
   std::vector<PetscReal> region_source(n_regions, 1.0);
   for (PetscInt r = 1; r <= n_regions; r++) {
      char opt[64];
      PetscReal interval[2];
      PetscInt n_vals = 2;
      PetscBool found = PETSC_FALSE;
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-region_%" PetscInt_FMT "_interval", r));
      PetscCall(PetscOptionsGetRealArray(NULL, NULL, opt, interval, &n_vals, &found));
      PetscCheck(found && n_vals == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, \
         "%s must be given as x0,x1", opt);
      intervals[r - 1] = {interval[0], interval[1], r};

      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-region_%" PetscInt_FMT "_density", r));
      PetscCall(PetscOptionsGetReal(NULL, NULL, opt, &region_density[r - 1], NULL));
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-region_%" PetscInt_FMT "_source", r));
      PetscCall(PetscOptionsGetReal(NULL, NULL, opt, &region_source[r - 1], NULL));
   }
   // The boundary condition on each face: vacuum (prescribed inflow, from the
   // rhs) or reflect. Keep at least one face vacuum: the last group's
   // scattering ratio is exactly 1 here, and an all-reflective group with no
   // absorption is singular
   const char *const bc_names[] = {"vacuum", "reflect"};
   PetscInt bc_left = 0, bc_right = 0;
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_left", bc_names, 2, &bc_left, NULL));
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_right", bc_names, 2, &bc_right, NULL));
   // The incoming flux prescribed on the vacuum faces, every group (it lands
   // on the Dirichlet rows of b and only means something there - reflective
   // rows owe the rhs a zero). -inflow 0 with a painted source region is a
   // cold slab driven by that region alone
   PetscReal inflow = 1.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-inflow", &inflow, NULL));
   // Write the scalar flux of the solution for inspection, one file per group:
   // -flux_vtk flux.vts writes flux_g0.vts, flux_g1.vts, ... The extension
   // picks the format, .vts or .vtr
   char flux_vtk[PETSC_MAX_PATH_LEN];
   PetscBool write_flux = PETSC_FALSE;
   PetscCall(PetscOptionsGetString(NULL, NULL, "-flux_vtk", flux_vtk, sizeof(flux_vtk), &write_flux));

   KSPConvergedReason reason = KSP_CONVERGED_ITERATING;

   // Device memory has to be gone before PetscFinalize takes Kokkos down
   {
      BCSpec bcs;
      if (bc_left == 1) bcs.set(StructuredFD1D::FACE_LEFT, BCType::REFLECT);
      if (bc_right == 1) bcs.set(StructuredFD1D::FACE_RIGHT, BCType::REFLECT);

      PhaseSpace ps;
      PetscCall(ps.create(PETSC_COMM_WORLD, n_cells, n_angles, n_groups));
      SNQuadrature quad;
      PetscCall(quad.create(n_angles));
      StructuredFD1D disc;
      PetscCall(disc.create(PETSC_COMM_WORLD, ps, length, quad, bcs));

      // ~~~~~~~~~~~~~
      // Cross sections, per group and local cell
      //
      // Absorption is the same in every group: sigma_t carries the within-group
      // scatter plus whatever leaves the group by downscatter, so adding
      // transfer moves reaction rate between groups instead of inventing it.
      // The last group has nowhere to scatter down to
      //
      // The group recipe goes into a MaterialSpec - the background at density
      // 1, each region the same recipe scaled by its density - and is expanded
      // onto the cells through the painted material indices. With -n_regions 0
      // this fills exactly what the constant setters used to
      // ~~~~~~~~~~~~~
      MaterialSpec mats;
      PetscCall(mats.create(n_regions + 1, n_groups));
      for (PetscInt m = 0; m <= n_regions; m++) {

         const PetscScalar density = (m == 0) ? 1.0 : (PetscScalar)region_density[m - 1];
         const PetscScalar source = (m == 0) ? 1.0 : (PetscScalar)region_source[m - 1];

         for (PetscInt g = 0; g < n_groups; g++) {

            const PetscBool has_downscatter = (g + 1 < n_groups) ? PETSC_TRUE : PETSC_FALSE;
            const PetscScalar out = has_downscatter ? (PetscScalar)sigma_transfer : 0.0;

            PetscCall(mats.set_sigma_t(m, g, density * ((PetscScalar)max_exponent + out)));
            PetscCall(mats.set_sigma_s(m, g, g, density * (PetscScalar)max_exponent));
            if (has_downscatter) PetscCall(mats.set_sigma_s(m, g, g + 1, density * out));
            PetscCall(mats.set_source(m, g, source));
         }
      }

      PetscIntKokkosView mat_id_d;
      PetscCall(disc.paint_intervals(0, intervals, mat_id_d));
      GroupXSections xs;
      PetscCall(xs.create(ps));
      PetscCall(xs.set_from_materials(mats, mat_id_d));

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free.
      // Built once for group 0 and re-pointed at each group's xsections below
      // ~~~~~~~~~~~~~
      StreamingTerm streaming;
      PetscCall(streaming.create(ps, disc, quad));
      RemovalTerm removal;
      PetscCall(removal.create(ps, disc, xs.sigma_t(0)));
      ScatteringTerm scattering;
      PetscCall(scattering.create(ps, disc, quad, xs.sigma_s(0, 0)));

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

         // Rhs: the incoming flux everywhere first (the Dirichlet rows keep
         // it), the per-material external source over the non-BC rows, and
         // everything downscattered out of the groups already solved. A
         // reflective row's rhs is zero, so the inflow comes off those rows
         // (add_source skips every BC row on its own)
         PetscCall(VecSet(b, (PetscScalar)inflow));
         PetscCall(UboltFillSource(ps, disc.boundary_info(), mats, mat_id_d, g, b));
         PetscCall(UboltZeroReflectRows(disc.boundary_info(), b));
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

      // Only a full sweep leaves every group's flux worth looking at
      if (write_flux && reason > 0) {
         const char *dot = strrchr(flux_vtk, '.');
         const size_t base_len = dot ? (size_t)(dot - flux_vtk) : strlen(flux_vtk);
         for (PetscInt g = 0; g < n_groups; g++) {
            char fname[PETSC_MAX_PATH_LEN];
            PetscCall(PetscSNPrintf(fname, sizeof(fname), "%.*s_g%" PetscInt_FMT "%s", \
               (int)base_len, flux_vtk, g, dot ? dot : ""));
            PetscCall(UboltWriteScalarFluxVTK(ps, disc, quad, psi[g], fname));
         }
      }

      PetscCall(solver.destroy());
      PetscCall(op.destroy());
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
