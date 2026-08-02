// The solve driver: everything physical comes from a JSON problem-definition
// file (see docs/problem_files.md), everything about HOW it is solved stays on
// the command line - PETSc options (-ksp_*, -pc_*) plus the strategy and
// verification knobs below. One driver for every dimension and any number of
// groups: the file picks the backend, and a single-group file IS the
// single-group problem, so there is no separate driver for it
//
// Group Gauss-Seidel with downscatter only: groups are ordered high energy to
// low, the within-group scatter stays on the lhs (matrix-free) and everything
// scattering out of an already-solved group goes to the rhs. With no upscatter
// the group system is block lower triangular, so one forward sweep is exact and
// there is no outer iteration - that arrives with upscatter. The sparsity is
// preallocated once and every group refills the same matrix through
// TransportOperator::assemble()
//
// Currently run with:
// make build_tests && ./transportk -problem problems/slab_st2.json -ksp_monitor

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include <petscksp.h>
#include "pflare.h"
#include <string>
#include <vector>
#include <cstring>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   PetscFunctionBeginUser;

   PetscCall(PetscInitialize(&argc, &args, (char*)0, NULL));
   // Register the pflare types
   PCRegister_PFLARE();

   // ~~~~~~~~~~~~~
   // Options: the problem file, then the solver-strategy knobs
   // ~~~~~~~~~~~~~
   char problem_path[PETSC_MAX_PATH_LEN];
   PetscBool have_problem = PETSC_FALSE;
   PetscCall(PetscOptionsGetString(NULL, NULL, "-problem", problem_path, sizeof(problem_path), \
      &have_problem));
   PetscCheck(have_problem, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, \
      "no problem file: run with -problem <file.json>");
   // Do we precondition with streaming or streaming/removal
   PetscBool precon_stream = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL));
   // Diagonally scale the assembled operator and the rhs
   PetscBool diag_scale = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));
   // Check the solution against the infinite-medium constant (below)
   PetscBool check_inf_medium = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_inf_medium", &check_inf_medium, NULL));
   // Write the scalar flux of the solution for inspection - overrides the
   // problem file's output.flux_vtk. A single-group problem writes the
   // filename as given, multigroup writes one file per group: -flux_vtk
   // flux.vts writes flux_g0.vts, flux_g1.vts, ... The extension picks the
   // format, .vts or .vtr
   char flux_vtk_cli[PETSC_MAX_PATH_LEN];
   PetscBool have_flux_cli = PETSC_FALSE;
   PetscCall(PetscOptionsGetString(NULL, NULL, "-flux_vtk", flux_vtk_cli, sizeof(flux_vtk_cli), \
      &have_flux_cli));

   KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
   PetscReal inf_medium_err = 0.0;

   // Device memory has to be gone before PetscFinalize takes Kokkos down
   {
      ProblemSpec spec;
      PetscCall(spec.create(PETSC_COMM_WORLD, problem_path));
      const PetscInt n_groups = spec.n_groups;
      const std::string flux_vtk = have_flux_cli ? std::string(flux_vtk_cli) : spec.flux_vtk;

      // The infinite-medium check needs nothing for the streaming term to do
      // and nowhere to leak: every face reflective, one uniform material, and
      // absorption in every group so the constant is finite
      const PetscInt bg = spec.background_material;
      if (check_inf_medium) {
         PetscCheck(spec.n_reflect_faces == 2 * spec.dimension, PETSC_COMM_WORLD, \
            PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs every face reflective");
         PetscCheck(spec.intervals.empty() && spec.boxes.empty() && spec.boxes_3d.empty(), \
            PETSC_COMM_WORLD, \
            PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs uniform xsections and source: no paint");
         for (PetscInt g = 0; g < n_groups; g++) {
            PetscCheck(PetscRealPart(spec.materials.sigma_s_host()[(bg * n_groups + g) * n_groups + g]) \
               < PetscRealPart(spec.materials.sigma_t_host()[bg * n_groups + g]), PETSC_COMM_WORLD, \
               PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs absorption: within-group Sigma_s below " \
               "Sigma_t in every group");
         }
      }

      // ~~~~~~~~~~~~~
      // The per-dimension pieces: the quadrature, the backend that owns the
      // mesh and the decomposition, the painting, and the streaming term that
      // owns the upwind slot convention. Everything after this block works
      // through the bases
      // ~~~~~~~~~~~~~
      PhaseSpace ps;
      SNQuadrature quad_1d;
      SNQuadrature2D quad_2d;
      SNQuadrature3D quad_3d;
      StructuredFD1D disc_1d;
      StructuredFD2D disc_2d;
      StructuredFD3D disc_3d;
      StreamingTerm streaming_1d;
      StreamingTerm2D streaming_2d;
      StreamingTerm3D streaming_3d;
      const AngularQuadrature *quad = NULL;
      Discretisation *disc = NULL;
      OperatorTerm *streaming = NULL;
      PetscIntKokkosView mat_id_d;

      if (spec.dimension == 1) {
         PetscCall(ps.create(PETSC_COMM_WORLD, spec.n_cells_x, spec.n_angles, n_groups));
         PetscCall(quad_1d.create(spec.n_angles));
         PetscCall(disc_1d.create(PETSC_COMM_WORLD, ps, spec.length_x, quad_1d, spec.bcs));
         PetscCall(disc_1d.paint_intervals(bg, spec.intervals, mat_id_d));
         PetscCall(streaming_1d.create(ps, disc_1d, quad_1d));
         quad = &quad_1d;
         disc = &disc_1d;
         streaming = &streaming_1d;
      }
      else if (spec.dimension == 2) {
         PetscCall(ps.create(PETSC_COMM_WORLD, spec.n_cells_x * spec.n_cells_y, spec.n_angles, \
            n_groups));
         PetscCall(quad_2d.create(spec.n_angles));
         PetscCall(disc_2d.create(PETSC_COMM_WORLD, ps, spec.n_cells_x, spec.n_cells_y, \
            spec.length_x, spec.length_y, quad_2d, spec.bcs));
         PetscCall(disc_2d.paint_boxes(bg, spec.boxes, mat_id_d));
         PetscCall(streaming_2d.create(ps, disc_2d, quad_2d));
         quad = &quad_2d;
         disc = &disc_2d;
         streaming = &streaming_2d;
      }
      else if (spec.dimension == 3) {
         PetscCall(ps.create(PETSC_COMM_WORLD, \
            spec.n_cells_x * spec.n_cells_y * spec.n_cells_z, spec.n_angles, n_groups));
         PetscCall(quad_3d.create(spec.n_angles));
         PetscCall(disc_3d.create(PETSC_COMM_WORLD, ps, spec.n_cells_x, spec.n_cells_y, \
            spec.n_cells_z, spec.length_x, spec.length_y, spec.length_z, quad_3d, spec.bcs));
         PetscCall(disc_3d.paint_boxes(bg, spec.boxes_3d, mat_id_d));
         PetscCall(streaming_3d.create(ps, disc_3d, quad_3d));
         quad = &quad_3d;
         disc = &disc_3d;
         streaming = &streaming_3d;
      }
      else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "no backend for dimension %" PetscInt_FMT, \
         spec.dimension);

      // ~~~~~~~~~~~~~
      // Cross sections per group and local cell: the file's material table
      // expanded through the painted indices. The terms never learn that
      // materials exist
      // ~~~~~~~~~~~~~
      GroupXSections xs;
      PetscCall(xs.create(ps));
      PetscCall(xs.set_from_materials(spec.materials, mat_id_d));

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free.
      // Built once for group 0 and re-pointed at each group's xsections below
      // ~~~~~~~~~~~~~
      RemovalTerm removal;
      PetscCall(removal.create(ps, *disc, xs.sigma_t(0)));
      ScatteringTerm scattering;
      PetscCall(scattering.create(ps, *disc, *quad, xs.sigma_s(0, 0)));

      TransportOperator op;
      PetscCall(op.create(PETSC_COMM_WORLD, ps, *disc));
      PetscCall(op.add_term(streaming));
      PetscCall(op.add_term(&removal));
      PetscCall(op.add_term(&scattering));

      GroupTransfer transfer;
      PetscCall(transfer.create(ps, *quad, xs, disc->boundary_info()));

      // ~~~~~~~~~~~~~
      // One angular flux Vec per group, plus a single rhs we rebuild per group
      // ~~~~~~~~~~~~~
      Vec b, diag_vec = NULL;
      std::vector<Vec> psi(n_groups, NULL);
      PetscCall(MatCreateVecs(op.assembled_mat(), &psi[0], &b));
      for (PetscInt g = 1; g < n_groups; g++) PetscCall(VecDuplicate(psi[0], &psi[g]));
      if (diag_scale) PetscCall(VecDuplicate(b, &diag_vec));

      // The streaming-only pmat does not depend on the group, so it is built
      // once here and PCAIR keeps its setup across the whole sweep. It also
      // stays UNSCALED under -diag_scale, matching what the removal shell
      // preconditions - only the assembled operator and the rhs are scaled
      Mat streaming_mat = NULL;
      if (precon_stream)
      {
         const OperatorTerm *streaming_only[] = {streaming};
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

         // Rhs: the incoming flux everywhere first (the Dirichlet rows keep
         // it), the per-material external source over the non-BC rows, and
         // everything downscattered out of the groups already solved. A
         // reflective row's rhs is zero, so the inflow comes off those rows
         // (add_source skips every BC row on its own)
         PetscCall(VecSet(b, (PetscScalar)spec.inflow));
         PetscCall(UboltFillSource(ps, disc->boundary_info(), *quad, spec.materials, mat_id_d, g, b));
         PetscCall(UboltZeroReflectRows(disc->boundary_info(), b));
         for (PetscInt g_from = 0; g_from < g; g_from++) {
            PetscCall(transfer.add_source(g_from, g, b));
         }

         // Diagonally scale this group's assembled operator and rhs - AFTER
         // the rhs is complete, and BEFORE the solver sees the matrix, so the
         // removal shell caches the scaled diagonal (create and refresh both
         // read it off the assembled matrix)
         if (diag_scale) {
            PetscCall(MatGetDiagonal(op.assembled_mat(), diag_vec));
            PetscCall(VecReciprocal(diag_vec));
            PetscCall(MatDiagonalScale(op.assembled_mat(), diag_vec, PETSC_NULLPTR));
            PetscCall(VecPointwiseMult(b, diag_vec, b));
         }

         // The shell only exists once something has been assembled
         if (g == 0) PetscCall(solver.create(PETSC_COMM_WORLD, op, \
            precon_stream ? streaming_mat : op.assembled_mat()));
         // sigma_t changed under the removal preconditioner
         else PetscCall(solver.refresh());

         PetscCall(solver.solve(b, psi[g]));
         reason = solver.converged_reason();
         if (reason <= 0) break;

         // This group's scalar flux is fixed now, and every group below it
         // scatters from it. Integrate once here rather than once per target
         PetscCall(transfer.set_scalar_flux(g, psi[g]));
      }

      // The infinite-medium check: uniform source, uniform xsections and no
      // boundary to leak through leave nothing for the streaming term to do,
      // so the exact discrete solution is constant in every cell and angle -
      // to solver tolerance, not discretisation error. Per group, by forward
      // substitution down the sweep:
      //   psi_g = (Source_g / sum_weights + sum_{g'<g} Sigma_s[g'][g] psi_g')
      //           / (Sigma_t[g] - Sigma_s[g][g])
      // which is 1 / (sigma_t - sigma_s) for the single-group Source 2.0
      // slab files (sum_weights is 2 in 1D) and
      // 1 / (sum_weights (sigma_t - sigma_s)) for the Source 1.0 box ones
      if (check_inf_medium && reason > 0) {
         std::vector<PetscScalar> expected(n_groups, 0.0);
         for (PetscInt g = 0; g < n_groups; g++) {
            PetscScalar coupled = spec.materials.source_host()[bg * n_groups + g] / quad->sum_weights();
            for (PetscInt g_from = 0; g_from < g; g_from++) {
               coupled += spec.materials.sigma_s_host()[(bg * n_groups + g_from) * n_groups + g] \
                  * expected[g_from];
            }
            expected[g] = coupled / (spec.materials.sigma_t_host()[bg * n_groups + g] \
               - spec.materials.sigma_s_host()[(bg * n_groups + g) * n_groups + g]);

            PetscReal err_g = 0.0;
            PetscCall(VecShift(psi[g], -expected[g]));
            PetscCall(VecNorm(psi[g], NORM_INFINITY, &err_g));
            PetscCall(VecShift(psi[g], expected[g]));
            inf_medium_err = PetscMax(inf_medium_err, err_g);
         }
      }

      // Only a full sweep leaves every group's flux worth looking at. Each file
      // carries that group's scalar flux plus the two group-dependent inputs
      // that produced it - sigma_t and the external source - so the file says
      // what was solved without going back to the problem definition. A
      // single-group problem writes the filename as given - the retired
      // single-group drivers' behaviour - multigroup suffixes _g<g>
      if (!flux_vtk.empty() && reason > 0) {
         const char *dot = strrchr(flux_vtk.c_str(), '.');
         const size_t base_len = dot ? (size_t)(dot - flux_vtk.c_str()) : flux_vtk.size();
         // The source is expanded onto the cells for output only, so one buffer
         // refilled per group rather than a table kept over the sweep
         PetscScalarKokkosView cell_source_d("cell_source_d", ps.local_cells);

         for (PetscInt g = 0; g < n_groups; g++) {
            char fname[PETSC_MAX_PATH_LEN];
            if (n_groups == 1) PetscCall(PetscStrncpy(fname, flux_vtk.c_str(), sizeof(fname)));
            else PetscCall(PetscSNPrintf(fname, sizeof(fname), "%.*s_g%" PetscInt_FMT "%s", \
               (int)base_len, flux_vtk.c_str(), g, dot ? dot : ""));

            PetscCall(UboltFillCellSource(ps, spec.materials, mat_id_d, g, cell_source_d));
            const UboltCellField extra[] = {{"sigma_t", xs.sigma_t(g)}, {"source", cell_source_d}};
            PetscCall(UboltWriteScalarFluxVTK(ps, *disc, *quad, psi[g], \
               (PetscInt)(sizeof(extra) / sizeof(extra[0])), extra, fname));
         }
      }

      PetscCall(solver.destroy());
      PetscCall(op.destroy());
      PetscCall(disc->destroy());
      PetscCall(MatDestroy(&streaming_mat));
      PetscCall(VecDestroy(&b));
      PetscCall(VecDestroy(&diag_vec));
      for (PetscInt g = 0; g < n_groups; g++) PetscCall(VecDestroy(&psi[g]));
   }

   // Did we converge - this is the pass/fail of the test. Diagnostics go to
   // stderr so they don't pollute the -ksp_monitor output the baselines in
   // tests/baselines are captured from
   if (reason <= 0) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "KSP did not converge: %s\n", KSPConvergedReasons[reason]));

   // Print what the infinite-medium check measured, so a run drifting towards
   // failure is visible before it fails
   const PetscBool inf_medium_ok = (PetscBool)(!check_inf_medium || inf_medium_err < 1e-9);
   if (check_inf_medium) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, \
      "infinite-medium check: max error %g against tolerance 1e-9 - %s\n", \
      (double)inf_medium_err, inf_medium_ok ? "pass" : "FAIL"));

   PetscCall(PetscFinalize());
   return (reason > 0 && inf_medium_ok) ? 0 : 1;
}
