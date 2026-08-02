// 1D slab single-group SN driver
//
// Currently run with:
// make build_tests && ./slab_1dk -ksp_monitor -sub_1_pc_air_print_stats_timings -ksp_pc_side right -precon_stream -sigma_t 2

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
   PetscReal sigma_t = 1.0;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-sigma_t", &sigma_t, NULL));
   // Scattering xsection, defaulting to the total one as before this option
   // existed - that is a scattering ratio of exactly 1, i.e. no absorption at
   // all. Set it lower to get some; an all-reflective slab NEEDS some, or the
   // operator has the constants in its kernel
   PetscReal sigma_scatter = sigma_t;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-sigma_scatter", &sigma_scatter, NULL));
   // The boundary condition on each face: vacuum (prescribed inflow, from the
   // rhs) or reflect
   const char *const bc_names[] = {"vacuum", "reflect"};
   PetscInt bc_left = 0, bc_right = 0;
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_left", bc_names, 2, &bc_left, NULL));
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_right", bc_names, 2, &bc_right, NULL));
   // Check the solve against the infinite-medium solution: with every face
   // reflective, a uniform unit source and uniform xsections the exact
   // discrete solution is psi = 1 / (sigma_t - sigma_s) everywhere
   PetscBool check_inf_medium = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_inf_medium", &check_inf_medium, NULL));
   PetscCheck(!check_inf_medium || (bc_left == 1 && bc_right == 1), PETSC_COMM_WORLD, \
      PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs -bc_left reflect -bc_right reflect");
   PetscCheck(!check_inf_medium || sigma_scatter < sigma_t, PETSC_COMM_WORLD, \
      PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs absorption: -sigma_scatter below -sigma_t");
   // Write the scalar flux of the solution for inspection, e.g.
   // -flux_vtk flux.vts - the extension picks the format, .vts or .vtr
   char flux_vtk[PETSC_MAX_PATH_LEN];
   PetscBool write_flux = PETSC_FALSE;
   PetscCall(PetscOptionsGetString(NULL, NULL, "-flux_vtk", flux_vtk, sizeof(flux_vtk), &write_flux));

   KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
   PetscReal inf_medium_err = 0.0;

   // Device memory has to be gone before PetscFinalize takes Kokkos down
   {
      BCSpec bcs;
      if (bc_left == 1) bcs.set(StructuredFD1D::FACE_LEFT, BCType::REFLECT);
      if (bc_right == 1) bcs.set(StructuredFD1D::FACE_RIGHT, BCType::REFLECT);

      PhaseSpace ps;
      PetscCall(ps.create(PETSC_COMM_WORLD, n_cells, n_angles));
      SNQuadrature quad;
      PetscCall(quad.create(n_angles));
      StructuredFD1D disc;
      PetscCall(disc.create(PETSC_COMM_WORLD, ps, length, quad, bcs));

      // ~~~~~~~~~~~~~
      // Cross sections, one per local cell
      // ~~~~~~~~~~~~~
      PetscScalarKokkosView sigma_t_d("sigma_t_d", ps.local_cells);
      PetscScalarKokkosView sigma_s_d("sigma_s_d", ps.local_cells);
      Kokkos::deep_copy(sigma_t_d, (PetscScalar)sigma_t);
      Kokkos::deep_copy(sigma_s_d, (PetscScalar)sigma_scatter);

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free
      // ~~~~~~~~~~~~~
      StreamingTerm streaming;
      PetscCall(streaming.create(ps, disc, quad));
      RemovalTerm removal;
      PetscCall(removal.create(ps, disc, sigma_t_d));
      ScatteringTerm scattering;
      PetscCall(scattering.create(ps, disc, quad, sigma_s_d));

      TransportOperator op;
      PetscCall(op.create(PETSC_COMM_WORLD, ps, disc));
      PetscCall(op.add_term(&streaming));
      PetscCall(op.add_term(&removal));
      PetscCall(op.add_term(&scattering));
      PetscCall(op.assemble());

      // ~~~~~~~~~~~~~
      // RHS: external source, and the incoming flux on the Dirichlet rows
      // ~~~~~~~~~~~~~
      Vec x, b, diag_vec = NULL;
      PetscCall(MatCreateVecs(op.assembled_mat(), &x, &b));
      PetscCall(VecSet(b, 1.0));
      // A reflective row's rhs is zero - psi_r = psi_partner - so the source
      // has to come off those rows again, before any diagonal scaling
      PetscCall(UboltZeroReflectRows(disc.boundary_info(), b));

      // Diagonally scale the streaming/removal operator
      if (diag_scale) {

         PetscCall(VecDuplicate(x, &diag_vec));
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

      // The infinite-medium check: uniform source, uniform xsections and no
      // boundary to leak through leave nothing for the streaming term to do,
      // so the exact discrete solution is the constant 1 / (sigma_t - sigma_s)
      // in every cell and angle - to solver tolerance, not discretisation error
      if (check_inf_medium) {
         const PetscScalar expected = 1.0 / ((PetscScalar)sigma_t - (PetscScalar)sigma_scatter);
         PetscCall(VecShift(x, -expected));
         PetscCall(VecNorm(x, NORM_INFINITY, &inf_medium_err));
         PetscCall(VecShift(x, expected));
      }

      if (write_flux) PetscCall(UboltWriteScalarFluxVTK(ps, disc, quad, x, flux_vtk));

      PetscCall(solver.destroy());
      PetscCall(op.destroy());
      PetscCall(disc.destroy());
      PetscCall(MatDestroy(&streaming_mat));
      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&b));
      PetscCall(VecDestroy(&diag_vec));
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
