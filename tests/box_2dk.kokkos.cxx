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
   // that is a scattering ratio of exactly 1, i.e. no absorption at all. Set it
   // lower to get some. Nothing in the preconditioner does anything about the
   // scattering, so the ratio is the knob to turn when asking whether that
   // matters (see the research note in ../TODO.md, where it once looked like it
   // did and the real answer was a bug in the operator)
   PetscReal sigma_scatter = (PetscReal)max_exponent;
   PetscCall(PetscOptionsGetReal(NULL, NULL, "-sigma_scatter", &sigma_scatter, NULL));
   // Heterogeneous regions: -n_regions N, then per region r (1-based) an
   // axis-aligned box -region_<r>_box x0,x1,y0,y1 painted over the background
   // in order, later regions winning, plus optional -region_<r>_sigma_t,
   // -region_<r>_sigma_s and -region_<r>_source (defaulting to the background
   // values above and a unit source). Region r is material index r; material 0
   // is the background. The default -n_regions 0 is the uniform problem
   PetscInt n_regions = 0;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_regions", &n_regions, NULL));
   PetscCheck(n_regions >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, \
      "-n_regions must not be negative, was given %" PetscInt_FMT, n_regions);

   MaterialSpec mats;
   PetscCall(mats.create(n_regions + 1, 1));
   PetscCall(mats.set_sigma_t(0, 0, (PetscScalar)max_exponent));
   PetscCall(mats.set_sigma_s(0, 0, 0, (PetscScalar)sigma_scatter));
   PetscCall(mats.set_source(0, 0, 1.0));

   std::vector<MaterialBox2D> boxes(n_regions);
   for (PetscInt r = 1; r <= n_regions; r++) {
      char opt[64];
      PetscReal box[4];
      PetscInt n_vals = 4;
      PetscBool found = PETSC_FALSE;
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-region_%" PetscInt_FMT "_box", r));
      PetscCall(PetscOptionsGetRealArray(NULL, NULL, opt, box, &n_vals, &found));
      PetscCheck(found && n_vals == 4, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, \
         "%s must be given as x0,x1,y0,y1", opt);
      boxes[r - 1] = {box[0], box[1], box[2], box[3], r};

      PetscReal sigma_t_r = (PetscReal)max_exponent, sigma_s_r = sigma_scatter, source_r = 1.0;
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-region_%" PetscInt_FMT "_sigma_t", r));
      PetscCall(PetscOptionsGetReal(NULL, NULL, opt, &sigma_t_r, NULL));
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-region_%" PetscInt_FMT "_sigma_s", r));
      PetscCall(PetscOptionsGetReal(NULL, NULL, opt, &sigma_s_r, NULL));
      PetscCall(PetscSNPrintf(opt, sizeof(opt), "-region_%" PetscInt_FMT "_source", r));
      PetscCall(PetscOptionsGetReal(NULL, NULL, opt, &source_r, NULL));
      PetscCall(mats.set_sigma_t(r, 0, (PetscScalar)sigma_t_r));
      PetscCall(mats.set_sigma_s(r, 0, 0, (PetscScalar)sigma_s_r));
      PetscCall(mats.set_source(r, 0, (PetscScalar)source_r));
   }
   // The boundary condition on each face: vacuum (prescribed inflow, from the
   // rhs) or reflect
   const char *const bc_names[] = {"vacuum", "reflect"};
   PetscInt bc_left = 0, bc_right = 0, bc_bottom = 0, bc_top = 0;
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_left", bc_names, 2, &bc_left, NULL));
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_right", bc_names, 2, &bc_right, NULL));
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_bottom", bc_names, 2, &bc_bottom, NULL));
   PetscCall(PetscOptionsGetEList(NULL, NULL, "-bc_top", bc_names, 2, &bc_top, NULL));
   // Check the solve against the infinite-medium solution: with every face
   // reflective, a uniform unit source and uniform xsections the exact
   // discrete solution is psi = 1 / (sigma_t - sigma_s) everywhere
   PetscBool check_inf_medium = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_inf_medium", &check_inf_medium, NULL));
   PetscCheck(!check_inf_medium || (bc_left == 1 && bc_right == 1 && bc_bottom == 1 && bc_top == 1), \
      PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs all four -bc_* reflect");
   PetscCheck(!check_inf_medium || sigma_scatter < (PetscReal)max_exponent, PETSC_COMM_WORLD, \
      PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs absorption: -sigma_scatter below -max_exponent");
   PetscCheck(!check_inf_medium || n_regions == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
      "-check_inf_medium needs uniform xsections and source: -n_regions 0");
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
      if (bc_left == 1) bcs.set(StructuredFD2D::FACE_LEFT, BCType::REFLECT);
      if (bc_right == 1) bcs.set(StructuredFD2D::FACE_RIGHT, BCType::REFLECT);
      if (bc_bottom == 1) bcs.set(StructuredFD2D::FACE_BOTTOM, BCType::REFLECT);
      if (bc_top == 1) bcs.set(StructuredFD2D::FACE_TOP, BCType::REFLECT);

      PhaseSpace ps;
      PetscCall(ps.create(PETSC_COMM_WORLD, n_cells_x * n_cells_y, n_angles));
      SNQuadrature2D quad;
      PetscCall(quad.create(n_angles));
      StructuredFD2D disc;
      PetscCall(disc.create(PETSC_COMM_WORLD, ps, n_cells_x, n_cells_y, length_x, length_y, quad, bcs));

      // ~~~~~~~~~~~~~
      // Cross sections, one per local cell: paint the material indices onto
      // the cells (all background when there are no regions) and expand the
      // material table through them
      // ~~~~~~~~~~~~~
      PetscIntKokkosView mat_id_d;
      PetscCall(disc.paint_boxes(0, boxes, mat_id_d));
      GroupXSections xs;
      PetscCall(xs.create(ps));
      PetscCall(xs.set_from_materials(mats, mat_id_d));

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free
      // ~~~~~~~~~~~~~
      StreamingTerm2D streaming;
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
      PetscCall(op.assemble());

      // ~~~~~~~~~~~~~
      // RHS: external source, and the incoming flux on the Dirichlet rows
      // ~~~~~~~~~~~~~
      Vec x, b;
      PetscCall(MatCreateVecs(op.assembled_mat(), &x, &b));
      // The inflow value everywhere first - the Dirichlet rows keep it - then
      // the per-material source over the non-BC rows
      PetscCall(VecSet(b, 1.0));
      PetscCall(UboltFillSource(ps, disc.boundary_info(), mats, mat_id_d, 0, b));
      // A reflective row's rhs is zero - psi_r = psi_partner - so the inflow
      // has to come off those rows again
      PetscCall(UboltZeroReflectRows(disc.boundary_info(), b));

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
         const PetscScalar expected = 1.0 / ((PetscScalar)max_exponent - (PetscScalar)sigma_scatter);
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
   }

   // Did we converge - this is the pass/fail of the test. Diagnostics go to
   // stderr so they don't pollute the -ksp_monitor output
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
