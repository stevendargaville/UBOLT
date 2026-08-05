// The solve driver: everything physical comes from a JSON problem-definition
// file (see docs/problem_files.md), everything about HOW it is solved stays on
// the command line - PETSc options (-ksp_*, -pc_*) plus the strategy and
// verification knobs below. One driver for every dimension and any number of
// groups: the file picks the backend, and a single-group file IS the
// single-group problem, so there is no separate driver for it
//
// Strategy knobs: -precon_stream (precondition with a streaming-only pmat),
// -matfree_removal (apply the removal term matrix-free, so the assembled matrix
// is streaming only), -precon_ref_shift + -precon_ref_k (put a representative
// removal back onto that streaming pmat, in k reference-shifted copies),
// -precon_dsa (add the DSA diffusion correction to the composite - its inner
// solve takes the -dsa_ prefix), -diag_scale, and the verification ones,
// -check_inf_medium, -check_matfree and the -flux_vtk output override
//
// Group Gauss-Seidel with downscatter only: groups are ordered high energy to
// low, the within-group scatter stays on the lhs (matrix-free) and everything
// scattering out of an already-solved group goes to the rhs. With no upscatter
// the group system is block lower triangular, so one forward sweep is exact and
// there is no outer iteration - that arrives with upscatter. The sparsity is
// preallocated once and every group refills the same matrix through
// TransportOperator::assemble()
//
// -matfree_removal is that last sentence taken away. With the removal applied
// matrix-free the assembled matrix carries STREAMING ONLY, which does not
// depend on the group, so it is assembled ONCE before the sweep and no group
// refills anything: a group is a pointer swap on the two terms plus
// TransportSolver::refresh(), and PCAIR sets up once for the whole sweep. The
// assembled matrix is then already the streaming-only pmat -precon_stream
// builds a second copy of, so this mode IMPLIES that preconditioner and an
// explicit -precon_stream alongside it is accepted and ignored (which is what
// makes a matfree recipe exactly its -precon_stream twin plus this flag - the
// two must agree on iteration count, see ../docs/dev/testing.md).
// -diag_scale is a checked error in this mode: it would scale the streaming
// part of the operator and leave the matrix-free removal and scatter unscaled
//
// -precon_ref_shift is what makes that mode usable when the removal is strong.
// A bare streaming pmat has no removal in it at all, and PCAIR set up on one
// diverges from roughly a mean free path per cell; the flag preconditions each
// group with L + alpha_g * D_ref instead, D_ref being group 0's per-cell
// Sigma_t and alpha_g that group's ratio to it, in k = -precon_ref_k shared
// copies (default: the fewest that keep every group close to the one it uses -
// see RefShiftPmats). It NEEDS -matfree_removal, which is the only mode whose
// pmat is missing its removal, and is a checked error without it. k pmats means
// k PCAIR hierarchies, which is the design's stated cost. It composes with
// -precon_dsa: the correction wants a removal-carrying pmat to attach to, and
// this is what puts one there
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

// -check_matfree: is the matrix-free removal the same operator as the
// assembled one, and is the composed diagonal the same diagonal?
//
// Both operators are built here from the same discretisation and the same
// streaming term, so the ONLY difference between them is which half of
// RemovalTerm is live - which is exactly the thing being claimed equivalent.
// Two measurements, per group, on the same random vectors:
//
//  1. the shells applied to random vectors, max_{r} |y_f - y_a| / max_r |y_a|.
//     NOT expected to be zero: the assembled path sums streaming and removal
//     into one matrix entry and multiplies once, the matrix-free path
//     multiplies twice and adds, so the two differ in the last bits. Rounding,
//     nothing else - a mistake in the apply is many orders larger
//  2. the composed diagonal against MatGetDiagonal of the fully assembled
//     matrix, max_r |d_f - d_a|. This one IS expected to be exactly 0.0: the
//     terms' add_diagonal write the same expressions in the same order as
//     their assemble_add, and the boundary rows get the same 1.0
//
// The assembled operator is refilled per group and the matrix-free one is not
// touched at all - the per-group behaviour, not just the operator at one group
static PetscErrorCode CheckMatfreeEquivalence(MPI_Comm comm, const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad, const OperatorTerm *streaming, \
   const GroupXSections &xs, PetscInt n_groups, PetscInt n_vectors, \
   PetscReal *max_rel_diff, PetscReal *max_diag_diff)
{
   PetscFunctionBeginUser;

   // The pair of removals, one each way, and a scatter apiece so neither
   // operator can be reading the other's scratch
   RemovalTerm removal_a, removal_f;
   PetscCall(removal_a.create(ps, disc, xs.sigma_t(0)));
   PetscCall(removal_f.create(ps, disc, xs.sigma_t(0)));
   removal_f.set_matrix_free(PETSC_TRUE);
   ScatteringTerm scattering_a, scattering_f;
   PetscCall(scattering_a.create(ps, disc, quad, xs.sigma_s(0, 0)));
   PetscCall(scattering_f.create(ps, disc, quad, xs.sigma_s(0, 0)));

   TransportOperator op_a, op_f;
   PetscCall(op_a.create(comm, ps, disc));
   PetscCall(op_a.add_term(streaming));
   PetscCall(op_a.add_term(&removal_a));
   PetscCall(op_a.add_term(&scattering_a));
   PetscCall(op_f.create(comm, ps, disc));
   PetscCall(op_f.add_term(streaming));
   PetscCall(op_f.add_term(&removal_f));
   PetscCall(op_f.add_term(&scattering_f));
   // Streaming only, and streaming does not depend on the group: this is the
   // one and only assembly the matrix-free operator gets
   PetscCall(op_f.assemble());

   Vec x, y_a, y_f, d_a, d_f;
   PetscCall(MatCreateVecs(op_a.assembled_mat(), &x, &y_a));
   PetscCall(VecDuplicate(y_a, &y_f));
   PetscCall(VecDuplicate(y_a, &d_a));
   PetscCall(VecDuplicate(y_a, &d_f));

   PetscRandom rand;
   PetscCall(PetscRandomCreate(comm, &rand));
   PetscCall(PetscRandomSetFromOptions(rand));

   for (PetscInt g = 0; g < n_groups; g++) {

      removal_a.set_sigma_t(xs.sigma_t(g));
      removal_f.set_sigma_t(xs.sigma_t(g));
      scattering_a.set_sigma_s(xs.sigma_s(g, g));
      scattering_f.set_sigma_s(xs.sigma_s(g, g));
      // The default path's per-group refill. The matrix-free one has nothing
      // to refill, which is the point of it
      PetscCall(op_a.assemble());

      for (PetscInt k = 0; k < n_vectors; k++) {

         PetscReal norm_a = 0.0, diff = 0.0;

         PetscCall(VecSetRandom(x, rand));
         PetscCall(MatMult(op_a.mat(), x, y_a));
         PetscCall(MatMult(op_f.mat(), x, y_f));

         PetscCall(VecNorm(y_a, NORM_INFINITY, &norm_a));
         PetscCall(VecAXPY(y_f, -1.0, y_a));
         PetscCall(VecNorm(y_f, NORM_INFINITY, &diff));
         *max_rel_diff = PetscMax(*max_rel_diff, norm_a > 0.0 ? diff / norm_a : diff);
      }

      // The composed diagonal against the assembled one. op_f is the operator
      // that has to compose - op_a's removal is in its matrix
      PetscReal diag_diff = 0.0;
      PetscCall(op_f.diagonal(d_f));
      PetscCall(MatGetDiagonal(op_a.assembled_mat(), d_a));
      PetscCall(VecAXPY(d_f, -1.0, d_a));
      PetscCall(VecNorm(d_f, NORM_INFINITY, &diag_diff));
      *max_diag_diff = PetscMax(*max_diag_diff, diag_diff);
   }

   PetscCall(PetscRandomDestroy(&rand));
   PetscCall(VecDestroy(&x));
   PetscCall(VecDestroy(&y_a));
   PetscCall(VecDestroy(&y_f));
   PetscCall(VecDestroy(&d_a));
   PetscCall(VecDestroy(&d_f));
   PetscCall(op_a.destroy());
   PetscCall(op_f.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

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
   // Do we apply the removal term matrix-free instead of assembling it - see
   // the header comment. Implies the streaming-only pmat, since the assembled
   // matrix IS one, and takes every assembly out of the group sweep
   PetscBool matfree_removal = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-matfree_removal", &matfree_removal, NULL));
   // Do we put a representative removal back onto the streaming-only pmat -
   // see the header comment. -precon_ref_k is how many reference-shifted
   // copies to build; unset (0) asks RefShiftPmats for its default rule
   PetscBool precon_ref_shift = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_ref_shift", &precon_ref_shift, NULL));
   PetscInt precon_ref_k = 0;
   PetscBool have_ref_k = PETSC_FALSE;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-precon_ref_k", &precon_ref_k, &have_ref_k));
   // The shift exists to repair a pmat that is missing its removal, and
   // -matfree_removal is the only mode that produces one. Under any other mode
   // the pmat already carries the removal (the default) or was deliberately
   // asked not to (-precon_stream), so the flag would either do nothing or
   // silently undo the choice
   PetscCheck(!precon_ref_shift || matfree_removal, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
      "-precon_ref_shift shifts the streaming-only pmat that -matfree_removal leaves " \
      "behind: run the two together");
   PetscCheck(!have_ref_k || precon_ref_shift, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
      "-precon_ref_k counts the reference-shifted pmats: it needs -precon_ref_shift");
   PetscCheck(precon_ref_k >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, \
      "-precon_ref_k must be positive (or left unset for the default), was given %" \
      PetscInt_FMT, precon_ref_k);
   // Do we add the DSA diffusion correction to the composite preconditioner
   // (index 2, its inner solve under the -dsa_ prefix - see DSAPrecon). Off by
   // default: nothing preconditions the scattering without it, which is what
   // an optically thick, high scattering ratio problem needs
   PetscBool precon_dsa = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_dsa", &precon_dsa, NULL));
   // Diagonally scale the assembled operator and the rhs
   PetscBool diag_scale = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));
   // Diagonal scaling scales the ASSEMBLED operator and the rhs, and under
   // -matfree_removal the assembled operator is the streaming part alone: the
   // removal and the scatter would go through the shell unscaled and the
   // system solved would not be the scaled one. Inconsistent by construction,
   // so it is an error rather than a caveat
   PetscCheck(!(diag_scale && matfree_removal), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
      "-diag_scale scales the assembled operator, which under -matfree_removal carries " \
      "streaming only - the matrix-free removal and scatter would stay unscaled");
   // Check the solution against the infinite-medium constant (below)
   PetscBool check_inf_medium = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_inf_medium", &check_inf_medium, NULL));
   // Check the matrix-free removal against the assembled one - the operator on
   // random vectors and the composed diagonal against the assembled diagonal,
   // per group (see CheckMatfreeEquivalence above). Independent of
   // -matfree_removal: it builds both operators itself, so it says the same
   // thing about whichever mode this run is solving in
   PetscBool check_matfree = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_matfree", &check_matfree, NULL));
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
   PetscReal matfree_err = 0.0, matfree_diag_err = 0.0;

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

      // The quadrature comes first in each branch: the file names an SN order
      // and how many ordinates that is, is the quadrature's answer - which is
      // what the phase space is sized on
      if (spec.dimension == 1) {
         PetscCall(quad_1d.create(spec.sn_order));
         PetscCall(ps.create(PETSC_COMM_WORLD, spec.n_cells_x, quad_1d.n_angles(), n_groups));
         PetscCall(disc_1d.create(PETSC_COMM_WORLD, ps, spec.length_x, quad_1d, spec.bcs));
         PetscCall(disc_1d.paint_intervals(bg, spec.intervals, mat_id_d));
         PetscCall(streaming_1d.create(ps, disc_1d, quad_1d));
         quad = &quad_1d;
         disc = &disc_1d;
         streaming = &streaming_1d;
      }
      else if (spec.dimension == 2) {
         PetscCall(quad_2d.create(spec.sn_order));
         PetscCall(ps.create(PETSC_COMM_WORLD, spec.n_cells_x * spec.n_cells_y, \
            quad_2d.n_angles(), n_groups));
         PetscCall(disc_2d.create(PETSC_COMM_WORLD, ps, spec.n_cells_x, spec.n_cells_y, \
            spec.length_x, spec.length_y, quad_2d, spec.bcs));
         PetscCall(disc_2d.paint_boxes(bg, spec.boxes, mat_id_d));
         PetscCall(streaming_2d.create(ps, disc_2d, quad_2d));
         quad = &quad_2d;
         disc = &disc_2d;
         streaming = &streaming_2d;
      }
      else if (spec.dimension == 3) {
         PetscCall(quad_3d.create(spec.sn_order));
         PetscCall(ps.create(PETSC_COMM_WORLD, \
            spec.n_cells_x * spec.n_cells_y * spec.n_cells_z, quad_3d.n_angles(), n_groups));
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
      // The DSA diffusion correction, if it was asked for. Per-dimension, like
      // the streaming term and for the same reason - it is built from the
      // geometry - so it goes here while the concrete backend is still in
      // scope
      //
      // CAREFUL with -diag_scale: the two are mechanically compatible, but
      // scaling the assembled operator breaks the R A P consistency the
      // /sum_weights scaling relies on, so the correction is no longer the
      // right one for the system being solved. Not special-cased, just said
      // ~~~~~~~~~~~~~
      DSAPrecon dsa;
      if (precon_dsa) {
         if (spec.dimension == 1) PetscCall(dsa.create(PETSC_COMM_WORLD, ps, disc_1d, \
            *quad, spec.bcs));
         else if (spec.dimension == 2) PetscCall(dsa.create(PETSC_COMM_WORLD, ps, disc_2d, \
            *quad, spec.bcs));
         else if (spec.dimension == 3) PetscCall(dsa.create(PETSC_COMM_WORLD, ps, disc_3d, \
            *quad, spec.bcs));
         else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, \
            "-precon_dsa has no backend for dimension %" PetscInt_FMT, spec.dimension);
      }

      // ~~~~~~~~~~~~~
      // Is the matrix-free removal the same operator as the assembled one?
      // Both are built inside the check, so this says nothing about which mode
      // the solve below runs in - see CheckMatfreeEquivalence
      // ~~~~~~~~~~~~~
      if (check_matfree) PetscCall(CheckMatfreeEquivalence(PETSC_COMM_WORLD, ps, *disc, *quad, \
         streaming, xs, n_groups, 3, &matfree_err, &matfree_diag_err));

      // ~~~~~~~~~~~~~
      // The operator: streaming + removal assembled, scattering matrix-free.
      // Built once for group 0 and re-pointed at each group's xsections below
      //
      // Under -matfree_removal the removal moves to the other half, so the
      // assembled matrix is streaming only. The mode goes on before the first
      // assemble() (the operator partitions its terms there) - add_term does
      // not care which side of it this call lands on
      // ~~~~~~~~~~~~~
      RemovalTerm removal;
      PetscCall(removal.create(ps, *disc, xs.sigma_t(0)));
      removal.set_matrix_free(matfree_removal);
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
      //
      // Not under -matfree_removal: the assembled matrix is already exactly
      // this matrix, so a second copy of it would be the same values twice.
      // An explicit -precon_stream is therefore redundant rather than wrong,
      // and is quietly ignored - see the header comment
      Mat streaming_mat = NULL;
      if (precon_stream && !matfree_removal)
      {
         const OperatorTerm *streaming_only[] = {streaming};
         PetscCall(op.assemble_subset(1, streaming_only, &streaming_mat));
      }

      // Streaming only, and streaming does not depend on the group: assemble
      // it ONCE, here, and the sweep below assembles nothing at all
      if (matfree_removal) PetscCall(op.assemble());

      // The reference-shifted pmats: k copies of that streaming matrix, each
      // with a representative removal on its interior diagonal, and a map from
      // group to which one. Built here because it reads the assembled matrix,
      // so it has to come after the assemble above
      RefShiftPmats ref_shift;
      if (precon_ref_shift) {
         PetscCall(ref_shift.create(PETSC_COMM_WORLD, ps, *disc, xs, op.assembled_mat(), \
            precon_ref_k));
         PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, \
            "reference-shifted pmat: %" PetscInt_FMT " hierarchies over %" PetscInt_FMT \
            " groups, worst mismatch %g\n", ref_shift.n_bins(), n_groups, \
            (double)ref_shift.worst_mismatch()));
      }

      // ~~~~~~~~~~~~~
      // Sweep the groups
      //
      // One solver per pmat: without -precon_ref_shift that is one for the
      // whole sweep, with it one per bin. A solver owns its KSP, its composite
      // PC and the PCAIR hierarchy underneath it, so this vector IS the memory
      // the mode costs - and each one is created lazily, the first time a group
      // that uses it comes round
      // ~~~~~~~~~~~~~
      std::vector<TransportSolver> solvers(precon_ref_shift ? ref_shift.n_bins() : 1);
      std::vector<PetscBool> solver_created(solvers.size(), PETSC_FALSE);
      for (PetscInt g = 0; g < n_groups; g++) {

         // Point the terms at this group and refill the values. No
         // re-preallocation - the sparsity is the discretisation's, not the
         // group's. Under -matfree_removal there is nothing to refill: both
         // re-pointed terms are matrix-free and read their xsections straight
         // through, so the pointer swaps above are the whole per-group update
         removal.set_sigma_t(xs.sigma_t(g));
         scattering.set_sigma_s(xs.sigma_s(g, g));
         if (!matfree_removal) PetscCall(op.assemble());
         // The diffusion operator is built from the same two xsections, so it
         // is refilled here rather than in TransportSolver::refresh() - the
         // solver has no idea which group is being solved
         if (precon_dsa) PetscCall(dsa.set_group(xs.sigma_t(g), xs.sigma_s(g, g)));

         // Rhs: zero, then the per-face inflow onto the Dirichlet rows (the
         // per-row value the backend computed at create - the winning face's
         // inflow, windowed), the per-material external source over the non-BC
         // rows, and everything downscattered out of the groups already
         // solved. A reflective row's rhs is zero - UboltZeroReflectRows
         // enforces that contract regardless of what the fills above wrote
         // (add_source skips every BC row on its own)
         PetscCall(VecSet(b, 0.0));
         PetscCall(UboltFillInflow(disc->boundary_info(), b));
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

         // The shell only exists once something has been assembled. Pmat is
         // this group's reference-shifted copy if there are any, the
         // streaming-only copy if one was built, and otherwise the assembled
         // matrix - which under -matfree_removal is streaming only in its own
         // right
         const PetscInt bin = precon_ref_shift ? ref_shift.bin_of_group(g) : 0;
         TransportSolver &solver = solvers[bin];
         if (!solver_created[bin]) {
            Mat pmat = precon_ref_shift ? ref_shift.pmat(bin) : \
               (streaming_mat ? streaming_mat : op.assembled_mat());
            PetscCall(solver.create(PETSC_COMM_WORLD, op, pmat, precon_dsa ? &dsa : nullptr));
            solver_created[bin] = PETSC_TRUE;
         }
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

      for (size_t s = 0; s < solvers.size(); s++) PetscCall(solvers[s].destroy());
      // After the solvers: the shell at composite index 2 pointed at it
      PetscCall(dsa.destroy());
      // Same order for the pmats the solvers were built on
      PetscCall(ref_shift.destroy());
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

   // Same for the matrix-free equivalence: the matvec is a rounding difference
   // against 1e-13, the diagonal is an identity and its tolerance is 0.0
   const PetscBool matfree_ok = (PetscBool)(!check_matfree || \
      (matfree_err < 1e-13 && matfree_diag_err == 0.0));
   if (check_matfree) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, \
      "matrix-free removal check: max relative matvec difference %g against tolerance 1e-13, " \
      "max diagonal difference %g against 0 - %s\n", \
      (double)matfree_err, (double)matfree_diag_err, matfree_ok ? "pass" : "FAIL"));

   PetscCall(PetscFinalize());
   return (reason > 0 && inf_medium_ok && matfree_ok) ? 0 : 1;
}
