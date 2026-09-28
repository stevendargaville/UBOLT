// The solve driver, and the example of how the library fits together.
// Everything physical comes from a JSON problem file (-problem, see
// docs/problem_files.md); everything about HOW it is solved stays on the command
// line, so one file can be solved many ways. One driver for every backend,
// dimension and group count.
//
// main, top to bottom: the options (each knob's rationale is at its parse
// site), the problem file, BuildBackend (the one per-backend switch: the
// quadrature, the discretisation that owns the mesh and the decomposition, the
// painting, the streaming term), the per-group xsections, the optional DSA,
// the operator and its terms, the group Gauss-Seidel sweep, the checks, the
// flux output.
//
// Knobs: -precon_stream, -matfree_removal, -precon_ref_shift / -precon_ref_k,
// -precon_dsa (+ the -dsa_ family, see DSAPrecon), -precon_block_scale,
// -diag_scale, -supg_zeta, and the verification ones, -check_inf_medium,
// -check_matfree, -check_ref_shift, plus the -flux_vtk output override.
//
// Currently run with:
// make build_tests && ./transportk -problem problems/slab_st2.json -ksp_monitor

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include "petsc_kokkos.hpp"
#include <petscksp.h>
#include "pflare.h"
#include <string>
#include <vector>
#include <cstring>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// -check_inf_medium's error: max over the rows of |psi - the infinite-medium
// solution|, which is the constant on every row of a one-dof-per-cell backend
// and, with a modal basis (DG1), the constant on basis 0 and zero on the rest.
// A host loop - it runs once, after the solve
static PetscErrorCode InfMediumError(const PhaseSpace &ps, Vec psi, PetscScalar expected, PetscReal *err)
{
   const PetscScalar *psi_a = nullptr;
   PetscInt local_rows = 0;
   PetscReal local_err = 0.0;

   PetscFunctionBeginUser;

   PetscCall(VecGetLocalSize(psi, &local_rows));
   PetscCall(VecGetArrayRead(psi, &psi_a));
   for (PetscInt r = 0; r < local_rows; r++) {
      const PetscInt basis = (r / ps.n_angles) % ps.n_basis;
      const PetscScalar target = (basis == 0) ? expected : (PetscScalar)0.0;
      local_err = PetscMax(local_err, PetscAbsScalar(psi_a[r] - target));
   }
   PetscCall(VecRestoreArrayRead(psi, &psi_a));
   PetscCallMPI(MPIU_Allreduce(&local_err, err, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)psi)));

   PetscFunctionReturn(PETSC_SUCCESS);
}

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
   PetscCall(op_a.create(disc));
   PetscCall(op_a.add_term(streaming));
   PetscCall(op_a.add_term(&removal_a));
   PetscCall(op_a.add_term(&scattering_a));
   PetscCall(op_f.create(disc));
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

// -check_ref_shift: is every group's reference-shifted pmat its FULL operator?
//
// Per group, the pmat of its bin against streaming + removal assembled for
// that group into a matrix of its own, max_{ij} |P - A| / max_c Sigma_t(g).
// That is the exact-coverage identity (see RefShiftPmats) at the matrix level,
// and it holds exactly when two things do: every group has a bin of its own
// alpha (the driver's "worst mismatch 1"), and the materials are
// density-scaled copies of one another OUTSIDE THE VOIDS. Voids included -
// which is what makes it the check of the void handling: a class reference
// that were nonzero in a group's void, or two supports sharing a bin, would put
// a removal into rows whose operator has none, an O(Sigma_t) difference. A
// problem that is not density-scaled fails it by design (the alphas are then a
// fit), so it is run on the files that are
//
// Rounding only: alpha_g is an exp of a log-mean, so alpha_g * Sigma_t(ref)
// differs from Sigma_t(g) in the last bits
static PetscErrorCode CheckRefShiftExact(const PhaseSpace &ps, const Discretisation &disc, \
   const TransportOperator &op, const OperatorTerm *streaming, const GroupXSections &xs, \
   const RefShiftPmats &ref_shift, PetscInt n_groups, PetscReal *max_rel_diff)
{
   PetscFunctionBeginUser;

   RemovalTerm removal;
   PetscCall(removal.create(ps, disc, xs.sigma_t(0)));
   const OperatorTerm *full_terms[] = {streaming, &removal};

   *max_rel_diff = 0.0;
   for (PetscInt g = 0; g < n_groups; g++) {

      Mat full;
      PetscReal diff = 0.0, sigma_max = 0.0;

      removal.set_sigma_t(xs.sigma_t(g));
      PetscCall(op.assemble_subset(2, full_terms, &full));
      // Same pattern: both come off the discretisation's create_matrix
      PetscCall(MatAXPY(full, -1.0, ref_shift.pmat(ref_shift.bin_of_group(g)), SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(full, NORM_MAX, &diff));
      PetscCall(MatDestroy(&full));

      auto sigma_t_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), xs.sigma_t(g));
      Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), sigma_t_h, xs.sigma_t(g));
      PetscGetKokkosExecutionSpace().fence();
      for (PetscInt c = 0; c < ps.local_cells; c++) {
         sigma_max = PetscMax(sigma_max, PetscAbsScalar(sigma_t_h(c)));
      }
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &sigma_max, 1, MPIU_REAL, MPIU_MAX, \
         PETSC_COMM_WORLD));

      // A streaming-only group's pmat is the streaming matrix itself, so the
      // difference is absolute there and exactly zero if the bin is right
      *max_rel_diff = PetscMax(*max_rel_diff, sigma_max > 0.0 ? diff / sigma_max : diff);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Everything the per-backend switch produces. Only the concrete members the
// problem's backend needs are ever created; the pointers are what the rest of
// main reads, through the bases
struct Backend {
   PhaseSpace ps;
   SNQuadrature quad_1d;
   SNQuadrature2D quad_2d;
   SNQuadrature3D quad_3d;
   StructuredFD1D fd_1d;
   StructuredFD2D fd_2d;
   StructuredFD3D fd_3d;
   UnstructuredDG dg;
   UnstructuredCG cg;
   StreamingTerm1D streaming_1d;
   StreamingTerm2D streaming_2d;
   StreamingTerm3D streaming_3d;
   StreamingTermDG0 streaming_dg0;
   StreamingTermDG1 streaming_dg1;

   const AngularQuadrature *quad = NULL;
   Discretisation *disc = NULL;
   // The group-INDEPENDENT streaming term. NULL on CG-SUPG, whose one
   // assembled term depends on sigma_t (tau does) and is built by main once
   // the xsections exist
   OperatorTerm *streaming = NULL;
   // The painted material index per material entry: local cell, or local
   // element on CG-SUPG
   PetscIntKokkosView mat_id_d;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The per-backend construction, the one place that names concrete classes.
// The quadrature comes first: the file names an SN order, and how many
// ordinates that is, is the quadrature's answer - which is what the phase space
// is sized on. The discretisation comes next because its create decides the
// decomposition, and everything sized off the phase space comes after it
//
// The plex backends are constructed in two stages: the MESH decides the
// global row count (a simplex box has 2 triangles or 6 tets per box cell, a
// file whatever it has; CG's rows are per vertex), so the phase space is sized
// off the mesh, never off the n_cells product. Their painting layers the
// file's "Cell Sets" under the paint boxes when there are any - a generated box
// has no "Cell Sets" label, so cell_sets on one is the backend's error
static PetscErrorCode BuildBackend(const ProblemSpec &spec, Backend &be)
{
   MPI_Comm comm = PETSC_COMM_WORLD;
   const PetscInt dim = spec.dimension, n_groups = spec.n_groups, bg = spec.background_material;
   const PetscInt *n = spec.mesh.n_cells;
   const PetscReal *len = spec.mesh.lengths;

   PetscFunctionBeginUser;

   if (dim == 1) {
      PetscCall(be.quad_1d.create(spec.sn_order));
      be.quad = &be.quad_1d;
   } else if (dim == 2) {
      PetscCall(be.quad_2d.create(spec.sn_order));
      be.quad = &be.quad_2d;
   } else {
      PetscCall(be.quad_3d.create(spec.sn_order));
      be.quad = &be.quad_3d;
   }
   const PetscInt n_angles = be.quad->n_angles();

   switch (spec.backend) {
   case BackendKind::STRUCTURED_FD:
      if (dim == 1) {
         PetscCall(be.ps.create(comm, n[0], n_angles, n_groups));
         PetscCall(be.fd_1d.create(comm, be.ps, len[0], be.quad_1d, spec.bcs));
         PetscCall(be.fd_1d.paint_intervals(bg, spec.intervals, be.mat_id_d));
         PetscCall(be.streaming_1d.create(be.ps, be.fd_1d, be.quad_1d));
         be.disc = &be.fd_1d;
         be.streaming = &be.streaming_1d;
      } else if (dim == 2) {
         PetscCall(be.ps.create(comm, n[0] * n[1], n_angles, n_groups));
         PetscCall(be.fd_2d.create(comm, be.ps, n[0], n[1], len[0], len[1], be.quad_2d, spec.bcs));
         PetscCall(be.fd_2d.paint_boxes(bg, spec.boxes, be.mat_id_d));
         PetscCall(be.streaming_2d.create(be.ps, be.fd_2d, be.quad_2d));
         be.disc = &be.fd_2d;
         be.streaming = &be.streaming_2d;
      } else {
         PetscCall(be.ps.create(comm, n[0] * n[1] * n[2], n_angles, n_groups));
         PetscCall(be.fd_3d.create(comm, be.ps, n[0], n[1], n[2], len[0], len[1], len[2], be.quad_3d, \
            spec.bcs));
         PetscCall(be.fd_3d.paint_boxes(bg, spec.boxes_3d, be.mat_id_d));
         PetscCall(be.streaming_3d.create(be.ps, be.fd_3d, be.quad_3d));
         be.disc = &be.fd_3d;
         be.streaming = &be.streaming_3d;
      }
      break;

   case BackendKind::UNSTRUCTURED_DG:
   case BackendKind::UNSTRUCTURED_CG: {
      PlexDiscretisation *plex = NULL;
      if (spec.backend == BackendKind::UNSTRUCTURED_DG) {
         PetscCall(be.dg.create_mesh(comm, spec.mesh));
         PetscCall(be.ps.create(comm, be.dg.n_global_cells(), n_angles, n_groups));
         PetscCall(be.dg.create(be.ps, *be.quad, spec.bcs, spec.order));
         plex = &be.dg;
      } else {
         PetscCall(be.cg.create_mesh(comm, spec.mesh));
         PetscCall(be.ps.create(comm, be.cg.n_global_vertices(), n_angles, n_groups));
         PetscCall(be.cg.create(be.ps, *be.quad, spec.bcs, spec.supg_zeta));
         plex = &be.cg;
      }
      if (!spec.cell_sets.empty()) {
         PetscCall(plex->paint_cell_sets(bg, spec.cell_sets, be.mat_id_d));
         if (dim == 2) PetscCall(plex->paint_boxes_over(spec.boxes, be.mat_id_d));
         else PetscCall(plex->paint_boxes_over(spec.boxes_3d, be.mat_id_d));
      }
      else if (dim == 2) PetscCall(plex->paint_boxes(bg, spec.boxes, be.mat_id_d));
      else PetscCall(plex->paint_boxes(bg, spec.boxes_3d, be.mat_id_d));
      be.disc = plex;
      // The DG streaming term is per order, like the structured ones are per
      // dimension: it owns the slot convention
      if (spec.backend == BackendKind::UNSTRUCTURED_DG && spec.order == 1) {
         PetscCall(be.streaming_dg1.create(be.ps, be.dg));
         be.streaming = &be.streaming_dg1;
      } else if (spec.backend == BackendKind::UNSTRUCTURED_DG) {
         PetscCall(be.streaming_dg0.create(be.ps, be.dg));
         be.streaming = &be.streaming_dg0;
      }
      break;
   }

   default:
      SETERRQ(comm, PETSC_ERR_SUP, "unknown backend kind %d", (int)spec.backend);
   }

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
   // Precondition with a streaming-only pmat instead of the assembled
   // streaming + removal operator
   PetscBool precon_stream = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_stream", &precon_stream, NULL));
   // Apply the removal matrix-free instead of assembling it. The assembled
   // matrix then carries STREAMING ONLY, which does not depend on the group, so
   // it is assembled ONCE before the sweep: a group is a pointer swap on the
   // terms plus TransportSolver::refresh(), and PCAIR sets up once for the
   // whole sweep. That matrix is already the streaming-only pmat -precon_stream
   // builds a copy of, so this mode IMPLIES that preconditioner and an explicit
   // -precon_stream alongside it is accepted and ignored - which makes a
   // matfree recipe exactly its -precon_stream twin plus this flag (the two
   // must agree on iteration count, see ../docs/dev/testing.md)
   PetscBool matfree_removal = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-matfree_removal", &matfree_removal, NULL));
   // Put a representative removal back onto that streaming-only pmat: what
   // makes -matfree_removal usable when the removal is strong, since PCAIR set
   // up on bare streaming diverges from roughly a mean free path per cell.
   // Each group is preconditioned with L + alpha_g * D_ref, D_ref a reference
   // group's Sigma_t (one reference per set of void cells, a void cell left
   // unshifted, so its pmat row is the bare streaming row the operator has
   // there) and alpha_g that group's ratio to it, in k = -precon_ref_k shared
   // copies - unset asks RefShiftPmats for its default rule. k pmats means k
   // PCAIR hierarchies, the design's stated cost. It composes with
   // -precon_dsa: the correction wants a removal-carrying pmat to attach to
   PetscBool precon_ref_shift = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_ref_shift", &precon_ref_shift, NULL));
   PetscInt precon_ref_k = 0;
   PetscBool have_ref_k = PETSC_FALSE;
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-precon_ref_k", &precon_ref_k, &have_ref_k));
   // The shift repairs a pmat that is missing its removal, and
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
      "-precon_ref_k must be non-negative (0, like leaving it unset, asks for the default), was given %" \
      PetscInt_FMT, precon_ref_k);
   // Add the DSA diffusion correction to the composite preconditioner (index
   // 2, its inner solve under the -dsa_ prefix - see DSAPrecon). Off by
   // default: nothing preconditions the scattering without it, which is what
   // an optically thick, high scattering ratio problem needs
   PetscBool precon_dsa = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_dsa", &precon_dsa, NULL));
   // Diagonally scale the assembled operator and the rhs. Only the ASSEMBLED
   // operator is scaled: the matrix-free scatter goes unscaled, which makes it
   // pathological with scattering (the DG0 box 7 -> 83 iterations; CG stalls)
   PetscBool diag_scale = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));
   // Under -matfree_removal the assembled operator is streaming alone: the
   // removal and the scatter would go through the shell unscaled and the
   // system solved would not be the scaled one. Inconsistent by construction
   PetscCheck(!(diag_scale && matfree_removal), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
      "-diag_scale scales the assembled operator, which under -matfree_removal carries " \
      "streaming only - the matrix-free removal and scatter would stay unscaled");
   // And the DSA's R A P consistency (the /sum_weights scaling of its
   // restriction) holds for the unscaled operator, so under -diag_scale the
   // correction would be the wrong one for the system being solved
   PetscCheck(!(diag_scale && precon_dsa), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
      "-diag_scale scales the operator the DSA correction is derived from, so the correction " \
      "would no longer match the system solved: run -precon_dsa without it");
   // Build the streaming stage's PCAIR on the element-block-scaled pmat,
   // D^{-1} pmat - see TransportSolver::create. The default depends on the
   // backend, so it is resolved once the problem file has been read: ON for
   // the unstructured backends (DG both orders - at DG0 it is the diagonal -
   // and CG-SUPG, point Jacobi at one dof per vertex), OFF for the structured
   // ones, whose baselines predate it. Either way the flag wins
   PetscBool precon_block_scale = PETSC_FALSE, have_block_scale = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-precon_block_scale", &precon_block_scale, &have_block_scale));
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
   // Check every reference-shifted pmat against its group's full operator
   // (see CheckRefShiftExact above) - the exact-coverage identity, voids
   // included, on a density-scaled problem with a bin per distinct alpha
   PetscBool check_ref_shift = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_ref_shift", &check_ref_shift, NULL));
   PetscCheck(!check_ref_shift || precon_ref_shift, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
      "-check_ref_shift checks the pmats -precon_ref_shift builds: it needs -precon_ref_shift");
   // Write the scalar flux of the solution for inspection - overrides the
   // problem file's output.flux_vtk. A single-group problem writes the
   // filename as given, multigroup writes one file per group: -flux_vtk
   // flux.vts writes flux_g0.vts, flux_g1.vts, ... The extension picks the
   // format: .vts or .vtr on a structured mesh, .vtu on an unstructured one
   char flux_vtk_cli[PETSC_MAX_PATH_LEN];
   PetscBool have_flux_cli = PETSC_FALSE;
   PetscCall(PetscOptionsGetString(NULL, NULL, "-flux_vtk", flux_vtk_cli, sizeof(flux_vtk_cli), \
      &have_flux_cli));

   KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
   PetscReal inf_medium_err = 0.0;
   PetscReal matfree_err = 0.0, matfree_diag_err = 0.0;
   PetscReal ref_shift_err = 0.0;

   // Device memory has to be gone before PetscFinalize takes Kokkos down
   {
      ProblemSpec spec;
      PetscCall(spec.create(PETSC_COMM_WORLD, problem_path));
      const PetscInt n_groups = spec.n_groups;
      const PetscInt bg = spec.background_material;
      // Checked here, not after the sweep, like the file's own output.flux_vtk
      if (have_flux_cli) PetscCall(spec.check_flux_vtk(flux_vtk_cli, "-flux_vtk"));
      const std::string flux_vtk = have_flux_cli ? std::string(flux_vtk_cli) : spec.flux_vtk;
      if (!have_block_scale) precon_block_scale = (spec.backend != BackendKind::STRUCTURED_FD) ? PETSC_TRUE : PETSC_FALSE;

      // The CG-SUPG backend's thin-cell parameter, overriding the file's
      // mesh.supg_zeta for sweeps
      PetscBool have_zeta = PETSC_FALSE;
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-supg_zeta", &spec.supg_zeta, &have_zeta));
      PetscCheck(!have_zeta || spec.backend == BackendKind::UNSTRUCTURED_CG, PETSC_COMM_WORLD, \
         PETSC_ERR_ARG_INCOMP, "-supg_zeta is the CG-SUPG backend's: the problem file's mesh is not \"cg_supg\"");

      // The infinite-medium check needs nothing for the streaming term to do
      // and nowhere to leak: one uniform material, absorption in every group
      // so the constant is finite, and every face either reflective or a
      // whole-face vacuum face whose inflow IS the infinite-medium flux - a
      // boundary that feeds in exactly what the medium holds changes nothing,
      // in either vacuum treatment, which is what lets the same oracle check
      // the vacuum rows. The face ids are 1 .. 2 * dimension on every box,
      // structured or unstructured, but a mesh file's "Face Sets" can be
      // anything, so a mesh file is refused
      if (check_inf_medium) {
         PetscCheck(spec.mesh.file.empty(), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
            "-check_inf_medium needs a box mesh, whose faces are \"Face Sets\" 1 .. 2 * dimension");
         PetscCheck(spec.intervals.empty() && spec.boxes.empty() && spec.boxes_3d.empty(), \
            PETSC_COMM_WORLD, \
            PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs uniform xsections and source: no paint");
         for (PetscInt g = 0; g < n_groups; g++) {
            PetscCheck(PetscRealPart(spec.materials.sigma_s_host()[(bg * n_groups + g) * n_groups + g]) \
               < PetscRealPart(spec.materials.sigma_t_host()[bg * n_groups + g]), PETSC_COMM_WORLD, \
               PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs absorption: within-group Sigma_s below " \
               "Sigma_t in every group");
         }
         for (PetscInt f = 1; f <= 2 * spec.dimension; f++) {
            const BCFace face = spec.bcs.face(f);
            PetscCheck(face.type == BCType::REFLECT || face.n_window_pairs == 0, PETSC_COMM_WORLD, \
               PETSC_ERR_ARG_INCOMP, "-check_inf_medium needs every vacuum face unwindowed");
         }
      }

      // ~~~~~~~~~~~~~
      // The per-backend pieces. Everything after this works through the bases
      // ~~~~~~~~~~~~~
      Backend be;
      PetscCall(BuildBackend(spec, be));
      const PhaseSpace &ps = be.ps;
      Discretisation *disc = be.disc;
      const AngularQuadrature *quad = be.quad;
      OperatorTerm *streaming = be.streaming;

      // Everything that works on a group-independent streaming matrix. CG-SUPG
      // has none: its one assembled term refills per group (tau depends on
      // sigma_t), and there is no diagonal removal to take out of it
      const char *needs_streaming = matfree_removal ? "-matfree_removal" : precon_stream ? "-precon_stream" : \
         precon_ref_shift ? "-precon_ref_shift" : check_matfree ? "-check_matfree" : NULL;
      PetscCheck(streaming || !needs_streaming, PETSC_COMM_WORLD, PETSC_ERR_SUP, \
         "%s needs a group-independent streaming matrix, and this backend has none: its assembled " \
         "operator refills per group (CG-SUPG's tau depends on sigma_t)", needs_streaming);

      // ~~~~~~~~~~~~~
      // Cross sections per group and material entry (local cell, or local
      // element on CG-SUPG): the file's material table expanded through the
      // painted indices. The terms never learn that materials exist
      // ~~~~~~~~~~~~~
      GroupXSections xs;
      PetscCall(xs.create(n_groups, disc->n_material_entries()));
      PetscCall(xs.set_from_materials(spec.materials, be.mat_id_d));

      // ~~~~~~~~~~~~~
      // The DSA diffusion correction, if it was asked for. Built from the
      // geometry, so DSAPrecon dispatches on the concrete backend
      // ~~~~~~~~~~~~~
      DSAPrecon dsa;
      if (precon_dsa) PetscCall(dsa.create(PETSC_COMM_WORLD, ps, *disc, *quad, spec.bcs));

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
      ScatteringTerm scattering;
      SUPGTermCG supg_cg;
      ScatteringTermCG scattering_cg;
      TransportOperator op;
      GroupTransfer transfer;
      GroupTransferCG transfer_cg;
      GroupSource *group_source = NULL;
      PetscCall(op.create(*disc));
      if (spec.backend == BackendKind::UNSTRUCTURED_CG) {
         // Streaming, SUPG and removal in the one assembled term; the scatter
         // and the transfer carry the SUPG weight too (terms_cg.hpp)
         PetscCall(supg_cg.create(ps, be.cg, xs.sigma_t(0)));
         PetscCall(scattering_cg.create(ps, be.cg, *quad, xs.sigma_t(0), xs.sigma_s(0, 0)));
         PetscCall(op.add_term(&supg_cg));
         PetscCall(op.add_term(&scattering_cg));
         PetscCall(transfer_cg.create(ps, be.cg, *quad, xs, spec.materials, be.mat_id_d));
         group_source = &transfer_cg;
      } else {
         PetscCall(removal.create(ps, *disc, xs.sigma_t(0)));
         removal.set_matrix_free(matfree_removal);
         PetscCall(scattering.create(ps, *disc, *quad, xs.sigma_s(0, 0)));
         PetscCall(op.add_term(streaming));
         PetscCall(op.add_term(&removal));
         PetscCall(op.add_term(&scattering));
         PetscCall(transfer.create(ps, *quad, xs, disc->boundary_info(), spec.materials, be.mat_id_d));
         group_source = &transfer;
      }

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
      // preconditions - only the assembled operator and the rhs are scaled.
      // Not under -matfree_removal: the assembled matrix is already exactly
      // this matrix (see -matfree_removal's parse)
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
         // Only when there is something to say: groups void in different
         // cells are binned apart, one reference per set of void cells
         if (ref_shift.n_classes() > 1) {
            PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, \
               "reference-shifted pmat: %" PetscInt_FMT " void patterns, binned apart\n", \
               ref_shift.n_classes()));
         }
         if (check_ref_shift) PetscCall(CheckRefShiftExact(ps, *disc, op, streaming, xs, \
            ref_shift, n_groups, &ref_shift_err));
      }

      // ~~~~~~~~~~~~~
      // Sweep the groups: Gauss-Seidel with downscatter only. Groups are
      // ordered high energy to low, the within-group scatter stays on the lhs
      // (matrix-free) and everything scattering out of an already-solved group
      // goes to the rhs. With no upscatter the group system is block lower
      // triangular, so one forward sweep is exact and there is no outer
      // iteration - that arrives with upscatter
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
         // through, so the pointer swaps are the whole per-group update
         PetscCall(op.set_group(xs, g));
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
         // (add_transfer skips every BC row on its own)
         PetscCall(VecSet(b, 0.0));
         PetscCall(UboltFillInflow(disc->boundary_info(), b));
         PetscCall(group_source->add_external(g, b));
         PetscCall(UboltZeroReflectRows(disc->boundary_info(), b));
         for (PetscInt g_from = 0; g_from < g; g_from++) PetscCall(group_source->add_transfer(g_from, g, b));

         // Diagonally scale this group's assembled operator and rhs - AFTER
         // the rhs is complete, and BEFORE the solver sees the matrix, so the
         // removal shell caches the scaled blocks (create and refresh both
         // read them off the assembled matrix)
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
            PetscCall(solver.create(PETSC_COMM_WORLD, op, pmat, precon_dsa ? &dsa : nullptr, \
               precon_block_scale));
            solver_created[bin] = PETSC_TRUE;
         }
         // sigma_t changed under the removal preconditioner
         else PetscCall(solver.refresh());

         PetscCall(solver.solve(b, psi[g]));
         reason = solver.converged_reason();
         if (reason <= 0) break;

         // This group's scalar flux is fixed now, and every group below it
         // scatters from it. Integrate once here rather than once per target
         PetscCall(group_source->set_scalar_flux(g, psi[g]));
      }

      // The infinite-medium check: uniform source, uniform xsections and no
      // boundary to leak through leave nothing for the streaming term to do,
      // so the exact discrete solution is constant in every cell and angle -
      // to solver tolerance, not discretisation error (at DG1: the constant on
      // basis 0 and zero slope on the rest). Per group, by forward
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

            // A vacuum face's per-ordinate inflow has to be that constant, or
            // the constant is not the solution and the comparison is meaningless
            for (PetscInt f = 1; f <= 2 * spec.dimension; f++) {
               const BCFace face = spec.bcs.face(f);
               if (face.type == BCType::REFLECT) continue;
               PetscCheck(PetscAbsScalar(face.inflow / quad->sum_weights() - expected[g]) \
                  <= 1e-12 * PetscAbsScalar(expected[g]), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, \
                  "-check_inf_medium needs every vacuum face's inflow / sum_weights to be the " \
                  "infinite-medium flux %g in every group (face %" PetscInt_FMT ", group %" \
                  PetscInt_FMT ")", (double)PetscRealPart(expected[g]), f, g);
            }

            PetscReal err_g = 0.0;
            PetscCall(InfMediumError(ps, psi[g], expected[g], &err_g));
            inf_medium_err = PetscMax(inf_medium_err, err_g);
         }
      }

      // Only a full sweep leaves every group's flux worth looking at. Each file
      // carries that group's scalar flux plus the two group-dependent inputs
      // that produced it - sigma_t and the external source - so the file says
      // what was solved without going back to the problem definition. A
      // single-group problem writes the filename as given, multigroup suffixes
      // _g<g> before the extension (the last dot of the file NAME, not of a
      // directory)
      if (!flux_vtk.empty() && reason > 0) {
         const char *slash = strrchr(flux_vtk.c_str(), '/');
         const char *dot = strrchr(slash ? slash + 1 : flux_vtk.c_str(), '.');
         const size_t base_len = dot ? (size_t)(dot - flux_vtk.c_str()) : flux_vtk.size();
         // The source is expanded onto the cells for output only, so one buffer
         // refilled per group rather than a table kept over the sweep
         PetscScalarKokkosView cell_source_d("cell_source_d", disc->n_material_entries());

         for (PetscInt g = 0; g < n_groups; g++) {
            char fname[PETSC_MAX_PATH_LEN];
            if (n_groups == 1) PetscCall(PetscStrncpy(fname, flux_vtk.c_str(), sizeof(fname)));
            else PetscCall(PetscSNPrintf(fname, sizeof(fname), "%.*s_g%" PetscInt_FMT "%s", \
               (int)base_len, flux_vtk.c_str(), g, dot ? dot : ""));

            PetscCall(UboltFillCellSource(spec.materials, be.mat_id_d, g, cell_source_d));
            std::vector<UboltCellField> extra = {{"sigma_t", xs.sigma_t(g)}, {"source", cell_source_d}};
            // DG1: scalar_flux is the cell average, so the slope rides along
            // and the file carries the whole linear solution
            std::vector<PetscScalarKokkosView> grad;
            if (spec.backend == BackendKind::UNSTRUCTURED_DG && spec.order == 1) {
               static const char *grad_names[3] = {"scalar_flux_grad_x", "scalar_flux_grad_y", "scalar_flux_grad_z"};
               PetscCall(be.dg.scalar_flux_gradient(psi[g], *quad, grad));
               for (size_t d = 0; d < grad.size(); d++) extra.push_back({grad_names[d], grad[d]});
            }
            PetscCall(UboltWriteScalarFluxVTK(ps, *disc, *quad, psi[g], (PetscInt)extra.size(), extra.data(), \
               fname));
         }
      }

      for (size_t s = 0; s < solvers.size(); s++) PetscCall(solvers[s].destroy());
      // After the solvers: the shell at composite index 2 pointed at it
      PetscCall(dsa.destroy());
      // Same order for the pmats the solvers were built on
      PetscCall(ref_shift.destroy());
      PetscCall(op.destroy());
      PetscCall(scattering_cg.destroy());
      PetscCall(transfer_cg.destroy());
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

   // And the reference-shift identity: rounding against 1e-12
   const PetscBool ref_shift_ok = (PetscBool)(!check_ref_shift || ref_shift_err < 1e-12);
   if (check_ref_shift) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, \
      "reference-shift check: max relative difference from the full operator %g against " \
      "tolerance 1e-12 - %s\n", (double)ref_shift_err, ref_shift_ok ? "pass" : "FAIL"));

   PetscCall(PetscFinalize());
   return (reason > 0 && inf_medium_ok && matfree_ok && ref_shift_ok) ? 0 : 1;
}
