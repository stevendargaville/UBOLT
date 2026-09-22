// Phase 6a verification for the unstructured DG0 backend (UnstructuredDG0 +
// StreamingTermDG0)
//
// Every check fails the process rather than printing something a human has to
// read, and every one runs serially AND in parallel - nothing below assumes
// rank 0 owns anything, or that either backend numbers its rows naturally:
//
//  1. FD twin, 2D quads. On a uniform quad box DG0 upwind IS the structured
//     upwind stencil (|mu| dy / (dx dy) = |mu| / dx), so the same problem is
//     built on StructuredFD2D and on the plex, and compared through a
//     permutation matched by cell centroid: the assembled matrices, the BC row
//     mask / reflect flags / Dirichlet values, the rhs, the matrix-free
//     scatter, the composed diagonal (bitwise, against the plex's own
//     MatGetDiagonal) and a full solve. Vacuum, reflective, mixed, windowed
//     inflow with the corner winning-face rule, and painted cases (by box and
//     by a "Cell Sets" label)
//  2. The same in 3D against StructuredFD3D, including the three-reflective-
//     face corner and a window in (y, z)
//  3. Simplex meshes (triangles, tets), which have no twin: an infinite medium
//     (all faces reflective - flat to 1e-10) and a vacuum box with a central
//     source (inside the exact discrete bounds, and its integrated scalar flux
//     per unit source within 25% of the quad/hex mesh's at the same nominal h
//     - a coarse consistency check, not a convergence claim)
//  4. Layout and geometry invariants on every mesh: the volumes sum to the
//     box's, every cell's outward normals close, the cell count is the box's,
//     and (on the simplex checks' meshes) each "Face Sets" id sits on the box
//     face the structured FACE_* constants give it
//  5. Error paths: a reflection partner that is itself a BC row, materials
//     for "Cell Sets" on a mesh without the label, a .vts name on the plex, a
//     2D quadrature on a 3D mesh
//  6. The .vtu writer writes each cell exactly once in parallel (the overlap
//     must not appear in the file)
//
// The FD twin comparison is exact in structure (the flags, which rows are
// which) and to rounding in value: the two backends reach the same
// coefficients through different arithmetic, so "reproduces" means ~1e-15,
// not bitwise
//
// Currently run with:
// make build_tests && ./verify_plexk

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include <petscksp.h>
#include <petscdmplex.h>
#include "pflare.h"
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// The materials every case solves: material 0 everywhere, material 1 where a
// case paints it. Distinct sigma_t, sigma_s and source, so a painting mismatch
// between the twins shows up in the matrix, the scatter and the rhs alike
static PetscErrorCode TwinMaterials(MaterialSpec &mats)
{
   PetscFunctionBeginUser;

   PetscCall(mats.create(2, 1));
   PetscCall(mats.set_sigma_t(0, 0, 1.5));
   PetscCall(mats.set_sigma_s(0, 0, 0, 0.7));
   PetscCall(mats.set_source(0, 0, 1.0));
   PetscCall(mats.set_sigma_t(1, 0, 0.7));
   PetscCall(mats.set_sigma_s(1, 0, 0, 0.3));
   PetscCall(mats.set_source(1, 0, 0.5));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Everything one backend needs for a single-group solve, built the way the
// driver builds it: the backend's streaming term plus removal (both
// assembled) and the matrix-free scatter
struct Solve {
   GroupXSections xs;
   RemovalTerm removal;
   ScatteringTerm scattering;
   TransportOperator op;
};

static PetscErrorCode BuildSolve(const PhaseSpace &ps, const Discretisation &disc, const OperatorTerm &streaming, \
   const AngularQuadrature &quad, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, Solve &s)
{
   PetscFunctionBeginUser;

   PetscCall(s.xs.create(ps));
   PetscCall(s.xs.set_from_materials(mats, mat_id_d));
   PetscCall(s.removal.create(ps, disc, s.xs.sigma_t(0)));
   PetscCall(s.scattering.create(ps, disc, quad, s.xs.sigma_s(0, 0)));
   PetscCall(s.op.create(PETSC_COMM_WORLD, ps, disc));
   PetscCall(s.op.add_term(&streaming));
   PetscCall(s.op.add_term(&s.removal));
   PetscCall(s.op.add_term(&s.scattering));
   PetscCall(s.op.assemble());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// The rhs exactly as the driver fills it
static PetscErrorCode FillRhs(const PhaseSpace &ps, const Discretisation &disc, const AngularQuadrature &quad, \
   const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, Vec b)
{
   PetscFunctionBeginUser;

   PetscCall(VecSet(b, 0.0));
   PetscCall(UboltFillInflow(disc.boundary_info(), b));
   PetscCall(UboltFillSource(ps, disc.boundary_info(), quad, mats, mat_id_d, 0, b));
   PetscCall(UboltZeroReflectRows(disc.boundary_info(), b));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Solve to well below the comparison tolerances. After the solver's create(),
// so this wins over anything on the command line
static PetscErrorCode SolveTight(const Solve &s, Vec b, Vec x, PetscBool *converged)
{
   TransportSolver solver;

   PetscFunctionBeginUser;

   PetscCall(solver.create(PETSC_COMM_WORLD, s.op, s.op.assembled_mat()));
   PetscCall(KSPSetTolerances(solver.ksp(), 1e-13, 1e-50, PETSC_CURRENT, 1000));
   PetscCall(VecZeroEntries(x));
   PetscCall(solver.solve(b, x));
   *converged = (PetscBool)(solver.converged_reason() > 0);
   PetscCall(solver.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySolve(Solve &s)
{
   PetscFunctionBeginUser;

   PetscCall(s.op.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A per-row device view onto a Vec over the same rows, through the host (this
// is a test, not a kernel path). The transform turns the BC mask and reflect
// slots into 0/1 flags
enum class RowField { IS_BC, IS_REFLECT, DIRICHLET };

static PetscErrorCode BoundaryToVec(const BoundaryInfo &boundary, RowField field, Vec v)
{
   PetscScalar *v_a = nullptr;
   PetscInt n = 0;

   PetscFunctionBeginUser;

   auto is_bc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary.is_bc_row_d);
   auto reflect_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary.reflect_slot_d);
   auto dirichlet_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary.dirichlet_value_d);

   PetscCall(VecGetLocalSize(v, &n));
   PetscCall(VecGetArray(v, &v_a));
   for (PetscInt r = 0; r < n; r++) {
      if (field == RowField::IS_BC) v_a[r] = is_bc_h(r) ? 1.0 : 0.0;
      else if (field == RowField::IS_REFLECT) v_a[r] = (reflect_h(r) >= 0) ? 1.0 : 0.0;
      else v_a[r] = dirichlet_h(r);
   }
   PetscCall(VecRestoreArray(v, &v_a));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The row permutation between the twins, from a key per local cell on each
// side - the cell's natural (lexicographic) index, used ONLY as a join key.
// Neither backend's global numbering is assumed: each side's row of local cell
// c is its own rstart + c * n_angles + a, the property each backend's layout
// check asserts (CheckDALayout / CheckPlexLayout)
//
// The join goes through a Vec indexed by key: the plex ranks write their row
// base into it, the FD ranks read theirs back. Out of it come
//  - P, the permutation matrix with P(fd_row, plex_row) = 1, so
//    P^T A_fd P is the FD matrix in plex order and in the plex's layout -
//    whatever the two decompositions are
//  - a VecScatter plex -> FD (and back, in reverse) for the vectors
struct TwinPerm {
   Mat P = NULL;
   VecScatter scatter = NULL;
};

static PetscErrorCode BuildTwinPerm(PetscInt n_cells, PetscInt n_angles, const std::vector<PetscInt> &key_fd, \
   PetscInt rstart_fd, const std::vector<PetscInt> &key_plex, PetscInt rstart_plex, Vec x_fd, Vec x_plex, \
   TwinPerm &perm, PetscBool *ok)
{
   Vec plex_base = NULL, hits = NULL, gathered = NULL;
   IS is_keys = NULL, is_fd = NULL, is_plex = NULL;
   VecScatter gather = NULL;
   PetscReal min_hits = 0.0, max_hits = 0.0;
   const PetscInt local_fd = (PetscInt)key_fd.size();
   const PetscInt local_plex = (PetscInt)key_plex.size();

   PetscFunctionBeginUser;

   PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n_cells, &plex_base));
   PetscCall(VecDuplicate(plex_base, &hits));
   for (PetscInt c = 0; c < local_plex; c++) {
      const PetscScalar base = (PetscScalar)(rstart_plex + c * n_angles);
      const PetscScalar one = 1.0;
      PetscCall(VecSetValue(plex_base, key_plex[c], base, INSERT_VALUES));
      PetscCall(VecSetValue(hits, key_plex[c], one, ADD_VALUES));
   }
   PetscCall(VecAssemblyBegin(plex_base));
   PetscCall(VecAssemblyEnd(plex_base));
   PetscCall(VecAssemblyBegin(hits));
   PetscCall(VecAssemblyEnd(hits));

   // Every box cell matched by exactly one plex cell: the centroid match is a
   // bijection, or the comparisons below would be meaningless
   PetscCall(VecMin(hits, NULL, &min_hits));
   PetscCall(VecMax(hits, NULL, &max_hits));
   if (min_hits != 1.0 || max_hits != 1.0) {
      *ok = PETSC_FALSE;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  the centroid match is not a bijection: cells hit %g to %g times\n", \
         (double)min_hits, (double)max_hits));
   }

   // Each FD rank reads the plex row base of its own cells
   PetscCall(ISCreateGeneral(PETSC_COMM_SELF, local_fd, key_fd.data(), PETSC_COPY_VALUES, &is_keys));
   PetscCall(VecCreateSeq(PETSC_COMM_SELF, local_fd, &gathered));
   PetscCall(VecScatterCreate(plex_base, is_keys, gathered, NULL, &gather));
   PetscCall(VecScatterBegin(gather, plex_base, gathered, INSERT_VALUES, SCATTER_FORWARD));
   PetscCall(VecScatterEnd(gather, plex_base, gathered, INSERT_VALUES, SCATTER_FORWARD));

   std::vector<PetscInt> fd_rows(local_fd * n_angles), plex_rows(local_fd * n_angles);
   {
      const PetscScalar *g_a = nullptr;
      PetscCall(VecGetArrayRead(gathered, &g_a));
      for (PetscInt c = 0; c < local_fd; c++) {
         const PetscInt base = (PetscInt)PetscRealPart(g_a[c]);
         for (PetscInt a = 0; a < n_angles; a++) {
            fd_rows[c * n_angles + a] = rstart_fd + c * n_angles + a;
            plex_rows[c * n_angles + a] = base + a;
         }
      }
      PetscCall(VecRestoreArrayRead(gathered, &g_a));
   }

   PetscInt m_fd = 0, m_plex = 0;
   PetscCall(VecGetLocalSize(x_fd, &m_fd));
   PetscCall(VecGetLocalSize(x_plex, &m_plex));
   PetscCall(MatCreate(PETSC_COMM_WORLD, &perm.P));
   PetscCall(MatSetSizes(perm.P, m_fd, m_plex, PETSC_DETERMINE, PETSC_DETERMINE));
   PetscCall(MatSetType(perm.P, MATAIJ));
   PetscCall(MatSeqAIJSetPreallocation(perm.P, 1, NULL));
   PetscCall(MatMPIAIJSetPreallocation(perm.P, 1, NULL, 1, NULL));
   for (PetscInt r = 0; r < local_fd * n_angles; r++) {
      PetscCall(MatSetValue(perm.P, fd_rows[r], plex_rows[r], 1.0, INSERT_VALUES));
   }
   PetscCall(MatAssemblyBegin(perm.P, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(perm.P, MAT_FINAL_ASSEMBLY));

   PetscCall(ISCreateGeneral(PETSC_COMM_SELF, local_fd * n_angles, plex_rows.data(), PETSC_COPY_VALUES, &is_plex));
   PetscCall(ISCreateGeneral(PETSC_COMM_SELF, local_fd * n_angles, fd_rows.data(), PETSC_COPY_VALUES, &is_fd));
   PetscCall(VecScatterCreate(x_plex, is_plex, x_fd, is_fd, &perm.scatter));

   PetscCall(VecScatterDestroy(&gather));
   PetscCall(ISDestroy(&is_keys));
   PetscCall(ISDestroy(&is_fd));
   PetscCall(ISDestroy(&is_plex));
   PetscCall(VecDestroy(&gathered));
   PetscCall(VecDestroy(&plex_base));
   PetscCall(VecDestroy(&hits));

   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyTwinPerm(TwinPerm &perm)
{
   PetscFunctionBeginUser;

   PetscCall(MatDestroy(&perm.P));
   PetscCall(VecScatterDestroy(&perm.scatter));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// || v_plex - (v_fd in plex order) ||_inf
static PetscErrorCode PermutedDiff(const TwinPerm &perm, Vec v_fd, Vec v_plex, PetscReal *diff)
{
   Vec w = NULL;

   PetscFunctionBeginUser;

   PetscCall(VecDuplicate(v_plex, &w));
   PetscCall(VecScatterBegin(perm.scatter, v_fd, w, INSERT_VALUES, SCATTER_REVERSE));
   PetscCall(VecScatterEnd(perm.scatter, v_fd, w, INSERT_VALUES, SCATTER_REVERSE));
   PetscCall(VecAXPY(w, -1.0, v_plex));
   PetscCall(VecNorm(w, NORM_INFINITY, diff));
   PetscCall(VecDestroy(&w));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Checks (a)-(f) of the FD twin, on any dimension: everything here goes
// through the dimension-independent surfaces
static PetscErrorCode CompareTwins(const char *desc, PetscInt n_cells, PetscInt n_angles, \
   const PhaseSpace &ps_fd, const Discretisation &disc_fd, const OperatorTerm &streaming_fd, \
   const PetscIntKokkosView &mat_id_fd, const std::vector<PetscInt> &key_fd, \
   const PhaseSpace &ps_plex, const Discretisation &disc_plex, const OperatorTerm &streaming_plex, \
   const PetscIntKokkosView &mat_id_plex, const std::vector<PetscInt> &key_plex, \
   const AngularQuadrature &quad, const MaterialSpec &mats, PetscBool *ok)
{
   Solve fd, plex;
   TwinPerm perm;
   Mat A_fd = NULL, A_plex = NULL, A_fd_perm = NULL;
   Vec x_fd = NULL, y_fd = NULL, x_plex = NULL, y_plex = NULL, d_composed = NULL, d_mat = NULL;
   PetscInt rstart_fd = 0, rstart_plex = 0;
   PetscReal norm_fd = 0.0, mat_diff = 0.0, flag_diff = 0.0, dirichlet_diff = 0.0, rhs_diff = 0.0;
   PetscReal scatter_diff = 0.0, diag_diff = 0.0, sol_diff = 0.0;
   PetscBool conv_fd = PETSC_FALSE, conv_plex = PETSC_FALSE;
   const PetscReal mat_tol = 1e-12, value_tol = 1e-14, sol_tol = 1e-9;

   PetscFunctionBeginUser;

   PetscCall(BuildSolve(ps_fd, disc_fd, streaming_fd, quad, mats, mat_id_fd, fd));
   PetscCall(BuildSolve(ps_plex, disc_plex, streaming_plex, quad, mats, mat_id_plex, plex));

   PetscCall(MatCreateVecs(fd.op.assembled_mat(), &x_fd, &y_fd));
   PetscCall(MatCreateVecs(plex.op.assembled_mat(), &x_plex, &y_plex));
   PetscCall(MatGetOwnershipRange(fd.op.assembled_mat(), &rstart_fd, NULL));
   PetscCall(MatGetOwnershipRange(plex.op.assembled_mat(), &rstart_plex, NULL));
   PetscCall(BuildTwinPerm(n_cells, n_angles, key_fd, rstart_fd, key_plex, rstart_plex, x_fd, x_plex, perm, ok));

   // (a) the assembled matrices (streaming + removal), plex order
   PetscCall(MatConvert(fd.op.assembled_mat(), MATAIJ, MAT_INITIAL_MATRIX, &A_fd));
   PetscCall(MatConvert(plex.op.assembled_mat(), MATAIJ, MAT_INITIAL_MATRIX, &A_plex));
   PetscCall(MatPtAP(A_fd, perm.P, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &A_fd_perm));
   PetscCall(MatNorm(A_fd, NORM_INFINITY, &norm_fd));
   PetscCall(MatAXPY(A_fd_perm, -1.0, A_plex, DIFFERENT_NONZERO_PATTERN));
   PetscCall(MatNorm(A_fd_perm, NORM_INFINITY, &mat_diff));

   // (b) the boundary info, flags exactly and the Dirichlet values to rounding
   {
      const RowField flags[2] = {RowField::IS_BC, RowField::IS_REFLECT};
      for (const RowField field : flags) {
         PetscReal d = 0.0;
         PetscCall(BoundaryToVec(disc_fd.boundary_info(), field, x_fd));
         PetscCall(BoundaryToVec(disc_plex.boundary_info(), field, x_plex));
         PetscCall(PermutedDiff(perm, x_fd, x_plex, &d));
         flag_diff = PetscMax(flag_diff, d);
      }
      PetscCall(BoundaryToVec(disc_fd.boundary_info(), RowField::DIRICHLET, x_fd));
      PetscCall(BoundaryToVec(disc_plex.boundary_info(), RowField::DIRICHLET, x_plex));
      PetscCall(PermutedDiff(perm, x_fd, x_plex, &dirichlet_diff));
   }

   // (c) the rhs, filled exactly as the driver fills it
   PetscCall(FillRhs(ps_fd, disc_fd, quad, mats, mat_id_fd, y_fd));
   PetscCall(FillRhs(ps_plex, disc_plex, quad, mats, mat_id_plex, y_plex));
   PetscCall(PermutedDiff(perm, y_fd, y_plex, &rhs_diff));

   // (f) the full solve, from that rhs
   {
      Vec psi_fd = NULL, psi_plex = NULL;
      PetscCall(VecDuplicate(x_fd, &psi_fd));
      PetscCall(VecDuplicate(x_plex, &psi_plex));
      PetscCall(SolveTight(fd, y_fd, psi_fd, &conv_fd));
      PetscCall(SolveTight(plex, y_plex, psi_plex, &conv_plex));
      PetscCall(PermutedDiff(perm, psi_fd, psi_plex, &sol_diff));
      PetscCall(VecDestroy(&psi_fd));
      PetscCall(VecDestroy(&psi_plex));
   }

   // (d) the matrix-free scatter on the same random vector
   {
      PetscRandom rand;
      PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
      PetscCall(PetscRandomSetSeed(rand, 0x5eed));
      PetscCall(PetscRandomSeed(rand));
      PetscCall(VecSetRandom(x_fd, rand));
      PetscCall(PetscRandomDestroy(&rand));
      PetscCall(VecScatterBegin(perm.scatter, x_fd, x_plex, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(perm.scatter, x_fd, x_plex, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecSet(y_fd, 0.0));
      PetscCall(VecSet(y_plex, 0.0));
      PetscCall(fd.scattering.apply_add(x_fd, y_fd));
      PetscCall(plex.scattering.apply_add(x_plex, y_plex));
      PetscCall(PermutedDiff(perm, y_fd, y_plex, &scatter_diff));
   }

   // (e) the composed diagonal against the plex's own assembled one - the
   // StreamingTermDG0::add_diagonal contract, which must hold BITWISE
   PetscCall(VecDuplicate(x_plex, &d_composed));
   PetscCall(VecDuplicate(x_plex, &d_mat));
   PetscCall(plex.op.diagonal(d_composed));
   PetscCall(MatGetDiagonal(plex.op.assembled_mat(), d_mat));
   PetscCall(VecAXPY(d_composed, -1.0, d_mat));
   PetscCall(VecNorm(d_composed, NORM_INFINITY, &diag_diff));

   const PetscBool pass = (PetscBool)(mat_diff <= mat_tol * norm_fd && flag_diff == 0.0 && \
      dirichlet_diff <= value_tol && rhs_diff <= value_tol && scatter_diff <= value_tol && diag_diff == 0.0 && \
      sol_diff <= sol_tol && conv_fd && conv_plex);
   if (!pass) *ok = PETSC_FALSE;

   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FD twin, %s:%s\n", desc, pass ? "" : " FAILED"));
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "    matrix %.3e (tol %.0e x |A| = %.3e), BC flags %.1e (exact), Dirichlet values %.3e (tol %.0e)\n", \
      (double)mat_diff, (double)mat_tol, (double)(mat_tol * norm_fd), (double)flag_diff, (double)dirichlet_diff, \
      (double)value_tol));
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "    rhs %.3e (tol %.0e), scatter %.3e (tol %.0e), composed diagonal %.1e (exact), solution %.3e (tol %.0e)%s\n", \
      (double)rhs_diff, (double)value_tol, (double)scatter_diff, (double)value_tol, (double)diag_diff, \
      (double)sol_diff, (double)sol_tol, (conv_fd && conv_plex) ? "" : " - a solve did NOT converge"));

   PetscCall(MatDestroy(&A_fd));
   PetscCall(MatDestroy(&A_plex));
   PetscCall(MatDestroy(&A_fd_perm));
   PetscCall(VecDestroy(&x_fd));
   PetscCall(VecDestroy(&y_fd));
   PetscCall(VecDestroy(&x_plex));
   PetscCall(VecDestroy(&y_plex));
   PetscCall(VecDestroy(&d_composed));
   PetscCall(VecDestroy(&d_mat));
   PetscCall(DestroyTwinPerm(perm));
   PetscCall(DestroySolve(fd));
   PetscCall(DestroySolve(plex));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 4 on one mesh: the volumes tile the box, every cell's outward
// area-weighted normals close (sum to zero - a closed polytope, and a normal
// that failed to be flipped outward would show up here as 2 nA), and on a
// quad/hex box the cell count is the box's (expected_cells < 0 skips that)
static PetscErrorCode CheckGeometry(const char *desc, const UnstructuredDG0 &disc, PetscReal box_volume, \
   PetscInt expected_cells, PetscBool *ok)
{
   PetscReal volume = 0.0, total = 0.0, closure = 0.0, max_closure = 0.0;
   const PetscReal vol_tol = 1e-12, closure_tol = 1e-12;

   PetscFunctionBeginUser;

   const std::vector<PetscReal> &vol_h = disc.volume_host();
   auto offset_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), disc.cell_face_offset_d());
   auto nA_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), disc.face_nA_d());

   for (PetscReal v : vol_h) volume += v;
   PetscCallMPI(MPI_Allreduce(&volume, &total, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

   for (PetscInt c = 0; c < (PetscInt)vol_h.size(); c++) {
      PetscReal sum[3] = {0.0, 0.0, 0.0}, area = 0.0;
      for (PetscInt k = offset_h(c); k < offset_h(c + 1); k++) {
         PetscReal a2 = 0.0;
         for (PetscInt d = 0; d < 3; d++) {
            sum[d] += PetscRealPart(nA_h(3 * k + d));
            a2 += PetscRealPart(nA_h(3 * k + d)) * PetscRealPart(nA_h(3 * k + d));
         }
         area += PetscSqrtReal(a2);
      }
      const PetscReal rel = PetscSqrtReal(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]) / area;
      closure = PetscMax(closure, rel);
   }
   PetscCallMPI(MPI_Allreduce(&closure, &max_closure, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

   const PetscReal vol_err = PetscAbsReal(total - box_volume) / box_volume;
   const PetscBool count_ok = (PetscBool)(expected_cells < 0 || disc.n_global_cells() == expected_cells);
   const PetscBool pass = (PetscBool)(vol_err <= vol_tol && max_closure <= closure_tol && count_ok);
   if (!pass) *ok = PETSC_FALSE;

   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  geometry, %s: %" PetscInt_FMT " cells%s, volume error %.3e (tol %.0e), normal closure %.3e (tol %.0e)%s\n", \
      desc, disc.n_global_cells(), expected_cells < 0 ? "" : (count_ok ? " (as the box)" : " (NOT the box's)"), \
      (double)vol_err, (double)vol_tol, (double)max_closure, (double)closure_tol, pass ? "" : " FAILED"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The natural (lexicographic) index of each plex local cell on a uniform box,
// from its centroid - the join key BuildTwinPerm matches the FD twin by
static void PlexKeys(const UnstructuredDG0 &disc, PetscInt dim, const PetscInt *n, const PetscReal *h, \
   std::vector<PetscInt> &key)
{
   const std::vector<PetscReal> &cen = disc.centroid_host();
   const PetscInt local_cells = (PetscInt)cen.size() / 3;

   key.resize(local_cells);
   for (PetscInt c = 0; c < local_cells; c++) {
      PetscInt idx[3] = {0, 0, 0};
      for (PetscInt d = 0; d < dim; d++) {
         idx[d] = (PetscInt)PetscFloorReal(cen[3 * c + d] / h[d]);
         idx[d] = PetscMin(PetscMax(idx[d], 0), n[d] - 1);
      }
      key[c] = (dim == 2) ? idx[1] * n[0] + idx[0] : (idx[2] * n[1] + idx[1]) * n[0] + idx[0];
   }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 1: one 2D case against StructuredFD2D
// With via_cell_sets the plex side is painted NOT by its boxes but through a
// "Cell Sets" label written here onto the cells whose centroid is inside them
// (value 7, an arbitrary label id, mapped onto material 1) - the path a Gmsh
// file's physical groups take - so paint_cell_sets is held to the same FD twin
static PetscErrorCode CheckTwin2D(PetscInt nx, PetscInt ny, PetscInt sn_order, const BCSpec &bcs, \
   const std::vector<MaterialBox2D> &boxes, PetscBool via_cell_sets, const char *bc_desc, PetscBool *ok)
{
   const PetscReal lx = 1.0, ly = 2.0;
   SNQuadrature2D quad;
   MaterialSpec mats;
   PhaseSpace ps_fd, ps_plex;
   StructuredFD2D fd;
   UnstructuredDG0 plex;
   StreamingTerm2D streaming_fd;
   StreamingTermDG0 streaming_plex;
   PetscIntKokkosView mat_id_fd, mat_id_plex;
   std::vector<PetscInt> key_fd, key_plex;
   char desc[256];

   PetscFunctionBeginUser;

   PetscCall(quad.create(sn_order));
   PetscCall(TwinMaterials(mats));
   const PetscInt n_angles = quad.n_angles();

   PetscCall(ps_fd.create(PETSC_COMM_WORLD, nx * ny, n_angles));
   PetscCall(fd.create(PETSC_COMM_WORLD, ps_fd, nx, ny, lx, ly, quad, bcs));
   PetscCall(streaming_fd.create(ps_fd, fd, quad));
   PetscCall(fd.paint_boxes(0, boxes, mat_id_fd));

   PlexMeshSpec mesh;
   mesh.dimension = 2;
   mesh.n_cells[0] = nx;
   mesh.n_cells[1] = ny;
   mesh.lengths[0] = lx;
   mesh.lengths[1] = ly;
   PetscCall(plex.create_mesh(PETSC_COMM_WORLD, mesh));
   PetscCall(ps_plex.create(PETSC_COMM_WORLD, plex.n_global_cells(), n_angles));
   PetscCall(plex.create(ps_plex, quad, bcs));
   PetscCall(streaming_plex.create(ps_plex, plex));
   if (!via_cell_sets) PetscCall(plex.paint_boxes(0, boxes, mat_id_plex));
   else {
      const PetscInt label_value = 7;
      const std::vector<PetscReal> &cen = plex.centroid_host();
      std::map<PetscInt, PetscInt> cell_sets;
      cell_sets[label_value] = 1;
      PetscCall(DMCreateLabel(plex.dm(), "Cell Sets"));
      for (PetscInt c = 0; c < ps_plex.local_cells; c++) {
         for (const MaterialBox2D &box : boxes) {
            if (cen[3 * c] >= box.x0 && cen[3 * c] <= box.x1 && cen[3 * c + 1] >= box.y0 && cen[3 * c + 1] <= box.y1) {
               PetscCall(DMSetLabelValue(plex.dm(), "Cell Sets", plex.cell_point_host()[c], label_value));
            }
         }
      }
      PetscCall(plex.paint_cell_sets(0, cell_sets, mat_id_plex));
   }

   // The FD local cell order is the DMDA patch's own lexicographic one
   key_fd.resize(ps_fd.local_cells);
   for (PetscInt c = 0; c < ps_fd.local_cells; c++) {
      const PetscInt i = fd.cell_start_x() + c % fd.local_cells_x();
      const PetscInt j = fd.cell_start_y() + c / fd.local_cells_x();
      key_fd[c] = j * nx + i;
   }
   const PetscInt n[3] = {nx, ny, 1};
   const PetscReal h[3] = {lx / nx, ly / ny, 1.0};
   PlexKeys(plex, 2, n, h, key_plex);

   PetscCall(PetscSNPrintf(desc, sizeof(desc), "2D quads %" PetscInt_FMT " x %" PetscInt_FMT ", S%" PetscInt_FMT \
      " (%" PetscInt_FMT " angles), %s", nx, ny, sn_order, n_angles, bc_desc));
   PetscCall(CheckGeometry(desc, plex, lx * ly, nx * ny, ok));
   PetscCall(CompareTwins(desc, nx * ny, n_angles, ps_fd, fd, streaming_fd, mat_id_fd, key_fd, \
      ps_plex, plex, streaming_plex, mat_id_plex, key_plex, quad, mats, ok));

   PetscCall(fd.destroy());
   PetscCall(plex.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 2: one 3D case against StructuredFD3D
static PetscErrorCode CheckTwin3D(PetscInt nx, PetscInt ny, PetscInt nz, PetscInt sn_order, const BCSpec &bcs, \
   const char *bc_desc, PetscBool *ok)
{
   const PetscReal lx = 1.0, ly = 2.0, lz = 1.5;
   SNQuadrature3D quad;
   MaterialSpec mats;
   PhaseSpace ps_fd, ps_plex;
   StructuredFD3D fd;
   UnstructuredDG0 plex;
   StreamingTerm3D streaming_fd;
   StreamingTermDG0 streaming_plex;
   PetscIntKokkosView mat_id_fd, mat_id_plex;
   std::vector<PetscInt> key_fd, key_plex;
   const std::vector<MaterialBox3D> no_boxes;
   char desc[256];

   PetscFunctionBeginUser;

   PetscCall(quad.create(sn_order));
   PetscCall(TwinMaterials(mats));
   const PetscInt n_angles = quad.n_angles();

   PetscCall(ps_fd.create(PETSC_COMM_WORLD, nx * ny * nz, n_angles));
   PetscCall(fd.create(PETSC_COMM_WORLD, ps_fd, nx, ny, nz, lx, ly, lz, quad, bcs));
   PetscCall(streaming_fd.create(ps_fd, fd, quad));
   PetscCall(fd.paint_boxes(0, no_boxes, mat_id_fd));

   PlexMeshSpec mesh;
   mesh.dimension = 3;
   mesh.n_cells[0] = nx;
   mesh.n_cells[1] = ny;
   mesh.n_cells[2] = nz;
   mesh.lengths[0] = lx;
   mesh.lengths[1] = ly;
   mesh.lengths[2] = lz;
   PetscCall(plex.create_mesh(PETSC_COMM_WORLD, mesh));
   PetscCall(ps_plex.create(PETSC_COMM_WORLD, plex.n_global_cells(), n_angles));
   PetscCall(plex.create(ps_plex, quad, bcs));
   PetscCall(streaming_plex.create(ps_plex, plex));
   PetscCall(plex.paint_boxes(0, no_boxes, mat_id_plex));

   key_fd.resize(ps_fd.local_cells);
   for (PetscInt c = 0; c < ps_fd.local_cells; c++) {
      const PetscInt xm = fd.local_cells_x(), ym = fd.local_cells_y();
      const PetscInt i = fd.cell_start_x() + c % xm;
      const PetscInt j = fd.cell_start_y() + (c / xm) % ym;
      const PetscInt k = fd.cell_start_z() + c / (xm * ym);
      key_fd[c] = (k * ny + j) * nx + i;
   }
   const PetscInt n[3] = {nx, ny, nz};
   const PetscReal h[3] = {lx / nx, ly / ny, lz / nz};
   PlexKeys(plex, 3, n, h, key_plex);

   PetscCall(PetscSNPrintf(desc, sizeof(desc), "3D hexes %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT \
      ", S%" PetscInt_FMT " (%" PetscInt_FMT " angles), %s", nx, ny, nz, sn_order, n_angles, bc_desc));
   PetscCall(CheckGeometry(desc, plex, lx * ly * lz, nx * ny * nz, ok));
   PetscCall(CompareTwins(desc, nx * ny * nz, n_angles, ps_fd, fd, streaming_fd, mat_id_fd, key_fd, \
      ps_plex, plex, streaming_plex, mat_id_plex, key_plex, quad, mats, ok));

   PetscCall(fd.destroy());
   PetscCall(plex.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// What a plex solve reports back to the simplex checks
struct PlexSolveResult {
   PetscBool converged = PETSC_FALSE;
   PetscReal psi_min = 0.0;
   PetscReal psi_max = 0.0;
   // sum_c V_c phi_c, the scalar flux integrated over the domain
   PetscReal integral = 0.0;
   // sum_c V_c q_c, the total source
   PetscReal source_integral = 0.0;
   PetscInt n_cells = 0;
};

// Is the .vtu PETSc wrote a file with exactly one entry per global cell? Its
// header gives a <Piece NumberOfCells="..."> per rank that wrote; they must
// sum to the mesh's cell count - an overlap cell written by two ranks would
// push it over. Read on rank 0 after a barrier, deleted afterwards, the result
// broadcast so every rank agrees
static PetscErrorCode CheckVTU(const char *filename, PetscInt expected_cells, PetscBool *ok)
{
   PetscMPIInt rank;
   PetscInt total = 0, n_pieces = 0;
   PetscInt found = 0;

   PetscFunctionBeginUser;

   PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
   PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
   if (rank == 0) {
      FILE *fp = fopen(filename, "rb");
      if (fp) {
         found = 1;
         std::string contents;
         char buf[4096];
         size_t n_read = 0;
         while ((n_read = fread(buf, 1, sizeof(buf), fp)) > 0) contents.append(buf, n_read);
         fclose(fp);
         const std::string tag = "NumberOfCells=\"";
         for (size_t pos = contents.find(tag); pos != std::string::npos; pos = contents.find(tag, pos + 1)) {
            total += (PetscInt)std::stoll(contents.substr(pos + tag.size(), 32));
            n_pieces++;
         }
         remove(filename);
      }
   }
   PetscCallMPI(MPI_Bcast(&found, 1, MPIU_INT, 0, PETSC_COMM_WORLD));
   PetscCallMPI(MPI_Bcast(&total, 1, MPIU_INT, 0, PETSC_COMM_WORLD));
   PetscCallMPI(MPI_Bcast(&n_pieces, 1, MPIU_INT, 0, PETSC_COMM_WORLD));

   const PetscBool pass = (PetscBool)(found && total == expected_cells);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  vtu output: %s %s, %" PetscInt_FMT " pieces with %" PetscInt_FMT \
      " cells in total (expected %" PetscInt_FMT ")%s\n", filename, found ? "written" : "NOT written", n_pieces, total, \
      expected_cells, pass ? "" : " FAILED"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Check 4, the box's "Face Sets": every face carrying id v lies on the box
// face the structured backends' FACE_* constant v names - PETSc's box
// convention, which simplex boxes get from a different code path
// (DMPlexSetBoxLabel_Internal) than tensor ones - and every id is used
static PetscErrorCode CheckFaceSets(const char *desc, const UnstructuredDG0 &disc, const PlexMeshSpec &mesh, PetscBool *ok)
{
   const PetscInt dim = mesh.dimension, n_ids = 2 * dim;
   // (axis, 0 = lower / 1 = upper) of each id, 1-based
   const PetscInt axis_2d[4][2] = {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
   const PetscInt axis_3d[6][2] = {{2, 0}, {2, 1}, {1, 0}, {1, 1}, {0, 1}, {0, 0}};
   PetscInt n_wrong = 0, n_global_wrong = 0;
   std::vector<PetscInt> count(n_ids, 0), global_count(n_ids, 0);

   PetscFunctionBeginUser;

   for (PetscInt v = 1; v <= n_ids; v++) {
      IS faces = NULL;
      const PetscInt *f = nullptr;
      PetscInt n = 0;
      const PetscInt axis = (dim == 2) ? axis_2d[v - 1][0] : axis_3d[v - 1][0];
      const PetscReal plane = ((dim == 2) ? axis_2d[v - 1][1] : axis_3d[v - 1][1]) ? mesh.lengths[axis] : 0.0;
      PetscCall(DMGetStratumIS(disc.dm(), "Face Sets", v, &faces));
      if (!faces) continue;
      PetscCall(ISGetLocalSize(faces, &n));
      PetscCall(ISGetIndices(faces, &f));
      for (PetscInt i = 0; i < n; i++) {
         PetscReal area = 0.0, cen[3] = {0.0, 0.0, 0.0}, normal[3] = {0.0, 0.0, 0.0};
         PetscCall(DMPlexComputeCellGeometryFVM(disc.dm(), f[i], &area, cen, normal));
         if (PetscAbsReal(cen[axis] - plane) > 1e-12) n_wrong++;
      }
      count[v - 1] = n;
      PetscCall(ISRestoreIndices(faces, &f));
      PetscCall(ISDestroy(&faces));
   }
   PetscCallMPI(MPI_Allreduce(&n_wrong, &n_global_wrong, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
   PetscCallMPI(MPI_Allreduce(count.data(), global_count.data(), (PetscMPIInt)n_ids, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

   PetscBool all_used = PETSC_TRUE;
   for (PetscInt v = 0; v < n_ids; v++) all_used = (PetscBool)(all_used && global_count[v] > 0);
   const PetscBool pass = (PetscBool)(n_global_wrong == 0 && all_used);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  face sets, %s: ids 1..%" PetscInt_FMT " %s, %" PetscInt_FMT \
      " faces off their box face (exact)%s\n", desc, n_ids, all_used ? "all present" : "NOT all present", \
      n_global_wrong, pass ? "" : " FAILED"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Paint material 1 over the central box [L/4, 3L/4] on every axis
static PetscErrorCode PaintCentral(const UnstructuredDG0 &disc, const PlexMeshSpec &mesh, PetscIntKokkosView &mat_id_d)
{
   PetscFunctionBeginUser;

   if (mesh.dimension == 2) {
      std::vector<MaterialBox2D> boxes(1);
      boxes[0].x0 = 0.25 * mesh.lengths[0];
      boxes[0].x1 = 0.75 * mesh.lengths[0];
      boxes[0].y0 = 0.25 * mesh.lengths[1];
      boxes[0].y1 = 0.75 * mesh.lengths[1];
      boxes[0].material = 1;
      PetscCall(disc.paint_boxes_over(boxes, mat_id_d));
   } else {
      std::vector<MaterialBox3D> boxes(1);
      boxes[0].x0 = 0.25 * mesh.lengths[0];
      boxes[0].x1 = 0.75 * mesh.lengths[0];
      boxes[0].y0 = 0.25 * mesh.lengths[1];
      boxes[0].y1 = 0.75 * mesh.lengths[1];
      boxes[0].z0 = 0.25 * mesh.lengths[2];
      boxes[0].z1 = 0.75 * mesh.lengths[2];
      boxes[0].material = 1;
      PetscCall(disc.paint_boxes_over(boxes, mat_id_d));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// One single-group solve on a plex box: geometry check, solve, the solution's
// range, its integrated scalar flux and the total source, and optionally the
// .vtu written from it (check 6). One cross section everywhere; the source is
// everywhere, or with central_source only in the central box
template <class Quad>
static PetscErrorCode SolveOnPlex(const char *desc, const PlexMeshSpec &mesh, PetscInt sn_order, const BCSpec &bcs, \
   PetscScalar sigma_t, PetscScalar sigma_s, PetscScalar source, PetscBool central_source, PetscInt expected_cells, \
   const char *vtu, PlexSolveResult *result, PetscBool *ok)
{
   Quad quad;
   MaterialSpec mats;
   PhaseSpace ps;
   UnstructuredDG0 disc;
   StreamingTermDG0 streaming;
   PetscIntKokkosView mat_id_d;
   Solve s;
   Vec b = NULL, psi = NULL;
   std::map<PetscInt, PetscInt> no_cell_sets;

   PetscFunctionBeginUser;

   PetscCall(quad.create(sn_order));
   // Material 1 is material 0 with the source switched on - painted over the
   // whole box, or only over its central half-width box
   PetscCall(mats.create(2, 1));
   for (PetscInt m = 0; m < 2; m++) {
      PetscCall(mats.set_sigma_t(m, 0, sigma_t));
      PetscCall(mats.set_sigma_s(m, 0, 0, sigma_s));
   }
   PetscCall(mats.set_source(1, 0, source));

   PetscCall(disc.create_mesh(PETSC_COMM_WORLD, mesh));
   PetscCall(ps.create(PETSC_COMM_WORLD, disc.n_global_cells(), quad.n_angles()));
   PetscCall(disc.create(ps, quad, bcs));
   PetscCall(streaming.create(ps, disc));
   // No cell sets on a generated box - the background everywhere, which is
   // also what exercises paint_cell_sets' empty-map path
   PetscCall(disc.paint_cell_sets(central_source ? 0 : 1, no_cell_sets, mat_id_d));
   if (central_source) PetscCall(PaintCentral(disc, mesh, mat_id_d));

   PetscReal box_volume = 1.0;
   for (PetscInt d = 0; d < mesh.dimension; d++) box_volume *= mesh.lengths[d];
   PetscCall(CheckGeometry(desc, disc, box_volume, expected_cells, ok));
   PetscCall(CheckFaceSets(desc, disc, mesh, ok));

   PetscCall(BuildSolve(ps, disc, streaming, quad, mats, mat_id_d, s));
   PetscCall(MatCreateVecs(s.op.assembled_mat(), &psi, &b));
   PetscCall(FillRhs(ps, disc, quad, mats, mat_id_d, b));
   PetscCall(SolveTight(s, b, psi, &result->converged));

   PetscCall(VecMin(psi, NULL, &result->psi_min));
   PetscCall(VecMax(psi, NULL, &result->psi_max));
   {
      PetscScalar2DKokkosView phi_d("phi_d", ps.local_cells, 1);
      PetscCall(UboltAngularIntegral(psi, ps.n_angles, quad.w_d(), phi_d));
      auto phi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi_d);
      PetscReal local = 0.0;
      for (PetscInt c = 0; c < ps.local_cells; c++) local += disc.volume_host()[c] * PetscRealPart(phi_h(c, 0));
      PetscCallMPI(MPI_Allreduce(&local, &result->integral, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
   }
   result->n_cells = disc.n_global_cells();
   {
      auto mat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat_id_d);
      PetscReal local = 0.0;
      for (PetscInt c = 0; c < ps.local_cells; c++) local += (mat_h(c) == 1) ? disc.volume_host()[c] * PetscRealPart(source) : 0.0;
      PetscCallMPI(MPI_Allreduce(&local, &result->source_integral, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
   }

   if (vtu) {
      UboltCellField sigma_t_field = {"sigma_t", s.xs.sigma_t(0)};
      PetscCall(UboltWriteScalarFluxVTK(ps, disc, quad, psi, 1, &sigma_t_field, vtu));
      PetscCall(CheckVTU(vtu, disc.n_global_cells(), ok));
   }

   PetscCall(VecDestroy(&b));
   PetscCall(VecDestroy(&psi));
   PetscCall(DestroySolve(s));
   PetscCall(disc.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 3: simplex meshes, which have no structured twin
//  (a) infinite medium - every face reflective, so the exact discrete
//      solution is flat: psi = q / (sum_weights (sigma_t - sigma_s)). That
//      pins reflection on the axis-aligned faces of triangles/tets and the
//      scatter reading the BC rows
//  (b) vacuum, a pure absorber, and a source only in the central box
//      [1/4, 3/4]^dim - converged, inside the exact discrete bounds
//      0 <= psi <= q / (sum_weights sigma_t) (DG0 upwind is monotone and the
//      infinite medium is an upper solution), and the scalar flux integrated
//      over the domain per unit source within 25% of the quad/hex mesh's at
//      the same nominal h. Catches a sign or scaling error in the streaming
//      geometry without pretending two first-order schemes on different meshes
//      agree closely
//
// Why (b) is not a uniform source with scattering: the vacuum BC here is the
// Dirichlet-CELL one, which zeroes the incoming flux over the whole of every
// boundary cell - an O(h) error whose constant is the boundary cells' volume,
// and a hex box has far more of that than a tet box at the same h (measured:
// tets 1.8x hexes at 3^3, still 1.25x at 12^3). With the source away from the
// boundary and no scattering, an incoming ray to a boundary cell's centre
// crosses nothing but that cell, so zero IS its exact value and the comparison
// measures the interior discretisation instead (3-5% at these sizes). The
// source is normalised out because centroid painting gives the two meshes
// different source volumes
template <class Quad>
static PetscErrorCode CheckSimplex(PetscInt dim, PetscInt n, PetscInt sn_order_inf, PetscInt sn_order_vac, \
   const char *vtu, PetscBool *ok)
{
   const PetscInt n_faces = 2 * dim;
   const PetscReal inf_tol = 1e-10, bound_tol = 1e-12, integral_tol = 0.25;
   PlexMeshSpec simplex, tensor;
   BCSpec reflect_all, vacuum_all;
   PlexSolveResult inf, vac_simplex, vac_tensor;
   char desc[256];
   const char *shape = (dim == 2) ? "triangles" : "tets";

   PetscFunctionBeginUser;

   simplex.dimension = dim;
   simplex.simplex = PETSC_TRUE;
   for (PetscInt d = 0; d < dim; d++) {
      simplex.n_cells[d] = n;
      simplex.lengths[d] = 1.0;
   }
   tensor = simplex;
   tensor.simplex = PETSC_FALSE;
   // Face Sets ids run 1 .. 2 * dim in PETSc's box convention either way
   for (PetscInt f = 1; f <= n_faces; f++) reflect_all.set(f, BCType::REFLECT);

   // (a)
   {
      Quad quad;
      PetscCall(quad.create(sn_order_inf));
      const PetscReal exact = 1.0 / (PetscRealPart(quad.sum_weights()) * (2.0 - 1.0));
      PetscCall(PetscSNPrintf(desc, sizeof(desc), "%" PetscInt_FMT "D %s %" PetscInt_FMT "^%" PetscInt_FMT \
         " box, S%" PetscInt_FMT ", all reflective", dim, shape, n, dim, sn_order_inf));
      PetscCall(SolveOnPlex<Quad>(desc, simplex, sn_order_inf, reflect_all, 2.0, 1.0, 1.0, PETSC_FALSE, -1, NULL, \
         &inf, ok));
      const PetscReal err = PetscMax(PetscAbsReal(inf.psi_max - exact), PetscAbsReal(inf.psi_min - exact));
      const PetscBool pass = (PetscBool)(inf.converged && err <= inf_tol);
      if (!pass) *ok = PETSC_FALSE;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  infinite medium, %s: psi in [%.12f, %.12f] against %.12f, " \
         "max error %.3e (tol %.0e)%s%s\n", desc, (double)inf.psi_min, (double)inf.psi_max, (double)exact, \
         (double)err, (double)inf_tol, inf.converged ? "" : " - did NOT converge", pass ? "" : " FAILED"));
   }

   // (b)
   {
      Quad quad;
      PetscCall(quad.create(sn_order_vac));
      const PetscReal sigma_t = 1.0, source = 1.0;
      const PetscReal upper = source / (PetscRealPart(quad.sum_weights()) * sigma_t);

      PetscCall(PetscSNPrintf(desc, sizeof(desc), "%" PetscInt_FMT "D %s %" PetscInt_FMT "^%" PetscInt_FMT \
         " box, S%" PetscInt_FMT, dim, shape, n, dim, sn_order_vac));
      PetscCall(SolveOnPlex<Quad>(desc, simplex, sn_order_vac, vacuum_all, sigma_t, 0.0, source, PETSC_TRUE, -1, \
         vtu, &vac_simplex, ok));
      char desc_tensor[256];
      PetscInt n_tensor = 1;
      for (PetscInt d = 0; d < dim; d++) n_tensor *= n;
      PetscCall(PetscSNPrintf(desc_tensor, sizeof(desc_tensor), "%" PetscInt_FMT "D %s %" PetscInt_FMT "^%" \
         PetscInt_FMT " box, S%" PetscInt_FMT, dim, dim == 2 ? "quads" : "hexes", n, dim, sn_order_vac));
      PetscCall(SolveOnPlex<Quad>(desc_tensor, tensor, sn_order_vac, vacuum_all, sigma_t, 0.0, source, PETSC_TRUE, \
         n_tensor, NULL, &vac_tensor, ok));

      const PetscReal per_source_simplex = vac_simplex.integral / vac_simplex.source_integral;
      const PetscReal per_source_tensor = vac_tensor.integral / vac_tensor.source_integral;
      const PetscReal rel = PetscAbsReal(per_source_simplex - per_source_tensor) / per_source_tensor;
      const PetscBool converged = (PetscBool)(vac_simplex.converged && vac_tensor.converged);
      const PetscBool bounded = (PetscBool)(vac_simplex.psi_min >= -bound_tol && vac_tensor.psi_min >= -bound_tol && \
         vac_simplex.psi_max <= upper + bound_tol && vac_tensor.psi_max <= upper + bound_tol);
      const PetscBool pass = (PetscBool)(converged && bounded && rel <= integral_tol);
      if (!pass) *ok = PETSC_FALSE;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  vacuum, central source, %s (%" PetscInt_FMT " cells) against %s:%s\n", \
         desc, vac_simplex.n_cells, desc_tensor, pass ? "" : " FAILED"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "    psi in [%.3e, %.3e] and [%.3e, %.3e], bounds [0, %.6f] (tol %.0e); " \
         "integrated scalar flux per unit source %.6f against %.6f, relative difference %.3f (tol %.2f)%s\n", \
         (double)vac_simplex.psi_min, (double)vac_simplex.psi_max, (double)vac_tensor.psi_min, \
         (double)vac_tensor.psi_max, (double)upper, (double)bound_tol, (double)per_source_simplex, \
         (double)per_source_tensor, (double)rel, (double)integral_tol, converged ? "" : " - did NOT converge"));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 5: configurations the backend must refuse, with an error rather than
// a wrong matrix (or a hang - the reflective one is found per rank)
static PetscErrorCode CheckErrorPaths(PetscBool *ok)
{
   PetscInt n_cases = 0, n_rejected = 0;

   PetscFunctionBeginUser;

   // The rejections below print nothing and come back as error codes
   PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));

   // Reflective on BOTH x faces of a single-cell-wide box: every reflection
   // partner comes in through the opposite face, so it is a BC row itself
   {
      SNQuadrature2D quad;
      PhaseSpace ps;
      UnstructuredDG0 disc;
      BCSpec bcs;
      PlexMeshSpec mesh;
      mesh.dimension = 2;
      mesh.n_cells[0] = 1;
      mesh.n_cells[1] = 4;
      mesh.lengths[0] = 1.0;
      mesh.lengths[1] = 4.0;
      bcs.set(StructuredFD2D::FACE_LEFT, BCType::REFLECT);
      bcs.set(StructuredFD2D::FACE_RIGHT, BCType::REFLECT);

      n_cases++;
      if (quad.create(2) || disc.create_mesh(PETSC_COMM_WORLD, mesh) || \
          ps.create(PETSC_COMM_WORLD, disc.n_global_cells(), quad.n_angles())) {
         *ok = PETSC_FALSE;
      } else if (disc.create(ps, quad, bcs)) n_rejected++;
      if (disc.destroy()) *ok = PETSC_FALSE;
   }

   // Materials for "Cell Sets" on a mesh that has no such label (a generated
   // box) - an error, not a silently unpainted mesh
   {
      SNQuadrature2D quad;
      PhaseSpace ps;
      UnstructuredDG0 disc;
      PlexMeshSpec mesh;
      PetscIntKokkosView mat_id_d;
      std::map<PetscInt, PetscInt> cell_sets;
      cell_sets[1] = 0;
      mesh.dimension = 2;
      for (PetscInt d = 0; d < 2; d++) {
         mesh.n_cells[d] = 2;
         mesh.lengths[d] = 1.0;
      }

      n_cases++;
      if (quad.create(2) || disc.create_mesh(PETSC_COMM_WORLD, mesh) || \
          ps.create(PETSC_COMM_WORLD, disc.n_global_cells(), quad.n_angles()) || disc.create(ps, quad)) {
         *ok = PETSC_FALSE;
      } else if (disc.paint_cell_sets(0, cell_sets, mat_id_d)) n_rejected++;
      if (disc.destroy()) *ok = PETSC_FALSE;
   }

   // A 2D quadrature handed to a 3D mesh, and a .vts name on the plex (with a
   // valid 3D setup around it, so the extension is the only thing wrong)
   {
      SNQuadrature2D quad_2d;
      SNQuadrature3D quad_3d;
      PhaseSpace ps;
      UnstructuredDG0 disc;
      PlexMeshSpec mesh;
      Mat A = NULL;
      Vec psi = NULL;
      mesh.dimension = 3;
      for (PetscInt d = 0; d < 3; d++) {
         mesh.n_cells[d] = 2;
         mesh.lengths[d] = 1.0;
      }

      PetscBool setup_failed = PETSC_FALSE;
      if (quad_2d.create(2) || quad_3d.create(2) || disc.create_mesh(PETSC_COMM_WORLD, mesh)) setup_failed = PETSC_TRUE;

      n_cases++;
      if (!setup_failed) {
         PhaseSpace ps_2d;
         if (ps_2d.create(PETSC_COMM_WORLD, disc.n_global_cells(), quad_2d.n_angles())) setup_failed = PETSC_TRUE;
         else if (disc.create(ps_2d, quad_2d)) n_rejected++;
      }

      n_cases++;
      if (!setup_failed) {
         if (ps.create(PETSC_COMM_WORLD, disc.n_global_cells(), quad_3d.n_angles()) || disc.create(ps, quad_3d) || \
             disc.create_matrix(&A) || MatCreateVecs(A, &psi, NULL)) setup_failed = PETSC_TRUE;
         else if (UboltWriteScalarFluxVTK(ps, disc, quad_3d, psi, 0, NULL, "verify_plexk_never_written.vts")) n_rejected++;
      }
      if (setup_failed) *ok = PETSC_FALSE;
      if (VecDestroy(&psi) || MatDestroy(&A) || disc.destroy()) *ok = PETSC_FALSE;
   }

   PetscCall(PetscPopErrorHandler());

   if (n_rejected != n_cases) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  error paths (reflection partner on a BC row, cell sets without the label, 2D quadrature on a 3D " \
      "mesh, .vts on the plex): %" \
      PetscInt_FMT " of %" PetscInt_FMT " rejected\n", n_rejected, n_cases));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   PetscBool ok = PETSC_TRUE;
   PetscMPIInt size;
   char vtu[PETSC_MAX_PATH_LEN];

   PetscFunctionBeginUser;

   PetscCall(PetscInitialize(&argc, &args, (char*)0, NULL));
   // Register the pflare types
   PCRegister_PFLARE();
   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

   // No block scope here, unlike the solve drivers: every device view lives
   // and dies inside the Check* functions, well before PetscFinalize
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Unstructured DG0 verification on %d rank(s)\n", (int)size));

   const BCType V = BCType::VACUUM, R = BCType::REFLECT;

   // ~~~~~~~~~~
   // 1. FD twin, 2D quads
   // ~~~~~~~~~~
   {
      const std::vector<MaterialBox2D> no_boxes;
      BCSpec vacuum;
      PetscCall(CheckTwin2D(4, 3, 2, vacuum, no_boxes, PETSC_FALSE, "vacuum", &ok));
      PetscCall(CheckTwin2D(4, 3, 4, vacuum, no_boxes, PETSC_FALSE, "vacuum", &ok));

      BCSpec reflect_lb;
      reflect_lb.set(StructuredFD2D::FACE_LEFT, R);
      reflect_lb.set(StructuredFD2D::FACE_BOTTOM, R);
      PetscCall(CheckTwin2D(5, 4, 2, reflect_lb, no_boxes, PETSC_FALSE, "reflect left + bottom", &ok));

      BCSpec mixed;
      mixed.set(StructuredFD2D::FACE_LEFT, R);
      mixed.set(StructuredFD2D::FACE_TOP, R);
      mixed.set(StructuredFD2D::FACE_RIGHT, V);
      mixed.set(StructuredFD2D::FACE_BOTTOM, V);
      PetscCall(CheckTwin2D(5, 4, 4, mixed, no_boxes, PETSC_FALSE, "reflect left + top, vacuum right + bottom", &ok));

      // Both faces of the corner cell at the origin let inflow in: the left
      // face wins (x before y) and its window excludes that cell's centre, so
      // the corner rows take 0 rather than the bottom's 0.3
      BCSpec inflow;
      const PetscReal window[2] = {0.5, 1.5};
      inflow.set_inflow(StructuredFD2D::FACE_LEFT, 1.0);
      inflow.set_window(StructuredFD2D::FACE_LEFT, 1, window);
      inflow.set_inflow(StructuredFD2D::FACE_BOTTOM, 0.3);
      PetscCall(CheckTwin2D(6, 4, 2, inflow, no_boxes, PETSC_FALSE, "inflow 1.0 on left in y [0.5, 1.5], 0.3 on bottom", \
         &ok));

      std::vector<MaterialBox2D> boxes(1);
      boxes[0].x0 = 0.3;
      boxes[0].x1 = 0.8;
      boxes[0].y0 = 0.4;
      boxes[0].y1 = 1.2;
      boxes[0].material = 1;
      PetscCall(CheckTwin2D(4, 3, 4, vacuum, boxes, PETSC_FALSE, "vacuum, painted box", &ok));
      PetscCall(CheckTwin2D(4, 3, 4, vacuum, boxes, PETSC_TRUE, "vacuum, the same box as a \"Cell Sets\" label", &ok));
   }

   // ~~~~~~~~~~
   // 2. FD twin, 3D hexes. CAREFUL with face names: in 3D bottom/top are the
   // Z faces (PETSc's box convention), not y as in 2D
   // ~~~~~~~~~~
   {
      BCSpec vacuum;
      PetscCall(CheckTwin3D(3, 2, 2, 2, vacuum, "vacuum", &ok));

      BCSpec corner;
      corner.set(StructuredFD3D::FACE_LEFT, R);
      corner.set(StructuredFD3D::FACE_FRONT, R);
      corner.set(StructuredFD3D::FACE_BOTTOM, R);
      PetscCall(CheckTwin3D(3, 3, 2, 4, corner, "reflect left + front + bottom", &ok));

      BCSpec inflow;
      const PetscReal window[4] = {0.0, 1.0, 0.5, 1.5};
      inflow.set_inflow(StructuredFD3D::FACE_LEFT, 1.0);
      inflow.set_window(StructuredFD3D::FACE_LEFT, 2, window);
      PetscCall(CheckTwin3D(4, 2, 3, 2, inflow, "inflow 1.0 on left in y [0, 1] x z [0.5, 1.5]", &ok));
   }

   // ~~~~~~~~~~
   // 3 + 6. Simplex meshes, and the .vtu written from the 2D vacuum solve
   // ~~~~~~~~~~
   PetscCall(PetscSNPrintf(vtu, sizeof(vtu), "verify_plexk_tmp_%d.vtu", (int)size));
   PetscCall(CheckSimplex<SNQuadrature2D>(2, 6, 4, 4, vtu, &ok));
#if defined(PETSC_HAVE_CTETGEN) || defined(PETSC_HAVE_TETGEN)
   PetscCall(CheckSimplex<SNQuadrature3D>(3, 3, 2, 2, NULL, &ok));
#else
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  3D tets skipped: PETSc was configured without a tet mesher " \
      "(--download-ctetgen)\n"));
#endif

   // ~~~~~~~~~~
   // 5. Error paths
   // ~~~~~~~~~~
   PetscCall(CheckErrorPaths(&ok));

   if (!ok) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "Unstructured DG0 verification FAILED\n"));

   PetscCall(PetscFinalize());
   return ok ? 0 : 1;
}
