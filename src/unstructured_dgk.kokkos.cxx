#include "ubolt/unstructured_dg.hpp"
#include "petsc_kokkos.hpp"
#include <petscdmplex.h>
#include <petscsf.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Which height-0 points of the local mesh this rank OWNS: everything that is
// not a leaf of the point SF. After a distribution with overlap the local mesh
// also holds the neighbouring ranks' cells across the partition boundary - the
// ghosts the upwind columns point into - and those are the SF's leaves. An SF
// with no graph (one rank, never distributed) has no leaves at all
static PetscErrorCode MarkOwnedCells(DM dm, PetscInt c_start, PetscInt c_end, std::vector<PetscInt> &owned)
{
   PetscSF sf = NULL;
   PetscInt n_roots = 0, n_leaves = 0;
   const PetscInt *leaves = nullptr;

   PetscFunctionBeginUser;

   owned.assign(c_end - c_start, 1);
   PetscCall(DMGetPointSF(dm, &sf));
   PetscCall(PetscSFGetGraph(sf, &n_roots, &n_leaves, &leaves, NULL));
   if (n_roots >= 0) {
      for (PetscInt l = 0; l < n_leaves; l++) {
         const PetscInt p = leaves ? leaves[l] : l;
         if (p >= c_start && p < c_end) owned[p - c_start] = 0;
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build (or read) the mesh and distribute it with a one-cell, face-adjacent
// overlap
//
// As in the DMDA backends we deliberately do NOT call DMSetFromOptions: it
// would expose -dm_plex_box_faces, -dm_refine and friends, any of which could
// resize the mesh out from under the PhaseSpace that every other object is
// about to be sized from
PetscErrorCode UnstructuredDG::create_mesh(MPI_Comm comm, const PlexMeshSpec &mesh)
{
   DM dm = NULL, dm_dist = NULL;
   PetscPartitioner part = NULL;
   PetscInt dm_dim = 0, n_owned = 0;
   std::vector<PetscInt> owned;

   PetscFunctionBeginUser;

   PetscCheck(!dm_, comm, PETSC_ERR_ARG_WRONGSTATE, "create_mesh has already built this backend's mesh");
   PetscCheck(mesh.dimension == 2 || mesh.dimension == 3, comm, PETSC_ERR_ARG_OUTOFRANGE, \
      "the unstructured backend is 2D or 3D, was asked for dimension %" PetscInt_FMT \
      " (the FD slab is the 1D backend)", mesh.dimension);

   comm_ = comm;

   if (!mesh.file.empty()) {

      // Interpolated, so the faces exist as points - the DG0 fluxes live on
      // them. Gmsh files come with "Cell Sets" and "Face Sets" labels from
      // their physical groups
      PetscCall(DMPlexCreateFromFile(comm, mesh.file.c_str(), "ubolt_mesh", PETSC_TRUE, &dm));

   } else {

      for (PetscInt d = 0; d < mesh.dimension; d++) {
         PetscCheck(mesh.n_cells[d] > 0 && mesh.lengths[d] > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
            "box axis %" PetscInt_FMT " needs a positive cell count and length, was given %" \
            PetscInt_FMT " cells over %g", d, mesh.n_cells[d], (double)mesh.lengths[d]);
      }
      // A simplex box is a generated mesh: PETSc triangulates the box
      // surface with an external mesher, so it has to have been configured in
      if (mesh.simplex && mesh.dimension == 2) {
#if !defined(PETSC_HAVE_TRIANGLE)
         SETERRQ(comm, PETSC_ERR_SUP, "a 2D simplex box mesh needs PETSc configured with " \
            "--download-triangle");
#endif
      }
      if (mesh.simplex && mesh.dimension == 3) {
#if !defined(PETSC_HAVE_CTETGEN) && !defined(PETSC_HAVE_TETGEN)
         SETERRQ(comm, PETSC_ERR_SUP, "a 3D simplex box mesh needs PETSc configured with " \
            "--download-ctetgen");
#endif
      }

      // Interpolated, which is also what gives the box its "Face Sets" label
      // with PETSc's box ids (DMPlexSetBoxLabel_Internal in plexcreate.c) -
      // for simplex and tensor cells alike
      const PetscReal lower[3] = {0.0, 0.0, 0.0};
      const DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
      PetscCall(DMPlexCreateBoxMesh(comm, mesh.dimension, mesh.simplex, mesh.n_cells, lower, mesh.lengths, \
         periodicity, PETSC_TRUE, 0, PETSC_FALSE, &dm));
   }

   PetscCall(DMGetDimension(dm, &dm_dim));
   PetscCheck(dm_dim == mesh.dimension, comm, PETSC_ERR_ARG_INCOMP, \
      "the mesh is %" PetscInt_FMT "D but dimension %" PetscInt_FMT " was asked for", dm_dim, mesh.dimension);

   // Face adjacency (cone, not closure), set BEFORE distributing: the overlap
   // is then the face neighbours only - what the upwind flux reads and nothing
   // more. The default (closure) adjacency would also bring in every cell
   // sharing a vertex
   PetscCall(DMSetAdjacency(dm, PETSC_DEFAULT, PETSC_TRUE, PETSC_FALSE));

   // Simple rather than whatever PETSc would pick: deterministic across
   // machines and CI images (ParMETIS may be absent there) and so iteration
   // counts pinned in parallel stay put. -petscpartitioner_type still wins
   PetscCall(DMPlexGetPartitioner(dm, &part));
   PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERSIMPLE));
   PetscCall(PetscPartitionerSetFromOptions(part));

   // On one rank there is nothing to distribute and PETSc hands back NULL:
   // the serial mesh is the mesh
   PetscCall(DMPlexDistribute(dm, 1, NULL, &dm_dist));
   if (dm_dist) {
      PetscCall(DMDestroy(&dm));
      dm = dm_dist;
   }
   PetscCall(DMGetCoordinatesLocalSetUp(dm));
   PetscCall(DMViewFromOptions(dm, NULL, "-ubolt_dm_view"));

   dm_ = dm;
   dim_ = dm_dim;

   // The mesh decides the global cell count - count once, cached
   PetscCall(DMPlexGetHeightStratum(dm_, 0, &c_start_, &c_end_));
   PetscCall(MarkOwnedCells(dm_, c_start_, c_end_, owned));
   for (const PetscInt o : owned) n_owned += o;
   PetscCallMPI(MPI_Allreduce(&n_owned, &n_global_cells_, 1, MPIU_INT, MPI_SUM, comm_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Where a cell's first dof sits in the GLOBAL numbering. The global section
// encodes a point this rank does not own - an overlap ghost - as -(g + 1), so
// the one lookup serves owned cells and the ghost neighbours a column points
// into alike
static PetscErrorCode GlobalCellOffset(PetscSection gsec, PetscInt c, PetscInt *g_out, PetscBool *is_owned)
{
   PetscInt g = 0;

   PetscFunctionBeginUser;

   PetscCall(PetscSectionGetOffset(gsec, c, &g));
   *is_owned = (PetscBool)(g >= 0);
   *g_out = (g >= 0) ? g : -(g + 1);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check the DM's layout is the one everything downstream is written against -
// the sibling of the DMDA backends' CheckDALayout, asserted rather than
// assumed:
//   (i)   the global section agrees with the point SF about which cells are
//         owned, and the owned cells over all ranks are the phase space's,
//   (ii)  this rank owns owned cells x n_basis x n_angles rows,
//   (iii) walking the owned cells in increasing POINT order, their offsets
//         are rstart + k * n_basis * n_angles, k = 0, 1, ... - contiguous,
//         and (basis, angle) inside a cell, angle fastest.
// (iii) is what makes "local cell k = the k-th owned cell in point order" the
// indexing every per-cell view uses, and what RemovalTerm's r / rows_per_cell
// relies on. PETSc builds the global section in point order today; this is
// what stops a change there from surfacing as a wrong answer
static PetscErrorCode CheckPlexLayout(DM dm, PetscSection gsec, const PhaseSpace &ps, PetscInt c_start, \
   PetscInt c_end, const std::vector<PetscInt> &owned, PetscInt *rstart_out)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)dm);
   PetscInt rstart = 0, rend = 0, n_owned = 0, n_global = 0, k = 0;
   Vec gv = NULL;

   PetscFunctionBeginUser;

   // A transient global vector, only to read the ownership range - it never
   // reaches the solve (see docs/dev/kokkos.md)
   PetscCall(DMCreateGlobalVector(dm, &gv));
   PetscCall(VecGetOwnershipRange(gv, &rstart, &rend));
   PetscCall(VecDestroy(&gv));

   for (PetscInt c = c_start; c < c_end; c++) {
      PetscInt g = 0, dof = 0;
      PetscBool is_owned = PETSC_FALSE;
      PetscCall(GlobalCellOffset(gsec, c, &g, &is_owned));
      PetscCall(PetscSectionGetDof(gsec, c, &dof));
      // A ghost's dof is encoded as -(dof + 1) in a global section too
      if (dof < 0) dof = -(dof + 1);
      PetscCheck(dof == ps.rows_per_cell(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "cell %" PetscInt_FMT \
         " carries %" PetscInt_FMT " dof, not the phase space's %" PetscInt_FMT " basis x %" PetscInt_FMT \
         " angles", c, dof, ps.n_basis, ps.n_angles);
      PetscCheck(is_owned == (PetscBool)(owned[c - c_start] != 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, \
         "the global section and the point SF disagree about who owns cell %" PetscInt_FMT, c);
      if (!is_owned) continue;

      const PetscInt expected = rstart + k * ps.rows_per_cell();
      PetscCheck(g == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "owned cell %" PetscInt_FMT \
         " (local cell %" PetscInt_FMT ") starts at global row %" PetscInt_FMT ", not the " \
         "point-ordered angle-fastest %" PetscInt_FMT, c, k, g, expected);
      k++;
   }
   n_owned = k;

   PetscCheck(rend - rstart == n_owned * ps.rows_per_cell(), PETSC_COMM_SELF, PETSC_ERR_PLIB, \
      "the DM owns %" PetscInt_FMT " rows but %" PetscInt_FMT " owned cells x %" PetscInt_FMT \
      " rows per cell", rend - rstart, n_owned, ps.rows_per_cell());
   PetscCallMPI(MPI_Allreduce(&n_owned, &n_global, 1, MPIU_INT, MPI_SUM, comm));
   PetscCheck(n_global == ps.n_cells, comm, PETSC_ERR_PLIB, "the mesh has %" PetscInt_FMT \
      " cells, the phase space %" PetscInt_FMT, n_global, ps.n_cells);

   *rstart_out = rstart;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A boundary configuration create() cannot build (a reflective face that is
// not axis-aligned, a reflection partner that is itself a boundary row, a
// window of the wrong shape) is found by whichever rank owns the offending
// cell - but create() is collective, and a rank that returned early would
// leave the others waiting in the next collective call. So the failure is
// recorded, create() carries on to this point on every rank, and everybody
// errors together: the rank that found it with its message, the rest pointing
// at it
static PetscErrorCode CollectiveFailure(MPI_Comm comm, PetscBool local_fail, const char *message)
{
   PetscMPIInt local = local_fail ? 1 : 0, any = 0;

   PetscFunctionBeginUser;

   PetscCallMPI(MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, comm));
   PetscCheck(!local_fail, PETSC_COMM_SELF, PETSC_ERR_SUP, "%s", message);
   PetscCheck(!any, comm, PETSC_ERR_SUP, "the unstructured backend cannot build this boundary " \
      "configuration - another rank's error says why");

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The dominant axis of a face normal: the one its largest component is on,
// lowest axis on a tie. On a box every boundary face is axis-aligned and this
// is simply its axis
static inline PetscInt DominantAxis(const PetscScalar *n, PetscInt dim)
{
   PetscInt best = 0;
   for (PetscInt d = 1; d < dim; d++) {
      if (PetscAbsScalar(n[d]) > PetscAbsScalar(n[best])) best = d;
   }
   return best;
}

// Omega_a . nA_f - the upwind test. Written with the same terms in the same
// order as the kernels in StreamingTermDG0, so the host classification and the
// device fill agree on the sign; where they could not (|s| at rounding level on
// a face tangent to the ordinate) the coefficient concerned is itself at
// rounding level
static inline PetscScalar FaceFlux(const PetscScalar *omega, PetscInt a, const PetscScalar *nA, PetscInt k)
{
   return omega[3 * a] * nA[3 * k] + omega[3 * a + 1] * nA[3 * k + 1] + omega[3 * a + 2] * nA[3 * k + 2];
}

// Does a face whose centroid has these tangential coordinates let this face
// family's inflow in? Inclusive at both ends, one [lo, hi] pair per tangential
// axis in ascending axis order - the structured backends' test, with the face
// centroid standing in for the boundary cell's centre (they coincide on a
// quad/hex box)
static inline bool InWindow(const BCFace &face, const PetscReal *t, PetscInt n_t)
{
   if (face.n_window_pairs == 0) return true;
   for (PetscInt p = 0; p < n_t; p++) {
      if (t[p] < face.window[2 * p] || t[p] > face.window[2 * p + 1]) return false;
   }
   return true;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// DG1 geometry: exact moments of polytopes as fans of simplices
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The zeroth, first and second moments of a region, about a reference point:
// m0 = int 1, m1 = int y, m2 = int y y^T (row-major 3x3), y = x - reference
struct FanMoments {
   PetscReal m0 = 0.0;
   PetscReal m1[3] = {0.0, 0.0, 0.0};
   PetscReal m2[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

// Add one d-simplex (d + 1 points, relative to the reference) of the given
// measure: the exact moments of a uniform density on a simplex,
//   int y     = measure * mean_k y_k
//   int y y^T = measure / ((d + 1)(d + 2)) * (sum_k y_k y_k^T + (sum_k y_k)(sum_k y_k)^T)
static void AddSimplex(FanMoments &m, PetscInt d, PetscReal measure, const PetscReal y[][3])
{
   PetscReal sum[3] = {0.0, 0.0, 0.0};
   for (PetscInt k = 0; k <= d; k++) {
      for (PetscInt a = 0; a < 3; a++) sum[a] += y[k][a];
   }
   m.m0 += measure;
   for (PetscInt a = 0; a < 3; a++) m.m1[a] += measure * sum[a] / (PetscReal)(d + 1);
   const PetscReal w = measure / (PetscReal)((d + 1) * (d + 2));
   for (PetscInt a = 0; a < 3; a++) {
      for (PetscInt b = 0; b < 3; b++) {
         PetscReal outer = sum[a] * sum[b];
         for (PetscInt k = 0; k <= d; k++) outer += y[k][a] * y[k][b];
         m.m2[3 * a + b] += w * outer;
      }
   }
}

// The centroid, and the second moment about it, of everything added
static void Centralise(const FanMoments &m, const PetscReal reference[3], PetscReal centroid[3], PetscReal central[9])
{
   for (PetscInt a = 0; a < 3; a++) centroid[a] = reference[a] + m.m1[a] / m.m0;
   for (PetscInt a = 0; a < 3; a++) {
      for (PetscInt b = 0; b < 3; b++) central[3 * a + b] = m.m2[3 * a + b] - m.m1[a] * m.m1[b] / m.m0;
   }
}

static inline void Sub3(const PetscReal a[3], const PetscReal b[3], PetscReal out[3])
{
   for (PetscInt d = 0; d < 3; d++) out[d] = a[d] - b[d];
}

// The two vertices of an edge (an interpolated mesh has edges as points), z = 0
// in 2D
static PetscErrorCode EdgeVertices(DM dm, PetscSection csec, const PetscScalar *coords, PetscInt dim, \
   PetscInt edge, PetscReal x[2][3])
{
   const PetscInt *cone = nullptr;
   PetscInt n_cone = 0;

   PetscFunctionBeginUser;

   PetscCall(DMPlexGetConeSize(dm, edge, &n_cone));
   PetscCheck(n_cone == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "edge %" PetscInt_FMT " has %" \
      PetscInt_FMT " vertices", edge, n_cone);
   PetscCall(DMPlexGetCone(dm, edge, &cone));
   for (PetscInt v = 0; v < 2; v++) {
      PetscInt off = 0, n_dof = 0;
      PetscCall(PetscSectionGetDof(csec, cone[v], &n_dof));
      PetscCheck(n_dof == dim, PETSC_COMM_SELF, PETSC_ERR_SUP, "vertex %" PetscInt_FMT " carries %" \
         PetscInt_FMT " coordinates, not %" PetscInt_FMT " - DG1 reads vertex coordinates", cone[v], n_dof, dim);
      PetscCall(PetscSectionGetOffset(csec, cone[v], &off));
      for (PetscInt d = 0; d < 3; d++) x[v][d] = (d < dim) ? PetscRealPart(coords[off + d]) : 0.0;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// A face's area, centroid and central second moment. In 2D a face is an edge
// (one segment); in 3D a polygon, fanned into triangles off `fref` - any point
// in its plane, the FVM face centroid here - one per edge, which needs no
// vertex ordering
static PetscErrorCode FaceMoments(DM dm, PetscSection csec, const PetscScalar *coords, PetscInt dim, \
   PetscInt f, const PetscReal fref[3], PetscReal *area, PetscReal centroid[3], PetscReal central[9])
{
   FanMoments m;

   PetscFunctionBeginUser;

   if (dim == 2) {
      PetscReal x[2][3], y[2][3];
      PetscCall(EdgeVertices(dm, csec, coords, dim, f, x));
      Sub3(x[0], fref, y[0]);
      Sub3(x[1], fref, y[1]);
      PetscReal e[3];
      Sub3(x[1], x[0], e);
      AddSimplex(m, 1, PetscSqrtReal(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]), y);
   } else {
      const PetscInt *cone = nullptr;
      PetscInt n_cone = 0;
      PetscCall(DMPlexGetConeSize(dm, f, &n_cone));
      PetscCall(DMPlexGetCone(dm, f, &cone));
      for (PetscInt e = 0; e < n_cone; e++) {
         PetscReal x[2][3], y[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
         PetscCall(EdgeVertices(dm, csec, coords, dim, cone[e], x));
         Sub3(x[0], fref, y[1]);
         Sub3(x[1], fref, y[2]);
         const PetscReal cr[3] = {y[1][1] * y[2][2] - y[1][2] * y[2][1], y[1][2] * y[2][0] - y[1][0] * y[2][2], \
            y[1][0] * y[2][1] - y[1][1] * y[2][0]};
         AddSimplex(m, 2, 0.5 * PetscSqrtReal(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]), y);
      }
   }
   *area = m.m0;
   Centralise(m, fref, centroid, central);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// A cell's volume, centroid and central second moment, fanned off `cref` (the
// FVM centroid, inside any star-shaped cell): 2D, one triangle per edge; 3D,
// one tetrahedron per (face, edge of the face) with the face's FVM centroid as
// its fourth point. Exact for any cell with planar faces
static PetscErrorCode CellMoments(DM dm, PetscSection csec, const PetscScalar *coords, PetscInt dim, \
   PetscInt c, const PetscReal cref[3], PetscReal *volume, PetscReal centroid[3], PetscReal central[9])
{
   FanMoments m;
   const PetscInt *cone = nullptr;
   PetscInt n_cone = 0;

   PetscFunctionBeginUser;

   PetscCall(DMPlexGetConeSize(dm, c, &n_cone));
   PetscCall(DMPlexGetCone(dm, c, &cone));
   for (PetscInt lf = 0; lf < n_cone; lf++) {
      if (dim == 2) {
         PetscReal x[2][3], y[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
         PetscCall(EdgeVertices(dm, csec, coords, dim, cone[lf], x));
         Sub3(x[0], cref, y[1]);
         Sub3(x[1], cref, y[2]);
         AddSimplex(m, 2, 0.5 * PetscAbsReal(y[1][0] * y[2][1] - y[1][1] * y[2][0]), y);
      } else {
         PetscReal farea = 0.0, fcen[3] = {0.0, 0.0, 0.0};
         PetscCall(DMPlexComputeCellGeometryFVM(dm, cone[lf], &farea, fcen, NULL));
         const PetscInt *fcone = nullptr;
         PetscInt n_fcone = 0;
         PetscCall(DMPlexGetConeSize(dm, cone[lf], &n_fcone));
         PetscCall(DMPlexGetCone(dm, cone[lf], &fcone));
         for (PetscInt e = 0; e < n_fcone; e++) {
            PetscReal x[2][3], y[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
            PetscCall(EdgeVertices(dm, csec, coords, dim, fcone[e], x));
            Sub3(fcen, cref, y[1]);
            Sub3(x[0], cref, y[2]);
            Sub3(x[1], cref, y[3]);
            const PetscReal det = y[1][0] * (y[2][1] * y[3][2] - y[2][2] * y[3][1]) \
               - y[1][1] * (y[2][0] * y[3][2] - y[2][2] * y[3][0]) + y[1][2] * (y[2][0] * y[3][1] - y[2][1] * y[3][0]);
            AddSimplex(m, 3, PetscAbsReal(det) / 6.0, y);
         }
      }
   }
   *volume = m.m0;
   Centralise(m, cref, centroid, central);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// The linear half of a cell's orthonormal modal basis: with M = central / V the
// cell's normalised second moments (dim x dim, SPD) and M = L L^T, the rows of
// L^{-1} are the gradients beta_k of phi_{1+k}(x) = beta_k . (x - x_c), and
// (1/V) int phi_{1+k} phi_{1+l} = L^{-1} M L^{-T} = I. beta is 3 x 3 row-major,
// zero outside the leading dim x dim block
static PetscErrorCode OrthonormalGradients(PetscInt dim, PetscInt c, PetscReal volume, const PetscReal central[9], \
   PetscReal beta[9])
{
   PetscReal L[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

   PetscFunctionBeginUser;

   for (PetscInt a = 0; a < dim; a++) {
      for (PetscInt b = 0; b <= a; b++) {
         PetscReal sum = central[3 * a + b] / volume;
         for (PetscInt k = 0; k < b; k++) sum -= L[3 * a + k] * L[3 * b + k];
         if (a == b) {
            PetscCheck(sum > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "cell %" PetscInt_FMT \
               " has a degenerate second moment - it is flat along some direction", c);
            L[3 * a + a] = PetscSqrtReal(sum);
         } else L[3 * a + b] = sum / L[3 * b + b];
      }
   }
   // Forward substitution for the lower-triangular inverse, column by column
   for (PetscInt i = 0; i < 9; i++) beta[i] = 0.0;
   for (PetscInt col = 0; col < dim; col++) {
      for (PetscInt a = col; a < dim; a++) {
         PetscReal sum = (a == col) ? 1.0 : 0.0;
         for (PetscInt k = col; k < a; k++) sum -= L[3 * a + k] * beta[3 * k + col];
         beta[3 * a + col] = sum / L[3 * a + a];
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredDG::create(PhaseSpace &ps, const SNQuadrature2D &quad, const BCSpec &bcs, PetscInt order)
{
   PetscFunctionBeginUser;

   PetscCall(create_common(ps, 2, quad.n_angles(), quad.sum_weights(), quad.mu_host(), quad.eta_host(), \
      nullptr, quad.reflect_mu_host(), quad.reflect_eta_host(), nullptr, bcs, order));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredDG::create(PhaseSpace &ps, const SNQuadrature3D &quad, const BCSpec &bcs, PetscInt order)
{
   PetscFunctionBeginUser;

   PetscCall(create_common(ps, 3, quad.n_angles(), quad.sum_weights(), quad.mu_host(), quad.eta_host(), \
      quad.xi_host(), quad.reflect_mu_host(), quad.reflect_eta_host(), quad.reflect_xi_host(), bcs, order));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The layout, the geometry, the BC classification and the COO sparsity
// This happens on the host but we only need to do it once
PetscErrorCode UnstructuredDG::create_common(PhaseSpace &ps, PetscInt quad_dim, PetscInt n_angles, \
   PetscScalar sum_weights, const PetscScalar *mu, const PetscScalar *eta, const PetscScalar *xi, \
   const PetscInt *reflect_mu, const PetscInt *reflect_eta, const PetscInt *reflect_xi, const BCSpec &bcs, \
   PetscInt order)
{
   PetscSection sec = NULL, gsec = NULL;
   DMLabel face_sets = NULL;
   PetscInt rstart = 0;
   std::vector<PetscInt> owned;
   PetscBool failed = PETSC_FALSE;
   char message[512] = "";

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(dm_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
      "UnstructuredDG::create_mesh has to come first - the mesh decides the cell count");
   PetscCheck(quad_dim == dim_, comm_, PETSC_ERR_ARG_INCOMP, "a %" PetscInt_FMT "D quadrature was " \
      "handed to a %" PetscInt_FMT "D mesh", quad_dim, dim_);
   PetscCheck(n_angles == ps.n_angles, comm_, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, n_angles, ps.n_angles);
   PetscCheck(ps.n_cells == n_global_cells_, comm_, PETSC_ERR_ARG_INCOMP, "the mesh has %" PetscInt_FMT \
      " cells but the phase space %" PetscInt_FMT " - size it off n_global_cells()", n_global_cells_, ps.n_cells);

   PetscCheck(order == 0 || order == 1, comm_, PETSC_ERR_ARG_OUTOFRANGE, "the unstructured backend is " \
      "DG0 or DG1, was asked for order %" PetscInt_FMT, order);
   // A cell with several dofs has no one row to make the identity, and a DG
   // face flux IS the ghost-flux inflow - see the header
   PetscCheck(order == 0 || bcs.ghost_flux_vacuum(), comm_, PETSC_ERR_SUP, "DG1 takes the ghost-flux " \
      "vacuum treatment only - there is no Dirichlet-cell row for a cell with more than one dof, so drop " \
      "\"vacuum_treatment\": \"dirichlet_cell\"");

   const PetscInt dim = dim_;
   const PetscInt *reflect[3] = {reflect_mu, reflect_eta, reflect_xi};
   order_ = order;
   n_basis_ = (order == 0) ? 1 : dim + 1;
   const PetscInt nb = n_basis_;
   // The discretisation decides the spatial dofs per cell, like the
   // decomposition, and it has to be in the phase space before the layout check
   ps.n_basis = nb;
   const PetscInt rows_per_cell = ps.rows_per_cell();

   // ~~~~~~~~~~
   // The layout: n_basis * n_angles dof on every cell (overlap ghosts included -
   // the global section needs them to say where a ghost column lives), nothing
   // anywhere else, basis-major inside the cell so a (cell, basis) node is a
   // contiguous run of angles. The DM keeps a reference to the section
   // ~~~~~~~~~~
   PetscInt p_start = 0, p_end = 0;
   PetscCall(DMPlexGetChart(dm_, &p_start, &p_end));
   PetscCall(PetscSectionCreate(comm_, &sec));
   PetscCall(PetscSectionSetChart(sec, p_start, p_end));
   for (PetscInt c = c_start_; c < c_end_; c++) PetscCall(PetscSectionSetDof(sec, c, rows_per_cell));
   PetscCall(PetscSectionSetUp(sec));
   PetscCall(DMSetLocalSection(dm_, sec));
   PetscCall(PetscSectionDestroy(&sec));
   PetscCall(DMGetGlobalSection(dm_, &gsec));

   PetscCall(MarkOwnedCells(dm_, c_start_, c_end_, owned));
   PetscCall(CheckPlexLayout(dm_, gsec, ps, c_start_, c_end_, owned, &rstart));

   cell_of_local_.clear();
   local_of_cell_.assign(c_end_ - c_start_, -1);
   for (PetscInt c = c_start_; c < c_end_; c++) {
      if (!owned[c - c_start_]) continue;
      local_of_cell_[c - c_start_] = (PetscInt)cell_of_local_.size();
      cell_of_local_.push_back(c);
   }
   const PetscInt local_cells = (PetscInt)cell_of_local_.size();

   // The DM decided the decomposition - the PhaseSpace is told
   ps.local_cells = local_cells;
   ps_ = ps;
   const PetscInt local_rows = ps.local_rows();

   // ~~~~~~~~~~
   // Geometry, once on the host. Per owned cell its volume and centroid; per
   // face in cone order its outward area-weighted normal, its centroid, its
   // neighbour's global row base (-1 on the boundary) and, on the boundary,
   // its "Face Sets" value
   // ~~~~~~~~~~
   PetscCall(DMGetLabel(dm_, "Face Sets", &face_sets));

   centroid_h_.assign(3 * local_cells, 0.0);
   volume_h_.assign(local_cells, 0.0);
   cell_face_offset_h_.assign(local_cells + 1, 0);
   face_nA_h_.clear();
   face_neighbour_row_h_.clear();
   face_label_h_.clear();
   std::vector<PetscReal> face_centroid;
   std::vector<PetscInt> face_axis;
   std::vector<PetscInt> face_point;

   for (PetscInt k = 0; k < local_cells; k++) {

      const PetscInt c = cell_of_local_[k];
      PetscReal vol = 0.0, cen[3] = {0.0, 0.0, 0.0};
      PetscCall(DMPlexComputeCellGeometryFVM(dm_, c, &vol, cen, NULL));
      PetscCheck(vol > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "cell %" PetscInt_FMT \
         " has non-positive volume %g", c, (double)vol);
      volume_h_[k] = vol;
      for (PetscInt d = 0; d < 3; d++) centroid_h_[3 * k + d] = cen[d];

      const PetscInt *cone = nullptr;
      PetscInt n_faces = 0;
      PetscCall(DMPlexGetConeSize(dm_, c, &n_faces));
      PetscCall(DMPlexGetCone(dm_, c, &cone));
      cell_face_offset_h_[k + 1] = cell_face_offset_h_[k] + n_faces;

      for (PetscInt lf = 0; lf < n_faces; lf++) {

         const PetscInt f = cone[lf];
         // In 2D a face is an edge: its "area" is its length and the normal
         // is the in-plane one. The normal's sign is PETSc's orientation, so
         // it is pointed out of this cell by the centroid test below
         PetscReal area = 0.0, fcen[3] = {0.0, 0.0, 0.0}, normal[3] = {0.0, 0.0, 0.0};
         PetscCall(DMPlexComputeCellGeometryFVM(dm_, f, &area, fcen, normal));
         PetscCheck(area > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "face %" PetscInt_FMT \
            " of cell %" PetscInt_FMT " has non-positive area %g", f, c, (double)area);

         PetscReal norm = 0.0, outward = 0.0;
         for (PetscInt d = 0; d < dim; d++) norm += normal[d] * normal[d];
         norm = PetscSqrtReal(norm);
         PetscCheck(norm > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "face %" PetscInt_FMT \
            " has no normal", f);
         for (PetscInt d = 0; d < dim; d++) outward += normal[d] * (fcen[d] - cen[d]);
         const PetscReal sign = (outward < 0.0) ? -1.0 : 1.0;

         PetscScalar nA[3] = {0.0, 0.0, 0.0};
         for (PetscInt d = 0; d < dim; d++) nA[d] = sign * area * normal[d] / norm;
         for (PetscInt d = 0; d < 3; d++) {
            face_nA_h_.push_back(nA[d]);
            face_centroid.push_back(fcen[d]);
         }
         face_axis.push_back(DominantAxis(nA, dim));
         face_point.push_back(f);

         const PetscInt *support = nullptr;
         PetscInt n_support = 0;
         PetscCall(DMPlexGetSupportSize(dm_, f, &n_support));
         PetscCall(DMPlexGetSupport(dm_, f, &support));
         PetscCheck(n_support == 1 || n_support == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
            "face %" PetscInt_FMT " is shared by %" PetscInt_FMT " cells", f, n_support);

         if (n_support == 2) {

            // An interior face - the neighbour is owned or an overlap ghost,
            // and the global section knows its row either way
            const PetscInt other = (support[0] == c) ? support[1] : support[0];
            PetscInt g = 0;
            PetscBool is_owned = PETSC_FALSE;
            PetscCall(GlobalCellOffset(gsec, other, &g, &is_owned));
            face_neighbour_row_h_.push_back(g);
            face_label_h_.push_back(-1);

         } else {

            // A boundary face. With a one-cell face-adjacent overlap, an OWNED
            // cell's partition-boundary faces all have their neighbour in the
            // local mesh, so support size 1 here really is the domain boundary
            PetscInt value = -1;
            if (face_sets) PetscCall(DMLabelGetValue(face_sets, f, &value));
            face_neighbour_row_h_.push_back(-1);
            face_label_h_.push_back(value);

            // The face's family is used below; validate its shape here, once
            const BCFace bc = bcs.face(value);
            if (!failed && bc.n_window_pairs != 0 && bc.n_window_pairs != dim - 1) {
               failed = PETSC_TRUE;
               PetscCall(PetscSNPrintf(message, sizeof(message), "a %" PetscInt_FMT "D face takes %" \
                  PetscInt_FMT " tangential [lo, hi] window pairs, face set %" PetscInt_FMT " was given %" \
                  PetscInt_FMT, dim, dim - 1, value, bc.n_window_pairs));
            }
            // The mirror of an ordinate in a general plane is not an ordinate
            // of the set, so reflection is only defined on axis-aligned faces
            if (!failed && bc.type == BCType::REFLECT && \
                PetscAbsScalar(nA[face_axis.back()]) < (1.0 - 1e-10) * area) {
               failed = PETSC_TRUE;
               PetscCall(PetscSNPrintf(message, sizeof(message), "face %" PetscInt_FMT " (face set %" \
                  PetscInt_FMT ") is reflective but not axis-aligned - reflection maps an ordinate onto " \
                  "an ordinate only across a face normal to x, y or z", f, value));
            }
         }
      }
   }
   PetscCall(CollectiveFailure(comm_, failed, message));

   // Every label the BCSpec names has to be on some boundary face of the mesh,
   // somewhere: a boundary condition on a "Face Sets" value no face carries is
   // a mistyped id, and the alternative is a face the user meant to drive or
   // reflect silently going cold. Collective - a rank may own no faces at all
   for (const auto &entry : bcs.faces()) {
      PetscMPIInt seen = 0, any = 0;
      for (const PetscInt value : face_label_h_) {
         if (value == entry.first) seen = 1;
      }
      PetscCallMPI(MPI_Allreduce(&seen, &any, 1, MPI_INT, MPI_MAX, comm_));
      PetscCheck(any, comm_, PETSC_ERR_ARG_WRONG, "a boundary condition was given for \"Face Sets\" value %" \
         PetscInt_FMT " but no boundary face of the mesh carries it", entry.first);
   }

   // ~~~~~~~~~~
   // The ordinates, flattened (a * 3 + d) - the layout the device view has,
   // so the host classification reads them exactly as the fill does
   // ~~~~~~~~~~
   std::vector<PetscScalar> omega(3 * n_angles, 0.0);
   for (PetscInt a = 0; a < n_angles; a++) {
      omega[3 * a] = mu[a];
      omega[3 * a + 1] = eta[a];
      omega[3 * a + 2] = xi ? xi[a] : 0.0;
   }

   // ~~~~~~~~~~
   // COO coordinates. DG0: n_faces + 1 entries per row, in slot order: one
   // per face in cone order, then the diagonal. DG1: the same with every
   // entry widened to n_basis - see the header and the DG1 branch below. The
   // rest of this comment is DG0's
   // ~~~~~~~~~~
   // A -1 row/col index in the COO format says just ignore this entry. A face
   // slot is live only on an interior row, across an interior face, for an
   // angle flowing IN through it: the upwind selection, made here once, so
   // the value fill never branches on which neighbour is upwind. A reflective
   // boundary row instead repurposes the slot of its first incoming reflective
   // face for the coupling to the mirrored angle, and a Dirichlet row keeps
   // only its diagonal
   //
   // Under VacuumTreatment::GHOST_FLUX a row that comes in only through
   // VACUUM faces is not a BC row at all: it keeps the interior row's slots -
   // whose boundary faces are nulled by the same "neighbour exists" test - and
   // the streaming term writes its full diagonal, so the one change is the
   // rhs, |Omega . nA_f| / V_c times the face's per-angle inflow for every
   // vacuum face it comes in through (windowed per face, summed over faces).
   // That is exactly the upwind face flux with the inflow as the ghost value
   // outside, which is the natural DG0 vacuum condition. A REFLECTIVE inflow
   // face is a face flux the same way, its ghost value the mirrored angle in
   // this same cell, so its slot points there - DG1's rule, face by face - and
   // a row coming in through both kinds takes both, with no precedence rule
   // (the old "reflect wins" corner never saw the vacuum inflow). Under
   // ghost-flux there are no BC rows at all
   const PetscBool ghost = bcs.ghost_flux_vacuum();
   oor_.clear();
   ooc_.clear();
   std::vector<PetscInt> row_slot_offset(local_rows + 1, 0);
   std::vector<PetscInt> diag_slot(local_rows, 0);
   std::vector<PetscInt> is_bc_row(local_rows, 0);
   std::vector<PetscInt> reflect_slot(local_rows, -1);
   std::vector<PetscScalar> dirichlet_value(local_rows, 0.0);
   std::vector<PetscScalar> ghost_inflow(local_rows, 0.0);

   if (order == 1) {

      // DG1: the basis, the fan geometry and the face matrices first - the
      // ghost inflow below reads the basis at the face centroids
      std::vector<PetscScalar> face_own, face_up, basis_grad;
      std::vector<PetscReal> face_basis_value;
      PetscCall(build_dg1_geometry(face_own, face_up, basis_grad, face_basis_value, bcs));

      // No BC rows: every face is a face flux. Per row, n_basis slots per face
      // in cone order - the upwind cell's basis j on an interior inflow face,
      // this cell's basis j at the mirrored angle on a reflective inflow face,
      // nulled otherwise - then this cell's own n_basis at the same angle, the
      // diagonal being own slot i. A vacuum inflow face nulls its slots and
      // puts -(s / V) phi_i(x_f) times its per-angle inflow into the rhs
      for (PetscInt k = 0; k < local_cells; k++) {

         const PetscInt k0 = cell_face_offset_h_[k];
         const PetscInt n_faces = cell_face_offset_h_[k + 1] - k0;

         for (PetscInt i = 0; i < nb; i++) {
            for (PetscInt a = 0; a < n_angles; a++) {

               const PetscInt r = (k * nb + i) * n_angles + a;
               const PetscInt row = rstart + r;
               row_slot_offset[r] = (PetscInt)oor_.size();

               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  const PetscInt kf = k0 + lf;
                  const PetscScalar s = FaceFlux(omega.data(), a, face_nA_h_.data(), kf);
                  const PetscBool inflow = (PetscBool)(PetscRealPart(s) < 0.0);
                  const PetscBool interior = (PetscBool)(face_neighbour_row_h_[kf] >= 0);
                  const PetscBool reflective = (PetscBool)(!interior && \
                     bcs.type(face_label_h_[kf]) == BCType::REFLECT);
                  const PetscBool live = (PetscBool)(inflow && (interior || reflective));

                  for (PetscInt j = 0; j < nb; j++) {
                     PetscInt col = -1;
                     if (live && interior) col = face_neighbour_row_h_[kf] + j * n_angles + a;
                     // Same cell, mirrored over this face's own axis - always
                     // rank-local
                     else if (live) col = rstart + (k * nb + j) * n_angles + reflect[face_axis[kf]][a];
                     oor_.push_back(live ? row : -1);
                     ooc_.push_back(col);
                  }

                  if (!inflow || interior || reflective) continue;
                  const BCFace bc = bcs.face(face_label_h_[kf]);
                  PetscReal t[2] = {0.0, 0.0};
                  PetscInt n_t = 0;
                  for (PetscInt d = 0; d < dim; d++) {
                     if (d != face_axis[kf]) t[n_t++] = face_centroid[3 * kf + d];
                  }
                  if (InWindow(bc, t, n_t)) ghost_inflow[r] += -s / volume_h_[k] * \
                     face_basis_value[kf * nb + i] * ((PetscScalar)bc.inflow / sum_weights);
               }

               // This cell, this angle, every basis function
               for (PetscInt j = 0; j < nb; j++) {
                  if (j == i) diag_slot[r] = (PetscInt)oor_.size();
                  oor_.push_back(row);
                  ooc_.push_back(rstart + (k * nb + j) * n_angles + a);
               }

               for (PetscInt sl = row_slot_offset[r]; sl < (PetscInt)oor_.size(); sl++) {
                  PetscCheck(ooc_[sl] >= 0 || oor_[sl] == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
                     "no global index for slot %" PetscInt_FMT " of cell %" PetscInt_FMT, \
                     sl - row_slot_offset[r], cell_of_local_[k]);
               }
            }
         }
      }

      // The device copies of the DG1 geometry StreamingTermDG1 reads
      face_own_d_ = PetscScalarKokkosView("face_own_d", face_own.size());
      face_up_d_ = PetscScalarKokkosView("face_up_d", face_up.size());
      basis_grad_d_ = PetscScalarKokkosView("basis_grad_d", basis_grad.size());
      PetscScalarKokkosViewHostUnmanaged face_own_h(face_own.data(), face_own.size());
      PetscScalarKokkosViewHostUnmanaged face_up_h(face_up.data(), face_up.size());
      PetscScalarKokkosViewHostUnmanaged basis_grad_h(basis_grad.data(), basis_grad.size());
      Kokkos::deep_copy(face_own_d_, face_own_h);
      Kokkos::deep_copy(face_up_d_, face_up_h);
      Kokkos::deep_copy(basis_grad_d_, basis_grad_h);

   } else {

      // DG0
      for (PetscInt k = 0; k < local_cells && !failed; k++) {

         const PetscInt k0 = cell_face_offset_h_[k];
         const PetscInt n_faces = cell_face_offset_h_[k + 1] - k0;

         for (PetscInt a = 0; a < n_angles && !failed; a++) {

            const PetscInt r = k * n_angles + a;
            const PetscInt row = rstart + r;

            // Which boundary faces this direction comes IN through: the physics
            // decides whether the row is a boundary row at all (the sign of s),
            // the label only which family each face belongs to
            PetscBool any_incoming = PETSC_FALSE, any_vacuum = PETSC_FALSE;
            PetscInt win = -1, first_reflect = -1, reflect_axes = 0;
            for (PetscInt lf = 0; lf < n_faces; lf++) {
               const PetscInt kf = k0 + lf;
               if (face_neighbour_row_h_[kf] >= 0) continue;
               if (!(PetscRealPart(FaceFlux(omega.data(), a, face_nA_h_.data(), kf)) < 0.0)) continue;
               any_incoming = PETSC_TRUE;
               if (bcs.type(face_label_h_[kf]) == BCType::VACUUM) {
                  any_vacuum = PETSC_TRUE;
                  // The winning face: lowest dominant axis, then lowest point
                  if (win < 0 || face_axis[kf] < face_axis[win] || \
                      (face_axis[kf] == face_axis[win] && face_point[kf] < face_point[win])) win = kf;
               } else {
                  if (first_reflect < 0) first_reflect = lf;
                  reflect_axes |= (1 << face_axis[kf]);
               }
            }

            row_slot_offset[r] = (PetscInt)oor_.size();

            if (!any_incoming || ghost) {

               // Interior row: the upwind neighbour across every inflow face. A
               // ghost-flux row is the same row, its vacuum inflow faces
               // nulled, their flux moved to the rhs, and its reflective inflow
               // faces pointed at the mirrored angle in this cell (owned, so
               // always rank-local)
               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  const PetscInt kf = k0 + lf;
                  const PetscScalar s = FaceFlux(omega.data(), a, face_nA_h_.data(), kf);
                  const PetscBool inflow = (PetscBool)(PetscRealPart(s) < 0.0);
                  const PetscBool interior = (PetscBool)(face_neighbour_row_h_[kf] >= 0);
                  const PetscBool reflective = (PetscBool)(!interior && \
                     bcs.type(face_label_h_[kf]) == BCType::REFLECT);
                  const PetscBool live = (PetscBool)(inflow && (interior || reflective));
                  oor_.push_back(live ? row : -1);
                  ooc_.push_back(!live ? -1 : (interior ? face_neighbour_row_h_[kf] + a : \
                     rstart + k * n_angles + reflect[face_axis[kf]][a]));

                  if (interior || reflective || !inflow) continue;
                  const BCFace bc = bcs.face(face_label_h_[kf]);
                  PetscReal t[2] = {0.0, 0.0};
                  PetscInt n_t = 0;
                  for (PetscInt d = 0; d < dim; d++) {
                     if (d != face_axis[kf]) t[n_t++] = face_centroid[3 * kf + d];
                  }
                  if (InWindow(bc, t, n_t)) ghost_inflow[r] += -s / volume_h_[k] * ((PetscScalar)bc.inflow / sum_weights);
               }

            } else if (any_vacuum) {

               // Dirichlet row: identity, and the rhs takes the winning face's
               // inflow if the face centroid is inside its window
               is_bc_row[r] = 1;
               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  oor_.push_back(-1);
                  ooc_.push_back(-1);
               }
               const BCFace bc = bcs.face(face_label_h_[win]);
               PetscReal t[2] = {0.0, 0.0};
               PetscInt n_t = 0;
               for (PetscInt d = 0; d < dim; d++) {
                  if (d != face_axis[win]) t[n_t++] = face_centroid[3 * win + d];
               }
               dirichlet_value[r] = InWindow(bc, t, n_t) ? (PetscScalar)bc.inflow / sum_weights : (PetscScalar)0.0;

            } else {

               // Reflective row (Dirichlet-cell only): psi(a) - psi(partner) = 0
               // in the same cell, the partner mirrored in every axis the
               // direction came in through
               is_bc_row[r] = 1;
               PetscInt partner = a;
               for (PetscInt d = 0; d < dim; d++) {
                  if (reflect_axes & (1 << d)) partner = reflect[d][partner];
               }

               // The partner is outgoing through every face this direction came
               // in through. Its row may still be a boundary row: coming in
               // through a VACUUM face - a cell where a reflective axis plane
               // meets a slanted or curved vacuum boundary, the usual
               // symmetry-reduced geometry - makes it a Dirichlet row, and
               // psi(a) = psi(partner) = the inflow is a perfectly good pair of
               // equations. Coming in through another REFLECTIVE face is not: the
               // two rows would each define the other (a single-cell-wide
               // direction between two reflective faces), and there is no
               // sensible matrix for that
               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  const PetscInt kf = k0 + lf;
                  if (face_neighbour_row_h_[kf] >= 0) continue;
                  if (bcs.type(face_label_h_[kf]) != BCType::REFLECT) continue;
                  if (PetscRealPart(FaceFlux(omega.data(), partner, face_nA_h_.data(), kf)) < 0.0 && !failed) {
                     failed = PETSC_TRUE;
                     PetscCall(PetscSNPrintf(message, sizeof(message), "the reflection partner of cell %" \
                        PetscInt_FMT " angle %" PetscInt_FMT " is itself a reflective boundary row - a " \
                        "reflective face on a single-cell-wide direction between two reflective faces is not " \
                        "supported", cell_of_local_[k], a));
                  }
               }

               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  // Same cell, mirrored angle - an owned row, always rank-local
                  oor_.push_back(lf == first_reflect ? row : -1);
                  ooc_.push_back(lf == first_reflect ? rstart + k * n_angles + partner : -1);
               }
               reflect_slot[r] = row_slot_offset[r] + first_reflect;
            }

            diag_slot[r] = (PetscInt)oor_.size();
            oor_.push_back(row);
            ooc_.push_back(row);

            // A ghost the overlap did not bring in would have no global row, and
            // a stray -1 column would silently drop a coefficient rather than
            // fail, so say it out loud. A slot that was deliberately nulled has
            // its ROW index at -1 too, which is what separates it from a lookup
            // that failed
            for (PetscInt s = row_slot_offset[r]; s < (PetscInt)oor_.size(); s++) {
               PetscCheck(ooc_[s] >= 0 || oor_[s] == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
                  "no global index for face slot %" PetscInt_FMT " of cell %" PetscInt_FMT, \
                  s - row_slot_offset[r], cell_of_local_[k]);
            }
         }
      }
   }
   PetscCall(CollectiveFailure(comm_, failed, message));
   row_slot_offset[local_rows] = (PetscInt)oor_.size();

   PetscCall(set_pattern(row_slot_offset, diag_slot, is_bc_row, reflect_slot, dirichlet_value, ghost_inflow, \
      ghost));

   // ~~~~~~~~~~
   // The device geometry the streaming term reads - flat rank-1 views only
   // ~~~~~~~~~~
   const PetscInt n_cell_faces = cell_face_offset_h_[local_cells];
   std::vector<PetscScalar> inv_volume(local_cells), centroid(3 * local_cells);
   for (PetscInt k = 0; k < local_cells; k++) inv_volume[k] = 1.0 / volume_h_[k];
   for (PetscInt i = 0; i < 3 * local_cells; i++) centroid[i] = centroid_h_[i];

   omega_d_ = PetscScalarKokkosView("omega_d", 3 * n_angles);
   cell_face_offset_d_ = PetscIntKokkosView("cell_face_offset_d", local_cells + 1);
   face_nA_d_ = PetscScalarKokkosView("face_nA_d", 3 * n_cell_faces);
   inv_volume_d_ = PetscScalarKokkosView("inv_volume_d", local_cells);
   centroid_d_ = PetscScalarKokkosView("centroid_d", 3 * local_cells);

   PetscScalarKokkosViewHostUnmanaged omega_h(omega.data(), 3 * n_angles);
   PetscIntKokkosViewHostUnmanaged cell_face_offset_h(cell_face_offset_h_.data(), local_cells + 1);
   PetscScalarKokkosViewHostUnmanaged face_nA_h(face_nA_h_.data(), 3 * n_cell_faces);
   PetscScalarKokkosViewHostUnmanaged inv_volume_h(inv_volume.data(), local_cells);
   PetscScalarKokkosViewHostUnmanaged centroid_h(centroid.data(), 3 * local_cells);
   Kokkos::deep_copy(omega_d_, omega_h);
   Kokkos::deep_copy(cell_face_offset_d_, cell_face_offset_h);
   Kokkos::deep_copy(face_nA_d_, face_nA_h);
   Kokkos::deep_copy(inv_volume_d_, inv_volume_h);
   Kokkos::deep_copy(centroid_d_, centroid_h);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DG1 basis and face matrices - host, once, like the rest of the geometry
//
// Every cell of the LOCAL mesh gets its basis (an overlap ghost's is what a
// face matrix into it reads). A basis function is affine,
// phi(x) = phi(x_f) + b . (x - x_f) about any point x_f, so over a face with
// area A, centroid x_f and central second moment C_f
//   int_f phi_i phi_j = A phi_i(x_f) phi_j(x_f) + b_i^T C_f b_j
// exactly, whichever cells the two come from
PetscErrorCode UnstructuredDG::build_dg1_geometry(std::vector<PetscScalar> &face_own, \
   std::vector<PetscScalar> &face_up, std::vector<PetscScalar> &basis_grad, \
   std::vector<PetscReal> &face_basis_value, const BCSpec &bcs)
{
   PetscSection csec = NULL;
   Vec coords_vec = NULL;
   const PetscScalar *coords = nullptr;
   const PetscInt dim = dim_, nb = n_basis_;
   const PetscInt local_cells = (PetscInt)cell_of_local_.size();
   const PetscInt n_mesh_cells = c_end_ - c_start_;

   PetscFunctionBeginUser;

   PetscCall(DMGetCoordinateSection(dm_, &csec));
   PetscCall(DMGetCoordinatesLocal(dm_, &coords_vec));
   PetscCall(VecGetArrayRead(coords_vec, &coords));

   // Per local-mesh cell: centroid, volume, and the linear gradients beta
   std::vector<PetscReal> centre(3 * n_mesh_cells), vol(n_mesh_cells), beta(9 * n_mesh_cells);
   for (PetscInt c = c_start_; c < c_end_; c++) {
      const PetscInt m = c - c_start_;
      PetscReal fvm_vol = 0.0, cref[3] = {0.0, 0.0, 0.0}, central[9];
      PetscCall(DMPlexComputeCellGeometryFVM(dm_, c, &fvm_vol, cref, NULL));
      PetscCall(CellMoments(dm_, csec, coords, dim, c, cref, &vol[m], &centre[3 * m], central));
      // The fan and PETSc's FVM geometry agree on any cell with planar faces;
      // a cell where they do not has a face whose single normal the face
      // fluxes would be wrong for
      PetscCheck(PetscAbsReal(vol[m] - fvm_vol) <= 1e-8 * fvm_vol, PETSC_COMM_SELF, PETSC_ERR_SUP, \
         "cell %" PetscInt_FMT ": fan volume %.15g against PETSc's %.15g - DG1 needs planar faces", \
         c, (double)vol[m], (double)fvm_vol);
      PetscCall(OrthonormalGradients(dim, c, vol[m], central, &beta[9 * m]));
   }

   // phi_i of mesh cell m at x, and its gradient
   auto value = [&](PetscInt m, PetscInt i, const PetscReal x[3]) -> PetscReal {
      if (i == 0) return 1.0;
      PetscReal v = 0.0;
      for (PetscInt d = 0; d < 3; d++) v += beta[9 * m + 3 * (i - 1) + d] * (x[d] - centre[3 * m + d]);
      return v;
   };
   auto grad = [&](PetscInt m, PetscInt i, PetscInt d) -> PetscReal {
      return (i == 0) ? 0.0 : beta[9 * m + 3 * (i - 1) + d];
   };

   const PetscInt n_cell_faces = cell_face_offset_h_[local_cells];
   face_own.assign(n_cell_faces * nb * nb, 0.0);
   face_up.assign(n_cell_faces * nb * nb, 0.0);
   face_basis_value.assign(n_cell_faces * nb, 0.0);
   basis_grad.assign(local_cells * nb * 3, 0.0);

   for (PetscInt k = 0; k < local_cells; k++) {

      const PetscInt c = cell_of_local_[k];
      const PetscInt m = c - c_start_;
      // The fan geometry is what the basis is orthonormal against, so it is
      // what the rows are divided by
      volume_h_[k] = vol[m];
      for (PetscInt d = 0; d < 3; d++) centroid_h_[3 * k + d] = centre[3 * m + d];
      for (PetscInt i = 0; i < nb; i++) {
         for (PetscInt d = 0; d < 3; d++) basis_grad[(k * nb + i) * 3 + d] = grad(m, i, d);
      }

      const PetscInt *cone = nullptr;
      PetscCall(DMPlexGetCone(dm_, c, &cone));
      for (PetscInt kf = cell_face_offset_h_[k]; kf < cell_face_offset_h_[k + 1]; kf++) {

         const PetscInt f = cone[kf - cell_face_offset_h_[k]];
         PetscReal fvm_area = 0.0, fref[3] = {0.0, 0.0, 0.0}, area = 0.0, xf[3], Cf[9];
         PetscCall(DMPlexComputeCellGeometryFVM(dm_, f, &fvm_area, fref, NULL));
         PetscCall(FaceMoments(dm_, csec, coords, dim, f, fref, &area, xf, Cf));
         PetscCheck(PetscAbsReal(area - fvm_area) <= 1e-8 * fvm_area, PETSC_COMM_SELF, PETSC_ERR_SUP, \
            "face %" PetscInt_FMT ": fan area %.15g against PETSc's %.15g - DG1 needs planar faces", \
            f, (double)area, (double)fvm_area);

         // The upwind cell across this face: the neighbour through an
         // interior face, this cell itself through a reflective one (the
         // mirrored angle is the column's business), none through vacuum
         PetscInt up = -1;
         const PetscInt *support = nullptr;
         PetscInt n_support = 0;
         PetscCall(DMPlexGetSupportSize(dm_, f, &n_support));
         PetscCall(DMPlexGetSupport(dm_, f, &support));
         if (n_support == 2) up = ((support[0] == c) ? support[1] : support[0]) - c_start_;
         else if (bcs.type(face_label_h_[kf]) == BCType::REFLECT) up = m;

         const PetscReal scale = 1.0 / (vol[m] * area);
         for (PetscInt i = 0; i < nb; i++) {
            const PetscReal vi = value(m, i, xf);
            face_basis_value[kf * nb + i] = vi;
            for (PetscInt j = 0; j < nb; j++) {
               PetscReal own = area * vi * value(m, j, xf);
               for (PetscInt a = 0; a < 3; a++) {
                  for (PetscInt b = 0; b < 3; b++) own += grad(m, i, a) * Cf[3 * a + b] * grad(m, j, b);
               }
               face_own[(kf * nb + i) * nb + j] = own * scale;
               if (up < 0) continue;
               PetscReal cross = area * vi * value(up, j, xf);
               for (PetscInt a = 0; a < 3; a++) {
                  for (PetscInt b = 0; b < 3; b++) cross += grad(m, i, a) * Cf[3 * a + b] * grad(up, j, b);
               }
               face_up[(kf * nb + i) * nb + j] = cross * scale;
            }
         }
      }
   }

   PetscCall(VecRestoreArrayRead(coords_vec, &coords));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda (see docs/dev/kokkos.md): the
// slope of the scalar flux on axis d, from the per-node scalar flux
static void ScalarFluxGradientKernel(PetscScalarKokkosView grad_d, PetscScalar2DKokkosView phi_d, \
   PetscScalarKokkosView basis_grad_d, PetscInt nb, PetscInt d, PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

         PetscScalar g = 0.0;
         for (PetscInt i = 1; i < nb; i++) g += phi_d(c * nb + i, 0) * basis_grad_d((c * nb + i) * 3 + d);
         grad_d(c) = g;
      });
}

PetscErrorCode UnstructuredDG::scalar_flux_gradient(Vec psi, const AngularQuadrature &quad, \
   std::vector<PetscScalarKokkosView> &grad) const
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());
   PetscCheck(order_ == 1, comm_, PETSC_ERR_ARG_WRONGSTATE, "the scalar flux has no in-cell slope at DG0");
   PetscCheck(quad.n_angles() == ps_.n_angles, comm_, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps_.n_angles);

   // The shared angular integral, per node
   PetscScalar2DKokkosView phi_d("phi_d", ps_.local_nodes(), 1);
   PetscCall(UboltAngularIntegral(psi, ps_.n_angles, quad.w_d(), phi_d));

   grad.resize(dim_);
   for (PetscInt d = 0; d < dim_; d++) {
      grad[d] = PetscScalarKokkosView("scalar_flux_grad_d", ps_.local_cells);
      ScalarFluxGradientKernel(grad[d], phi_d, basis_grad_d_, n_basis_, d, ps_.local_cells);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda. The boxes ride along as flat views -
// 2 * dim (lo, hi) values per box, axis by axis - because a view of structs is
// not one of the typedefs types.hpp allows
static void PaintBoxesKernel(PetscIntKokkosView mat_id_d, PetscScalarKokkosView centroid_d, \
   PetscScalarKokkosView box_lohi_d, PetscIntKokkosView box_material_d, PetscInt n_boxes, \
   PetscInt dim, PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

         PetscInt m = mat_id_d(c);
         for (PetscInt b = 0; b < n_boxes; b++) {
            bool inside = true;
            for (PetscInt d = 0; d < dim; d++) {
               const PetscScalar x = centroid_d(3 * c + d);
               if (PetscRealPart(x) < PetscRealPart(box_lohi_d(2 * dim * b + 2 * d)) || \
                   PetscRealPart(x) > PetscRealPart(box_lohi_d(2 * dim * b + 2 * d + 1))) inside = false;
            }
            if (inside) m = box_material_d(b);
         }
         mat_id_d(c) = m;
      });
}

// Upload the flattened boxes and paint them over mat_id_d, later ones winning
PetscErrorCode UnstructuredDG::paint_flat_boxes(PetscInt n_boxes, const std::vector<PetscScalar> &box_lohi, \
   const std::vector<PetscInt> &box_material, PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps_.check_decomposed());
   PetscCheck((PetscInt)mat_id_d.extent(0) == ps_.local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "mat_id_d covers %" PetscInt_FMT " cells but there are %" PetscInt_FMT " local cells", \
      (PetscInt)mat_id_d.extent(0), ps_.local_cells);
   for (PetscInt b = 0; b < n_boxes; b++) {
      PetscCheck(box_material[b] >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
         "box %" PetscInt_FMT "'s material index %" PetscInt_FMT " is negative", b, box_material[b]);
      for (PetscInt d = 0; d < dim_; d++) {
         PetscCheck(PetscRealPart(box_lohi[2 * dim_ * b + 2 * d]) <= PetscRealPart(box_lohi[2 * dim_ * b + 2 * d + 1]), \
            comm_, PETSC_ERR_ARG_OUTOFRANGE, "box %" PetscInt_FMT " is inside out on axis %" PetscInt_FMT \
            " - give each axis as lo, hi with lo <= hi", b, d);
      }
   }

   PetscScalarKokkosView box_lohi_d("box_lohi_d", 2 * dim_ * n_boxes);
   PetscIntKokkosView box_material_d("box_material_d", n_boxes);
   if (n_boxes > 0) {
      PetscScalarKokkosViewHostUnmanaged box_lohi_h(const_cast<PetscScalar *>(box_lohi.data()), 2 * dim_ * n_boxes);
      PetscIntKokkosViewHostUnmanaged box_material_h(const_cast<PetscInt *>(box_material.data()), n_boxes);
      Kokkos::deep_copy(box_lohi_d, box_lohi_h);
      Kokkos::deep_copy(box_material_d, box_material_h);
   }

   PaintBoxesKernel(mat_id_d, centroid_d_, box_lohi_d, box_material_d, n_boxes, dim_, ps_.local_cells);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredDG::paint_boxes_over(const std::vector<MaterialBox2D> &boxes, PetscIntKokkosView &mat_id_d) const
{
   const PetscInt n_boxes = (PetscInt)boxes.size();

   PetscFunctionBeginUser;

   PetscCheck(dim_ == 2, comm_, PETSC_ERR_ARG_INCOMP, "2D boxes painted onto a %" PetscInt_FMT "D mesh", dim_);

   std::vector<PetscScalar> box_lohi(4 * n_boxes);
   std::vector<PetscInt> box_material(n_boxes);
   for (PetscInt b = 0; b < n_boxes; b++) {
      box_lohi[4 * b]     = boxes[b].x0;
      box_lohi[4 * b + 1] = boxes[b].x1;
      box_lohi[4 * b + 2] = boxes[b].y0;
      box_lohi[4 * b + 3] = boxes[b].y1;
      box_material[b] = boxes[b].material;
   }
   PetscCall(paint_flat_boxes(n_boxes, box_lohi, box_material, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredDG::paint_boxes_over(const std::vector<MaterialBox3D> &boxes, PetscIntKokkosView &mat_id_d) const
{
   const PetscInt n_boxes = (PetscInt)boxes.size();

   PetscFunctionBeginUser;

   PetscCheck(dim_ == 3, comm_, PETSC_ERR_ARG_INCOMP, "3D boxes painted onto a %" PetscInt_FMT "D mesh", dim_);

   std::vector<PetscScalar> box_lohi(6 * n_boxes);
   std::vector<PetscInt> box_material(n_boxes);
   for (PetscInt b = 0; b < n_boxes; b++) {
      box_lohi[6 * b]     = boxes[b].x0;
      box_lohi[6 * b + 1] = boxes[b].x1;
      box_lohi[6 * b + 2] = boxes[b].y0;
      box_lohi[6 * b + 3] = boxes[b].y1;
      box_lohi[6 * b + 4] = boxes[b].z0;
      box_lohi[6 * b + 5] = boxes[b].z1;
      box_material[b] = boxes[b].material;
   }
   PetscCall(paint_flat_boxes(n_boxes, box_lohi, box_material, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// See the declaration for the painting rules (background, later boxes win,
// membership by cell centroid)
PetscErrorCode UnstructuredDG::paint_boxes(PetscInt background_material, const std::vector<MaterialBox2D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps_.check_decomposed());
   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);

   mat_id_d = PetscIntKokkosView("mat_id_d", ps_.local_cells);
   Kokkos::deep_copy(mat_id_d, background_material);
   PetscCall(paint_boxes_over(boxes, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredDG::paint_boxes(PetscInt background_material, const std::vector<MaterialBox3D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps_.check_decomposed());
   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);

   mat_id_d = PetscIntKokkosView("mat_id_d", ps_.local_cells);
   Kokkos::deep_copy(mat_id_d, background_material);
   PetscCall(paint_boxes_over(boxes, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DMPlex half of the material split: which cells are which material is a
// label on the mesh, and the label's arbitrary values are remapped here onto
// MaterialSpec's dense indices - a host step, the same place BCSpec is
// consulted
PetscErrorCode UnstructuredDG::paint_cell_sets(PetscInt background_material, \
   const std::map<PetscInt, PetscInt> &label_to_material, PetscIntKokkosView &mat_id_d) const
{
   DMLabel cell_sets = NULL;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps_.check_decomposed());
   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);
   for (const auto &entry : label_to_material) {
      PetscCheck(entry.second >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, "cell set %" PetscInt_FMT \
         " maps to a negative material index %" PetscInt_FMT, entry.first, entry.second);
   }

   PetscCall(DMGetLabel(dm_, "Cell Sets", &cell_sets));
   PetscCheck(cell_sets || label_to_material.empty(), comm_, PETSC_ERR_ARG_WRONG, \
      "cell sets were given materials but the mesh has no \"Cell Sets\" label");

   const PetscInt local_cells = ps_.local_cells;
   std::vector<PetscInt> mat_id(local_cells, background_material);
   if (cell_sets) {
      for (PetscInt k = 0; k < local_cells; k++) {
         PetscInt value = -1;
         PetscCall(DMLabelGetValue(cell_sets, cell_of_local_[k], &value));
         const auto found = label_to_material.find(value);
         if (found != label_to_material.end()) mat_id[k] = found->second;
      }
   }

   mat_id_d = PetscIntKokkosView("mat_id_d", local_cells);
   PetscIntKokkosViewHostUnmanaged mat_id_h(mat_id.data(), local_cells);
   Kokkos::deep_copy(mat_id_d, mat_id_h);

   PetscFunctionReturn(PETSC_SUCCESS);
}
