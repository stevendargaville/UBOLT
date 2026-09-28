#include "ubolt/unstructured_dg.hpp"
#include "plex_commonk.hpp"
#include <petscdmplex.h>
#include <petscsf.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build (or read) the mesh and distribute it with a one-cell, face-adjacent
// overlap: cone, not closure, adjacency, so the overlap is the face neighbours
// only - what the upwind flux reads and nothing more. The default (closure)
// adjacency would also bring in every cell sharing a vertex
PetscErrorCode UnstructuredDG::create_mesh(MPI_Comm comm, const PlexMeshSpec &mesh)
{
   PetscFunctionBeginUser;

   PetscCall(create_plex_mesh(comm, mesh, PETSC_TRUE, PETSC_FALSE));
   // The mesh decides the global cell count - count once, cached
   PetscCall(CountOwnedPoints(dm_, comm_, c_start_, c_end_, &n_global_cells_));

   PetscFunctionReturn(PETSC_SUCCESS);
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

// The layout, the geometry, the BC classification and the COO sparsity
// This happens on the host but we only need to do it once
PetscErrorCode UnstructuredDG::create(PhaseSpace &ps, const AngularQuadrature &quad, const BCSpec &bcs, \
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
   PetscCheck(quad.dimension() == dim_, comm_, PETSC_ERR_ARG_INCOMP, "a %" PetscInt_FMT "D quadrature was " \
      "handed to a %" PetscInt_FMT "D mesh", quad.dimension(), dim_);
   PetscCheck(quad.n_angles() == ps.n_angles, comm_, PETSC_ERR_ARG_INCOMP, "quadrature has %" PetscInt_FMT \
      " angles but the phase space has %" PetscInt_FMT, quad.n_angles(), ps.n_angles);
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
   const PetscInt n_angles = quad.n_angles();
   const PetscScalar sum_weights = quad.sum_weights();
   // The ordinates flattened (a * 3 + d) - the layout the device view has, so
   // the host classification reads them exactly as the fill does
   const PetscScalar *omega = quad.omega_host();
   const PetscInt *reflect[3] = {quad.reflect_host(0), quad.reflect_host(1), quad.reflect_host(2)};
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

   PetscCall(MarkOwnedPoints(dm_, c_start_, c_end_, owned));
   PetscCall(CheckPlexStratumLayout(dm_, gsec, c_start_, c_end_, owned, rows_per_cell, ps.n_cells, "cell", &rstart));

   // Local cell k is the k-th owned cell in point order - what painting paints
   paint_point_.clear();
   for (PetscInt c = c_start_; c < c_end_; c++) {
      if (owned[c - c_start_]) paint_point_.push_back(c);
   }
   const std::vector<PetscInt> &cell_of_local = paint_point_;
   const PetscInt local_cells = (PetscInt)cell_of_local.size();

   // The DM decided the decomposition - the PhaseSpace is told
   ps.local_cells = local_cells;
   ps_ = ps;
   const PetscInt local_rows = ps.local_rows();

   // ~~~~~~~~~~
   // Geometry, once on the host. Per owned cell its volume and centroid; per
   // face in cone order its outward area-weighted normal, its centroid, its
   // neighbour's global row base (-1 on the boundary) and, on the boundary,
   // its "Face Sets" value. A bad cell or face is recorded and everybody
   // errors together below
   // ~~~~~~~~~~
   PetscCall(DMGetLabel(dm_, "Face Sets", &face_sets));

   centroid_h_.assign(3 * local_cells, 0.0);
   volume_h_.assign(local_cells, 0.0);
   cell_face_offset_h_.assign(local_cells + 1, 0);
   face_nA_h_.clear();
   face_neighbour_row_h_.clear();
   face_label_h_.clear();
   face_distance_h_.clear();
   std::vector<PetscReal> face_centroid;
   std::vector<PetscInt> face_axis;
   std::vector<PetscInt> face_point;

   for (PetscInt k = 0; k < local_cells; k++) {

      const PetscInt c = cell_of_local[k];
      PetscReal vol = 0.0, cen[3] = {0.0, 0.0, 0.0};
      PetscCall(DMPlexComputeCellGeometryFVM(dm_, c, &vol, cen, NULL));
      if (!(vol > 0.0)) PetscCall(RecordFailure(&failed, message, sizeof(message), "cell %" PetscInt_FMT \
         " has non-positive volume %g", c, (double)vol));
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
         // is the in-plane one
         PetscReal area = 0.0, fcen[3] = {0.0, 0.0, 0.0}, normal[3] = {0.0, 0.0, 0.0};
         PetscCall(DMPlexComputeCellGeometryFVM(dm_, f, &area, fcen, normal));
         if (!(area > 0.0)) PetscCall(RecordFailure(&failed, message, sizeof(message), "face %" PetscInt_FMT \
            " of cell %" PetscInt_FMT " has non-positive area %g", f, c, (double)area));

         PetscScalar nA[3];
         if (!OutwardAreaNormal(dim, area, normal, fcen, cen, nA)) PetscCall(RecordFailure(&failed, message, \
            sizeof(message), "face %" PetscInt_FMT " has no normal", f));
         for (PetscInt d = 0; d < 3; d++) {
            face_nA_h_.push_back(nA[d]);
            face_centroid.push_back(fcen[d]);
         }
         face_axis.push_back(DominantAxis(nA, dim));
         face_point.push_back(f);

         // This cell's centroid to the face's plane, along the outward unit
         // normal - the half-distance a two-point flux across the face uses
         // (DSAPrecon). Positive for any convex cell
         PetscReal dist_own = 0.0;
         for (PetscInt d = 0; d < dim; d++) dist_own += PetscRealPart(nA[d]) / area * (fcen[d] - cen[d]);
         if (!(dist_own > 0.0)) PetscCall(RecordFailure(&failed, message, sizeof(message), "the centroid of cell %" \
            PetscInt_FMT " is not inside its face %" PetscInt_FMT, c, f));
         face_distance_h_.push_back(dist_own);

         const PetscInt *support = nullptr;
         PetscInt n_support = 0;
         PetscCall(DMPlexGetSupportSize(dm_, f, &n_support));
         PetscCall(DMPlexGetSupport(dm_, f, &support));
         if (n_support != 1 && n_support != 2) PetscCall(RecordFailure(&failed, message, sizeof(message), \
            "face %" PetscInt_FMT " is shared by %" PetscInt_FMT " cells", f, n_support));

         if (n_support == 2) {

            // An interior face - the neighbour is owned or an overlap ghost,
            // and the global section knows its row either way
            const PetscInt other = (support[0] == c) ? support[1] : support[0];
            PetscInt g = 0;
            PetscBool is_owned = PETSC_FALSE;
            PetscCall(GlobalPointOffset(gsec, other, &g, &is_owned));
            face_neighbour_row_h_.push_back(g);
            face_label_h_.push_back(-1);

            // And the face's plane to the neighbour's centroid, the other half
            PetscReal ncen[3] = {0.0, 0.0, 0.0}, dist_nb = 0.0;
            PetscCall(DMPlexComputeCellGeometryFVM(dm_, other, NULL, ncen, NULL));
            for (PetscInt d = 0; d < dim; d++) dist_nb += PetscRealPart(nA[d]) / area * (ncen[d] - fcen[d]);
            if (!(dist_nb > 0.0)) PetscCall(RecordFailure(&failed, message, sizeof(message), "the centroid of " \
               "cell %" PetscInt_FMT " is not inside its face %" PetscInt_FMT, other, f));
            face_distance_h_.push_back(dist_nb);

         } else {

            // A boundary face. With a one-cell face-adjacent overlap, an OWNED
            // cell's partition-boundary faces all have their neighbour in the
            // local mesh, so support size 1 here really is the domain boundary
            PetscInt value = -1;
            if (face_sets) PetscCall(DMLabelGetValue(face_sets, f, &value));
            face_neighbour_row_h_.push_back(-1);
            face_label_h_.push_back(value);
            face_distance_h_.push_back(0.0);
            PetscCall(CheckBoundaryFaceBC(bcs, dim, value, nA, face_axis.back(), area, f, &failed, message, \
               sizeof(message)));
         }
      }
   }
   PetscCall(CollectiveFailure(comm_, failed, message));
   PetscCall(CheckBCLabelsSeen(comm_, bcs, face_label_h_));

   // ~~~~~~~~~~
   // COO coordinates. DG0: n_faces + 1 entries per row, in slot order: one
   // per face in cone order, then the diagonal. DG1: the same with every
   // entry widened to n_basis - see the header and the DG1 branch below. The
   // rest of this comment is DG0's
   // ~~~~~~~~~~
   // A -1 row/col index in the COO format says just ignore this entry. A face
   // slot is live only on an interior row, across an interior face, for an
   // angle flowing IN through it: the upwind selection, made here once, so
   // the value fill never branches on which neighbour is upwind
   //
   // Under GHOST_FLUX a row keeps the interior row's slots - its boundary
   // faces nulled by the same "neighbour exists" test - and the streaming
   // term writes its full diagonal; |Omega . nA_f| / V_c times each vacuum
   // inflow face's per-angle inflow (windowed per face, summed over faces)
   // goes to the rhs - the upwind face flux with the inflow as the ghost value
   // outside. A REFLECTIVE inflow face is a face flux the same way, its ghost
   // value the mirrored angle in this cell, so its slot points there, face by
   // face - a row coming in through both kinds takes both, with no precedence
   // rule. Under DIRICHLET_CELL a reflective row repurposes the slot of its
   // first incoming reflective face for the coupling to the mirrored angle,
   // and a Dirichlet row keeps only its diagonal
   const PetscBool ghost = bcs.ghost_flux_vacuum();
   oor_.clear();
   ooc_.clear();
   std::vector<PetscInt> row_slot_offset(local_rows + 1, 0);
   std::vector<PetscInt> diag_slot(local_rows, 0);
   BoundaryRows rows;
   rows.reset(local_rows, ghost);

   if (order == 1) {

      // DG1: the basis, the fan geometry and the face matrices first - the
      // ghost inflow below reads the basis at the face centroids
      std::vector<PetscScalar> face_own, face_up, basis_grad;
      std::vector<PetscReal> face_basis_value;
      PetscCall(build_dg1_geometry(face_own, face_up, basis_grad, face_basis_value, bcs, &failed, message, \
         sizeof(message)));

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
                  const PetscScalar s = FaceFlux(omega, a, face_nA_h_.data(), kf);
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
                  const PetscInt n_t = FaceWindowCoords(&face_centroid[3 * kf], face_axis[kf], dim, t);
                  if (InWindow(bc, t, n_t)) rows.ghost_inflow[r] += -s / volume_h_[k] * \
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
                     sl - row_slot_offset[r], cell_of_local[k]);
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
      // The host buffers move into members below, so they outlive the copies
      Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), face_own_d_, face_own_h);
      Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), face_up_d_, face_up_h);
      Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), basis_grad_d_, basis_grad_h);
      face_own_h_ = std::move(face_own);
      face_up_h_ = std::move(face_up);
      basis_grad_h_ = std::move(basis_grad);

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
               if (!(PetscRealPart(FaceFlux(omega, a, face_nA_h_.data(), kf)) < 0.0)) continue;
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
                  const PetscScalar s = FaceFlux(omega, a, face_nA_h_.data(), kf);
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
                  const PetscInt n_t = FaceWindowCoords(&face_centroid[3 * kf], face_axis[kf], dim, t);
                  if (InWindow(bc, t, n_t)) rows.ghost_inflow[r] += -s / volume_h_[k] * \
                     ((PetscScalar)bc.inflow / sum_weights);
               }

            } else if (any_vacuum) {

               // Dirichlet row: identity, and the rhs takes the winning face's
               // inflow if the face centroid is inside its window
               rows.is_bc_row[r] = 1;
               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  oor_.push_back(-1);
                  ooc_.push_back(-1);
               }
               const BCFace bc = bcs.face(face_label_h_[win]);
               PetscReal t[2] = {0.0, 0.0};
               const PetscInt n_t = FaceWindowCoords(&face_centroid[3 * win], face_axis[win], dim, t);
               rows.dirichlet_value[r] = InWindow(bc, t, n_t) ? (PetscScalar)bc.inflow / sum_weights : \
                  (PetscScalar)0.0;

            } else {

               // Reflective row (Dirichlet-cell only): psi(a) - psi(partner) = 0
               // in the same cell, the partner mirrored in every axis the
               // direction came in through
               rows.is_bc_row[r] = 1;
               PetscInt partner = a;
               for (PetscInt d = 0; d < dim; d++) {
                  if (reflect_axes & (1 << d)) partner = reflect[d][partner];
               }

               // The partner is outgoing through every face this direction came
               // in through. Its row may still be a Dirichlet row - a
               // reflective axis plane meeting a slanted vacuum face, where
               // psi(a) = psi(partner) = the inflow is a good pair of
               // equations - but not a REFLECTIVE one: the two rows would each
               // define the other (a single-cell-wide direction between two
               // reflective faces)
               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  const PetscInt kf = k0 + lf;
                  if (face_neighbour_row_h_[kf] >= 0) continue;
                  if (bcs.type(face_label_h_[kf]) != BCType::REFLECT) continue;
                  if (PetscRealPart(FaceFlux(omega, partner, face_nA_h_.data(), kf)) < 0.0) \
                     PetscCall(RecordFailure(&failed, message, sizeof(message), "the reflection partner of cell %" \
                     PetscInt_FMT " angle %" PetscInt_FMT " is itself a reflective boundary row - a reflective " \
                     "face on a single-cell-wide direction between two reflective faces is not supported", \
                     cell_of_local[k], a));
               }

               for (PetscInt lf = 0; lf < n_faces; lf++) {
                  // Same cell, mirrored angle - an owned row, always rank-local
                  oor_.push_back(lf == first_reflect ? row : -1);
                  ooc_.push_back(lf == first_reflect ? rstart + k * n_angles + partner : -1);
               }
               rows.reflect_slot[r] = row_slot_offset[r] + first_reflect;
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
                  s - row_slot_offset[r], cell_of_local[k]);
            }
         }
      }
   }
   PetscCall(CollectiveFailure(comm_, failed, message));
   row_slot_offset[local_rows] = (PetscInt)oor_.size();

   PetscCall(set_pattern(row_slot_offset, diag_slot, rows));

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
   paint_centroid_d_ = PetscScalarKokkosView("centroid_d", 3 * local_cells);

   // const_cast only because the unmanaged host view type is non-const; the
   // deep_copy reads it
   PetscScalarKokkosViewHostUnmanaged omega_h(const_cast<PetscScalar *>(omega), 3 * n_angles);
   PetscIntKokkosViewHostUnmanaged cell_face_offset_h(cell_face_offset_h_.data(), local_cells + 1);
   PetscScalarKokkosViewHostUnmanaged face_nA_h(face_nA_h_.data(), 3 * n_cell_faces);
   PetscScalarKokkosViewHostUnmanaged inv_volume_h(inv_volume.data(), local_cells);
   PetscScalarKokkosViewHostUnmanaged centroid_h(centroid.data(), 3 * local_cells);
   auto exec = PetscGetKokkosExecutionSpace();
   Kokkos::deep_copy(exec, omega_d_, omega_h);
   Kokkos::deep_copy(exec, cell_face_offset_d_, cell_face_offset_h);
   Kokkos::deep_copy(exec, face_nA_d_, face_nA_h);
   Kokkos::deep_copy(exec, inv_volume_d_, inv_volume_h);
   Kokkos::deep_copy(exec, paint_centroid_d_, centroid_h);
   // The copies are asynchronous and inv_volume and centroid die here
   exec.fence();

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
   std::vector<PetscReal> &face_basis_value, const BCSpec &bcs, PetscBool *failed, char *message, size_t len)
{
   PetscSection csec = NULL;
   Vec coords_vec = NULL;
   const PetscScalar *coords = nullptr;
   const PetscInt dim = dim_, nb = n_basis_;
   const PetscInt local_cells = (PetscInt)paint_point_.size();
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
      if (!(PetscAbsReal(vol[m] - fvm_vol) <= 1e-8 * fvm_vol)) PetscCall(RecordFailure(failed, message, len, \
         "cell %" PetscInt_FMT ": fan volume %.15g against PETSc's %.15g - DG1 needs planar faces", \
         c, (double)vol[m], (double)fvm_vol));
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
   face_nb_grad_h_.assign(n_cell_faces * nb * 3, 0.0);

   for (PetscInt k = 0; k < local_cells; k++) {

      const PetscInt c = paint_point_[k];
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
         if (!(PetscAbsReal(area - fvm_area) <= 1e-8 * fvm_area)) PetscCall(RecordFailure(failed, message, len, \
            "face %" PetscInt_FMT ": fan area %.15g against PETSc's %.15g - DG1 needs planar faces", \
            f, (double)area, (double)fvm_area));

         // The upwind cell across this face: the neighbour through an
         // interior face, this cell itself through a reflective one (the
         // mirrored angle is the column's business), none through vacuum
         PetscInt up = -1;
         const PetscInt *support = nullptr;
         PetscInt n_support = 0;
         PetscCall(DMPlexGetSupportSize(dm_, f, &n_support));
         PetscCall(DMPlexGetSupport(dm_, f, &support));
         if (n_support == 2) {
            up = ((support[0] == c) ? support[1] : support[0]) - c_start_;
            for (PetscInt j = 0; j < nb; j++) {
               for (PetscInt d = 0; d < 3; d++) face_nb_grad_h_[(kf * nb + j) * 3 + d] = grad(up, j, d);
            }
         }
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
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

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
