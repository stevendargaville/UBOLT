#ifndef UBOLT_PLEX_COMMON_HPP
#define UBOLT_PLEX_COMMON_HPP

// The DMPlex plumbing the two unstructured backends share - UnstructuredDG
// (cell dofs) and UnstructuredCG (vertex dofs): building and distributing the
// mesh, reading point ownership and global offsets, the collective failure
// idiom, the face helpers and the box painting kernel
//
// INTERNAL: it lives under src/ so it can never enter the public include tree
// (it is not a library interface, just two translation units' shared code).
// Everything here is static inline (no unused-function warning in the TU that
// skips a helper), so each TU gets its own copy; the Makefile
// declares the two objects' dependency on it by hand

#include "ubolt/types.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/unstructured_dg.hpp"
#include "petsc_kokkos.hpp"
#include <petscdmplex.h>
#include <petscsf.h>
#include <vector>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Which points in [start, end) of the local mesh this rank OWNS: everything
// that is not a leaf of the point SF. After a distribution with overlap the
// local mesh also holds the neighbouring ranks' points across the partition
// boundary, and those are the SF's leaves. An SF with no graph (one rank,
// never distributed) has no leaves at all. Works on any stratum - cells for
// the DG backend, vertices for the CG one
static inline PetscErrorCode MarkOwnedPoints(DM dm, PetscInt start, PetscInt end, std::vector<PetscInt> &owned)
{
   PetscSF sf = NULL;
   PetscInt n_roots = 0, n_leaves = 0;
   const PetscInt *leaves = nullptr;

   PetscFunctionBeginUser;

   owned.assign(end - start, 1);
   PetscCall(DMGetPointSF(dm, &sf));
   PetscCall(PetscSFGetGraph(sf, &n_roots, &n_leaves, &leaves, NULL));
   if (n_roots >= 0) {
      for (PetscInt l = 0; l < n_leaves; l++) {
         const PetscInt p = leaves ? leaves[l] : l;
         if (p >= start && p < end) owned[p - start] = 0;
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build (or read) the mesh and distribute it with a one-cell overlap, under
// the given adjacency - (useCone, !useClosure) is face adjacency, the DG
// backend's: the overlap is the face neighbours only. (!useCone, useClosure)
// is PETSc's FEM adjacency, the CG backend's: every cell sharing any point -
// so any vertex - with an owned cell, which puts the whole star of every owned
// vertex on this rank
//
// As in the DMDA backends we deliberately do NOT call DMSetFromOptions: it
// would expose -dm_plex_box_faces, -dm_refine and friends, any of which could
// resize the mesh out from under the PhaseSpace that every other object is
// about to be sized from
static inline PetscErrorCode UboltCreatePlexMesh(MPI_Comm comm, const PlexMeshSpec &mesh, PetscBool use_cone, \
   PetscBool use_closure, DM *dm_out)
{
   DM dm = NULL, dm_dist = NULL;
   PetscPartitioner part = NULL;
   PetscInt dm_dim = 0;

   PetscFunctionBeginUser;

   PetscCheck(mesh.dimension == 2 || mesh.dimension == 3, comm, PETSC_ERR_ARG_OUTOFRANGE, \
      "the unstructured backend is 2D or 3D, was asked for dimension %" PetscInt_FMT \
      " (the FD slab is the 1D backend)", mesh.dimension);

   if (!mesh.file.empty()) {

      // Interpolated, so the faces exist as points - the DG0 fluxes live on
      // them, and the CG boundary integrals too. Gmsh files come with "Cell
      // Sets" and "Face Sets" labels from their physical groups
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

   // Set BEFORE distributing: it decides what the overlap brings in
   PetscCall(DMSetAdjacency(dm, PETSC_DEFAULT, use_cone, use_closure));

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

   *dm_out = dm;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Where a point's first dof sits in the GLOBAL numbering. The global section
// encodes a point this rank does not own - an overlap ghost - as -(g + 1), so
// the one lookup serves owned points and the ghosts a column points into alike
static inline PetscErrorCode GlobalPointOffset(PetscSection gsec, PetscInt p, PetscInt *g_out, PetscBool *is_owned)
{
   PetscInt g = 0;

   PetscFunctionBeginUser;

   PetscCall(PetscSectionGetOffset(gsec, p, &g));
   *is_owned = (PetscBool)(g >= 0);
   *g_out = (g >= 0) ? g : -(g + 1);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A boundary configuration create() cannot build (a reflective face that is
// not axis-aligned, a reflection partner that is itself a boundary row, a
// window of the wrong shape) is found by whichever rank owns the offending
// point - but create() is collective, and a rank that returned early would
// leave the others waiting in the next collective call. So the failure is
// recorded, create() carries on to this point on every rank, and everybody
// errors together: the rank that found it with its message, the rest pointing
// at it
static inline PetscErrorCode CollectiveFailure(MPI_Comm comm, PetscBool local_fail, const char *message)
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
// Painting
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda. The boxes ride along as flat views -
// 2 * dim (lo, hi) values per box, axis by axis - because a view of structs is
// not one of the typedefs types.hpp allows
static inline void PaintBoxesKernel(PetscIntKokkosView mat_id_d, PetscScalarKokkosView centroid_d, \
   PetscScalarKokkosView box_lohi_d, PetscIntKokkosView box_material_d, PetscInt n_boxes, \
   PetscInt dim, PetscInt n_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, n_cells), KOKKOS_LAMBDA(PetscInt c) {

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

// Check and upload the flattened boxes and paint them over mat_id_d (n_cells
// entries, centroids 3 per cell), later ones winning
static inline PetscErrorCode PaintFlatBoxes(MPI_Comm comm, PetscInt dim, PetscInt n_cells, \
   const PetscScalarKokkosView &centroid_d, PetscInt n_boxes, const std::vector<PetscScalar> &box_lohi, \
   const std::vector<PetscInt> &box_material, PetscIntKokkosView &mat_id_d)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck((PetscInt)mat_id_d.extent(0) == n_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "mat_id_d covers %" PetscInt_FMT " cells but there are %" PetscInt_FMT " local cells", \
      (PetscInt)mat_id_d.extent(0), n_cells);
   for (PetscInt b = 0; b < n_boxes; b++) {
      PetscCheck(box_material[b] >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "box %" PetscInt_FMT "'s material index %" PetscInt_FMT " is negative", b, box_material[b]);
      for (PetscInt d = 0; d < dim; d++) {
         PetscCheck(PetscRealPart(box_lohi[2 * dim * b + 2 * d]) <= PetscRealPart(box_lohi[2 * dim * b + 2 * d + 1]), \
            comm, PETSC_ERR_ARG_OUTOFRANGE, "box %" PetscInt_FMT " is inside out on axis %" PetscInt_FMT \
            " - give each axis as lo, hi with lo <= hi", b, d);
      }
   }

   PetscScalarKokkosView box_lohi_d("box_lohi_d", 2 * dim * n_boxes);
   PetscIntKokkosView box_material_d("box_material_d", n_boxes);
   if (n_boxes > 0) {
      PetscScalarKokkosViewHostUnmanaged box_lohi_h(const_cast<PetscScalar *>(box_lohi.data()), 2 * dim * n_boxes);
      PetscIntKokkosViewHostUnmanaged box_material_h(const_cast<PetscInt *>(box_material.data()), n_boxes);
      Kokkos::deep_copy(box_lohi_d, box_lohi_h);
      Kokkos::deep_copy(box_material_d, box_material_h);
   }

   PaintBoxesKernel(mat_id_d, centroid_d, box_lohi_d, box_material_d, n_boxes, dim, n_cells);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// The boxes, flattened to 2 * dim (lo, hi) pairs per box, axis by axis
static inline void FlattenBoxes(const std::vector<MaterialBox2D> &boxes, std::vector<PetscScalar> &box_lohi, \
   std::vector<PetscInt> &box_material)
{
   const PetscInt n_boxes = (PetscInt)boxes.size();
   box_lohi.resize(4 * n_boxes);
   box_material.resize(n_boxes);
   for (PetscInt b = 0; b < n_boxes; b++) {
      box_lohi[4 * b]     = boxes[b].x0;
      box_lohi[4 * b + 1] = boxes[b].x1;
      box_lohi[4 * b + 2] = boxes[b].y0;
      box_lohi[4 * b + 3] = boxes[b].y1;
      box_material[b] = boxes[b].material;
   }
}

static inline void FlattenBoxes(const std::vector<MaterialBox3D> &boxes, std::vector<PetscScalar> &box_lohi, \
   std::vector<PetscInt> &box_material)
{
   const PetscInt n_boxes = (PetscInt)boxes.size();
   box_lohi.resize(6 * n_boxes);
   box_material.resize(n_boxes);
   for (PetscInt b = 0; b < n_boxes; b++) {
      box_lohi[6 * b]     = boxes[b].x0;
      box_lohi[6 * b + 1] = boxes[b].x1;
      box_lohi[6 * b + 2] = boxes[b].y0;
      box_lohi[6 * b + 3] = boxes[b].y1;
      box_lohi[6 * b + 4] = boxes[b].z0;
      box_lohi[6 * b + 5] = boxes[b].z1;
      box_material[b] = boxes[b].material;
   }
}

// "Cell Sets" label value -> material, over the given cell points (any
// subset of the local mesh, in the order the mat_id_d view is indexed in).
// The background everywhere else; errors if the map is non-empty and the mesh
// has no "Cell Sets" label
static inline PetscErrorCode PaintCellSets(DM dm, MPI_Comm comm, PetscInt background_material, \
   const std::map<PetscInt, PetscInt> &label_to_material, const std::vector<PetscInt> &cell_points, \
   PetscIntKokkosView &mat_id_d)
{
   DMLabel cell_sets = NULL;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(background_material >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);
   for (const auto &entry : label_to_material) {
      PetscCheck(entry.second >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "cell set %" PetscInt_FMT \
         " maps to a negative material index %" PetscInt_FMT, entry.first, entry.second);
   }

   PetscCall(DMGetLabel(dm, "Cell Sets", &cell_sets));
   PetscCheck(cell_sets || label_to_material.empty(), comm, PETSC_ERR_ARG_WRONG, \
      "cell sets were given materials but the mesh has no \"Cell Sets\" label");

   const PetscInt n_cells = (PetscInt)cell_points.size();
   std::vector<PetscInt> mat_id(n_cells, background_material);
   if (cell_sets) {
      for (PetscInt k = 0; k < n_cells; k++) {
         PetscInt value = -1;
         PetscCall(DMLabelGetValue(cell_sets, cell_points[k], &value));
         const auto found = label_to_material.find(value);
         if (found != label_to_material.end()) mat_id[k] = found->second;
      }
   }

   mat_id_d = PetscIntKokkosView("mat_id_d", n_cells);
   PetscIntKokkosViewHostUnmanaged mat_id_h(mat_id.data(), n_cells);
   Kokkos::deep_copy(mat_id_d, mat_id_h);

   PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
