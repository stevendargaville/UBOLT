#ifndef UBOLT_PLEX_DISCRETISATION_HPP
#define UBOLT_PLEX_DISCRETISATION_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/material_regions.hpp"
#include "ubolt/plex_mesh_spec.hpp"
#include <map>
#include <vector>

// What the two DMPlex backends (UnstructuredDG, UnstructuredCG) share: the
// mesh, built or read and distributed with a one-cell overlap by each
// backend's create_mesh, and the painting of materials onto it
//
// Painting covers each backend's material unit - the owned cells on the DG
// backend, every local element (overlap included) on the CG one - by CENTROID,
// with the structured paint_boxes semantics: the background everywhere, then
// the boxes in order with the later ones winning, inclusive at both ends.
// "Face Sets" ids are PETSc's box convention - the structured backends'
// FACE_* constants - so a BCSpec means the same thing on either kind of
// backend; a boundary face with no label takes BCSpec::face(-1), cold vacuum
// by default
class PETSC_VISIBILITY_PUBLIC PlexDiscretisation : public Discretisation {
public:
   PetscInt dimension() const { return dim_; }

   // Allocates mat_id_d sized n_material_entries(). The 2D overload requires
   // dimension 2, the 3D one dimension 3
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox2D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox3D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;
   // Paint boxes OVER an existing mat_id_d (from paint_cell_sets), later boxes
   // winning - how the driver layers "Cell Sets" then "paint".
   // paint_boxes(bg, boxes, out) is allocate + background + paint_boxes_over
   PetscErrorCode paint_boxes_over(const std::vector<MaterialBox2D> &boxes, PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_boxes_over(const std::vector<MaterialBox3D> &boxes, PetscIntKokkosView &mat_id_d) const;
   // "Cell Sets" label value -> material index. Allocates and fills mat_id_d:
   // the background everywhere, then each cell whose label value is in the map
   // takes that material. Errors if the map is non-empty and the mesh has no
   // "Cell Sets" label
   PetscErrorCode paint_cell_sets(PetscInt background_material, \
      const std::map<PetscInt, PetscInt> &label_to_material, PetscIntKokkosView &mat_id_d) const;

protected:
   PlexDiscretisation() = default;

   // Build (or read) and distribute the mesh with a one-cell overlap under the
   // given adjacency, and record the dimension and the cell range. No
   // DMSetFromOptions: the mesh must not be resized out from under the phase
   // space. Partitioned by PETSc's "simple" partitioner (deterministic, no
   // external package), overridable with -petscpartitioner_type
   PetscErrorCode create_plex_mesh(MPI_Comm comm, const PlexMeshSpec &mesh, PetscBool use_cone, \
      PetscBool use_closure);

   PetscInt dim_ = 0;
   // Height-0 point range of the local (overlapped) mesh
   PetscInt c_start_ = 0;
   PetscInt c_end_ = 0;
   // What the painting paints, set by the derived create(): the cell points in
   // mat_id_d order and their centroids, 3 per cell (z = 0 in 2D), on device
   std::vector<PetscInt> paint_point_;
   PetscScalarKokkosView paint_centroid_d_;

private:
   PetscErrorCode paint_flat_boxes(PetscInt box_dim, PetscInt n_boxes, const std::vector<PetscScalar> &box_lohi, \
      const std::vector<PetscInt> &box_material, PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode allocate_background(PetscInt background_material, PetscIntKokkosView &mat_id_d) const;
};

#endif
