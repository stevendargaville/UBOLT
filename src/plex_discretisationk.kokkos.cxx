#include "ubolt/plex_discretisation.hpp"
#include "plex_commonk.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode PlexDiscretisation::create_plex_mesh(MPI_Comm comm, const PlexMeshSpec &mesh, PetscBool use_cone, \
   PetscBool use_closure)
{
   PetscFunctionBeginUser;

   PetscCheck(!dm_, comm, PETSC_ERR_ARG_WRONGSTATE, "create_mesh has already built this backend's mesh");

   comm_ = comm;
   PetscCall(UboltCreatePlexMesh(comm, mesh, use_cone, use_closure, &dm_));
   dim_ = mesh.dimension;
   PetscCall(DMPlexGetHeightStratum(dm_, 0, &c_start_, &c_end_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Painting, over paint_point_ by paint_centroid_d_
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode PlexDiscretisation::paint_flat_boxes(PetscInt box_dim, PetscInt n_boxes, \
   const std::vector<PetscScalar> &box_lohi, const std::vector<PetscInt> &box_material, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCheck(dim_ == box_dim, comm_, PETSC_ERR_ARG_INCOMP, "%" PetscInt_FMT "D boxes painted onto a %" \
      PetscInt_FMT "D mesh", box_dim, dim_);
   PetscCall(ps_.check_decomposed());
   PetscCall(PaintFlatBoxes(comm_, dim_, (PetscInt)paint_point_.size(), paint_centroid_d_, n_boxes, box_lohi, \
      box_material, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PlexDiscretisation::paint_boxes_over(const std::vector<MaterialBox2D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   std::vector<PetscScalar> box_lohi;
   std::vector<PetscInt> box_material;

   PetscFunctionBeginUser;

   UboltFlattenBoxes(boxes, box_lohi, box_material);
   PetscCall(paint_flat_boxes(2, (PetscInt)boxes.size(), box_lohi, box_material, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PlexDiscretisation::paint_boxes_over(const std::vector<MaterialBox3D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   std::vector<PetscScalar> box_lohi;
   std::vector<PetscInt> box_material;

   PetscFunctionBeginUser;

   UboltFlattenBoxes(boxes, box_lohi, box_material);
   PetscCall(paint_flat_boxes(3, (PetscInt)boxes.size(), box_lohi, box_material, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode PlexDiscretisation::allocate_background(PetscInt background_material, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps_.check_decomposed());
   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);

   mat_id_d = PetscIntKokkosView("mat_id_d", (PetscInt)paint_point_.size());
   Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), mat_id_d, background_material);

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PlexDiscretisation::paint_boxes(PetscInt background_material, const std::vector<MaterialBox2D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCall(allocate_background(background_material, mat_id_d));
   PetscCall(paint_boxes_over(boxes, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PlexDiscretisation::paint_boxes(PetscInt background_material, const std::vector<MaterialBox3D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCall(allocate_background(background_material, mat_id_d));
   PetscCall(paint_boxes_over(boxes, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DMPlex half of the material split: which cells are which material is a
// label on the mesh, and the label's arbitrary values are remapped here onto
// MaterialSpec's dense indices - a host step, the same place BCSpec is
// consulted
PetscErrorCode PlexDiscretisation::paint_cell_sets(PetscInt background_material, \
   const std::map<PetscInt, PetscInt> &label_to_material, PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());
   PetscCall(PaintCellSets(dm_, comm_, background_material, label_to_material, paint_point_, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
