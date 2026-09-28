#ifndef UBOLT_MATERIAL_REGIONS_HPP
#define UBOLT_MATERIAL_REGIONS_HPP

#include <petscsys.h>
#include <vector>

// The axis-aligned regions the backends' painting methods take: one material
// each, membership decided by the cell centre (the element centroid on the
// plex backends), inclusive at both ends, later regions winning

// An interval, for StructuredFD1D::paint_intervals
struct PETSC_VISIBILITY_PUBLIC MaterialInterval1D {
   PetscScalar x0 = 0.0, x1 = 0.0;
   PetscInt material = 0;
};

// A box, for the 2D paint_boxes / paint_boxes_over
struct PETSC_VISIBILITY_PUBLIC MaterialBox2D {
   static constexpr PetscInt dim = 2;
   PetscScalar x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0;
   PetscInt material = 0;
   // (lo, hi) per axis, axis by axis - the flat layout the paint kernels read
   void lohi(PetscScalar *out) const { out[0] = x0; out[1] = x1; out[2] = y0; out[3] = y1; }
};

// A box, for the 3D paint_boxes / paint_boxes_over
struct PETSC_VISIBILITY_PUBLIC MaterialBox3D {
   static constexpr PetscInt dim = 3;
   PetscScalar x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0, z0 = 0.0, z1 = 0.0;
   PetscInt material = 0;
   void lohi(PetscScalar *out) const
   {
      out[0] = x0; out[1] = x1; out[2] = y0; out[3] = y1; out[4] = z0; out[5] = z1;
   }
};

// The boxes flattened for the device - 2 * dim (lo, hi) values per box, axis
// by axis, and the materials - because a view of structs is not one of the
// typedefs types.hpp allows
template <class Box>
inline void UboltFlattenBoxes(const std::vector<Box> &boxes, std::vector<PetscScalar> &box_lohi, \
   std::vector<PetscInt> &box_material)
{
   const PetscInt n_boxes = (PetscInt)boxes.size();
   box_lohi.resize(2 * Box::dim * n_boxes);
   box_material.resize(n_boxes);
   for (PetscInt b = 0; b < n_boxes; b++) {
      boxes[b].lohi(&box_lohi[2 * Box::dim * b]);
      box_material[b] = boxes[b].material;
   }
}

#endif
