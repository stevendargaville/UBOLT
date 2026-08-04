#ifndef UBOLT_BC_SPEC_HPP
#define UBOLT_BC_SPEC_HPP

#include <petscsys.h>
#include <map>

// Which boundary condition family each piece of the boundary belongs to, and
// what a vacuum face lets in
//
// Keyed by an integer boundary label id, the way a DMPlex "Face Sets" label is:
// the structured backends define their face ids to match PETSc's box-mesh
// convention (see the FACE_* constants on StructuredFD1D / StructuredFD2D), and
// the unstructured phase will feed real "Face Sets" values through this same
// spec. The label only answers which family a face belongs to - what actually
// happens to each row is decided per direction by the physics (the sign of
// direction dot normal, which is the discretisation's upwind test), the same
// split PFLARE's adv_dg_upwind example makes
//
// A label id that was never set is VACUUM with inflow 0, so a
// default-constructed BCSpec is a cold box: every incoming direction is
// prescribed, and prescribed to zero
enum class BCType { VACUUM, REFLECT };

// One face's boundary condition: the family, the prescribed incoming flux on
// a vacuum face (0.0 = a cold vacuum face, the default) - an ISOTROPIC,
// ANGLE-INTEGRATED strength exactly like a material's Source, shared over the
// ordinates as inflow / sum_weights by the backend - and an optional
// tangential window restricting where that inflow enters. The window
// is [lo, hi] pairs over the face's TANGENTIAL axes in ascending global-axis
// order (one pair for a 2D face, two for a 3D face; a 1D face is a point and
// takes none), membership decided by the boundary cell's centre, inclusive at
// both ends - the same convention paint_intervals/paint_boxes use
struct PETSC_VISIBILITY_PUBLIC BCFace {
   BCType type = BCType::VACUUM;
   PetscReal inflow = 0.0;
   PetscInt n_window_pairs = 0;                  // 0 = whole face
   PetscReal window[4] = {0.0, 0.0, 0.0, 0.0};   // lo0, hi0, lo1, hi1
};

// A dumb holder: the window's shape against the dimension is validated by the
// only producer (the problem file parser) and again by each backend's create
class PETSC_VISIBILITY_PUBLIC BCSpec {
public:
   void set(PetscInt label_id, BCType type) { map_[label_id].type = type; }

   void set_inflow(PetscInt label_id, PetscReal value) { map_[label_id].inflow = value; }

   // n_pairs is 1 or 2; lo_hi holds 2 * n_pairs values, lo before hi per pair
   void set_window(PetscInt label_id, PetscInt n_pairs, const PetscReal *lo_hi)
   {
      BCFace &face = map_[label_id];
      face.n_window_pairs = n_pairs;
      for (PetscInt p = 0; p < 2 * n_pairs; p++) face.window[p] = lo_hi[p];
   }

   BCType type(PetscInt label_id) const { return face(label_id).type; }

   BCFace face(PetscInt label_id) const
   {
      const auto found = map_.find(label_id);
      return found == map_.end() ? BCFace() : found->second;
   }

private:
   std::map<PetscInt, BCFace> map_;
};

#endif
