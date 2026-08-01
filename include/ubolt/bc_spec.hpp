#ifndef UBOLT_BC_SPEC_HPP
#define UBOLT_BC_SPEC_HPP

#include <petscsys.h>
#include <map>

// Which boundary condition family each piece of the boundary belongs to
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
// A label id that was never set is VACUUM, so a default-constructed BCSpec is
// today's behaviour: prescribed inflow on every incoming direction, with the
// value taken from the right hand side
enum class BCType { VACUUM, REFLECT };

class PETSC_VISIBILITY_PUBLIC BCSpec {
public:
   void set(PetscInt label_id, BCType type) { map_[label_id] = type; }

   BCType type(PetscInt label_id) const
   {
      const auto found = map_.find(label_id);
      return found == map_.end() ? BCType::VACUUM : found->second;
   }

private:
   std::map<PetscInt, BCType> map_;
};

#endif
