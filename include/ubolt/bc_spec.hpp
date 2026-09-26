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

// How a VACUUM face is discretised. Two treatments of the same physics - a
// prescribed incoming flux - that differ in whether the boundary cell stays an
// unknown
//
// GHOST_FLUX (the default since Sep 2026): the boundary cell stays an ordinary
// unknown. Its row carries the full upwind stencil - the same diagonal
// sum_a |Omega_a| / h_a every interior row has - and the one off-diagonal whose
// upwind neighbour lies outside the domain is simply absent, its contribution
// |Omega_a| / h_a * psi_in moved to the rhs. That is the usual upwind flux with
// a ghost cell holding psi_in, and what a DG face flux does anyway
//
// A REFLECTIVE face is treated the same way under GHOST_FLUX: the ghost cell
// behind it holds the mirrored direction's flux in the same boundary cell, so
// the outside-pointing entry couples to that mirrored angle instead of being
// dropped, and there are NO boundary rows at all (the reflective row
// psi(a) - psi(mirror a) = 0 belongs to DIRICHLET_CELL only)
//
// Why it is the default: with no boundary rows left, A_{-d} is the exact
// transpose of A_d on EVERY row rather than only the interior ones, because a
// row's stencil no longer depends on which direction it is for. With P
// swapping (cell, Omega) and (cell, -Omega): A^T = P A P for ANY combination
// of vacuum and reflective faces - inflow values and windows only touch the
// rhs - on the assembled streaming + removal operator (the block PCAIR
// inverts; sigma_t may vary cell to cell), in every structured backend. On the
// unstructured DG backend it is the volume-weighted (V A)^T = P (V A) P (see
// unstructured_dg.hpp), which is the plain identity on equal volumes. It does
// NOT hold under DIRICHLET_CELL, and it is not a property of the matrix-free
// isotropic scatter, which commutes with P but is not symmetric once the
// quadrature weights differ (level-symmetric beyond S4). See the
// transposed-solve study for what that buys
//
// DIRICHLET_CELL (opt-in, and everything UBOLT did before Sep 2026): the
// boundary cell's row for an incoming direction is REPLACED by the identity
// and the rhs there carries the incoming flux. The cell is not an unknown for
// that direction; its equation says psi = psi_in. It puts the boundary at the
// boundary cell's centre rather than its face, so the two differ by O(h) there
enum class VacuumTreatment { DIRICHLET_CELL, GHOST_FLUX };

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

   // How every vacuum face of this spec is discretised - see VacuumTreatment.
   // A whole-spec setting rather than a per-face one: it is a choice about the
   // discretisation, not about the physics of one face
   void set_vacuum_treatment(VacuumTreatment treatment) { vacuum_treatment_ = treatment; }
   VacuumTreatment vacuum_treatment() const { return vacuum_treatment_; }
   PetscBool ghost_flux_vacuum() const
   {
      return (PetscBool)(vacuum_treatment_ == VacuumTreatment::GHOST_FLUX);
   }

   BCFace face(PetscInt label_id) const
   {
      const auto found = map_.find(label_id);
      return found == map_.end() ? BCFace() : found->second;
   }

   // Every label id the spec was given something for. A backend checks these
   // against the faces its mesh actually has, so a boundary condition on a
   // label no face carries is an error rather than a silently cold face
   const std::map<PetscInt, BCFace> &faces() const { return map_; }

private:
   std::map<PetscInt, BCFace> map_;
   VacuumTreatment vacuum_treatment_ = VacuumTreatment::GHOST_FLUX;
};

#endif
