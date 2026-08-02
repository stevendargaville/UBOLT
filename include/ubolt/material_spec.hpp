#ifndef UBOLT_MATERIAL_SPEC_HPP
#define UBOLT_MATERIAL_SPEC_HPP

#include "ubolt/types.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/coo_pattern.hpp"
#include "ubolt/sn_quadrature.hpp"
#include <vector>

// What each material is made of: per-material, per-group cross sections and an
// external source
//
// This is BCSpec's sibling for cell data. The split is the same one: the spec
// says what a material means, and WHICH cells are which material is geometry,
// so it stays on the concrete backends - a per-local-cell material index view
// painted by StructuredFD1D::paint_intervals / StructuredFD2D::paint_boxes,
// and by a future unstructured backend reading a DMPlex "Cell Sets" label.
// Unlike BCSpec the tables here have to reach device kernels, so materials are
// DENSE indices 0 .. n_materials - 1 rather than arbitrary label ids - the
// unstructured backend will remap its label values to indices at paint time,
// a host-side step, the same place BCSpec is consulted at create
//
// The expansion onto the cells goes through the surfaces the terms already
// consume: GroupXSections::set_from_materials fills the per-cell xsection
// tables, and UboltFillSource below writes the external source into a rhs.
// The terms never learn that materials exist
//
// Groups are ordered high energy (0) to low, as in GroupXSections. A value
// that is never set is zero, so an untouched material is void
class PETSC_VISIBILITY_PUBLIC MaterialSpec {
public:
   PetscErrorCode create(PetscInt n_materials, PetscInt n_groups);

   PetscErrorCode set_sigma_t(PetscInt mat, PetscInt g, PetscScalar value);
   // g_from == g_to is the within-group scattering that stays on the lhs
   PetscErrorCode set_sigma_s(PetscInt mat, PetscInt g_from, PetscInt g_to, PetscScalar value);
   // The external source, as the angle-integrated (isotropic) strength -
   // UboltFillSource shares it out over the ordinates by dividing by the
   // quadrature's sum_weights
   PetscErrorCode set_source(PetscInt mat, PetscInt g, PetscScalar value);

   PetscInt n_materials() const { return n_materials_; }
   PetscInt n_groups() const { return n_groups_; }

   // The host tables the fill steps upload, material slowest:
   // sigma_t[mat * n_groups + g], sigma_s[(mat * n_groups + g_from) * n_groups
   // + g_to], source[mat * n_groups + g]
   const std::vector<PetscScalar> &sigma_t_host() const { return sigma_t_; }
   const std::vector<PetscScalar> &sigma_s_host() const { return sigma_s_; }
   const std::vector<PetscScalar> &source_host() const { return source_; }

private:
   PetscInt n_materials_ = 0;
   PetscInt n_groups_ = 0;
   std::vector<PetscScalar> sigma_t_;
   std::vector<PetscScalar> sigma_s_;
   std::vector<PetscScalar> source_;
};

// b = source(material(cell), g) / sum_weights on every row of every cell -
// the isotropic strength shared out over the quadrature's angular domain -
// EXCEPT the BC rows: the rhs there belongs to the boundary condition - the
// incoming flux on a Dirichlet row, zero on a reflective one - the same
// contract the terms and GroupTransfer::add_source have with the BC row mask.
// So fill b with the inflow value first (VecSet), call this, and let
// UboltZeroReflectRows take the inflow back off the reflective rows as usual
PETSC_EXTERN PetscErrorCode UboltFillSource(const PhaseSpace &ps, const BoundaryInfo &boundary, \
   const AngularQuadrature &quad, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, \
   PetscInt g, Vec b);

#endif
