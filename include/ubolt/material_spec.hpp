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
// so it stays on the concrete backends - a per-local-cell (per local element
// on the CG backend) material index view painted by the backends'
// paint_intervals / paint_boxes / paint_cell_sets. Unlike BCSpec the tables
// here have to reach device kernels, so materials are DENSE indices 0 ..
// n_materials - 1 rather than arbitrary label ids - the plex backends remap
// their "Cell Sets" label values to indices at paint time, on the host
//
// The expansion onto the cells goes through the surfaces the terms already
// consume: GroupXSections::set_from_materials fills the per-cell xsection
// tables, and MaterialSourceTable (through GroupSource::add_external) writes
// the external source into a rhs. The terms never learn that materials exist
//
// Groups are ordered high energy (0) to low, as in GroupXSections. A value
// that is never set is zero, so an untouched material is void
class PETSC_VISIBILITY_PUBLIC MaterialSpec {
public:
   PetscErrorCode create(PetscInt n_materials, PetscInt n_groups);

   PetscErrorCode set_sigma_t(PetscInt mat, PetscInt g, PetscScalar value);
   // g_from == g_to is the within-group scattering that stays on the lhs
   PetscErrorCode set_sigma_s(PetscInt mat, PetscInt g_from, PetscInt g_to, PetscScalar value);
   // The external source, as the angle-integrated (isotropic) strength - the
   // fills share it out over the ordinates by dividing by the quadrature's
   // sum_weights
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

// Error unless every painted index in mat_id_d is a material of mats - an
// index outside the tables would silently read garbage. One device reduction
PETSC_EXTERN PetscErrorCode UboltCheckMaterialIds(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d);

// The spec's external source table on the device, read through a painted
// material id view. The ids are checked and the table uploaded ONCE, at
// create, so each per-group fill below is one kernel launch
//
// Device views only, so no destroy(); it must be gone before PetscFinalize
class PETSC_VISIBILITY_PUBLIC MaterialSourceTable {
public:
   PetscErrorCode create(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d);

   // out(e) = source(material(e), g) for every entry of mat_id_d, as the
   // isotropic strength the spec holds (NOT divided by sum_weights). out is
   // as long as mat_id_d
   PetscErrorCode fill_entries(PetscInt g, const PetscScalarKokkosView &out) const;

   // b += source(material(cell), g) / sum_weights on every angle of every
   // local cell (mat_id_d is per local cell here), EXCEPT the BC rows: the rhs
   // there belongs to the boundary condition, the same contract the terms and
   // GroupSource::add_transfer have with the BC row mask
   //
   // It ADDS: under the default GHOST_FLUX a boundary cell's rhs carries the
   // external source AND the inflow UboltFillInflow put there first (under
   // DIRICHLET_CELL the rows touched are exactly the ones it leaves at zero,
   // so on a zeroed b adding is assigning, bitwise)
   //
   // At n_basis > 1 (DG1) only basis 0's rows: the basis is modal and
   // orthonormal with basis 0 the constant, and a cell-constant source
   // projects onto nothing else
   PetscErrorCode add_isotropic(PetscInt g, PetscInt n_angles, PetscInt n_basis, PetscScalar sum_weights, \
      const PetscIntKokkosView &is_bc_row_d, Vec b) const;

private:
   PetscInt n_groups_ = 0;
   PetscIntKokkosView mat_id_d_;
   PetscScalarKokkosView source_tab_d_;
};

// One-shot MaterialSourceTable::add_isotropic, with the phase space's checks:
// zero b and write the inflow first (VecSet + UboltFillInflow), call this, then
// UboltZeroReflectRows. The group sweep uses GroupSource::add_external instead,
// which keeps the table
PETSC_EXTERN PetscErrorCode UboltFillSource(const PhaseSpace &ps, const BoundaryInfo &boundary, \
   const AngularQuadrature &quad, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, \
   PetscInt g, Vec b);

// One-shot MaterialSourceTable::fill_entries: each entry's material's group-g
// source, one value per entry of mat_id_d (local cells, or local elements on
// the CG backend) - the source as an output field, the sibling of the sigma_t
// GroupXSections holds, not what a solve consumes
PETSC_EXTERN PetscErrorCode UboltFillCellSource(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, \
   PetscInt g, PetscScalarKokkosView cell_source_d);

#endif
