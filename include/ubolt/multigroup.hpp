#ifndef UBOLT_MULTIGROUP_HPP
#define UBOLT_MULTIGROUP_HPP

#include "ubolt/types.hpp"
#include "ubolt/coo_pattern.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/material_spec.hpp"
#include <vector>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Multigroup cross sections, per local cell
//
// Groups are ordered high energy (0) to low, so g_from < g_to is downscatter
//
// The tables are LayoutRight with the group index (or pair) slowest, so fixing
// the group slices out a contiguous per-cell view. That is the whole point:
// RemovalTerm and ScatteringTerm keep taking a plain 1D per-cell view and never
// learn that groups exist - the group sweep just points them at a different
// slice and refills the values
class PETSC_VISIBILITY_PUBLIC GroupXSections {
public:
   PetscErrorCode create(const PhaseSpace &ps);
   // Sized explicitly rather than off the phase space's local cells: the CG
   // backend's rows are per vertex but its xsections per local ELEMENT
   // (UnstructuredCG::n_local_elements()). create(ps) is create(ps.n_groups,
   // ps.local_cells)
   PetscErrorCode create(PetscInt n_groups, PetscInt n_entries);

   // Constant across the slab, for problems with a single material.
   // Spatially varying data comes in through set_from_materials, or fills the
   // views through the accessors below
   PetscErrorCode set_sigma_t(PetscInt g, PetscScalar value);
   PetscErrorCode set_sigma_s(PetscInt g_from, PetscInt g_to, PetscScalar value);

   // Expand a material table onto the cells: every group's sigma_t and every
   // group pair's sigma_s, from each cell's painted material index (see
   // MaterialSpec for where mat_id_d comes from)
   PetscErrorCode set_from_materials(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d);

   // Total xsection for group g, indexed by local cell
   PetscScalarKokkosView sigma_t(PetscInt g) const;
   // Scattering from g_from into g_to, indexed by local cell. g_from == g_to is
   // the within-group scattering that stays on the lhs
   PetscScalarKokkosView sigma_s(PetscInt g_from, PetscInt g_to) const;

private:
   PetscInt n_groups_ = 0;
   PetscInt local_cells_ = 0;
   PetscScalar2DRightKokkosView sigma_t_d_;  // (group, cell)
   PetscScalar3DRightKokkosView sigma_s_d_;  // (group from, group to, cell)
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The part of group g's right hand side that is not the boundary: the
// external source and what scatters in from other groups
//
// Not an OperatorTerm: it does not act on the unknowns of the group being
// solved, it builds that group's rhs. The group sweep calls, per group g,
//   add_external(g, b), then add_transfer(g_from, g, b) for every g_from
//   already solved, solve, then set_scalar_flux(g, psi_g)
// Every call ADDS to b and leaves the BC rows alone (their rhs belongs to the
// boundary condition). One implementation per family of discretisation:
// GroupTransfer (per node) and GroupTransferCG (SUPG-weighted)
//
// In the group Gauss-Seidel sweep the within-group block sigma_s(g -> g)
// stays on the lhs and the off-diagonal blocks come here, evaluated with the
// flux of whichever groups have already been solved. With downscatter only
// that sweep is a block lower triangular solve, so one forward pass is exact
class PETSC_VISIBILITY_PUBLIC GroupSource {
public:
   virtual ~GroupSource() = default;

   // b += group g's external source
   virtual PetscErrorCode add_external(PetscInt g, Vec b) const = 0;
   // Integrate group g's angular flux and keep it. Once per solved group:
   // every group below it scatters from the same flux, so integrating on
   // demand in add_transfer would redo it once per target
   virtual PetscErrorCode set_scalar_flux(PetscInt g, Vec psi_g) = 0;
   // b += the scatter from g_from into g_to. set_scalar_flux(g_from, ...)
   // must have been called first
   virtual PetscErrorCode add_transfer(PetscInt g_from, PetscInt g_to, Vec b) const = 0;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// GroupSource for the per-node discretisations (the structured backends and
// UnstructuredDG): the external source (MaterialSourceTable::add_isotropic)
// and b += sigma_s(g_from -> g_to) phi(g_from) / sum_weights on every angle
class PETSC_VISIBILITY_PUBLIC GroupTransfer : public GroupSource {
public:
   // xs and boundary must outlive the object; mat_id_d is per local cell.
   // Any quadrature: like the scatter it only needs the weights
   PetscErrorCode create(const PhaseSpace &ps, const AngularQuadrature &quad, const GroupXSections &xs, \
      const BoundaryInfo &boundary, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d);

   PetscErrorCode add_external(PetscInt g, Vec b) const override;
   PetscErrorCode set_scalar_flux(PetscInt g, Vec psi_g) override;
   PetscErrorCode add_transfer(PetscInt g_from, PetscInt g_to, Vec b) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt n_basis_ = 1;
   PetscInt local_nodes_ = 0;
   PetscScalar sum_weights_ = 0.0;
   const GroupXSections *xs_ = nullptr;
   PetscScalar2DKokkosView w_d_;
   PetscIntKokkosView is_bc_row_d_;
   MaterialSourceTable source_;
   // The cached scalar flux of each group, (local_nodes, 1) apiece. Separate
   // allocations rather than one (group, cell) table: the angular integral
   // writes a 2D gemm output, and a slice of a 2D table would be a rank-2 view
   // whose layout no longer matches the default one on a device backend
   std::vector<PetscScalar2DKokkosView> phi_;
   std::vector<PetscBool> phi_set_;
   // Persistent scratch - never allocate device memory inside an apply
   PetscScalar2DKokkosView scalar_flux_d_;
};

#endif
