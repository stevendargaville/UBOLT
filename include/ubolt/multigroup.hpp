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
//   with a cached flux, solve, then set_scalar_flux(g, psi_g, ...)
// Every call ADDS to b and leaves the BC rows alone (their rhs belongs to the
// boundary condition). One implementation per family of discretisation:
// GroupTransfer (per node) and GroupTransferCG (SUPG-weighted)
//
// In the group Gauss-Seidel sweep the within-group block sigma_s(g -> g)
// stays on the lhs and the off-diagonal blocks come here, evaluated with the
// latest cached flux of every other group. With downscatter only that sweep
// is a block lower triangular solve, so one forward pass is exact. With
// upscatter the cache is what carries one sweep's flux into the next (the
// outer iteration in tests/transportk): a group below g contributes its flux
// from the previous sweep, and before the first sweep the driver caches a
// zero flux (set_scalar_flux of a zero psi) for every group read before it is
// solved
class PETSC_VISIBILITY_PUBLIC GroupSource {
public:
   virtual ~GroupSource() = default;

   // b += group g's external source
   virtual PetscErrorCode add_external(PetscInt g, Vec b) const = 0;
   // Integrate group g's angular flux and keep it, replacing whatever was
   // cached. Once per solved group: every group it scatters into reads the
   // same flux, so integrating on demand in add_transfer would redo it once
   // per target
   //
   // change_sq and norm_sq measure the overwrite, for an outer iteration's
   // convergence test: both nullptr (nothing measured, no extra work) or both
   // given. Then *change_sq += sum |phi_new - phi_old|^2 and
   // *norm_sq += sum |phi_new|^2, each sum GLOBAL over psi_g's communicator
   // with every node (CG: vertex) counted once, so a caller accumulates any
   // number of groups and takes one square root. Measuring never changes the
   // cached values
   virtual PetscErrorCode set_scalar_flux(PetscInt g, Vec psi_g, PetscReal *change_sq, PetscReal *norm_sq) = 0;
   // b += the scatter from g_from into g_to, either direction.
   // set_scalar_flux(g_from, ...) must have been called first
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
   // With the current enabled (enable_current) this also integrates and caches
   // group g's current J alongside its scalar flux. The scalar flux is the same
   // gemm either way, so it is bitwise what it is without the current
   PetscErrorCode set_scalar_flux(PetscInt g, Vec psi_g, PetscReal *change_sq, PetscReal *norm_sq) override;
   // add_transfer(g, g, b) is legal too: b += sigma_s(g -> g) phi_g / W from
   // the CACHED phi_g - the within-group scatter frozen at that flux, which is
   // what a reconstruction's fixed rhs b + M sigma_s phi_g is
   PetscErrorCode add_transfer(PetscInt g_from, PetscInt g_to, Vec b) const override;

   // ~~~~~~~~~~~~~
   // The moments store (tests/transportk -outer_store moments): keep each
   // group's first two angular moments instead of its angular flux, and expand
   // them back into a starting psi. GroupTransfer-only rather than on
   // GroupSource: the expansion is per NODE with an isotropic M = 1 / W, which
   // is this class's scatter; GroupTransferCG's source and scatter are
   // SUPG-weighted (a vertex's whole star, tau per angle), so the same
   // reconstruction there would need that weighting, and the driver refuses
   // the store on CG-SUPG instead
   // ~~~~~~~~~~~~~

   // Cache the current J(node, d) = sum_a w_a Omega_a[d] psi(node, a) of every
   // group too, d < quad.dimension(), from the next set_scalar_flux on (a
   // group set before this has J = 0). Allocates n_groups (local_nodes, dim)
   // views - dim / n_angles of the angular flux each - and uploads the P1
   // tables once; call before the sweep. The quadrature must be the one create
   // was given and must integrate Omega to zero (every SN set here does, being
   // reflection-symmetric), so that the P1 expansion below reproduces phi
   PetscErrorCode enable_current(const AngularQuadrature &quad);
   PetscBool current_enabled() const { return (PetscBool)(n_dim_ > 0); }

   // psi(node, a) = phi_g(node) / W + Omega_a . T^{-1} J_g(node), the P1
   // expansion of group g's cached moments, T = sum_a w_a Omega_a Omega_a^T.
   // Its scalar flux and current are the cached ones exactly (to rounding).
   // Every row, BC rows included (it is a starting guess, not a rhs). Needs
   // enable_current and a cached flux for g. On the device, no allocation
   PetscErrorCode expand_p1(PetscInt g, Vec psi) const;

   // The cached scalar flux of group g, (local_nodes, 1): what -flux_vtk writes
   // under the moments store (UboltWriteScalarFluxVTKFromPhi). Valid once
   // set_scalar_flux(g, ...) has run
   const PetscScalar2DKokkosView &scalar_flux(PetscInt g) const { return phi_[g]; }

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
   // Persistent scratch - never allocate device memory inside an apply. Also
   // where a measured set_scalar_flux integrates before overwriting the cache
   PetscScalar2DKokkosView scalar_flux_d_;
   // The current (enable_current): 0 until enabled, then the quadrature's
   // dimension. w_omega_d_(a, d) = w_a Omega_a[d], the gemm the current is;
   // p1_d_(a, d) = (T^{-1} Omega_a)[d], what expand_p1 dots J with; one
   // (local_nodes, n_dim) J per group
   PetscInt n_dim_ = 0;
   PetscScalar2DKokkosView w_omega_d_;
   PetscScalar2DKokkosView p1_d_;
   std::vector<PetscScalar2DKokkosView> current_;
};

#endif
