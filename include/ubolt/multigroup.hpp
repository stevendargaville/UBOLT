#ifndef UBOLT_MULTIGROUP_HPP
#define UBOLT_MULTIGROUP_HPP

#include "ubolt/types.hpp"
#include "ubolt/coo_pattern.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
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

   // Constant across the slab, which is all the current test problems need.
   // Spatially varying data fills the views through the accessors below
   PetscErrorCode set_sigma_t(PetscInt g, PetscScalar value);
   PetscErrorCode set_sigma_s(PetscInt g_from, PetscInt g_to, PetscScalar value);

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

// Group-to-group scattering source
//
// In the group Gauss-Seidel sweep the within-group block sigma_s(g -> g) stays
// on the lhs (matrix-free, as ScatteringTerm) and the off-diagonal blocks move
// to the rhs, evaluated with the flux of whichever groups have already been
// solved. With downscatter only that sweep is a block lower triangular solve,
// so one forward pass over the groups is exact - no outer iteration
//
// Not an OperatorTerm: it does not act on the unknowns of the group being
// solved, it builds that group's right hand side
class PETSC_VISIBILITY_PUBLIC GroupTransfer {
public:
   // xs and boundary must outlive the object
   // Any quadrature: like the scatter it only needs the weights
   PetscErrorCode create(const PhaseSpace &ps, const AngularQuadrature &quad, \
      const GroupXSections &xs, const BoundaryInfo &boundary);

   // Integrate group g's angular flux and keep it. Call once, as soon as that
   // group is solved
   //
   // A group's scalar flux is fixed the moment the group is solved, and every
   // group below it scatters from the same one, so integrating on demand inside
   // add_source() would redo it once per target group: G(G-1)/2 angular
   // integrals over a sweep instead of G
   PetscErrorCode set_scalar_flux(PetscInt g, Vec psi_g);

   // b += sigma_s(g_from -> g_to) phi(g_from) / sum_weights, on every angle.
   // set_scalar_flux(g_from, ...) must have been called first
   //
   // BC rows are skipped: the rhs there belongs to the boundary condition (the
   // incoming flux value on a Dirichlet row, zero on a reflective one) and a
   // scattering source must not touch it, the same contract the terms have
   // with the BC row mask
   PetscErrorCode add_source(PetscInt g_from, PetscInt g_to, Vec b) const;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_cells_ = 0;
   PetscScalar sum_weights_ = 0.0;
   const GroupXSections *xs_ = nullptr;
   PetscScalar2DKokkosView w_d_;
   PetscIntKokkosView is_bc_row_d_;
   // The cached scalar flux of each group, (local_cells, 1) apiece. Separate
   // allocations rather than one (group, cell) table: the angular integral
   // writes a 2D gemm output, and a slice of a 2D table would be a rank-2 view
   // whose layout no longer matches the default one on a device backend
   std::vector<PetscScalar2DKokkosView> phi_;
   std::vector<PetscBool> phi_set_;
   // Persistent scratch - never allocate device memory inside an apply
   PetscScalar2DKokkosView scalar_flux_d_;
};

#endif
