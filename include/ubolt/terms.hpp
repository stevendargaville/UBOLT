#ifndef UBOLT_TERMS_HPP
#define UBOLT_TERMS_HPP

#include "ubolt/operator_term.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming: mu dpsi/dx, upwinded onto the discretisation's stencil
//
// Takes the 1D backend by concrete type, not the Discretisation base: it needs
// the geometry (dx) and it owns the upwind slot convention, both of which are
// per-dimension. A 2D mesh wants a 2D sibling, not this term
class PETSC_VISIBILITY_PUBLIC StreamingTerm : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const StructuredFD1D &disc, const SNQuadrature &quad);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalar dx_ = 0.0;
   PetscScalarKokkosView mu_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming in 2D: mu dpsi/dx + eta dpsi/dy, each upwinded onto its own axis
//
// The sibling of StreamingTerm, not a generalisation of it: the slot convention
// is per-dimension, and this is the term that owns it. It writes StructuredFD2D's
// three slots positionally - upwind-x, upwind-y, diagonal - which is the same
// contract the 1D pair have, one axis wider
class PETSC_VISIBILITY_PUBLIC StreamingTerm2D : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const StructuredFD2D &disc, const SNQuadrature2D &quad);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalar dx_ = 0.0;
   PetscScalar dy_ = 0.0;
   PetscScalarKokkosView mu_d_;
   PetscScalarKokkosView eta_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming in 3D: mu dpsi/dx + eta dpsi/dy + xi dpsi/dz, each upwinded onto
// its own axis
//
// The next sibling along: it writes StructuredFD3D's four slots positionally -
// upwind-x, upwind-y, upwind-z, diagonal - the same contract the 1D and 2D
// pairs have, one axis wider again
class PETSC_VISIBILITY_PUBLIC StreamingTerm3D : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const StructuredFD3D &disc, const SNQuadrature3D &quad);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalar dx_ = 0.0;
   PetscScalar dy_ = 0.0;
   PetscScalar dz_ = 0.0;
   PetscScalarKokkosView mu_d_;
   PetscScalarKokkosView eta_d_;
   PetscScalarKokkosView xi_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Removal: sigma_t psi, a pure diagonal contribution
//
// Nothing here is dimension-specific - it writes the diagonal slot and skips
// the Dirichlet rows - so it takes any Discretisation
class PETSC_VISIBILITY_PUBLIC RemovalTerm : public OperatorTerm {
public:
   // sigma_t_d is indexed by local cell and must outlive the term
   PetscErrorCode create(const PhaseSpace &ps, const Discretisation &disc, const PetscScalarKokkosView &sigma_t_d);

   // Point the term at a different xsection. The group sweep calls this and
   // then TransportOperator::assemble() to refill the values, which is why
   // nothing here caches anything derived from sigma_t
   void set_sigma_t(const PetscScalarKokkosView &sigma_t_d) { sigma_t_d_ = sigma_t_d; }

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalarKokkosView sigma_t_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Isotropic scattering, applied matrix-free: integrate the angular flux to get
// the scalar flux, scale by sigma_s, and subtract it from every angle (it is on
// the lhs). Never assembled - that is the whole point of the MatShell
class PETSC_VISIBILITY_PUBLIC ScatteringTerm : public OperatorTerm {
public:
   // sigma_s_d is indexed by local cell and must outlive the term. In a
   // multigroup sweep this is the within-group block sigma_s(g -> g); the
   // off-diagonal blocks go to the rhs through GroupTransfer
   // Any discretisation and any quadrature: the scatter is cell-local, so all
   // it wants from the discretisation is the Dirichlet mask, and all it wants
   // from the quadrature is the weights, never the directions
   PetscErrorCode create(const PhaseSpace &ps, const Discretisation &disc, \
      const AngularQuadrature &quad, const PetscScalarKokkosView &sigma_s_d);

   // Point the term at a different xsection - see RemovalTerm::set_sigma_t
   void set_sigma_s(const PetscScalarKokkosView &sigma_s_d) { sigma_s_d_ = sigma_s_d; }

   PetscBool matrix_free() const override { return PETSC_TRUE; }
   PetscErrorCode apply_add(Vec x, Vec y) const override;

private:
   PetscInt n_angles_ = 0;
   PetscScalar sum_weights_ = 0.0;
   PetscScalarKokkosView sigma_s_d_;
   PetscScalar2DKokkosView w_d_;
   BoundaryInfo boundary_;
   // Persistent scratch - never allocate device memory inside an apply
   PetscScalar2DKokkosView scalar_flux_d_;
};

#endif
