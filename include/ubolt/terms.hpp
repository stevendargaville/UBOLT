#ifndef UBOLT_TERMS_HPP
#define UBOLT_TERMS_HPP

#include "ubolt/operator_term.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/structured_fd_1d.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming: mu dpsi/dx, upwinded onto the discretisation's stencil
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

// Removal: sigma_t psi, a pure diagonal contribution
class PETSC_VISIBILITY_PUBLIC RemovalTerm : public OperatorTerm {
public:
   // sigma_t_d is indexed by local cell and must outlive the term
   PetscErrorCode create(const PhaseSpace &ps, const StructuredFD1D &disc, const PetscScalarKokkosView &sigma_t_d);

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
   PetscErrorCode create(const PhaseSpace &ps, const SNQuadrature &quad, const PetscScalarKokkosView &sigma_s_d);

   // Point the term at a different xsection - see RemovalTerm::set_sigma_t
   void set_sigma_s(const PetscScalarKokkosView &sigma_s_d) { sigma_s_d_ = sigma_s_d; }

   PetscBool matrix_free() const override { return PETSC_TRUE; }
   PetscErrorCode apply_add(Vec x, Vec y) const override;

private:
   PetscInt n_angles_ = 0;
   PetscScalar sum_weights_ = 0.0;
   PetscScalarKokkosView sigma_s_d_;
   PetscScalar2DKokkosView w_d_;
   // Persistent scratch - never allocate device memory inside an apply
   PetscScalar2DKokkosView scalar_flux_d_;
};

#endif
