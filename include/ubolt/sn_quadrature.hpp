#ifndef UBOLT_SN_QUADRATURE_HPP
#define UBOLT_SN_QUADRATURE_HPP

#include "ubolt/types.hpp"
#include <petscvec.h>
#include <vector>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// What an angular quadrature owes the consumers that do not care which
// directions the ordinates point in: how many there are, what their weights sum
// to, and the weight matrix the angular integral is a gemm against
//
// That is everything the scattering term and the group transfer need, so they
// take this base and work in any dimension. The direction cosines are on the
// concrete classes, because the only thing that wants them is the streaming
// term, which is per-dimension anyway - the same split the Discretisation base
// makes with the geometry
class PETSC_VISIBILITY_PUBLIC AngularQuadrature {
public:
   virtual ~AngularQuadrature() = default;

   PetscInt n_angles() const { return n_angles_; }
   // The measure of the angular domain the ordinates cover. An isotropic source
   // is shared out over the angles by dividing by this
   PetscScalar sum_weights() const { return sum_weights_; }
   // (n_angles, 1) so the angular integral is a gemm against the flux. SN uses
   // one moment; a Pn set widens that column dimension (see TODO.md)
   const PetscScalar2DKokkosView &w_d() const { return w_d_; }

protected:
   AngularQuadrature() = default;

   // Copy the ordinate weights to the device
   //
   // sum_weights is passed in rather than summed here: it is the exact measure
   // of the angular domain (2 for 1D's [-1, 1], 4 pi for 2D's sphere), and a
   // floating point sum of the weights would agree with it only to rounding
   PetscErrorCode set_weights(const std::vector<PetscScalar> &w, PetscScalar sum_weights);

   PetscInt n_angles_ = 0;
   PetscScalar sum_weights_ = 0.0;
   PetscScalar2DKokkosView w_d_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 1D SN quadrature (Gauss-Legendre for n_angles = 2 or 4)
//
// Keeps the ordinates on the host (the COO preallocation is a host loop) and on
// the device (the assembly and scatter kernels)
class PETSC_VISIBILITY_PUBLIC SNQuadrature : public AngularQuadrature {
public:
   PetscErrorCode create(PetscInt n_angles);

   const PetscScalar *mu_host() const { return mu_h_.data(); }
   const PetscScalarKokkosView &mu_d() const { return mu_d_; }
   // reflect_mu_host()[a] is the ordinate with mu = -mu_a and the same weight -
   // the partner a reflective boundary couples angle a to. Host only: the COO
   // preallocation is the only consumer
   const PetscInt *reflect_mu_host() const { return reflect_mu_h_.data(); }

private:
   std::vector<PetscScalar> mu_h_;
   std::vector<PetscInt> reflect_mu_h_;
   PetscScalarKokkosView mu_d_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 2D (XY geometry) SN quadrature: the level-symmetric sets, S2 (4 ordinates)
// and S4 (12), which is 4 quadrants x {1, 3} directions each
//
// XY geometry is symmetric about the z = 0 plane, so only the directions with
// xi > 0 are tracked and their weights are doubled to stand in for the mirrored
// half. The weights therefore sum to the full 4 pi, and xi itself is never
// needed - the operator only ever sees mu and eta
class PETSC_VISIBILITY_PUBLIC SNQuadrature2D : public AngularQuadrature {
public:
   PetscErrorCode create(PetscInt n_angles);

   const PetscScalar *mu_host() const { return mu_h_.data(); }
   const PetscScalar *eta_host() const { return eta_h_.data(); }
   const PetscScalarKokkosView &mu_d() const { return mu_d_; }
   const PetscScalarKokkosView &eta_d() const { return eta_d_; }
   // The reflection partners: reflect_mu_host()[a] flips the sign of mu and
   // keeps eta (an x-face reflection), reflect_eta_host()[a] the other way
   // round. Host only: the COO preallocation is the only consumer
   const PetscInt *reflect_mu_host() const { return reflect_mu_h_.data(); }
   const PetscInt *reflect_eta_host() const { return reflect_eta_h_.data(); }

private:
   std::vector<PetscScalar> mu_h_;
   std::vector<PetscScalar> eta_h_;
   std::vector<PetscInt> reflect_mu_h_;
   std::vector<PetscInt> reflect_eta_h_;
   PetscScalarKokkosView mu_d_;
   PetscScalarKokkosView eta_d_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The angular integral: scalar_flux(cell, 0) = sum_a w_a psi(cell, a), over the
// LOCAL part of psi only. Runs on the device
//
// scalar_flux_d is caller-owned persistent scratch sized (local_cells, 1) -
// never allocate device memory inside an apply. Shared by every term that needs
// a scalar flux, so they all do bit-identical arithmetic
PETSC_EXTERN PetscErrorCode UboltAngularIntegral(Vec psi, PetscInt n_angles, \
   const PetscScalar2DKokkosView &w_d, const PetscScalar2DKokkosView &scalar_flux_d);

#endif
