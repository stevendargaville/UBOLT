#ifndef UBOLT_SN_QUADRATURE_HPP
#define UBOLT_SN_QUADRATURE_HPP

#include "ubolt/types.hpp"
#include <petscvec.h>
#include <vector>

// SN quadrature (Gauss-Legendre for n_angles = 2 or 4)
//
// Keeps the ordinates on the host (the COO preallocation is a host loop) and on
// the device (the assembly and scatter kernels)
class PETSC_VISIBILITY_PUBLIC SNQuadrature {
public:
   PetscErrorCode create(PetscInt n_angles);

   PetscInt n_angles() const { return n_angles_; }
   // In 1D we integrate over [-1, 1] so the sum of weights is 2
   PetscScalar sum_weights() const { return sum_weights_; }

   const PetscScalar *mu_host() const { return mu_h_.data(); }
   const PetscScalarKokkosView &mu_d() const { return mu_d_; }
   // (n_angles, 1) so the angular integral is a gemm against the flux
   const PetscScalar2DKokkosView &w_d() const { return w_d_; }

private:
   PetscInt n_angles_ = 0;
   PetscScalar sum_weights_ = 0.0;
   std::vector<PetscScalar> mu_h_;
   std::vector<PetscScalar> w_h_;
   PetscScalarKokkosView mu_d_;
   PetscScalar2DKokkosView w_d_;
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
