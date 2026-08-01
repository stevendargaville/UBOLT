#include "ubolt/sn_quadrature.hpp"
#include <KokkosBlas.hpp>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode SNQuadrature::create(PetscInt n_angles)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   n_angles_ = n_angles;
   mu_h_.resize(n_angles);
   w_h_.resize(n_angles);

   PetscScalar *mu = mu_h_.data();
   PetscScalar *w  = w_h_.data();

   if (n_angles == 2) {
      mu[0] = -0.5773502692; mu[1] = 0.5773502692;
      w[0] = 1.0;            w[1] = 1.0;
   } else if (n_angles == 4) {
      mu[0] = -0.8611363116; mu[1] = -0.3399810436;
      mu[2] =  0.3399810436; mu[3] =  0.8611363116;
      w[0] = 0.3478548451;   w[1] = 0.6521451549;
      w[2] = 0.6521451549;   w[3] = 0.3478548451;
   } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
         "SN quadrature only implemented for 2 or 4 angles, was given %" PetscInt_FMT, n_angles);
   }
   // In 1D we integrate over [-1, 1] so the sum of weights is 2
   sum_weights_ = 2.0;

   // Copy the quadrature to device memory
   mu_d_ = PetscScalarKokkosView("mu_d", n_angles);
   w_d_  = PetscScalar2DKokkosView("w_d", n_angles, 1);
   PetscScalarKokkosViewHostUnmanaged mu_host_view(mu, n_angles);
   PetscScalar2DKokkosViewHostUnmanaged w_host_view(w, n_angles, 1);
   Kokkos::deep_copy(mu_d_, mu_host_view);
   Kokkos::deep_copy(w_d_, w_host_view);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UboltAngularIntegral(Vec psi, PetscInt n_angles, \
   const PetscScalar2DKokkosView &w_d, const PetscScalar2DKokkosView &scalar_flux_d)
{
   PetscFunctionBeginUser;

   // Everything here works on the local part of psi only - this is how many
   // local cells we have
   PetscInt local_rows;
   PetscCall(VecGetLocalSize(psi, &local_rows));
   const PetscInt local_cells = local_rows / n_angles;

   // Get the device view, const since we only read it
   PetscScalarConstKokkosView psi_d;
   PetscCall(VecGetKokkosView(psi, &psi_d));

   // Reshape (without copying) the 1D view into a 2D one. The 2D view uses
   // LayoutRight so we get the correct ordering of contiguous in angle
   PetscScalar2DConstKokkosView psi_d_2d(psi_d.data(), local_cells, n_angles);

   // The integral over angle is just a dgemm (on the device)
   const double alpha = double(1.0);
   const double beta  = double(0.0);
   KokkosBlas::gemm("N", "N", alpha, psi_d_2d, w_d, beta, scalar_flux_d);

   PetscCall(VecRestoreKokkosView(psi, &psi_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
