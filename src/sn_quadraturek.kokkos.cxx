#include "ubolt/sn_quadrature.hpp"
#include <KokkosBlas.hpp>
#include <utility>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AngularQuadrature
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode AngularQuadrature::set_weights(const std::vector<PetscScalar> &w, PetscScalar sum_weights)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   n_angles_ = (PetscInt)w.size();
   sum_weights_ = sum_weights;

   // (n_angles, 1): one moment, so the angular integral is a gemm with a single
   // column rather than a gemv - see the Pn note in TODO.md
   w_d_ = PetscScalar2DKokkosView("w_d", n_angles_, 1);
   // const_cast only because the unmanaged host view type is non-const; the
   // deep_copy below reads it
   PetscScalar2DKokkosViewHostUnmanaged w_host_view(const_cast<PetscScalar *>(w.data()), n_angles_, 1);
   Kokkos::deep_copy(w_d_, w_host_view);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SNQuadrature (1D)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode SNQuadrature::create(PetscInt n_angles)
{
   std::vector<PetscScalar> w_h;

   PetscFunctionBeginUser;

   mu_h_.resize(n_angles);
   w_h.resize(n_angles);

   PetscScalar *mu = mu_h_.data();
   PetscScalar *w  = w_h.data();

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
   PetscCall(set_weights(w_h, 2.0));

   // Copy the ordinates to device memory
   mu_d_ = PetscScalarKokkosView("mu_d", n_angles);
   PetscScalarKokkosViewHostUnmanaged mu_host_view(mu, n_angles);
   Kokkos::deep_copy(mu_d_, mu_host_view);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SNQuadrature2D (XY geometry)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The level-symmetric (LQn) sets. A level-symmetric set is built from a short
// list of direction cosines whose permutations, in every sign combination, give
// the ordinates - so the set looks the same from any axis, which is what makes
// it a sensible SN set on a structured grid
//
// S2 has one direction per octant, (c, c, c) with c = 1/sqrt(3); S4 has three,
// the permutations of (mu1, mu1, mu2). In XY we drop xi and take each (mu, eta)
// once per quadrant, so those become 4 and 12 ordinates. Equal weights within a
// set: 4 pi / n_angles, which is the doubling for the mirrored xi < 0 half
// already folded in
PetscErrorCode SNQuadrature2D::create(PetscInt n_angles)
{
   // The |mu|, |eta| of one quadrant - the sign combinations come after
   std::vector<std::pair<PetscScalar, PetscScalar>> quadrant;
   std::vector<PetscScalar> w_h;

   PetscFunctionBeginUser;

   if (n_angles == 4) {
      // S2: c = 1/sqrt(3), the same cosine 1D's S2 uses
      const PetscScalar c = 0.5773502692;
      quadrant = {{c, c}};
   } else if (n_angles == 12) {
      // S4: 2 mu1^2 + mu2^2 = 1
      const PetscScalar mu1 = 0.3500212;
      const PetscScalar mu2 = 0.8688903;
      quadrant = {{mu1, mu1}, {mu1, mu2}, {mu2, mu1}};
   } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
         "2D SN quadrature only implemented for 4 (S2) or 12 (S4) angles, was given %" \
         PetscInt_FMT, n_angles);
   }

   mu_h_.clear();
   eta_h_.clear();
   // The whole sphere is 4 pi and the ordinates of a level-symmetric set of
   // this order all carry the same share of it
   const PetscScalar w_each = 4.0 * PETSC_PI / n_angles;

   // Ordinate order: quadrant slowest, so the four sign combinations of each
   // base pair sit together. Nothing depends on the order - the streaming term
   // reads mu and eta per row and the integral is a weighted sum - but it is
   // fixed here because the COO preallocation and the value fills have to agree
   // on which angle index is which
   for (const auto &p : quadrant) {
      for (PetscInt sy = -1; sy <= 1; sy += 2) {
         for (PetscInt sx = -1; sx <= 1; sx += 2) {
            mu_h_.push_back(sx * p.first);
            eta_h_.push_back(sy * p.second);
            w_h.push_back(w_each);
         }
      }
   }
   PetscCall(set_weights(w_h, 4.0 * PETSC_PI));

   // Copy the ordinates to device memory
   mu_d_  = PetscScalarKokkosView("mu_d", n_angles);
   eta_d_ = PetscScalarKokkosView("eta_d", n_angles);
   PetscScalarKokkosViewHostUnmanaged mu_host_view(mu_h_.data(), n_angles);
   PetscScalarKokkosViewHostUnmanaged eta_host_view(eta_h_.data(), n_angles);
   Kokkos::deep_copy(mu_d_, mu_host_view);
   Kokkos::deep_copy(eta_d_, eta_host_view);

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
