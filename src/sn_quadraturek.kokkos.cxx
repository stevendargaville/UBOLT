#include "ubolt/sn_quadrature.hpp"
#include <KokkosBlas.hpp>
#include <array>
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

// Find the ordinate whose cosines equal (mu, eta, xi) exactly - a cosine the
// dimension does not carry passes NULL and is ignored (1D has only mu, 2D has
// no xi). The cosine tables are sign-symmetric literals, so exact comparison
// holds; the PetscCheck is the load-bearing part, making a future
// non-symmetric set fail loudly rather than silently reflect into a wrong angle
static PetscErrorCode FindOrdinate(const PetscScalar *mu, const PetscScalar *eta, \
   const PetscScalar *xi, PetscInt n_angles, PetscScalar mu_want, PetscScalar eta_want, \
   PetscScalar xi_want, PetscInt *found)
{
   PetscFunctionBeginUser;

   *found = -1;
   for (PetscInt b = 0; b < n_angles; b++) {
      if (mu[b] == mu_want && (!eta || eta[b] == eta_want) && (!xi || xi[b] == xi_want)) \
         { *found = b; break; }
   }
   PetscCheck(*found >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
      "Quadrature set is not symmetric: no ordinate at the reflection of (%g, %g, %g)", \
      (double)PetscRealPart(mu_want), (double)PetscRealPart(eta_want), \
      (double)PetscRealPart(xi_want));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SNQuadrature (1D)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode SNQuadrature::create(PetscInt sn_order)
{
   std::vector<PetscScalar> w_h;

   PetscFunctionBeginUser;

   if (sn_order == 2) {
      mu_h_ = {-0.5773502692, 0.5773502692};
      w_h   = {1.0, 1.0};
   } else if (sn_order == 4) {
      mu_h_ = {-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116};
      w_h   = {0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451};
   } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
         "1D SN quadrature only implemented for order 2 or 4, was given S%" \
         PetscInt_FMT, sn_order);
   }
   // In 1D the SN order IS the ordinate count: SN is the N point
   // Gauss-Legendre rule on [-1, 1]
   const PetscInt n_angles = sn_order;
   PetscScalar *mu = mu_h_.data();
   // In 1D we integrate over [-1, 1] so the sum of weights is 2
   PetscCall(set_weights(w_h, 2.0));

   // The reflection map: the Gauss points come in +/- pairs with equal weights,
   // and the search (rather than index arithmetic) is what a reflective
   // boundary's correctness rests on
   reflect_mu_h_.resize(n_angles);
   for (PetscInt a = 0; a < n_angles; a++) {
      PetscCall(FindOrdinate(mu, NULL, NULL, n_angles, -mu[a], 0.0, 0.0, &reflect_mu_h_[a]));
      PetscCheck(w_h[reflect_mu_h_[a]] == w_h[a], PETSC_COMM_SELF, PETSC_ERR_PLIB, \
         "Quadrature weights are not symmetric under mu -> -mu");
   }

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
// once per quadrant, so those become 4 and 12 ordinates - N (N + 2) / 2 for
// order N. Equal weights within a set: 4 pi / n_angles, which is the doubling
// for the mirrored xi < 0 half already folded in
PetscErrorCode SNQuadrature2D::create(PetscInt sn_order)
{
   // The |mu|, |eta| of one quadrant - the sign combinations come after
   std::vector<std::pair<PetscScalar, PetscScalar>> quadrant;
   std::vector<PetscScalar> w_h;

   PetscFunctionBeginUser;

   if (sn_order == 2) {
      // S2: c = 1/sqrt(3), the same cosine 1D's S2 uses
      const PetscScalar c = 0.5773502692;
      quadrant = {{c, c}};
   } else if (sn_order == 4) {
      // S4: 2 mu1^2 + mu2^2 = 1
      const PetscScalar mu1 = 0.3500212;
      const PetscScalar mu2 = 0.8688903;
      quadrant = {{mu1, mu1}, {mu1, mu2}, {mu2, mu1}};
   } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
         "2D SN quadrature only implemented for order 2 or 4, was given S%" \
         PetscInt_FMT, sn_order);
   }
   // One (mu, eta) per quadrant of each base pair: N (N + 2) / 2 ordinates
   const PetscInt n_angles = sn_order * (sn_order + 2) / 2;

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

   // The reflection maps: a level-symmetric set carries every sign combination
   // of its base cosines (equal weights within the set), so both single-axis
   // flips land on an ordinate. Built by search so the assertion in
   // FindOrdinate, not the quadrant ordering above, is what reflective
   // boundaries rest on
   reflect_mu_h_.resize(n_angles);
   reflect_eta_h_.resize(n_angles);
   for (PetscInt a = 0; a < n_angles; a++) {
      PetscCall(FindOrdinate(mu_h_.data(), eta_h_.data(), NULL, n_angles, -mu_h_[a], eta_h_[a], \
         0.0, &reflect_mu_h_[a]));
      PetscCall(FindOrdinate(mu_h_.data(), eta_h_.data(), NULL, n_angles, mu_h_[a], -eta_h_[a], \
         0.0, &reflect_eta_h_[a]));
   }

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
// SNQuadrature3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The same level-symmetric sets as 2D, but with nothing to fold: 3D has no
// symmetry plane, so every octant is a real octant and xi is a real cosine.
// S2 keeps its one direction per octant, S4 its three - all three permutations
// of (mu1, mu1, mu2) now, where XY could drop the xi component - so the counts
// are 8 and 24 against 2D's 4 and 12, N (N + 2) for order N
PetscErrorCode SNQuadrature3D::create(PetscInt sn_order)
{
   // The |mu|, |eta|, |xi| of one octant - the sign combinations come after
   std::vector<std::array<PetscScalar, 3>> octant;
   std::vector<PetscScalar> w_h;

   PetscFunctionBeginUser;

   if (sn_order == 2) {
      // S2: c = 1/sqrt(3), the same cosine the 1D and 2D S2 sets use
      const PetscScalar c = 0.5773502692;
      octant = {{c, c, c}};
   } else if (sn_order == 4) {
      // S4: 2 mu1^2 + mu2^2 = 1, the same pair the 2D S4 set uses
      const PetscScalar mu1 = 0.3500212;
      const PetscScalar mu2 = 0.8688903;
      octant = {{mu1, mu1, mu2}, {mu1, mu2, mu1}, {mu2, mu1, mu1}};
   } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
         "3D SN quadrature only implemented for order 2 or 4, was given S%" \
         PetscInt_FMT, sn_order);
   }
   // Every sign combination of each base triple, nothing folded away:
   // N (N + 2) ordinates, twice the same-order 2D set
   const PetscInt n_angles = sn_order * (sn_order + 2);

   mu_h_.clear();
   eta_h_.clear();
   xi_h_.clear();
   // The whole sphere is 4 pi and the ordinates of a level-symmetric set of
   // this order all carry the same share of it
   const PetscScalar w_each = 4.0 * PETSC_PI / n_angles;

   // Ordinate order: base triple slowest, so the eight sign combinations of
   // each sit together - the 2D quadrant loop one axis deeper. Nothing depends
   // on the order, but it is fixed here because the COO preallocation and the
   // value fills have to agree on which angle index is which
   for (const auto &p : octant) {
      for (PetscInt sz = -1; sz <= 1; sz += 2) {
         for (PetscInt sy = -1; sy <= 1; sy += 2) {
            for (PetscInt sx = -1; sx <= 1; sx += 2) {
               mu_h_.push_back(sx * p[0]);
               eta_h_.push_back(sy * p[1]);
               xi_h_.push_back(sz * p[2]);
               w_h.push_back(w_each);
            }
         }
      }
   }
   PetscCall(set_weights(w_h, 4.0 * PETSC_PI));

   // The reflection maps, one single-axis flip each: a level-symmetric set
   // carries every sign combination of its base cosines (equal weights within
   // the set), so all three land on an ordinate. Built by search so the
   // assertion in FindOrdinate, not the octant ordering above, is what
   // reflective boundaries rest on
   reflect_mu_h_.resize(n_angles);
   reflect_eta_h_.resize(n_angles);
   reflect_xi_h_.resize(n_angles);
   for (PetscInt a = 0; a < n_angles; a++) {
      PetscCall(FindOrdinate(mu_h_.data(), eta_h_.data(), xi_h_.data(), n_angles, \
         -mu_h_[a], eta_h_[a], xi_h_[a], &reflect_mu_h_[a]));
      PetscCall(FindOrdinate(mu_h_.data(), eta_h_.data(), xi_h_.data(), n_angles, \
         mu_h_[a], -eta_h_[a], xi_h_[a], &reflect_eta_h_[a]));
      PetscCall(FindOrdinate(mu_h_.data(), eta_h_.data(), xi_h_.data(), n_angles, \
         mu_h_[a], eta_h_[a], -xi_h_[a], &reflect_xi_h_[a]));
   }

   // Copy the ordinates to device memory
   mu_d_  = PetscScalarKokkosView("mu_d", n_angles);
   eta_d_ = PetscScalarKokkosView("eta_d", n_angles);
   xi_d_  = PetscScalarKokkosView("xi_d", n_angles);
   PetscScalarKokkosViewHostUnmanaged mu_host_view(mu_h_.data(), n_angles);
   PetscScalarKokkosViewHostUnmanaged eta_host_view(eta_h_.data(), n_angles);
   PetscScalarKokkosViewHostUnmanaged xi_host_view(xi_h_.data(), n_angles);
   Kokkos::deep_copy(mu_d_, mu_host_view);
   Kokkos::deep_copy(eta_d_, eta_host_view);
   Kokkos::deep_copy(xi_d_, xi_host_view);

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
   const PetscScalar alpha = 1.0;
   const PetscScalar beta  = 0.0;
   KokkosBlas::gemm("N", "N", alpha, psi_d_2d, w_d, beta, scalar_flux_d);

   PetscCall(VecRestoreKokkosView(psi, &psi_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
