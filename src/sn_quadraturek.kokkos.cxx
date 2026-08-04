#include "ubolt/sn_quadrature.hpp"
// The generated level-symmetric table lives under src/ so it cannot be
// included from the public headers - a quote include resolves against this
// file's directory. Regenerate it with src/sn_lqn_table.py, never by hand
#include "sn_lqn_table.hpp"
#include <KokkosBlas.hpp>

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

   // sum_weights is the EXACT measure of the angular domain, passed in rather
   // than summed here, and the infinite-medium check rests on the two agreeing
   // (see docs/dev/testing.md). Generated weights agree only to rounding, so
   // say out loud how far apart they are allowed to be - a set whose weights
   // do not cover the domain it claims would otherwise surface as a wrong
   // answer in an isotropic source rather than an error
   PetscScalar w_sum = 0.0;
   for (PetscInt a = 0; a < n_angles_; a++) w_sum += w[a];
   PetscCheck(PetscAbsScalar(w_sum - sum_weights) <= \
      1e-12 * PetscAbsScalar(sum_weights) * n_angles_, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
      "Quadrature weights sum to %g, not the angular measure %g they are declared over", \
      (double)PetscRealPart(w_sum), (double)PetscRealPart(sum_weights));

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
// no xi). Exact comparison holds because a sign-flipped ordinate is built by
// negating the SAME cosine value, never by recomputing it; the PetscCheck is
// the load-bearing part, making a future non-symmetric set fail loudly rather
// than silently reflect into a wrong angle
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

// P_n(x) and P'_n(x), from the recurrence
// k P_k = (2k - 1) x P_(k-1) - (k - 1) P_(k-2) and
// P'_n = n (x P_n - P_(n-1)) / (x^2 - 1)
static void LegendreAndDerivative(PetscInt n, PetscReal x, PetscReal *p, PetscReal *dp)
{
   PetscReal p_k = 1.0, p_km1 = 0.0;

   for (PetscInt k = 1; k <= n; k++) {
      const PetscReal p_km2 = p_km1;
      p_km1 = p_k;
      p_k = ((2 * k - 1) * x * p_km1 - (k - 1) * p_km2) / k;
   }
   *p = p_k;
   *dp = n * (x * p_k - p_km1) / (x * x - 1.0);
}

// The positive half of the n_points Gauss-Legendre rule on [-1, 1], to machine
// precision, ascending - the negative half is its mirror and the caller builds
// it by negating these values rather than recomputing them
//
// Newton from the standard asymptotic seed, which is close enough that this
// converges in a handful of iterations at any order. n_points is even, so no
// root sits at zero and the two halves really are disjoint
static PetscErrorCode GaussLegendrePositiveHalf(PetscInt n_points, \
   std::vector<PetscReal> &mu, std::vector<PetscReal> &w)
{
   const PetscInt half = n_points / 2;
   const PetscInt max_newton = 100;

   PetscFunctionBeginUser;

   mu.resize(half);
   w.resize(half);

   for (PetscInt i = 0; i < half; i++) {

      // The seeds run from the largest root down, so fill the arrays back to
      // front to come out ascending
      PetscReal x = PetscCosReal(PETSC_PI * (i + 0.75) / (n_points + 0.5));
      PetscReal p = 0.0, dp = 0.0;
      PetscInt it = 0;

      for (; it < max_newton; it++) {
         LegendreAndDerivative(n_points, x, &p, &dp);
         const PetscReal dx = -p / dp;
         x += dx;
         if (PetscAbsReal(dx) <= 4.0 * PETSC_MACHINE_EPSILON * PetscAbsReal(x)) break;
      }
      PetscCheck(it < max_newton, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
         "Gauss-Legendre Newton iteration did not converge for node %" PetscInt_FMT \
         " of %" PetscInt_FMT, i, n_points);

      // One more evaluation after the step went below the tolerance, so the
      // weight is built from the derivative AT the converged node
      LegendreAndDerivative(n_points, x, &p, &dp);
      mu[half - 1 - i] = x;
      w[half - 1 - i] = 2.0 / ((1.0 - x * x) * dp * dp);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNQuadrature::create(PetscInt sn_order)
{
   std::vector<PetscScalar> w_h;
   std::vector<PetscReal> mu_pos, w_pos;

   PetscFunctionBeginUser;

   // In 1D the SN order IS the ordinate count: SN is the N point
   // Gauss-Legendre rule on [-1, 1], generated here rather than tabulated, so
   // there is no upper bound on the order
   PetscCheck(sn_order > 0 && sn_order % 2 == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, \
      "1D SN quadrature needs a positive even order, was given S%" PetscInt_FMT, sn_order);
   const PetscInt n_angles = sn_order;
   const PetscInt half = n_angles / 2;

   PetscCall(GaussLegendrePositiveHalf(n_angles, mu_pos, w_pos));

   // Ascending in mu, the negative half mirroring the positive one. The mirror
   // is an exact negation of the SAME value and an exact copy of its weight,
   // which is what lets the reflection map below compare with == and what
   // makes the weight symmetry hold rather than nearly hold
   mu_h_.resize(n_angles);
   w_h.resize(n_angles);
   for (PetscInt t = 0; t < half; t++) {
      mu_h_[half + t] = mu_pos[t];
      w_h[half + t]   = w_pos[t];
      mu_h_[half - 1 - t] = -mu_pos[t];
      w_h[half - 1 - t]   = w_pos[t];
   }
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
// The level-symmetric sets, shared by 2D and 3D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A level-symmetric (LQn) set is built from a short list of direction cosines
// whose permutations, in every sign combination, give the ordinates - so the
// set looks the same from any axis, which is what makes it a sensible SN set
// on a structured grid. One octant holds N (N + 2) / 8 of them; 3D takes all
// eight octants, XY takes the four quadrants of the xi > 0 half and doubles
// the weights to stand in for the mirrored half
//
// The constants come from the generated table (src/sn_lqn_table.py). Beyond S4
// the weights are NOT equal within a set: they are constant on the permutation
// classes of the level-index triple, of which there are 2 at S6, 3 at S8 and
// up to 10 at S18 - which is where the table stops, because above it the
// weights stop being all positive
//
// Each cosine is COPIED out of the level array, so two points on the same level
// carry the bit-identical value and the reflection search below can compare
// with ==. Same for the weights within a permutation class
static PetscErrorCode LevelSymmetricOctant(PetscInt sn_order, const char *dim, \
   std::vector<PetscScalar> &mu, std::vector<PetscScalar> &eta, std::vector<PetscScalar> &xi, \
   std::vector<PetscScalar> &fraction)
{
   PetscFunctionBeginUser;

   PetscCheck(sn_order >= ubolt_lqn::lqn_min_order && sn_order % 2 == 0, PETSC_COMM_SELF, \
      PETSC_ERR_SUP, "%s SN quadrature needs an even order of at least S%" PetscInt_FMT \
      ", was given S%" PetscInt_FMT, dim, ubolt_lqn::lqn_min_order, sn_order);
   PetscCheck(sn_order <= ubolt_lqn::lqn_max_order, PETSC_COMM_SELF, PETSC_ERR_SUP, \
      "%s SN quadrature is level-symmetric, and those sets stop having all-positive " \
      "weights above S%" PetscInt_FMT ", was given S%" PetscInt_FMT, dim, \
      ubolt_lqn::lqn_max_order, sn_order);

   const ubolt_lqn::LqnOrder *table = ubolt_lqn::lqn_find(sn_order);
   PetscCheck(table, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
      "%s SN quadrature: no level-symmetric table for S%" PetscInt_FMT, dim, sn_order);

   mu.clear();
   eta.clear();
   xi.clear();
   fraction.clear();
   for (PetscInt p = 0; p < table->n_points; p++) {
      const ubolt_lqn::LqnPoint &point = table->points[p];
      mu.push_back(table->mu[point.i - 1]);
      eta.push_back(table->mu[point.j - 1]);
      xi.push_back(table->mu[point.k - 1]);
      fraction.push_back(point.weight);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SNQuadrature2D (XY geometry)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// In XY we drop xi and take each (mu, eta) once per quadrant, so the octant's
// N (N + 2) / 8 points become N (N + 2) / 2 ordinates - 4 at S2, 12 at S4, 40
// at S8. The xi component is what decides the third level index, so the octant
// points project onto distinct (mu, eta) pairs and nothing collides
PetscErrorCode SNQuadrature2D::create(PetscInt sn_order)
{
   // The |mu|, |eta| of one quadrant - the sign combinations come after
   std::vector<PetscScalar> base_mu, base_eta, base_xi, fraction;
   std::vector<PetscScalar> w_h;

   PetscFunctionBeginUser;

   PetscCall(LevelSymmetricOctant(sn_order, "2D", base_mu, base_eta, base_xi, fraction));
   // One (mu, eta) per quadrant of each base pair: N (N + 2) / 2 ordinates
   const PetscInt n_angles = sn_order * (sn_order + 2) / 2;

   mu_h_.clear();
   eta_h_.clear();

   // Ordinate order: quadrant slowest, so the four sign combinations of each
   // base pair sit together. Nothing depends on the order - the streaming term
   // reads mu and eta per row and the integral is a weighted sum - but it is
   // fixed here because the COO preallocation and the value fills have to agree
   // on which angle index is which
   for (PetscInt p = 0; p < (PetscInt)base_mu.size(); p++) {
      // The table's weights are fractions of the whole sphere; the extra 2 is
      // the mirrored xi < 0 half this ordinate stands in for
      const PetscScalar w_p = 2.0 * fraction[p] * (4.0 * PETSC_PI);
      for (PetscInt sy = -1; sy <= 1; sy += 2) {
         for (PetscInt sx = -1; sx <= 1; sx += 2) {
            mu_h_.push_back(sx * base_mu[p]);
            eta_h_.push_back(sy * base_eta[p]);
            w_h.push_back(w_p);
         }
      }
   }
   PetscCall(set_weights(w_h, 4.0 * PETSC_PI));

   // The reflection maps: a level-symmetric set carries every sign combination
   // of its base cosines, so both single-axis flips land on an ordinate, and a
   // flip stays inside its permutation class so the partner's weight is the
   // same value. Built by search so the assertion in FindOrdinate, not the
   // quadrant ordering above, is what reflective boundaries rest on
   reflect_mu_h_.resize(n_angles);
   reflect_eta_h_.resize(n_angles);
   for (PetscInt a = 0; a < n_angles; a++) {
      PetscCall(FindOrdinate(mu_h_.data(), eta_h_.data(), NULL, n_angles, -mu_h_[a], eta_h_[a], \
         0.0, &reflect_mu_h_[a]));
      PetscCall(FindOrdinate(mu_h_.data(), eta_h_.data(), NULL, n_angles, mu_h_[a], -eta_h_[a], \
         0.0, &reflect_eta_h_[a]));
      PetscCheck(w_h[reflect_mu_h_[a]] == w_h[a] && w_h[reflect_eta_h_[a]] == w_h[a], \
         PETSC_COMM_SELF, PETSC_ERR_PLIB, \
         "Quadrature weights are not symmetric under a single cosine flip");
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
// That doubles the ordinate count of the same-order 2D set - N (N + 2), so 8
// at S2, 24 at S4, 80 at S8 - and the weights are the table's fractions of the
// sphere without the fold's factor of two
PetscErrorCode SNQuadrature3D::create(PetscInt sn_order)
{
   // The |mu|, |eta|, |xi| of one octant - the sign combinations come after
   std::vector<PetscScalar> base_mu, base_eta, base_xi, fraction;
   std::vector<PetscScalar> w_h;

   PetscFunctionBeginUser;

   PetscCall(LevelSymmetricOctant(sn_order, "3D", base_mu, base_eta, base_xi, fraction));
   // Every sign combination of each base triple, nothing folded away:
   // N (N + 2) ordinates, twice the same-order 2D set
   const PetscInt n_angles = sn_order * (sn_order + 2);

   mu_h_.clear();
   eta_h_.clear();
   xi_h_.clear();

   // Ordinate order: base triple slowest, so the eight sign combinations of
   // each sit together - the 2D quadrant loop one axis deeper. Nothing depends
   // on the order, but it is fixed here because the COO preallocation and the
   // value fills have to agree on which angle index is which
   for (PetscInt p = 0; p < (PetscInt)base_mu.size(); p++) {
      const PetscScalar w_p = fraction[p] * (4.0 * PETSC_PI);
      for (PetscInt sz = -1; sz <= 1; sz += 2) {
         for (PetscInt sy = -1; sy <= 1; sy += 2) {
            for (PetscInt sx = -1; sx <= 1; sx += 2) {
               mu_h_.push_back(sx * base_mu[p]);
               eta_h_.push_back(sy * base_eta[p]);
               xi_h_.push_back(sz * base_xi[p]);
               w_h.push_back(w_p);
            }
         }
      }
   }
   PetscCall(set_weights(w_h, 4.0 * PETSC_PI));

   // The reflection maps, one single-axis flip each: a level-symmetric set
   // carries every sign combination of its base cosines, so all three land on
   // an ordinate, and a flip stays inside its permutation class so the
   // partner's weight is the same value. Built by search so the assertion in
   // FindOrdinate, not the octant ordering above, is what reflective
   // boundaries rest on
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
      PetscCheck(w_h[reflect_mu_h_[a]] == w_h[a] && w_h[reflect_eta_h_[a]] == w_h[a] && \
         w_h[reflect_xi_h_[a]] == w_h[a], PETSC_COMM_SELF, PETSC_ERR_PLIB, \
         "Quadrature weights are not symmetric under a single cosine flip");
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
