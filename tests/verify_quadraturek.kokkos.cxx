// Verification for the angular quadratures themselves
//
// The other verify_* drivers check a discretisation against an operator; this
// one checks the quadrature sets in isolation, because since Aug 2026 they are
// GENERATED rather than tabulated - 1D is a Gauss-Legendre rule built by Newton
// at create() time, and 2D/3D read the level-symmetric table that
// src/sn_lqn_table.py emits. Neither is anything a reader can eyeball.
//
// What it pins:
//
//  1. 1D, at a spread of even orders: the defining property of an N point
//     Gauss rule, exactness on every polynomial up to degree 2N - 1, plus the
//     +/- pairing and weight symmetry that a reflective boundary rests on.
//  2. 2D and 3D, at EVERY order the table carries: the even axis moments
//     through order N, which are what defines a level-symmetric set and the
//     real test of the generated table; the ordinate counts; unit direction
//     vectors; the three reflection maps; and, in 3D, invariance under
//     permuting the axes.
//  3. That the 2D set really is the 3D set folded through the z plane.
//  4. The two closed-form cosines, which is what ties the generated table back
//     to something checkable by hand.
//  5. That an unsupported order fails rather than returning something.
//
// Serial by nature - the quadrature is replicated on every rank - but it runs
// clean under mpiexec, which is what the parallel recipe checks.
//
// Currently run with:
// make build_tests && ./verify_quadraturek

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include <petscsys.h>
#include <vector>

// The orders the level-symmetric table carries - kept here rather than read
// out of the library so a silently shortened table would fail this driver.
// S18 is the last one whose weights are all positive; the generator stops
// there on its own and S20 has to be REJECTED, which is checked below
static const PetscInt LQN_MIN_ORDER = 2;
static const PetscInt LQN_MAX_ORDER = 18;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The weights only ever live on the device - the angular integral is a gemm
// against them - so every check here starts by bringing them back
static std::vector<PetscScalar> WeightsToHost(const AngularQuadrature &quad)
{
   const PetscInt n_angles = quad.n_angles();
   auto mirror = Kokkos::create_mirror_view(quad.w_d());
   Kokkos::deep_copy(mirror, quad.w_d());

   std::vector<PetscScalar> w(n_angles);
   for (PetscInt a = 0; a < n_angles; a++) w[a] = mirror(a, 0);
   return w;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 1D: Gauss-Legendre
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 1: the N point Gauss-Legendre rule on [-1, 1]
//
// Exactness up to degree 2N - 1 is the whole of what a Gauss rule promises, and
// nothing weaker would catch a node converged to the wrong root or a weight
// built from the derivative at the wrong point. The symmetry half is separate
// and is about EXACT equality rather than a tolerance: the reflection map is
// built by searching the cosines for -mu with ==, so the mirrored node has to
// be the bitwise negation of its partner and not merely close to it
static PetscErrorCode CheckGaussLegendre(PetscInt sn_order, PetscBool *ok)
{
   SNQuadrature quad;
   PetscReal worst_moment = 0.0, worst_sum = 0.0;
   PetscBool symmetric = PETSC_TRUE, ordered = PETSC_TRUE, involution = PETSC_TRUE;
   const PetscReal moment_tol = 1e-13;
   const PetscReal sum_tol = 1e-14;

   PetscFunctionBeginUser;

   PetscCall(quad.create(sn_order));
   const PetscInt n_angles = quad.n_angles();
   PetscCheck(n_angles == sn_order, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
      "1D S%" PetscInt_FMT " gave %" PetscInt_FMT " ordinates, not %" PetscInt_FMT, \
      sn_order, n_angles, sn_order);

   const PetscScalar *mu = quad.mu_host();
   const PetscInt *reflect = quad.reflect_mu_host();
   const std::vector<PetscScalar> w = WeightsToHost(quad);

   // The weights cover [-1, 1], which is what sum_weights claims
   PetscScalar sum = 0.0;
   for (PetscInt a = 0; a < n_angles; a++) sum += w[a];
   worst_sum = PetscAbsScalar(sum - 2.0) / (2.0 * n_angles);

   // Strictly ascending inside (-1, 1), and mirrored bit for bit
   for (PetscInt a = 0; a < n_angles; a++) {
      if (PetscRealPart(mu[a]) <= -1.0 || PetscRealPart(mu[a]) >= 1.0) ordered = PETSC_FALSE;
      if (a > 0 && PetscRealPart(mu[a]) <= PetscRealPart(mu[a - 1])) ordered = PETSC_FALSE;
      if (mu[n_angles - 1 - a] != -mu[a] || w[n_angles - 1 - a] != w[a]) symmetric = PETSC_FALSE;
      // The reflection map is that mirror, found by search rather than index
      if (reflect[a] != n_angles - 1 - a || reflect[reflect[a]] != a || \
         w[reflect[a]] != w[a]) involution = PETSC_FALSE;
   }

   // Exact on every polynomial of degree < 2N: odd powers integrate to zero,
   // even ones to 2 / (k + 1)
   for (PetscInt k = 0; k < 2 * sn_order; k++) {
      PetscScalar got = 0.0;
      for (PetscInt a = 0; a < n_angles; a++) got += w[a] * PetscPowScalarInt(mu[a], k);
      const PetscReal want = (k % 2 == 0) ? 2.0 / (k + 1) : 0.0;
      // Odd powers cancel to zero, so those get an absolute comparison
      const PetscReal scale = (k % 2 == 0) ? want : 1.0;
      worst_moment = PetscMax(worst_moment, PetscAbsScalar(got - want) / scale);
   }

   if (worst_moment > moment_tol || worst_sum > sum_tol || !symmetric || !ordered || !involution) \
      *ok = PETSC_FALSE;

   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  1D S%-3" PetscInt_FMT " %3" PetscInt_FMT " ordinates: moments to degree %3" PetscInt_FMT \
      " %.3e (tol %.0e), sum of weights %.3e (tol %.0e), ascending %s, mirrored exactly %s, "
      "reflection map %s\n", sn_order, n_angles, 2 * sn_order - 1, (double)worst_moment, \
      (double)moment_tol, (double)worst_sum, (double)sum_tol, ordered ? "yes" : "NO", \
      symmetric ? "yes" : "NO", involution ? "yes" : "NO"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 2: the 1D S2 and S4 sets against their published values
//
// The moment checks above would pass for any correct Gauss rule of the right
// order, so they cannot tell a right answer from a right answer to the wrong
// question. These two literals are the outside reference - and they are the
// values the code carried as truncated constants until the generator replaced
// them, so they also say the change did not move the set it already had
static PetscErrorCode CheckGaussLegendreReference(PetscBool *ok)
{
   SNQuadrature s2, s4;
   PetscReal worst = 0.0;
   const PetscReal tol = 1e-15;

   PetscFunctionBeginUser;

   PetscCall(s2.create(2));
   PetscCall(s4.create(4));

   // Ascending in mu, so the negative half comes first
   const PetscReal mu2[2] = {-0.57735026918962576, 0.57735026918962576};
   const PetscReal w2[2]  = {1.0, 1.0};
   const PetscReal mu4[4] = {-0.86113631159405258, -0.33998104358485626, \
                              0.33998104358485626,  0.86113631159405258};
   const PetscReal w4[4]  = {0.34785484513745386, 0.65214515486254614, \
                             0.65214515486254614, 0.34785484513745386};

   const std::vector<PetscScalar> ws2 = WeightsToHost(s2);
   const std::vector<PetscScalar> ws4 = WeightsToHost(s4);
   for (PetscInt a = 0; a < 2; a++) {
      worst = PetscMax(worst, PetscAbsScalar(s2.mu_host()[a] - mu2[a]));
      worst = PetscMax(worst, PetscAbsScalar(ws2[a] - w2[a]));
   }
   for (PetscInt a = 0; a < 4; a++) {
      worst = PetscMax(worst, PetscAbsScalar(s4.mu_host()[a] - mu4[a]));
      worst = PetscMax(worst, PetscAbsScalar(ws4[a] - w4[a]));
   }

   if (worst > tol) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  1D S2 and S4 against the published nodes and weights: worst difference %.3e (tol %.0e)\n", \
      (double)worst, (double)tol));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 2D and 3D: the level-symmetric sets
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The even axis moments of one cosine over the sphere: sum_a w_a c_a^2m should
// be 4 pi / (2m + 1). Returns the worst RELATIVE error over m = 0 .. N/2, which
// is the full set of conditions that define a level-symmetric set of order N -
// so this is the real test of what src/sn_lqn_table.py generated
static PetscReal WorstAxisMoment(PetscInt sn_order, PetscInt n_angles, \
   const PetscScalar *cosine, const std::vector<PetscScalar> &w)
{
   PetscReal worst = 0.0;

   for (PetscInt m = 0; m <= sn_order / 2; m++) {
      PetscScalar got = 0.0;
      for (PetscInt a = 0; a < n_angles; a++) {
         got += w[a] * PetscPowScalarInt(cosine[a], 2 * m);
      }
      const PetscReal want = 4.0 * PETSC_PI / (2 * m + 1);
      worst = PetscMax(worst, PetscAbsScalar(got - want) / want);
   }
   return worst;
}

// A reflection map has to be an involution that flips exactly its own cosine
// and preserves the weight - the last part is what makes a reflective boundary
// conserve, and it holds exactly because a sign flip stays inside a permutation
// class, so the partner's weight is a copy of the same value
static PetscBool ReflectionMapIsSound(PetscInt n_angles, const PetscInt *map, PetscInt axis, \
   const PetscScalar *mu, const PetscScalar *eta, const PetscScalar *xi, \
   const std::vector<PetscScalar> &w)
{
   const PetscScalar *cosine[3] = {mu, eta, xi};

   for (PetscInt a = 0; a < n_angles; a++) {
      const PetscInt b = map[a];
      if (b < 0 || b >= n_angles || map[b] != a || b == a) return PETSC_FALSE;
      if (w[b] != w[a]) return PETSC_FALSE;
      for (PetscInt d = 0; d < 3; d++) {
         if (!cosine[d]) continue;
         const PetscScalar want = (d == axis) ? -cosine[d][a] : cosine[d][a];
         if (cosine[d][b] != want) return PETSC_FALSE;
      }
   }
   return PETSC_TRUE;
}

// Check 3: one order of the 2D (XY) level-symmetric set
//
// XY folds the xi > 0 half onto the whole sphere, so xi is gone but the
// weights still have to integrate the sphere: the axis moments are the same
// conditions the 3D set satisfies. mu^2 + eta^2 < 1 strictly, because the
// missing xi is never zero
static PetscErrorCode CheckLevelSymmetric2D(PetscInt sn_order, PetscBool *ok)
{
   SNQuadrature2D quad;
   PetscReal worst_moment = 0.0, worst_sum = 0.0, smallest_gap = 1.0, smallest_weight = 0.0;
   PetscBool maps_sound = PETSC_TRUE;
   const PetscReal moment_tol = 1e-12;

   PetscFunctionBeginUser;

   PetscCall(quad.create(sn_order));
   const PetscInt n_angles = quad.n_angles();
   PetscCheck(n_angles == sn_order * (sn_order + 2) / 2, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
      "2D S%" PetscInt_FMT " gave %" PetscInt_FMT " ordinates, not N (N + 2) / 2 = %" PetscInt_FMT, \
      sn_order, n_angles, sn_order * (sn_order + 2) / 2);

   const PetscScalar *mu = quad.mu_host(), *eta = quad.eta_host();
   const std::vector<PetscScalar> w = WeightsToHost(quad);

   PetscScalar sum = 0.0;
   smallest_weight = PetscRealPart(w[0]);
   for (PetscInt a = 0; a < n_angles; a++) {
      sum += w[a];
      smallest_weight = PetscMin(smallest_weight, PetscRealPart(w[a]));
      // The dropped xi is what is left of the unit vector, and it is never zero
      smallest_gap = PetscMin(smallest_gap, \
         1.0 - PetscRealPart(mu[a] * mu[a] + eta[a] * eta[a]));
   }
   worst_sum = PetscAbsScalar(sum - 4.0 * PETSC_PI) / (4.0 * PETSC_PI);

   worst_moment = PetscMax(WorstAxisMoment(sn_order, n_angles, mu, w), \
                           WorstAxisMoment(sn_order, n_angles, eta, w));

   maps_sound = (PetscBool)(ReflectionMapIsSound(n_angles, quad.reflect_mu_host(), 0, mu, eta, \
                                                 NULL, w) &&
                            ReflectionMapIsSound(n_angles, quad.reflect_eta_host(), 1, mu, eta, \
                                                 NULL, w));

   if (worst_moment > moment_tol || worst_sum > moment_tol || smallest_gap <= 0.0 || \
      smallest_weight <= 0.0 || !maps_sound) *ok = PETSC_FALSE;

   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  2D S%-3" PetscInt_FMT " %3" PetscInt_FMT " ordinates: axis moments %.3e, sum of weights "
      "%.3e (tol %.0e), smallest 1 - mu^2 - eta^2 %.3e, smallest weight %.3e, reflection maps %s\n", \
      sn_order, n_angles, (double)worst_moment, (double)worst_sum, (double)moment_tol, \
      (double)smallest_gap, (double)smallest_weight, maps_sound ? "sound" : "BROKEN"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Check 4: one order of the 3D level-symmetric set
//
// Same conditions as 2D with xi a real cosine, plus the two things 3D can say
// that 2D cannot: the directions are unit vectors, and the set is invariant
// under permuting the axes - which is what "level symmetric" means and what a
// table with a class weight attached to the wrong permutation would break
static PetscErrorCode CheckLevelSymmetric3D(PetscInt sn_order, PetscBool *ok)
{
   SNQuadrature3D quad;
   PetscReal worst_moment = 0.0, worst_sum = 0.0, worst_norm = 0.0, smallest_weight = 0.0;
   PetscBool maps_sound = PETSC_TRUE, axis_symmetric = PETSC_TRUE;
   const PetscReal moment_tol = 1e-12;
   const PetscReal norm_tol = 1e-14;

   PetscFunctionBeginUser;

   PetscCall(quad.create(sn_order));
   const PetscInt n_angles = quad.n_angles();
   PetscCheck(n_angles == sn_order * (sn_order + 2), PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
      "3D S%" PetscInt_FMT " gave %" PetscInt_FMT " ordinates, not N (N + 2) = %" PetscInt_FMT, \
      sn_order, n_angles, sn_order * (sn_order + 2));

   const PetscScalar *mu = quad.mu_host(), *eta = quad.eta_host(), *xi = quad.xi_host();
   const std::vector<PetscScalar> w = WeightsToHost(quad);

   PetscScalar sum = 0.0;
   smallest_weight = PetscRealPart(w[0]);
   for (PetscInt a = 0; a < n_angles; a++) {
      sum += w[a];
      smallest_weight = PetscMin(smallest_weight, PetscRealPart(w[a]));
      worst_norm = PetscMax(worst_norm, \
         PetscAbsScalar(mu[a] * mu[a] + eta[a] * eta[a] + xi[a] * xi[a] - 1.0));
   }
   worst_sum = PetscAbsScalar(sum - 4.0 * PETSC_PI) / (4.0 * PETSC_PI);

   worst_moment = PetscMax(WorstAxisMoment(sn_order, n_angles, mu, w), \
                  PetscMax(WorstAxisMoment(sn_order, n_angles, eta, w), \
                           WorstAxisMoment(sn_order, n_angles, xi, w)));

   maps_sound = (PetscBool)(ReflectionMapIsSound(n_angles, quad.reflect_mu_host(), 0, mu, eta, \
                                                 xi, w) &&
                            ReflectionMapIsSound(n_angles, quad.reflect_eta_host(), 1, mu, eta, \
                                                 xi, w) &&
                            ReflectionMapIsSound(n_angles, quad.reflect_xi_host(), 2, mu, eta, \
                                                 xi, w));

   // Every permutation of every ordinate's cosines is an ordinate of the set,
   // carrying the same weight - by search, so the ordinate ordering is not
   // what this rests on
   {
      const PetscScalar *cosine[3] = {mu, eta, xi};
      const PetscInt perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, \
                                    {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
      for (PetscInt a = 0; a < n_angles && axis_symmetric; a++) {
         for (PetscInt p = 0; p < 6 && axis_symmetric; p++) {
            const PetscScalar want[3] = {cosine[perms[p][0]][a], cosine[perms[p][1]][a], \
                                         cosine[perms[p][2]][a]};
            PetscBool found = PETSC_FALSE;
            for (PetscInt b = 0; b < n_angles; b++) {
               if (mu[b] == want[0] && eta[b] == want[1] && xi[b] == want[2]) {
                  found = (PetscBool)(w[b] == w[a]);
                  break;
               }
            }
            if (!found) axis_symmetric = PETSC_FALSE;
         }
      }
   }

   if (worst_moment > moment_tol || worst_sum > moment_tol || worst_norm > norm_tol || \
      smallest_weight <= 0.0 || !maps_sound || !axis_symmetric) *ok = PETSC_FALSE;

   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  3D S%-3" PetscInt_FMT " %3" PetscInt_FMT " ordinates: axis moments %.3e, sum of weights "
      "%.3e (tol %.0e), unit vectors %.3e (tol %.0e), smallest weight %.3e, reflection maps %s, "
      "axis permutations %s\n", sn_order, n_angles, (double)worst_moment, (double)worst_sum, \
      (double)moment_tol, (double)worst_norm, (double)norm_tol, (double)smallest_weight, \
      maps_sound ? "sound" : "BROKEN", axis_symmetric ? "invariant" : "BROKEN"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 5: the 2D set is the xi > 0 half of the 3D set, weights doubled
//
// That is the whole of what XY geometry does to a level-symmetric set, and it
// is the one statement that ties the two classes together - they build their
// ordinates from the same table but by separate loops, so a divergence between
// them is exactly the kind of thing nothing else would catch. Matched by (mu,
// eta) with ==, since both come from the same table entries
static PetscErrorCode CheckFoldedSets(PetscInt sn_order, PetscBool *ok)
{
   SNQuadrature2D quad2d;
   SNQuadrature3D quad3d;
   PetscReal worst = 0.0;
   PetscBool matched = PETSC_TRUE;
   const PetscReal tol = 1e-15;

   PetscFunctionBeginUser;

   PetscCall(quad2d.create(sn_order));
   PetscCall(quad3d.create(sn_order));
   const std::vector<PetscScalar> w2 = WeightsToHost(quad2d), w3 = WeightsToHost(quad3d);
   const PetscScalar *mu2 = quad2d.mu_host(), *eta2 = quad2d.eta_host();
   const PetscScalar *mu3 = quad3d.mu_host(), *eta3 = quad3d.eta_host(), *xi3 = quad3d.xi_host();

   PetscInt n_matched = 0;
   for (PetscInt a = 0; a < quad2d.n_angles(); a++) {
      PetscBool found = PETSC_FALSE;
      for (PetscInt b = 0; b < quad3d.n_angles(); b++) {
         if (PetscRealPart(xi3[b]) > 0.0 && mu3[b] == mu2[a] && eta3[b] == eta2[a]) {
            worst = PetscMax(worst, PetscAbsScalar(w2[a] - 2.0 * w3[b]) / PetscAbsScalar(w2[a]));
            found = PETSC_TRUE;
            n_matched++;
            break;
         }
      }
      if (!found) matched = PETSC_FALSE;
   }
   // Every xi > 0 ordinate of the 3D set has to be used, not just covered
   if (n_matched != quad3d.n_angles() / 2) matched = PETSC_FALSE;

   if (worst > tol || !matched) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  S%-3" PetscInt_FMT " 2D set is the xi > 0 half of the 3D set: %3" PetscInt_FMT \
      " of %3" PetscInt_FMT " ordinates matched (%s), doubled weights differ by %.3e (tol %.0e)\n", \
      sn_order, n_matched, quad3d.n_angles() / 2, matched ? "all" : "INCOMPLETE", \
      (double)worst, (double)tol));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 6: the two level cosines that have a closed form
//
// S2 is forced by the unit vector condition alone, 3 c^2 = 1. S4's mu_1 is the
// root of 2 mu_1^4 + (1 - 2 mu_1^2)^2 = 3/5, which is the fifth moment
// condition written out, and it is the only other order whose cosine falls out
// in radicals. Everything above S4 rests on the generator's own assertions, so
// these two are what says the generator is solving the right problem
static PetscErrorCode CheckClosedFormCosines(PetscBool *ok)
{
   SNQuadrature2D s2_2d, s4_2d;
   SNQuadrature3D s2_3d, s4_3d;
   PetscReal worst = 0.0;
   const PetscReal tol = 1e-15;

   PetscFunctionBeginUser;

   PetscCall(s2_2d.create(2));
   PetscCall(s4_2d.create(4));
   PetscCall(s2_3d.create(2));
   PetscCall(s4_3d.create(4));

   const PetscReal c2 = PetscSqrtReal(1.0 / 3.0);
   const PetscReal mu1 = PetscSqrtReal((5.0 - PetscSqrtReal(10.0)) / 15.0);

   // The first ordinate of each set is the base point with both signs flipped
   worst = PetscMax(worst, PetscAbsScalar(PetscAbsScalar(s2_2d.mu_host()[0]) - c2));
   worst = PetscMax(worst, PetscAbsScalar(PetscAbsScalar(s2_3d.mu_host()[0]) - c2));
   worst = PetscMax(worst, PetscAbsScalar(PetscAbsScalar(s2_3d.xi_host()[0]) - c2));
   worst = PetscMax(worst, PetscAbsScalar(PetscAbsScalar(s4_2d.mu_host()[0]) - mu1));
   worst = PetscMax(worst, PetscAbsScalar(PetscAbsScalar(s4_2d.eta_host()[0]) - mu1));
   worst = PetscMax(worst, PetscAbsScalar(PetscAbsScalar(s4_3d.mu_host()[0]) - mu1));
   worst = PetscMax(worst, PetscAbsScalar(PetscAbsScalar(s4_3d.eta_host()[0]) - mu1));

   if (worst > tol) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  S2 and S4 base cosines against their closed forms: worst difference %.3e (tol %.0e)\n", \
      (double)worst, (double)tol));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 7: an order the set does not have fails
//
// 1D generates its rule, so only an odd or non-positive order is out of range;
// 2D and 3D are level-symmetric, whose weights stop being all positive above
// S18, so an order past the table has to fail rather than silently reach for
// something else. The error handler is swapped for the returning one so an
// expected error does not abort the run (CI puts -on_error_abort in
// PETSC_OPTIONS) or print a stack trace into the output
static PetscErrorCode CheckUnsupportedOrders(PetscBool *ok)
{
   PetscInt n_rejected = 0, n_cases = 0;

   PetscFunctionBeginUser;

   PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
   {
      SNQuadrature odd_1d;
      SNQuadrature2D odd_2d, past_2d;
      SNQuadrature3D odd_3d, past_3d;

      n_cases = 5;
      if (odd_1d.create(3)) n_rejected++;
      if (odd_2d.create(3)) n_rejected++;
      if (odd_3d.create(3)) n_rejected++;
      if (past_2d.create(LQN_MAX_ORDER + 2)) n_rejected++;
      if (past_3d.create(LQN_MAX_ORDER + 2)) n_rejected++;
   }
   PetscCall(PetscPopErrorHandler());

   if (n_rejected != n_cases) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  unsupported orders (odd in every dimension, S%" PetscInt_FMT " in 2D and 3D): %" \
      PetscInt_FMT " of %" PetscInt_FMT " rejected\n", LQN_MAX_ORDER + 2, n_rejected, n_cases));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   PetscBool ok = PETSC_TRUE;

   PetscFunctionBeginUser;

   PetscCall(PetscInitialize(&argc, &args, (char*)0, NULL));

   // No block scope here, unlike the solve drivers: every device view lives
   // and dies inside the Check* functions, well before PetscFinalize
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Quadrature verification\n"));

   // 1D generates its rule, so the orders are a spread rather than a list: the
   // two with published values, a few in between, and one well past anything
   // the table-driven dimensions can reach
   const PetscInt orders_1d[] = {2, 4, 6, 8, 12, 16, 32};
   for (PetscInt k = 0; k < (PetscInt)(sizeof(orders_1d) / sizeof(orders_1d[0])); k++) {
      PetscCall(CheckGaussLegendre(orders_1d[k], &ok));
   }
   PetscCall(CheckGaussLegendreReference(&ok));

   // 2D and 3D are tabulated, so EVERY order the table carries is checked -
   // there are only ten of them and the axis moments are what the table is for
   for (PetscInt sn_order = LQN_MIN_ORDER; sn_order <= LQN_MAX_ORDER; sn_order += 2) {
      PetscCall(CheckLevelSymmetric2D(sn_order, &ok));
   }
   for (PetscInt sn_order = LQN_MIN_ORDER; sn_order <= LQN_MAX_ORDER; sn_order += 2) {
      PetscCall(CheckLevelSymmetric3D(sn_order, &ok));
   }
   for (PetscInt sn_order = LQN_MIN_ORDER; sn_order <= LQN_MAX_ORDER; sn_order += 2) {
      PetscCall(CheckFoldedSets(sn_order, &ok));
   }

   PetscCall(CheckClosedFormCosines(&ok));
   PetscCall(CheckUnsupportedOrders(&ok));

   if (!ok) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "Quadrature verification FAILED\n"));

   PetscCall(PetscFinalize());
   return ok ? 0 : 1;
}
