// The CG-SUPG backend's verification (UnstructuredCG + terms_cg.hpp), serial
// and in parallel (-n 2, -n 4). Every check is parallel-safe: sums and maxima
// are reduced over the ranks, and every mesh is small
//
//   1. The element tables, on quads, triangles, hexes, tets and an irregular
//      Gmsh triangulation: sum_ij M_ij = V (partition of unity), sum_j G_ij =
//      sum_j K_ij = 0, and the linear exactness sum_j G^d_ij x^d'_j =
//      delta_dd' int phi_i (int phi_i = sum_j M_ij) - which is also what fails
//      if a basis function were matched to the wrong vertex
//   2. The layout: the rows are the global vertices times the angles, and the
//      lumped masses of the owned vertices sum to the domain volume over ranks
//   3. Consistency: a constant is the exact discrete solution of an infinite
//      medium with reflective and matching-inflow vacuum faces - with a VOID
//      box painted in (tau = h / zeta there), at zeta 0.5 and 2, on every cell
//      shape - so A c - b is zero to rounding through the library's operator
//      AND rhs (weighted source, weak inflow, reflective couplings, the
//      matrix-free SUPG scatter). And the term's add_diagonal is bitwise the
//      assembled matrix's diagonal. 3b: the global balance of a LINEAR flux
//      (the lumped-mass sum of A psi - b is outflow minus inflow, exactly) -
//      what fixes the SCALE of the weak boundary terms, which a constant
//      cannot see
//   4. Order: the pure absorber with left inflow and reflective y faces
//      (verify_plexk's DG1 order check's problem), nodal scalar flux against
//      the exact discrete-ordinates solution, quads and triangles
//   5. Benchmark A - the SAAF-LS void slab (Laboure et al., arXiv 1605.05388,
//      s4.3, where GMRES + BoomerAMG on SAAF-tau takes 801 to 8120
//      iterations): a source region, a void and an absorber, reflective at
//      x = 0, as a quasi-1D strip (reflective y faces) against the EXACT
//      discrete-ordinates solution of the same quadrature. Converges, at zeta
//      0.5 and 2
//   6. Benchmark B - Hammer, Morel & Wang's thin/thick two-region slab (arXiv
//      1902.08729 s III.B): sigma 0.1 | sigma 10, a source everywhere,
//      vacuum both ends. Converges, and the flux dip next to the interface at
//      their 8 cells per region (the effect they report for SAAF-tau) is
//      printed at zeta 0.5 and 2
//   7. Error paths: dirichlet_cell, a slanted reflective face, an unknown
//      "Face Sets" id, zeta <= 0
//
// Currently run with: make build_tests && ./verify_cgk

#include "ubolt/ubolt.hpp"
#include <petscksp.h>
#include "pflare.h"
#include <cmath>
#include <functional>
#include <type_traits>
#include <string>
#include <vector>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Everything a single-group CG solve needs, built the way the driver builds it
template <class Quad>
struct CGProblem {
   Quad quad;
   PhaseSpace ps;
   UnstructuredCG disc;
   PetscIntKokkosView mat_id_d;
   GroupXSections xs;
   SUPGTermCG supg;
   ScatteringTermCG scattering;
   TransportOperator op;
};

template <class Quad, class Box>
static PetscErrorCode BuildCG(const PlexMeshSpec &mesh, PetscInt sn_order, const BCSpec &bcs, PetscReal zeta, \
   const MaterialSpec &mats, const std::vector<Box> &boxes, CGProblem<Quad> &p)
{
   PetscFunctionBeginUser;

   PetscCall(p.quad.create(sn_order));
   PetscCall(p.disc.create_mesh(PETSC_COMM_WORLD, mesh));
   PetscCall(p.ps.create(PETSC_COMM_WORLD, p.disc.n_global_vertices(), p.quad.n_angles()));
   PetscCall(p.disc.create(p.ps, p.quad, bcs, zeta));
   PetscCall(p.disc.paint_boxes(0, boxes, p.mat_id_d));
   PetscCall(p.xs.create(1, p.disc.n_local_elements()));
   PetscCall(p.xs.set_from_materials(mats, p.mat_id_d));
   PetscCall(p.supg.create(p.ps, p.disc, p.xs.sigma_t(0)));
   PetscCall(p.scattering.create(p.ps, p.disc, p.quad, p.xs.sigma_t(0), p.xs.sigma_s(0, 0)));
   PetscCall(p.op.create(PETSC_COMM_WORLD, p.ps, p.disc));
   PetscCall(p.op.add_term(&p.supg));
   PetscCall(p.op.add_term(&p.scattering));
   PetscCall(p.op.assemble());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// The rhs exactly as the driver fills it
template <class Quad>
static PetscErrorCode FillRhsCG(const CGProblem<Quad> &p, const MaterialSpec &mats, Vec b)
{
   PetscFunctionBeginUser;

   PetscCall(VecSet(b, 0.0));
   PetscCall(UboltFillInflow(p.disc.boundary_info(), b));
   PetscCall(UboltFillSourceCG(p.disc, p.quad, mats, p.mat_id_d, p.xs.sigma_t(0), 0, b));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Solve to well below every comparison tolerance, with the driver's default
// on an unstructured mesh: PCAIR on the element-block (at one dof per vertex,
// point Jacobi) scaled pmat. On CG it makes no measurable difference (the void
// slab takes 9 - 10 at rtol 1e-13 from 40 to 640 cells either way)
template <class Quad>
static PetscErrorCode SolveTight(const CGProblem<Quad> &p, Vec b, Vec x, PetscBool *converged, PetscInt *its)
{
   TransportSolver solver;

   PetscFunctionBeginUser;

   PetscCall(solver.create(PETSC_COMM_WORLD, p.op, p.op.assembled_mat(), nullptr, PETSC_TRUE));
   PetscCall(KSPSetTolerances(solver.ksp(), 1e-13, 1e-50, PETSC_CURRENT, 2000));
   PetscCall(VecZeroEntries(x));
   PetscCall(solver.solve(b, x));
   *converged = (PetscBool)(solver.converged_reason() > 0);
   PetscCall(KSPGetIterationNumber(solver.ksp(), its));
   PetscCall(solver.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Quad>
static PetscErrorCode DestroyCG(CGProblem<Quad> &p)
{
   PetscFunctionBeginUser;

   PetscCall(p.op.destroy());
   PetscCall(p.scattering.destroy());
   PetscCall(p.disc.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// The nodal scalar flux of the owned vertices, on the host
template <class Quad>
static PetscErrorCode NodalScalarFlux(const CGProblem<Quad> &p, Vec psi, std::vector<PetscReal> &phi)
{
   PetscFunctionBeginUser;

   auto w_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.quad.w_d());
   const PetscScalar *a;
   phi.assign(p.ps.local_cells, 0.0);
   PetscCall(VecGetArrayRead(psi, &a));
   for (PetscInt k = 0; k < p.ps.local_cells; k++) {
      for (PetscInt q = 0; q < p.quad.n_angles(); q++) phi[k] += PetscRealPart(w_h(q, 0) * a[k * p.ps.n_angles + q]);
   }
   PetscCall(VecRestoreArrayRead(psi, &a));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A 1D slab, region by region: [x0, x1) with sigma_t = sigma_a and an
// isotropic source strength q
struct SlabRegion {
   PetscReal x0, x1, sigma, q;
};

// The EXACT discrete-ordinates scalar flux of a pure-absorber slab on
// [regions.front().x0, regions.back().x1], vacuum (no inflow) on the right,
// vacuum or reflective on the left, for the given ordinates' x cosines and
// weights: each ordinate is an ODE along x solved region by region in closed
// form, psi -> q / (W sigma) + (psi - q / (W sigma)) exp(-sigma s) over a path
// length s = dx / |mu| (psi + q s / W in a void). What a quasi-1D strip with
// reflective y faces must converge to
static PetscReal ExactSlabFlux(const std::vector<SlabRegion> &regions, PetscBool reflect_left, \
   const std::vector<PetscReal> &mu, const std::vector<PetscReal> &w, PetscReal sum_weights, PetscReal x)
{
   auto march = [&](PetscReal psi, PetscReal abs_mu, PetscReal from, PetscReal to) {
      // Along the travel direction, region by region
      const PetscReal lo = PetscMin(from, to), hi = PetscMax(from, to);
      const bool rightward = to > from;
      const PetscInt n = (PetscInt)regions.size();
      for (PetscInt i = 0; i < n; i++) {
         const SlabRegion &r = rightward ? regions[i] : regions[n - 1 - i];
         const PetscReal a = PetscMax(r.x0, lo), b = PetscMin(r.x1, hi);
         if (b <= a) continue;
         const PetscReal s = (b - a) / abs_mu;
         if (r.sigma > 0.0) {
            const PetscReal eq = r.q / (sum_weights * r.sigma);
            psi = eq + (psi - eq) * PetscExpReal(-r.sigma * s);
         } else psi += r.q * s / sum_weights;
      }
      return psi;
   };
   const PetscReal xl = regions.front().x0, xr = regions.back().x1;
   PetscReal phi = 0.0;
   for (size_t a = 0; a < mu.size(); a++) {
      const PetscReal m = PetscAbsReal(mu[a]);
      PetscReal psi;
      if (mu[a] < 0.0) psi = march(0.0, m, xr, x);
      else {
         const PetscReal at_left = reflect_left ? march(0.0, m, xr, xl) : 0.0;
         psi = march(at_left, m, xl, x);
      }
      phi += w[a] * psi;
   }
   return phi;
}

// Solve one quasi-1D slab (a strip of n_x x 1 square cells, reflective top and
// bottom) and measure the nodal scalar flux against ExactSlabFlux: the
// lumped-mass-weighted RMS error and the max error, both relative to the max
// exact flux. Also hands back the nodal (x, phi_h, phi_exact) triples and the
// iteration count
static PetscErrorCode SolveSlab(const std::vector<SlabRegion> &regions, PetscBool reflect_left, PetscInt n_x, \
   PetscInt sn_order, PetscReal zeta, PetscBool simplex, PetscReal *rms, PetscReal *max_err, PetscInt *its, \
   PetscBool *converged, std::vector<PetscReal> *profile)
{
   const PetscReal length = regions.back().x1 - regions.front().x0;
   const PetscInt n_mat = (PetscInt)regions.size();
   CGProblem<SNQuadrature2D> p;
   MaterialSpec mats;
   BCSpec bcs;
   PlexMeshSpec mesh;
   std::vector<MaterialBox2D> boxes;
   Vec psi = NULL, b = NULL;

   PetscFunctionBeginUser;

   mesh.dimension = 2;
   mesh.simplex = simplex;
   mesh.n_cells[0] = n_x;
   mesh.n_cells[1] = 1;
   mesh.lengths[0] = length;
   mesh.lengths[1] = length / n_x;

   PetscCall(mats.create(n_mat, 1));
   for (PetscInt i = 0; i < n_mat; i++) {
      PetscCall(mats.set_sigma_t(i, 0, regions[i].sigma));
      PetscCall(mats.set_source(i, 0, regions[i].q));
      if (i == 0) continue;
      MaterialBox2D box;
      box.x0 = regions[i].x0;
      box.x1 = regions[i].x1;
      box.y0 = -1.0;
      box.y1 = 1.0 + mesh.lengths[1];
      box.material = i;
      boxes.push_back(box);
   }
   bcs.set(StructuredFD2D::FACE_BOTTOM, BCType::REFLECT);
   bcs.set(StructuredFD2D::FACE_TOP, BCType::REFLECT);
   if (reflect_left) bcs.set(StructuredFD2D::FACE_LEFT, BCType::REFLECT);

   PetscCall(BuildCG(mesh, sn_order, bcs, zeta, mats, boxes, p));
   PetscCall(MatCreateVecs(p.op.assembled_mat(), &psi, &b));
   PetscCall(FillRhsCG(p, mats, b));
   PetscCall(SolveTight(p, b, psi, converged, its));

   std::vector<PetscReal> phi, mu, w;
   PetscCall(NodalScalarFlux(p, psi, phi));
   auto w_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.quad.w_d());
   for (PetscInt a = 0; a < p.quad.n_angles(); a++) {
      mu.push_back(PetscRealPart(p.quad.mu_host()[a]));
      w.push_back(PetscRealPart(w_h(a, 0)));
   }
   PetscReal sums[2] = {0.0, 0.0}, maxes[2] = {0.0, 0.0};
   if (profile) profile->clear();
   for (PetscInt k = 0; k < p.ps.local_cells; k++) {
      const PetscReal x = p.disc.vertex_coord_host()[3 * k];
      const PetscReal exact = ExactSlabFlux(regions, reflect_left, mu, w, PetscRealPart(p.quad.sum_weights()), x);
      const PetscReal m = p.disc.lumped_mass_host()[k];
      sums[0] += m * (phi[k] - exact) * (phi[k] - exact);
      sums[1] += m;
      maxes[0] = PetscMax(maxes[0], PetscAbsReal(phi[k] - exact));
      maxes[1] = PetscMax(maxes[1], PetscAbsReal(exact));
      if (profile) {
         profile->push_back(x);
         profile->push_back(phi[k]);
         profile->push_back(exact);
      }
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sums, 2, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, maxes, 2, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
   *rms = PetscSqrtReal(sums[0] / sums[1]) / maxes[1];
   *max_err = maxes[0] / maxes[1];

   PetscCall(VecDestroy(&psi));
   PetscCall(VecDestroy(&b));
   PetscCall(DestroyCG(p));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 1: the element tables
template <class Quad>
static PetscErrorCode CheckTables(const char *where, const PlexMeshSpec &mesh, PetscBool *ok)
{
   Quad quad;
   PhaseSpace ps;
   UnstructuredCG disc;
   const PetscReal tol = 1e-12;
   PetscReal err[4] = {0.0, 0.0, 0.0, 0.0};

   PetscFunctionBeginUser;

   PetscCall(quad.create(2));
   PetscCall(disc.create_mesh(PETSC_COMM_WORLD, mesh));
   PetscCall(ps.create(PETSC_COMM_WORLD, disc.n_global_vertices(), quad.n_angles()));
   PetscCall(disc.create(ps, quad));

   const PetscInt nv = disc.n_element_vertices(), dim = disc.dimension();
   const auto &M = disc.mass_host();
   const auto &G = disc.grad_host();
   const auto &K = disc.stiff_host();
   const auto &ev = disc.elem_vertex_host();
   const auto &xyz = disc.local_vertex_coord_host();
   for (PetscInt e = 0; e < disc.n_local_elements(); e++) {
      const PetscReal vol = disc.element_volume_host()[e];
      const PetscReal h = PetscPowReal(vol, 1.0 / dim);
      PetscReal sum_m = 0.0;
      for (PetscInt i = 0; i < nv; i++) {
         PetscReal int_phi = 0.0;
         for (PetscInt j = 0; j < nv; j++) {
            sum_m += PetscRealPart(M[(e * nv + i) * nv + j]);
            int_phi += PetscRealPart(M[(e * nv + i) * nv + j]);
         }
         for (PetscInt d = 0; d < 3; d++) {
            PetscReal g_sum = 0.0;
            for (PetscInt j = 0; j < nv; j++) g_sum += PetscRealPart(G[((e * nv + i) * nv + j) * 3 + d]);
            err[1] = PetscMax(err[1], PetscAbsReal(g_sum) * h / vol);
            for (PetscInt dd = 0; dd < 3; dd++) {
               PetscReal k_sum = 0.0, lin = 0.0;
               for (PetscInt j = 0; j < nv; j++) {
                  k_sum += PetscRealPart(K[((e * nv + i) * nv + j) * 9 + 3 * d + dd]);
                  lin += PetscRealPart(G[((e * nv + i) * nv + j) * 3 + d]) * xyz[3 * ev[e * nv + j] + dd];
               }
               err[2] = PetscMax(err[2], PetscAbsReal(k_sum) * h * h / vol);
               if (d < dim && dd < dim) {
                  const PetscReal expect = (d == dd) ? int_phi : 0.0;
                  err[3] = PetscMax(err[3], PetscAbsReal(lin - expect) / vol);
               }
            }
         }
      }
      err[0] = PetscMax(err[0], PetscAbsReal(sum_m - vol) / vol);
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, err, 4, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
   const PetscBool pass = (PetscBool)(err[0] <= tol && err[1] <= tol && err[2] <= tol && err[3] <= tol);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  element tables, %s: sum M = V %.1e, sum_j G = 0 %.1e, sum_j K = 0 " \
      "%.1e, G linear exactness %.1e (tol %.0e)%s\n", where, (double)err[0], (double)err[1], (double)err[2], \
      (double)err[3], (double)tol, pass ? "" : " FAILED"));

   // Check 2 rides along: the layout and the lumped masses
   PetscReal mass = 0.0, volume = 0.0;
   for (const PetscReal m : disc.lumped_mass_host()) mass += m;
   for (PetscInt e = 0; e < disc.n_local_elements(); e++) {
      if (disc.element_owned_host()[e]) volume += disc.element_volume_host()[e];
   }
   PetscInt rows = ps.local_rows(), global_rows = 0;
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &mass, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &volume, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
   PetscCallMPI(MPIU_Allreduce(&rows, &global_rows, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
   const PetscReal mass_err = PetscAbsReal(mass - volume) / volume;
   const PetscBool layout_pass = (PetscBool)(mass_err <= tol && global_rows == ps.global_rows() && \
      ps.n_cells == disc.n_global_vertices());
   if (!layout_pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  layout, %s: %" PetscInt_FMT " vertices x %" PetscInt_FMT \
      " angles = %" PetscInt_FMT " rows, lumped masses sum to the volume %.1e (tol %.0e)%s\n", where, \
      disc.n_global_vertices(), ps.n_angles, global_rows, (double)mass_err, (double)tol, layout_pass ? "" : " FAILED"));

   PetscCall(disc.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 3: the constant through the operator and the rhs. Material 0 absorbs
// and scatters (constant c = q / (W (sigma_t - sigma_s)) per ordinate),
// material 1 is a void painted over part of the domain (holding any constant)
template <class Quad, class Box>
static PetscErrorCode CheckConstant(const char *where, const PlexMeshSpec &mesh, const BCSpec &bcs_in, \
   const std::vector<Box> &void_box, PetscReal zeta, PetscBool *ok)
{
   CGProblem<Quad> p;
   MaterialSpec mats;
   BCSpec bcs = bcs_in;
   Vec c = NULL, b = NULL, r = NULL, d_comp = NULL, d_mat = NULL;
   const PetscReal sigma_t = 1.5, sigma_s = 0.7, q = 1.0, tol = 1e-12;

   PetscFunctionBeginUser;

   PetscCall(mats.create(2, 1));
   PetscCall(mats.set_sigma_t(0, 0, sigma_t));
   PetscCall(mats.set_sigma_s(0, 0, 0, sigma_s));
   PetscCall(mats.set_source(0, 0, q));
   // Every vacuum face fed exactly the medium's flux: inflow / W = c
   const PetscReal inflow = q / (sigma_t - sigma_s);
   for (PetscInt f = 1; f <= 2 * mesh.dimension; f++) {
      if (bcs.type(f) == BCType::VACUUM) bcs.set_inflow(f, inflow);
   }

   PetscCall(BuildCG(mesh, 4, bcs, zeta, mats, void_box, p));
   PetscCall(MatCreateVecs(p.op.assembled_mat(), &c, &b));
   PetscCall(VecDuplicate(b, &r));
   PetscCall(FillRhsCG(p, mats, b));
   PetscCall(VecSet(c, inflow / PetscRealPart(p.quad.sum_weights())));
   PetscCall(MatMult(p.op.mat(), c, r));
   PetscCall(VecAXPY(r, -1.0, b));
   PetscReal r_norm = 0.0, b_norm = 0.0;
   PetscCall(VecNorm(r, NORM_INFINITY, &r_norm));
   PetscCall(VecNorm(b, NORM_INFINITY, &b_norm));
   const PetscReal rel = r_norm / b_norm;

   // add_diagonal against the assembled matrix's diagonal: bitwise
   PetscCall(VecDuplicate(b, &d_comp));
   PetscCall(VecDuplicate(b, &d_mat));
   PetscCall(VecSet(d_comp, 0.0));
   PetscCall(p.supg.add_diagonal(d_comp));
   PetscCall(MatGetDiagonal(p.op.assembled_mat(), d_mat));
   PetscCall(VecAXPY(d_mat, -1.0, d_comp));
   PetscReal d_err = 0.0;
   PetscCall(VecNorm(d_mat, NORM_INFINITY, &d_err));

   const PetscBool pass = (PetscBool)(rel <= tol && d_err == 0.0);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  constant, %s, zeta %g: |A c - b| / |b| %.1e (tol %.0e), " \
      "add_diagonal vs the matrix %.1e (tol 0)%s\n", where, (double)zeta, (double)rel, (double)tol, (double)d_err, \
      pass ? "" : " FAILED"));

   PetscCall(VecDestroy(&c));
   PetscCall(VecDestroy(&b));
   PetscCall(VecDestroy(&r));
   PetscCall(VecDestroy(&d_comp));
   PetscCall(VecDestroy(&d_mat));
   PetscCall(DestroyCG(p));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 3b: the global balance, which fixes the weak boundary terms' SCALE (a
// constant cannot: it makes psi - psi_in vanish whatever the weight). Pure
// streaming on a box [0, L]^dim, psi = 1 + c . x on every ordinate, vacuum
// faces with inflow q_in. Weighted by the lumped masses the rows sum to
//   sum_i m_i (A psi - b)_i = int Omega . grad psi + sum_{inflow f} |Omega . n| int_f (psi - q_in / W)
// (the SUPG parts drop out, sum_i grad phi_i = 0), and int Omega . grad psi is
// the outflow minus the inflow of psi, so per ordinate it is
//   sum_{outflow faces} |Omega . n| int_f psi - sum_{inflow faces} |Omega . n| A_f q_in / W
// exact for a linear psi (the lumped face mass integrates it exactly)
template <class Quad>
static PetscErrorCode CheckBalance(const char *where, const PlexMeshSpec &mesh, PetscBool *ok)
{
   CGProblem<Quad> p;
   MaterialSpec mats;
   BCSpec bcs;
   const std::vector<typename std::conditional<std::is_same<Quad, SNQuadrature2D>::value, MaterialBox2D, \
      MaterialBox3D>::type> no_boxes;
   const PetscReal q_in = 0.7, c[3] = {0.3, -0.45, 0.2}, tol = 1e-12;
   const PetscInt dim = mesh.dimension;
   Vec psi = NULL, b = NULL, y = NULL;

   PetscFunctionBeginUser;

   PetscCall(mats.create(1, 1));
   for (PetscInt f = 1; f <= 2 * dim; f++) bcs.set_inflow(f, q_in);
   PetscCall(BuildCG(mesh, 4, bcs, 0.5, mats, no_boxes, p));
   PetscCall(MatCreateVecs(p.op.assembled_mat(), &psi, &b));
   PetscCall(VecDuplicate(b, &y));
   PetscCall(FillRhsCG(p, mats, b));

   const PetscInt na = p.ps.n_angles;
   {
      PetscScalar *a = nullptr;
      PetscCall(VecGetArray(psi, &a));
      for (PetscInt k = 0; k < p.ps.local_cells; k++) {
         PetscReal v = 1.0;
         for (PetscInt d = 0; d < dim; d++) v += c[d] * p.disc.vertex_coord_host()[3 * k + d];
         for (PetscInt q = 0; q < na; q++) a[k * na + q] = v;
      }
      PetscCall(VecRestoreArray(psi, &a));
   }
   PetscCall(MatMult(p.op.mat(), psi, y));
   PetscCall(VecAXPY(y, -1.0, b));

   std::vector<PetscReal> sums(na, 0.0);
   {
      const PetscScalar *a = nullptr;
      PetscCall(VecGetArrayRead(y, &a));
      for (PetscInt k = 0; k < p.ps.local_cells; k++) {
         for (PetscInt q = 0; q < na; q++) sums[q] += p.disc.lumped_mass_host()[k] * PetscRealPart(a[k * na + q]);
      }
      PetscCall(VecRestoreArrayRead(y, &a));
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sums.data(), na, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));

   PetscReal err = 0.0, scale = 0.0;
   const PetscReal W = PetscRealPart(p.quad.sum_weights());
   for (PetscInt q = 0; q < na; q++) {
      PetscReal om[3] = {PetscRealPart(p.quad.mu_host()[q]), PetscRealPart(p.quad.eta_host()[q]), 0.0};
      if constexpr (std::is_same<Quad, SNQuadrature3D>::value) om[2] = PetscRealPart(p.quad.xi_host()[q]);
      PetscReal expect = 0.0;
      for (PetscInt d = 0; d < dim; d++) {
         PetscReal area = 1.0, centre_rest = 1.0;
         for (PetscInt e = 0; e < dim; e++) {
            if (e == d) continue;
            area *= mesh.lengths[e];
            centre_rest += c[e] * 0.5 * mesh.lengths[e];
         }
         // The low face (normal -e_d) and the high one (+e_d)
         for (PetscInt side = 0; side < 2; side++) {
            const PetscReal on = side ? om[d] : -om[d];
            const PetscReal psi_face = centre_rest + (side ? c[d] * mesh.lengths[d] : 0.0);
            if (on > 0.0) expect += on * area * psi_face;
            else expect -= -on * area * q_in / W;
         }
      }
      err = PetscMax(err, PetscAbsReal(sums[q] - expect));
      scale = PetscMax(scale, PetscAbsReal(expect));
   }
   const PetscReal rel = err / scale;
   const PetscBool pass = (PetscBool)(rel <= tol);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  balance, %s: lumped-mass sum of A psi - b against outflow - inflow, " \
      "linear psi, per ordinate %.1e (tol %.0e)%s\n", where, (double)rel, (double)tol, pass ? "" : " FAILED"));

   PetscCall(VecDestroy(&psi));
   PetscCall(VecDestroy(&b));
   PetscCall(VecDestroy(&y));
   PetscCall(DestroyCG(p));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 4: the pure absorber on [0, 1]^2, inflow 1 on the left, reflective y
// faces, cold right face, sigma 1, S4. The nodal scalar flux against the exact
// SN solution (ExactSlabFlux), n = 8, 16, 32
static PetscErrorCode CheckOrder(const char *where, PetscBool simplex, PetscBool *ok)
{
   const PetscInt n_levels = 3, n_base = 8;
   const PetscReal sigma = 1.0, inflow = 1.0, min_order = 1.8;
   PetscReal err[n_levels];
   PetscBool converged_all = PETSC_TRUE;

   PetscFunctionBeginUser;

   for (PetscInt l = 0; l < n_levels; l++) {
      CGProblem<SNQuadrature2D> p;
      MaterialSpec mats;
      BCSpec bcs;
      PlexMeshSpec mesh;
      const std::vector<MaterialBox2D> no_boxes;
      Vec psi = NULL, b = NULL;
      PetscBool converged = PETSC_FALSE;
      PetscInt its = 0;
      const PetscInt n = n_base << l;

      mesh.dimension = 2;
      mesh.simplex = simplex;
      for (PetscInt d = 0; d < 2; d++) {
         mesh.n_cells[d] = n;
         mesh.lengths[d] = 1.0;
      }
      bcs.set(StructuredFD2D::FACE_LEFT, BCType::VACUUM);
      bcs.set_inflow(StructuredFD2D::FACE_LEFT, inflow);
      bcs.set(StructuredFD2D::FACE_BOTTOM, BCType::REFLECT);
      bcs.set(StructuredFD2D::FACE_TOP, BCType::REFLECT);
      PetscCall(mats.create(1, 1));
      PetscCall(mats.set_sigma_t(0, 0, sigma));

      PetscCall(BuildCG(mesh, 4, bcs, 0.5, mats, no_boxes, p));
      PetscCall(MatCreateVecs(p.op.assembled_mat(), &psi, &b));
      PetscCall(FillRhsCG(p, mats, b));
      PetscCall(SolveTight(p, b, psi, &converged, &its));
      if (!converged) converged_all = PETSC_FALSE;

      std::vector<PetscReal> phi;
      PetscCall(NodalScalarFlux(p, psi, phi));
      auto w_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.quad.w_d());
      const PetscReal psi_in = inflow / PetscRealPart(p.quad.sum_weights());
      PetscReal sums[2] = {0.0, 0.0};
      for (PetscInt k = 0; k < p.ps.local_cells; k++) {
         const PetscReal x = p.disc.vertex_coord_host()[3 * k];
         PetscReal exact = 0.0;
         for (PetscInt a = 0; a < p.quad.n_angles(); a++) {
            const PetscReal mu = PetscRealPart(p.quad.mu_host()[a]);
            if (mu > 0.0) exact += PetscRealPart(w_h(a, 0)) * psi_in * PetscExpReal(-sigma * x / mu);
         }
         const PetscReal m = p.disc.lumped_mass_host()[k];
         sums[0] += m * (phi[k] - exact) * (phi[k] - exact);
         sums[1] += m;
      }
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sums, 2, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
      err[l] = PetscSqrtReal(sums[0] / sums[1]);

      PetscCall(VecDestroy(&psi));
      PetscCall(VecDestroy(&b));
      PetscCall(DestroyCG(p));
   }

   const PetscReal order = PetscLog2Real(err[n_levels - 2] / err[n_levels - 1]);
   const PetscBool pass = (PetscBool)(converged_all && order >= min_order);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  order, %s, S4, pure absorber with left inflow: nodal RMS error " \
      "%.3e / %.3e / %.3e at n = %d / %d / %d, observed order %.2f (min %.1f)%s\n", where, (double)err[0], \
      (double)err[1], (double)err[2], (int)n_base, (int)(2 * n_base), (int)(4 * n_base), (double)order, \
      (double)min_order, pass ? "" : " FAILED"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Checks 5 and 6: a slab benchmark over a refinement sequence, at one zeta.
// Asserts convergence (every level converged, the finest error below
// max_final, the last observed order at least min_order) and prints each
// level's relative RMS error, max error and iteration count
static PetscErrorCode CheckSlabBenchmark(const char *name, const std::vector<SlabRegion> &regions, \
   PetscBool reflect_left, const std::vector<PetscInt> &n_x, PetscInt sn_order, PetscReal zeta, \
   PetscReal min_order, PetscReal max_final, PetscBool *ok)
{
   const PetscInt n_levels = (PetscInt)n_x.size();
   std::vector<PetscReal> rms(n_levels), max_err(n_levels);
   std::vector<PetscInt> its(n_levels);
   PetscBool converged_all = PETSC_TRUE;

   PetscFunctionBeginUser;

   for (PetscInt l = 0; l < n_levels; l++) {
      PetscBool converged = PETSC_FALSE;
      PetscCall(SolveSlab(regions, reflect_left, n_x[l], sn_order, zeta, PETSC_FALSE, &rms[l], &max_err[l], &its[l], \
         &converged, NULL));
      if (!converged) converged_all = PETSC_FALSE;
   }
   const PetscReal order = PetscLog2Real(rms[n_levels - 2] / rms[n_levels - 1]);
   const PetscBool pass = (PetscBool)(converged_all && order >= min_order && rms[n_levels - 1] <= max_final);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s, S%" PetscInt_FMT ", zeta %g: relative RMS / max error (its) by " \
      "cells", name, sn_order, (double)zeta));
   for (PetscInt l = 0; l < n_levels; l++) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %d: %.2e / %.2e (%d)", (int)n_x[l], (double)rms[l], (double)max_err[l], \
         (int)its[l]));
   }
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "; order %.2f (min %.1f), finest %.1e (tol %.0e)%s\n", (double)order, \
      (double)min_order, (double)rms[n_levels - 1], (double)max_final, pass ? "" : " FAILED"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Check 6's printed effect: at the benchmark's own 8 cells per region, how far
// below the exact flux the CG flux dips on the thin side of the interface and
// whether it oscillates (a local minimum of phi_h - phi_exact inside the thin
// region). Printed, not asserted - it is the scheme's behaviour at a coarse
// mesh, which zeta changes
static PetscErrorCode ReportInterfaceDip(const std::vector<SlabRegion> &regions, PetscInt n_x, PetscInt sn_order, \
   PetscReal zeta)
{
   std::vector<PetscReal> prof;
   PetscReal rms = 0.0, max_err = 0.0;
   PetscInt its = 0;
   PetscBool converged = PETSC_FALSE;
   PetscFunctionBeginUser;

   PetscCall(SolveSlab(regions, PETSC_FALSE, n_x, sn_order, zeta, PETSC_FALSE, &rms, &max_err, &its, &converged, &prof));
   // Largest relative undershoot of phi_h below phi_exact in the thin region
   // (x < the interface), and the vertex it is at
   const PetscReal x_if = regions[0].x1;
   PetscReal worst = 0.0, at = -1.0;
   for (size_t i = 0; i < prof.size(); i += 3) {
      if (prof[i] > x_if) continue;
      const PetscReal rel = (prof[i + 2] - prof[i + 1]) / prof[i + 2];
      if (rel > worst) {
         worst = rel;
         at = prof[i];
      }
   }
   // The worst over the ranks, and where: the rank holding it reports the x
   PetscReal global_worst = 0.0;
   PetscCallMPI(MPIU_Allreduce(&worst, &global_worst, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
   PetscReal global_at = (worst == global_worst) ? at : -PETSC_MAX_REAL;
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &global_at, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  thin/thick interface at %d cells, zeta %g: worst undershoot on the thin " \
      "side %.1f%% at x = %.3f (relative RMS %.2e)\n", (int)n_x, (double)zeta, (double)(100.0 * global_worst), \
      (double)global_at, (double)rms));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 7: configurations the backend must refuse. One case: build the mesh
// and try create() with the error handler silenced, counting a refusal
static PetscErrorCode TryErrorCase(const PlexMeshSpec &mesh, const BCSpec &bcs, PetscReal zeta, PetscInt *n_cases, \
   PetscInt *n_rejected)
{
   SNQuadrature2D quad;
   PhaseSpace ps;
   UnstructuredCG disc;
   PetscErrorCode ierr;

   PetscFunctionBeginUser;

   (*n_cases)++;
   PetscCall(quad.create(2));
   PetscCall(disc.create_mesh(PETSC_COMM_WORLD, mesh));
   PetscCall(ps.create(PETSC_COMM_WORLD, disc.n_global_vertices(), quad.n_angles()));
   PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
   ierr = disc.create(ps, quad, bcs, zeta);
   PetscCall(PetscPopErrorHandler());
   if (ierr) (*n_rejected)++;
   PetscCall(disc.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckErrorPaths(PetscBool *ok)
{
   PetscInt n_cases = 0, n_rejected = 0;

   PetscFunctionBeginUser;

   PlexMeshSpec box;
   box.dimension = 2;
   box.n_cells[0] = box.n_cells[1] = 3;
   box.lengths[0] = box.lengths[1] = 1.0;

   BCSpec dirichlet;
   dirichlet.set_vacuum_treatment(VacuumTreatment::DIRICHLET_CELL);
   PetscCall(TryErrorCase(box, dirichlet, 0.5, &n_cases, &n_rejected));

   BCSpec plain;
   PetscCall(TryErrorCase(box, plain, 0.0, &n_cases, &n_rejected));
   PetscCall(TryErrorCase(box, plain, -1.0, &n_cases, &n_rejected));

   BCSpec unknown;
   unknown.set(77, BCType::REFLECT);
   PetscCall(TryErrorCase(box, unknown, 0.5, &n_cases, &n_rejected));

   // The slanted mesh's "Face Sets" 12 is its hypotenuse
   PlexMeshSpec slanted;
   slanted.dimension = 2;
   slanted.file = "meshes/tri_slanted.msh";
   BCSpec slanted_reflect;
   slanted_reflect.set(12, BCType::REFLECT);
   PetscCall(TryErrorCase(slanted, slanted_reflect, 0.5, &n_cases, &n_rejected));

   const PetscBool pass = (PetscBool)(n_rejected == n_cases);
   if (!pass) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  error paths: %" PetscInt_FMT " of %" PetscInt_FMT " rejected%s\n", \
      n_rejected, n_cases, pass ? "" : " FAILED"));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   PetscBool ok = PETSC_TRUE;
   PetscMPIInt size;

   PetscFunctionBeginUser;

   PetscCall(PetscInitialize(&argc, &args, (char*)0, NULL));
   PCRegister_PFLARE();
   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CG-SUPG verification on %d rank(s)\n", (int)size));

   PlexMeshSpec quads, tris, hexes, tets, irregular;
   quads.dimension = tris.dimension = 2;
   hexes.dimension = tets.dimension = 3;
   for (PetscInt d = 0; d < 2; d++) {
      quads.n_cells[d] = tris.n_cells[d] = 6;
      quads.lengths[d] = tris.lengths[d] = 1.0;
   }
   quads.lengths[1] = tris.lengths[1] = 0.7;
   tris.simplex = PETSC_TRUE;
   for (PetscInt d = 0; d < 3; d++) {
      hexes.n_cells[d] = tets.n_cells[d] = 3;
      hexes.lengths[d] = tets.lengths[d] = 1.0 + 0.2 * d;
   }
   tets.simplex = PETSC_TRUE;
   irregular.dimension = 2;
   irregular.file = "meshes/square_irregular_tri.msh";
   const PetscBool have_tri = PetscDefined(HAVE_TRIANGLE) ? PETSC_TRUE : PETSC_FALSE;
   const PetscBool have_tet = (PetscDefined(HAVE_CTETGEN) || PetscDefined(HAVE_TETGEN)) ? PETSC_TRUE : PETSC_FALSE;

   // ~~~~~~~~~~
   // 1 + 2. Element tables and layout
   // ~~~~~~~~~~
   PetscCall(CheckTables<SNQuadrature2D>("quads", quads, &ok));
   if (have_tri) PetscCall(CheckTables<SNQuadrature2D>("triangles", tris, &ok));
   else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  triangles skipped (no Triangle)\n"));
   PetscCall(CheckTables<SNQuadrature3D>("hexes", hexes, &ok));
   if (have_tet) PetscCall(CheckTables<SNQuadrature3D>("tets", tets, &ok));
   else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  tets skipped (no tet mesher)\n"));
   PetscCall(CheckTables<SNQuadrature2D>("irregular Gmsh triangles", irregular, &ok));

   // ~~~~~~~~~~
   // 3. The constant, with a void box, reflective and vacuum faces
   // ~~~~~~~~~~
   {
      BCSpec bcs_2d;
      bcs_2d.set(StructuredFD2D::FACE_LEFT, BCType::REFLECT);
      bcs_2d.set(StructuredFD2D::FACE_BOTTOM, BCType::REFLECT);
      std::vector<MaterialBox2D> void_2d(1);
      void_2d[0].x0 = 0.3;
      void_2d[0].x1 = 0.7;
      void_2d[0].y0 = 0.2;
      void_2d[0].y1 = 0.5;
      void_2d[0].material = 1;
      BCSpec bcs_3d;
      bcs_3d.set(StructuredFD3D::FACE_LEFT, BCType::REFLECT);
      bcs_3d.set(StructuredFD3D::FACE_FRONT, BCType::REFLECT);
      bcs_3d.set(StructuredFD3D::FACE_BOTTOM, BCType::REFLECT);
      std::vector<MaterialBox3D> void_3d(1);
      void_3d[0].x0 = 0.3;
      void_3d[0].x1 = 0.8;
      void_3d[0].y0 = 0.3;
      void_3d[0].y1 = 0.9;
      void_3d[0].z0 = 0.5;
      void_3d[0].z1 = 1.5;
      void_3d[0].material = 1;
      for (const PetscReal zeta : {0.5, 2.0}) {
         PetscCall(CheckConstant<SNQuadrature2D>("quads", quads, bcs_2d, void_2d, zeta, &ok));
         if (have_tri) PetscCall(CheckConstant<SNQuadrature2D>("triangles", tris, bcs_2d, void_2d, zeta, &ok));
         PetscCall(CheckConstant<SNQuadrature3D>("hexes", hexes, bcs_3d, void_3d, zeta, &ok));
         if (have_tet) PetscCall(CheckConstant<SNQuadrature3D>("tets", tets, bcs_3d, void_3d, zeta, &ok));
      }
   }

   // ~~~~~~~~~~
   // 3b. The global balance - the weak boundary terms' scale
   // ~~~~~~~~~~
   PetscCall(CheckBalance<SNQuadrature2D>("quads", quads, &ok));
   if (have_tri) PetscCall(CheckBalance<SNQuadrature2D>("triangles", tris, &ok));
   PetscCall(CheckBalance<SNQuadrature3D>("hexes", hexes, &ok));
   if (have_tet) PetscCall(CheckBalance<SNQuadrature3D>("tets", tets, &ok));

   // ~~~~~~~~~~
   // 4. Order
   // ~~~~~~~~~~
   PetscCall(CheckOrder("quads", PETSC_FALSE, &ok));
   if (have_tri) PetscCall(CheckOrder("triangles", PETSC_TRUE, &ok));

   // ~~~~~~~~~~
   // 5. Benchmark A: source | void | absorber, reflective at x = 0
   // ~~~~~~~~~~
   {
      const PetscReal delta = 2.5;
      const std::vector<SlabRegion> regions = {{0.0, delta, 0.5, 1.0}, {delta, 3.0 * delta, 0.0, 0.0}, \
         {3.0 * delta, 4.0 * delta, 0.8, 0.0}};
      for (const PetscReal zeta : {0.5, 2.0}) {
         PetscCall(CheckSlabBenchmark("SAAF-LS void slab", regions, PETSC_TRUE, {40, 80, 160, 320}, 8, zeta, 1.8, \
            1e-4, &ok));
      }
   }

   // ~~~~~~~~~~
   // 6. Benchmark B: thin | thick, a source everywhere, vacuum both ends
   // ~~~~~~~~~~
   {
      const std::vector<SlabRegion> regions = {{0.0, 1.0, 0.1, 1.0}, {1.0, 2.0, 10.0, 1.0}};
      for (const PetscReal zeta : {0.5, 2.0}) {
         PetscCall(CheckSlabBenchmark("thin/thick slab", regions, PETSC_FALSE, {16, 32, 64, 128}, 8, zeta, 1.5, \
            5e-3, &ok));
         PetscCall(ReportInterfaceDip(regions, 16, 8, zeta));
      }
   }

   // ~~~~~~~~~~
   // 7. Error paths
   // ~~~~~~~~~~
   PetscCall(CheckErrorPaths(&ok));

   if (!ok) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "CG-SUPG verification FAILED\n"));
   PetscCall(PetscFinalize());
   return ok ? 0 : 1;
}
