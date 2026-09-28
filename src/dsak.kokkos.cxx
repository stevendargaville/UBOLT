#include "dsa_operatork.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"
#include "ubolt/unstructured_dg.hpp"
#include "ubolt/unstructured_cg.hpp"

// DSAPrecon: what every backend shares - the restriction and prolongation,
// the inner KSP and the void policy. The operators are in dsa_*k.kokkos.cxx.
// Kernels are file-static free functions taking view copies, never `this`
// (docs/dev/kokkos.md)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// R: the 0th angular moment per (cell, basis) node, times the backend's weight
// (V, or m_i on CG) when it has one. Not UboltAngularIntegral: this has to
// leave the BC rows out, which carry the boundary condition, not a balance -
// and P zeroes the same rows, so the two halves agree. A MASKED void's nodes
// restrict to zero, which keeps the inner solution zero on their identity rows
static void RestrictKernel(PetscScalarConstKokkosView x_d, PetscScalarKokkosView rhs_d, \
   PetscScalarConstKokkosView weight_d, PetscBool weighted, PetscScalar2DKokkosView w_d, \
   PetscIntKokkosView is_bc_row_d, PetscIntKokkosView void_cell_d, PetscInt n_angles, PetscInt n_basis, \
   PetscInt n_nodes)
{
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), n_nodes, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         const PetscInt node = t.league_rank();

         PetscScalar moment = 0.0;
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt a, PetscScalar &acc) {

               const PetscInt r = node * n_angles + a;
               if (!is_bc_row_d(r)) acc += w_d(a, 0) * x_d(r);
            }, moment);

         Kokkos::single(Kokkos::PerTeam(t), [&]() {
            rhs_d(node) = void_cell_d(node / n_basis) ? (PetscScalar)0.0 : \
               (weighted ? moment * weight_d(node) : moment);
         });
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// P: the correction broadcast isotropically, / sum_weights. Every row is
// written; BC rows and a MASKED void's rows get zero (the BC contract, and no
// correction lives in a masked void)
static void ProlongKernel(PetscScalarConstKokkosView node_d, PetscScalarKokkosView y_d, \
   PetscIntKokkosView is_bc_row_d, PetscIntKokkosView void_cell_d, PetscInt n_angles, PetscInt n_basis, \
   PetscScalar sum_weights, PetscInt local_rows)
{
   const PetscInt rows_per_cell = n_basis * n_angles;
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         y_d(r) = (is_bc_row_d(r) || void_cell_d(r / rows_per_cell)) ? (PetscScalar)0.0 : \
            node_d(r / n_angles) / sum_weights;
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAOperator::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(destroy_own());
   PetscCall(MatDestroy(&mat));
   PetscCall(VecDestroy(&rhs));
   PetscCall(VecDestroy(&sol));
   PetscCall(VecDestroy(&weight));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// create
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD1D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;
   PetscCall(init_common(comm, ps, disc, quad));
   PetscCall(DSAOperatorCreate(comm, ps, disc, quad, bcs, &op_));
   PetscCall(create_ksp());
   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD2D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;
   PetscCall(init_common(comm, ps, disc, quad));
   PetscCall(DSAOperatorCreate(comm, ps, disc, quad, bcs, &op_));
   PetscCall(create_ksp());
   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD3D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;
   PetscCall(init_common(comm, ps, disc, quad));
   PetscCall(DSAOperatorCreate(comm, ps, disc, quad, bcs, &op_));
   PetscCall(create_ksp());
   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;
   PetscCall(init_common(comm, ps, disc, quad));
   PetscCall(DSAOperatorCreate(comm, ps, disc, quad, bcs, &op_));
   PetscCall(create_ksp());
   PetscFunctionReturn(PETSC_SUCCESS);
}

// One diffusion unknown per owned vertex, so n_basis is 1 whatever the phase
// space says
PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredCG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;
   PetscCall(init_common(comm, ps, disc, quad));
   n_basis_ = 1;
   PetscCall(DSAOperatorCreate(comm, ps, disc, quad, bcs, &op_));
   PetscCall(create_ksp());
   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;

   if (auto *d1 = dynamic_cast<const StructuredFD1D *>(&disc)) PetscCall(create(comm, ps, *d1, quad, bcs));
   else if (auto *d2 = dynamic_cast<const StructuredFD2D *>(&disc)) PetscCall(create(comm, ps, *d2, quad, bcs));
   else if (auto *d3 = dynamic_cast<const StructuredFD3D *>(&disc)) PetscCall(create(comm, ps, *d3, quad, bcs));
   else if (auto *dg = dynamic_cast<const UnstructuredDG *>(&disc)) PetscCall(create(comm, ps, *dg, quad, bcs));
   else if (auto *cg = dynamic_cast<const UnstructuredCG *>(&disc)) PetscCall(create(comm, ps, *cg, quad, bcs));
   else SETERRQ(comm, PETSC_ERR_SUP, "DSAPrecon has no diffusion operator for this discretisation backend");

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Either vacuum treatment is fine: under ghost-flux the inflow rows are not BC
// rows, so R sums them and P corrects them; under Dirichlet-cell both mask them
PetscErrorCode DSAPrecon::init_common(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
   const AngularQuadrature &quad)
{
   PetscFunctionBeginUser;

   // Device views below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);

   comm_ = comm;
   n_angles_ = ps.n_angles;
   n_basis_ = ps.n_basis;
   local_cells_ = ps.local_cells;
   sum_weights_ = quad.sum_weights();
   w_d_ = quad.w_d();
   is_bc_row_d_ = disc.boundary_info().is_bc_row_d;
   n_cells_global_ = local_cells_;
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &n_cells_global_, 1, MPIU_INT, MPI_SUM, comm_));

   // A cell at or below the threshold is a void (0: only a true one); bridge
   // the voids rather than mask them; a fixed free-flight D for them in place
   // of the chord's L / 3
   PetscCall(PetscOptionsGetReal(NULL, "dsa_", "-void_sigma_t", &void_sigma_t_, NULL));
   PetscCall(PetscOptionsGetBool(NULL, "dsa_", "-void_bridge", &void_bridge_, NULL));
   PetscCall(PetscOptionsGetReal(NULL, "dsa_", "-void_d", &void_d_, NULL));

   // The mask, zero on both sides until a group masks something
   void_cell_d_ = PetscIntKokkosView("dsa_void_cell", local_cells_);
   void_cell_h_.assign(local_cells_, 0);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The inner diffusion solve: the paper's single BoomerAMG V-cycle becomes one
// PCGAMG application (no hypre in the CI images); KSPSetFromOptions last so
// -dsa_ksp_type, -dsa_pc_type hypre and friends select anything at runtime
PetscErrorCode DSAPrecon::create_ksp()
{
   PC pc = NULL;

   PetscFunctionBeginUser;

   PetscCall(KSPCreate(comm_, &ksp_));
   PetscCall(KSPSetOptionsPrefix(ksp_, "dsa_"));
   PetscCall(KSPSetType(ksp_, KSPPREONLY));
   PetscCall(KSPGetPC(ksp_, &pc));
   PetscCall(PCSetType(pc, PCGAMG));
   PetscCall(KSPSetOperators(ksp_, op_->mat, op_->mat));
   PetscCall(KSPSetFromOptions(ksp_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPrecon::set_group(const PetscScalarKokkosView &sigma_t_d, \
   const PetscScalarKokkosView &sigma_s_within_d)
{
   PetscFunctionBeginUser;

   sigma_t_d_ = sigma_t_d;
   sigma_s_d_ = sigma_s_within_d;
   PetscCall(assemble());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The void policy, then the backend's refill. Every collective here (and in
// void_volume_surface) sits under a condition built from globally reduced or
// globally identical values only
PetscErrorCode DSAPrecon::assemble()
{
   const PetscInt n_units = op_->n_units();
   const PetscInt *owned = op_->owned_units();
   const PetscBool can_mask = op_->can_mask();
   PetscInt n_void = 0;
   // The largest Sigma_a anywhere, and in the non-void units only: the
   // singular guard counts the second unless the voids are bridged
   PetscReal max_sigma_a[2] = {0.0, 0.0};

   PetscFunctionBeginUser;

   PetscCheck(sigma_t_d_.extent(0) == (size_t)n_units && sigma_s_d_.extent(0) == (size_t)n_units, \
      comm_, PETSC_ERR_ARG_INCOMP, "the group xsections cover %" PetscInt_FMT " and %" PetscInt_FMT \
      " %s but there are %" PetscInt_FMT " local %s", (PetscInt)sigma_t_d_.extent(0), \
      (PetscInt)sigma_s_d_.extent(0), op_->unit_name(), n_units, op_->unit_name());

   // The assembly is a host loop
   auto sigma_t_h = DSAHostCopy(sigma_t_d_);
   auto sigma_s_h = DSAHostCopy(sigma_s_d_);

   // The census, per group (a material can be a void in some groups only)
   is_void_.assign(n_units, 0);
   for (PetscInt u = 0; u < n_units; u++) {
      const PetscReal sigma_t = PetscRealPart(sigma_t_h(u));
      const PetscReal sigma_a = sigma_t - PetscRealPart(sigma_s_h(u));
      is_void_[u] = (sigma_t <= void_sigma_t_) ? 1 : 0;
      if (is_void_[u]) {
         if (!owned || owned[u]) n_void++;
      }
      else max_sigma_a[1] = PetscMax(max_sigma_a[1], sigma_a);
      max_sigma_a[0] = PetscMax(max_sigma_a[0], sigma_a);
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &n_void, 1, MPIU_INT, MPI_SUM, comm_));
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, max_sigma_a, 2, MPIU_REAL, MPIU_MAX, comm_));
   n_void_cells_ = n_void;

   // A backend that cannot mask has nothing to fall back on, so its guard
   // comes first and does not count on a void
   if (!can_mask) {
      PetscCheck(op_->any_vacuum || max_sigma_a[0] > 0.0, comm_, PETSC_ERR_ARG_WRONGSTATE, \
         "the DSA diffusion operator is singular: every face is reflective (pure Neumann) and " \
         "Sigma_a = Sigma_t - Sigma_s is zero everywhere in this group. Leave one face vacuum, or give " \
         "the material absorption");
   }

   // Bridge when asked and there is a void. Where the mask is the fallback,
   // also only when something is not void and the bridged operator is
   // nonsingular (the voids are IN it with no absorption, so a vacuum face or
   // some absorption has to hold it). Bridged iff a face ends a flight
   bridged_ = PETSC_FALSE;
   void_chord_ = 0.0;
   if (void_bridge_ && n_void > 0 && \
      (!can_mask || (n_void < n_cells_global_ && (op_->any_vacuum || max_sigma_a[0] > 0.0)))) {
      PetscReal vs[2] = {0.0, 0.0};
      PetscCall(op_->void_volume_surface(is_void_.data(), vs));
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, vs, 2, MPIU_REAL, MPIU_SUM, comm_));
      void_chord_ = vs[1] > 0.0 ? 4.0 * vs[0] / vs[1] : 0.0;
      bridged_ = (PetscBool)(void_chord_ > 0.0);
   }

   // D per unit: 1/(3 sigma_t); in a bridged void 1/3 of the flight
   // 1/(sigma_t + 1/L) (Wigner's rational form, L/3 in a true void) unless
   // -dsa_void_d fixes it; 0 in an unbridged void
   d_unit_.assign(n_units, 0.0);
   for (PetscInt u = 0; u < n_units; u++) {
      if (!is_void_[u]) d_unit_[u] = 1.0 / (3.0 * sigma_t_h(u));
      else if (bridged_) {
         d_unit_[u] = void_d_ > 0.0 ? (PetscScalar)void_d_ : \
            (PetscScalar)(1.0 / (3.0 * (PetscMax(PetscRealPart(sigma_t_h(u)), 0.0) + 1.0 / void_chord_)));
      }
   }

   if (can_mask) {

      // The mask the restriction/prolongation read, uploaded only on a change
      PetscBool changed = PETSC_FALSE;
      for (PetscInt c = 0; c < local_cells_; c++) {
         const PetscInt masked = (is_void_[c] && !bridged_) ? 1 : 0;
         if (masked != void_cell_h_[c]) changed = PETSC_TRUE;
         void_cell_h_[c] = masked;
      }
      if (changed) {
         Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), void_cell_d_, \
            PetscIntKokkosViewHostUnmanaged(void_cell_h_.data(), local_cells_));
         PetscGetKokkosExecutionSpace().fence();
      }

      // With every face reflective the operator is pure Neumann and only
      // absorption keeps it nonsingular - the transport's own constraint (see
      // docs/dev/testing.md). A face into a MASKED void is a Marshak face, so
      // any masked void is as good as a vacuum face here; bridging was only
      // chosen above where a vacuum face or some absorption holds it
      const PetscReal sigma_a = bridged_ ? max_sigma_a[0] : max_sigma_a[1];
      PetscCheck(op_->any_vacuum || n_void > 0 || sigma_a > 0.0, comm_, PETSC_ERR_ARG_WRONGSTATE, \
         "the DSA diffusion operator is singular: every face is reflective (pure Neumann), there " \
         "is no void, and Sigma_a = Sigma_t - Sigma_s is zero everywhere in this group. Leave one " \
         "face vacuum, or give the material absorption");
   }

   DSAGroup g;
   g.sigma_t_h = sigma_t_h.data();
   g.sigma_s_h = sigma_s_h.data();
   g.sigma_t_d = sigma_t_d_;
   g.is_void = is_void_.data();
   g.d = d_unit_.data();
   g.masked = void_cell_h_.data();
   g.n_void = n_void;
   g.bridged = bridged_;
   PetscCall(op_->assemble(g));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// y = P D_diff^-1 R x
PetscErrorCode DSAPrecon::apply(Vec x, Vec y)
{
   PetscInt local_rows = 0;

   PetscFunctionBeginUser;

   PetscCall(VecGetLocalSize(x, &local_rows));
   PetscCheck(local_rows == local_cells_ * n_basis_ * n_angles_, comm_, PETSC_ERR_ARG_INCOMP, \
      "x has %" PetscInt_FMT " local rows but this DSAPrecon covers %" PetscInt_FMT, \
      local_rows, local_cells_ * n_basis_ * n_angles_);

   // Restrict, weighted to match a volume-weighted (or weak-form) matrix
   {
      PetscScalarConstKokkosView x_d, weight_d;
      PetscScalarKokkosView rhs_d;
      const PetscBool weighted = (PetscBool)(op_->weight != NULL);
      PetscCall(VecGetKokkosView(x, &x_d));
      if (weighted) PetscCall(VecGetKokkosView(op_->weight, &weight_d));
      PetscCall(VecGetKokkosViewWrite(op_->rhs, &rhs_d));
      RestrictKernel(x_d, rhs_d, weight_d, weighted, w_d_, is_bc_row_d_, void_cell_d_, n_angles_, n_basis_, \
         local_cells_ * n_basis_);
      PetscCall(VecRestoreKokkosViewWrite(op_->rhs, &rhs_d));
      if (weighted) PetscCall(VecRestoreKokkosView(op_->weight, &weight_d));
      PetscCall(VecRestoreKokkosView(x, &x_d));
   }

   // Invert the diffusion operator, inexactly
   PetscCall(KSPSolve(ksp_, op_->rhs, op_->sol));

   // Prolong: the scalar correction back onto every ordinate
   {
      PetscScalarConstKokkosView sol_d;
      PetscScalarKokkosView y_d;
      PetscCall(VecGetKokkosView(op_->sol, &sol_d));
      PetscCall(VecGetKokkosViewWrite(y, &y_d));
      ProlongKernel(sol_d, y_d, is_bc_row_d_, void_cell_d_, n_angles_, n_basis_, sum_weights_, local_rows);
      PetscCall(VecRestoreKokkosViewWrite(y, &y_d));
      PetscCall(VecRestoreKokkosView(op_->sol, &sol_d));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Safe on a never-created object. The xsection views are not ours to drop
PetscErrorCode DSAPrecon::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(KSPDestroy(&ksp_));
   if (op_) {
      PetscCall(op_->destroy());
      delete op_;
      op_ = nullptr;
   }
   *this = DSAPrecon();

   PetscFunctionReturn(PETSC_SUCCESS);
}
