#include "ubolt/multigroup.hpp"
#include "petsc_kokkos.hpp"

// The kernels below capture plain values and shallow view copies, never `this`
// - a member access inside a KOKKOS_LAMBDA would dereference a host pointer on
// the device

// The per-group slices handed to the terms must be CONTIGUOUS, not strided.
// Cell is the fastest extent of both tables, so fixing the group index (or
// pair) leaves one dense run of local_cells. That is what the kernels want on
// either backend: a stride-1 walk vectorises on a CPU and coalesces on a GPU,
// where consecutive threads would otherwise be n_groups apart.
//
// Test the LAYOUT, not assignability: Kokkos allows assigning a rank-1 view of
// ANY layout to another (is_assignable_layout is true whenever the destination
// rank is 1), so a strided slice compiles fine and then aborts at runtime with
// "View assignment must have compatible layouts". A subview that is not
// contiguous comes back as LayoutStride, and that is the thing to catch
namespace {
   using SigmaTSlice = decltype(Kokkos::subview(PetscScalar2DRightKokkosView(), 0, Kokkos::ALL()));
   using SigmaSSlice = decltype(Kokkos::subview(PetscScalar3DRightKokkosView(), 0, 0, Kokkos::ALL()));
   static_assert(!std::is_same_v<typename SigmaTSlice::array_layout, Kokkos::LayoutStride>,
      "sigma_t slice is strided - cell must stay the fastest extent of sigma_t_d_");
   static_assert(!std::is_same_v<typename SigmaSSlice::array_layout, Kokkos::LayoutStride>,
      "sigma_s slice is strided - cell must stay the fastest extent of sigma_s_d_");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GroupXSections
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupXSections::create(const PhaseSpace &ps)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());

   n_groups_ = ps.n_groups;
   local_cells_ = ps.local_cells;

   // Zero initialised, so a group pair that is never set simply does not couple
   sigma_t_d_ = PetscScalar2DRightKokkosView("sigma_t_d", n_groups_, local_cells_);
   sigma_s_d_ = PetscScalar3DRightKokkosView("sigma_s_d", n_groups_, n_groups_, local_cells_);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupXSections::destroy()
{
   PetscFunctionBeginUser;

   // The device views have to go before PetscFinalize takes Kokkos down
   sigma_t_d_ = PetscScalar2DRightKokkosView();
   sigma_s_d_ = PetscScalar3DRightKokkosView();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscScalarKokkosView GroupXSections::sigma_t(PetscInt g) const
{
   // Contiguous in cell, so this is a plain 1D view of the group's row
   return Kokkos::subview(sigma_t_d_, g, Kokkos::ALL());
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscScalarKokkosView GroupXSections::sigma_s(PetscInt g_from, PetscInt g_to) const
{
   return Kokkos::subview(sigma_s_d_, g_from, g_to, Kokkos::ALL());
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupXSections::set_sigma_t(PetscInt g, PetscScalar value)
{
   PetscFunctionBeginUser;

   PetscCheck(g >= 0 && g < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, n_groups_);

   Kokkos::deep_copy(sigma_t(g), value);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupXSections::set_sigma_s(PetscInt g_from, PetscInt g_to, PetscScalar value)
{
   PetscFunctionBeginUser;

   PetscCheck(g_from >= 0 && g_from < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g_from, n_groups_);
   PetscCheck(g_to >= 0 && g_to < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g_to, n_groups_);

   Kokkos::deep_copy(sigma_s(g_from, g_to), value);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GroupTransfer
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::create(const PhaseSpace &ps, const AngularQuadrature &quad, \
   const GroupXSections &xs, const BoundaryInfo &boundary)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());

   n_angles_ = ps.n_angles;
   local_cells_ = ps.local_cells;
   sum_weights_ = quad.sum_weights();
   xs_ = &xs;
   w_d_ = quad.w_d();
   is_bc_row_d_ = boundary.is_bc_row_d;
   scalar_flux_d_ = PetscScalar2DKokkosView("scalar_flux_d", local_cells_, 1);

   phi_.resize(ps.n_groups);
   phi_set_.assign(ps.n_groups, PETSC_FALSE);
   for (PetscInt g = 0; g < ps.n_groups; g++) {
      phi_[g] = PetscScalar2DKokkosView("phi_d", local_cells_, 1);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::destroy()
{
   PetscFunctionBeginUser;

   // The device views have to go before PetscFinalize takes Kokkos down
   w_d_ = PetscScalar2DKokkosView();
   is_bc_row_d_ = PetscIntKokkosView();
   scalar_flux_d_ = PetscScalar2DKokkosView();
   phi_.clear();
   phi_set_.clear();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::set_scalar_flux(PetscInt g, Vec psi_g)
{
   PetscFunctionBeginUser;

   PetscCheck(g >= 0 && g < (PetscInt)phi_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, (PetscInt)phi_.size());

   PetscCall(UboltAngularIntegral(psi_g, n_angles_, w_d_, phi_[g]));
   phi_set_[g] = PETSC_TRUE;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Take the source group's cached scalar flux, scale it by the transfer xsection
// and spread it isotropically over the target group's angles
// This happens entirely on the device
PetscErrorCode GroupTransfer::add_source(PetscInt g_from, PetscInt g_to, Vec b) const
{
   const PetscInt n_angles = n_angles_;
   const PetscInt local_cells = local_cells_;
   const PetscScalar sum_weights = sum_weights_;
   const PetscScalar2DKokkosView scalar_flux_d = scalar_flux_d_;
   const PetscIntKokkosView is_bc_row_d = is_bc_row_d_;
   const PetscScalarKokkosView sigma_s_d = xs_->sigma_s(g_from, g_to);

   PetscFunctionBeginUser;

   PetscCheck(g_from >= 0 && g_from < (PetscInt)phi_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g_from, (PetscInt)phi_.size());
   PetscCheck(phi_set_[g_from], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
      "no scalar flux cached for group %" PetscInt_FMT " - set_scalar_flux() must be called " \
      "when that group is solved, before anything scatters out of it", g_from);

   const PetscScalar2DKokkosView phi_d = phi_[g_from];

   // Scale by the transfer xsection, and by the sum of weights to get the
   // amount going into each angle. Into the scratch, not the cache: the source
   // group scatters into every group below it and each wants its own xsection
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt i) {

         scalar_flux_d(i, 0) = phi_d(i, 0) * (sigma_s_d(i) / sum_weights);
      });

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));

   // Plus the source, unlike the within-group scatter which is on the lhs
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_cells, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // cell
         PetscInt i = t.league_rank();

         // For all the angles
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt j) {

               const PetscInt r = i * n_angles + j;
               // The rhs on a BC row belongs to the boundary condition
               if (is_bc_row_d(r)) return;
               b_d(r) += scalar_flux_d(i, 0);
         });
   });

   PetscCall(VecRestoreKokkosView(b, &b_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
