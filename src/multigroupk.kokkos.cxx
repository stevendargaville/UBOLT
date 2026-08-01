#include "ubolt/multigroup.hpp"
#include "petsc_kokkos.hpp"

// The kernels below capture plain values and shallow view copies, never `this`
// - a member access inside a KOKKOS_LAMBDA would dereference a host pointer on
// the device

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GroupXSections
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupXSections::create(const PhaseSpace &ps)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

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

PetscErrorCode GroupTransfer::create(const PhaseSpace &ps, const SNQuadrature &quad, \
   const GroupXSections &xs, const BoundaryInfo &boundary)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   n_angles_ = ps.n_angles;
   sum_weights_ = quad.sum_weights();
   xs_ = &xs;
   w_d_ = quad.w_d();
   is_dirichlet_row_d_ = boundary.is_dirichlet_row_d;
   scalar_flux_d_ = PetscScalar2DKokkosView("scalar_flux_d", ps.local_cells, 1);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::destroy()
{
   PetscFunctionBeginUser;

   // The device views have to go before PetscFinalize takes Kokkos down
   w_d_ = PetscScalar2DKokkosView();
   is_dirichlet_row_d_ = PetscIntKokkosView();
   scalar_flux_d_ = PetscScalar2DKokkosView();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Integrate the source group's angular flux, scale by the transfer xsection and
// spread it isotropically over the target group's angles
// This happens entirely on the device
PetscErrorCode GroupTransfer::add_source(PetscInt g_from, PetscInt g_to, Vec psi_from, Vec b) const
{
   const PetscInt n_angles = n_angles_;
   const PetscScalar sum_weights = sum_weights_;
   const PetscScalar2DKokkosView scalar_flux_d = scalar_flux_d_;
   const PetscIntKokkosView is_dirichlet_row_d = is_dirichlet_row_d_;
   const PetscScalarKokkosView sigma_s_d = xs_->sigma_s(g_from, g_to);

   PetscFunctionBeginUser;

   PetscInt local_rows;
   PetscCall(VecGetLocalSize(psi_from, &local_rows));
   const PetscInt local_cells = local_rows / n_angles;

   // The scalar flux of the group we are scattering out of
   PetscCall(UboltAngularIntegral(psi_from, n_angles, w_d_, scalar_flux_d));

   // Scale by the transfer xsection, and by the sum of weights to get the
   // amount going into each angle
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt i) {

         scalar_flux_d(i, 0) *= sigma_s_d(i) / sum_weights;
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
               // The rhs on a Dirichlet row is the incoming flux
               if (is_dirichlet_row_d(r)) return;
               b_d(r) += scalar_flux_d(i, 0);
         });
   });

   PetscCall(VecRestoreKokkosView(b, &b_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
