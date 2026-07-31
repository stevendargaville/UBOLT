#include "ubolt/terms.hpp"
#include <KokkosBlas.hpp>
#include "petsc_kokkos.hpp"
#include <math.h>

// The kernels below capture plain values and shallow view copies, never `this`
// - a member access inside a KOKKOS_LAMBDA would dereference a host pointer on
// the device

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// StreamingTerm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode StreamingTerm::create(const PhaseSpace &ps, const StructuredFD1D &disc, const SNQuadrature &quad)
{
   PetscFunctionBeginUser;

   n_angles_ = ps.n_angles;
   local_rows_ = ps.local_rows();
   dx_ = disc.dx();
   mu_d_ = quad.mu_d();
   pattern_ = disc.coo_pattern();
   boundary_ = disc.boundary_info();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Add the upwinded mu dpsi/dx stencil into the shared COO values
// This happens entirely on the device
PetscErrorCode StreamingTerm::assemble_add(PetscScalarKokkosView &coo_v_d) const
{
   const PetscInt n_angles = n_angles_;
   const PetscScalar dx = dx_;
   const PetscScalarKokkosView mu_d = mu_d_;
   const PetscIntKokkosView row_slot_offset_d = pattern_.row_slot_offset_d;
   const PetscIntKokkosView diag_slot_d = pattern_.diag_slot_d;
   const PetscIntKokkosView is_dirichlet_row_d = boundary_.is_dirichlet_row_d;

   PetscFunctionBeginUser;

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         // Dirichlet rows keep only the identity the assembly puts on them
         if (is_dirichlet_row_d(r)) return;

         const PetscInt a = r % n_angles;
         const PetscInt diag = diag_slot_d(r);

         coo_v_d(diag) += fabs(mu_d(a)) / dx;

         // Everything else in the row is an upwind neighbour. The neighbour
         // always sits on the opposite side of the diagonal, whichever way the
         // angle points: mu dpsi/dx is |mu|/dx (psi_i - psi_upwind)
         for (PetscInt s = row_slot_offset_d(r); s < row_slot_offset_d(r + 1); s++) {
            if (s == diag) continue;
            coo_v_d(s) += -fabs(mu_d(a)) / dx;
         }
      });

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// RemovalTerm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode RemovalTerm::create(const PhaseSpace &ps, const StructuredFD1D &disc, const PetscScalarKokkosView &sigma_t_d)
{
   PetscFunctionBeginUser;

   n_angles_ = ps.n_angles;
   local_rows_ = ps.local_rows();
   sigma_t_d_ = sigma_t_d;
   pattern_ = disc.coo_pattern();
   boundary_ = disc.boundary_info();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Add sigma_t onto the diagonal
// This happens entirely on the device
PetscErrorCode RemovalTerm::assemble_add(PetscScalarKokkosView &coo_v_d) const
{
   const PetscInt n_angles = n_angles_;
   const PetscScalarKokkosView sigma_t_d = sigma_t_d_;
   const PetscIntKokkosView diag_slot_d = pattern_.diag_slot_d;
   const PetscIntKokkosView is_dirichlet_row_d = boundary_.is_dirichlet_row_d;

   PetscFunctionBeginUser;

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         // Add nothing to the bcs
         if (is_dirichlet_row_d(r)) return;

         // The xsections are per cell, the rows are per cell and angle
         coo_v_d(diag_slot_d(r)) += sigma_t_d(r / n_angles);
      });

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ScatteringTerm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ScatteringTerm::create(const PhaseSpace &ps, const SNQuadrature &quad, const PetscScalarKokkosView &sigma_s_d)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   n_angles_ = ps.n_angles;
   sum_weights_ = quad.sum_weights();
   sigma_s_d_ = sigma_s_d;
   w_d_ = quad.w_d();
   // Persistent scratch for the scalar flux, sized by the local cells - the
   // scatter is applied to the local part of the vectors only
   scalar_flux_d_ = PetscScalar2DKokkosView("scalar_flux_d", ps.local_cells, 1);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Integrate the angular flux, scale by sigma_s and take it off y (the
// scattering term is on the lhs)
// This happens entirely on the device
PetscErrorCode ScatteringTerm::apply_add(Vec x, Vec y) const
{
   const PetscInt n_angles = n_angles_;
   const PetscScalar sum_weights = sum_weights_;
   const PetscScalarKokkosView sigma_s_d = sigma_s_d_;
   const PetscScalar2DKokkosView scalar_flux_d = scalar_flux_d_;

   PetscFunctionBeginUser;

   // All the kernels below operate on the local part of the vectors only -
   // this is how many local cells we have
   PetscInt local_rows;
   PetscCall(VecGetLocalSize(x, &local_rows));
   const PetscInt local_cells = local_rows / n_angles;

   // Get the device views
   // Making sure to use a const view for the input
   PetscScalarConstKokkosView x_d;
   PetscCall(VecGetKokkosView(x, &x_d));
   PetscScalarKokkosView y_d;
   PetscCall(VecGetKokkosView(y, &y_d));

   // Let's reshape (without copying) the 1D input view into a 2D view
   // The 2D view uses LayoutRight so we have the correct ordering of contiguous
   // in angle
   PetscScalar2DConstKokkosView x_d_2d(x_d.data(), local_cells, n_angles);

   // Now let's integrate the angular flux to get the scalar flux
   // This is just a dgemm (on the device)
   const double alpha = double(1.0);
   const double beta = double(0.0);
   KokkosBlas::gemm("N","N",alpha,x_d_2d,w_d_,beta,scalar_flux_d);

   // Now let's multiply by the scattering xsection
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(int i) {

         // We have to divide by the sum of weights to get the amount going
         // into each angle
         scalar_flux_d(i, 0) *= sigma_s_d(i) / sum_weights;
      });

   // Now let's put the scatter back into y
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_cells, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // cell
         PetscInt i = t.league_rank();

         // For all the angles
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt j) {

               // Minus the scattering term as it's on the lhs
               y_d(i * n_angles + j) -= scalar_flux_d(i, 0);
         });
   });

   // Restore the views
   PetscCall(VecRestoreKokkosView(x, &x_d));
   PetscCall(VecRestoreKokkosView(y, &y_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
