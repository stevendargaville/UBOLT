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

   PetscCall(ps.check_decomposed());

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
// StreamingTerm2D
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode StreamingTerm2D::create(const PhaseSpace &ps, const StructuredFD2D &disc, const SNQuadrature2D &quad)
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());

   n_angles_ = ps.n_angles;
   local_rows_ = ps.local_rows();
   dx_ = disc.dx();
   dy_ = disc.dy();
   mu_d_ = quad.mu_d();
   eta_d_ = quad.eta_d();
   pattern_ = disc.coo_pattern();
   boundary_ = disc.boundary_info();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Add the upwinded mu dpsi/dx + eta dpsi/dy stencil into the shared COO values
// This happens entirely on the device
PetscErrorCode StreamingTerm2D::assemble_add(PetscScalarKokkosView &coo_v_d) const
{
   const PetscInt n_angles = n_angles_;
   const PetscScalar dx = dx_;
   const PetscScalar dy = dy_;
   const PetscScalarKokkosView mu_d = mu_d_;
   const PetscScalarKokkosView eta_d = eta_d_;
   const PetscIntKokkosView row_slot_offset_d = pattern_.row_slot_offset_d;
   const PetscIntKokkosView diag_slot_d = pattern_.diag_slot_d;
   const PetscIntKokkosView is_dirichlet_row_d = boundary_.is_dirichlet_row_d;

   PetscFunctionBeginUser;

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         // Dirichlet rows keep only the identity the assembly puts on them
         if (is_dirichlet_row_d(r)) return;

         const PetscInt a = r % n_angles;
         // Slot order is the discretisation's: upwind-x, upwind-y, diagonal.
         // Unlike 1D there is more than one off-diagonal, so the two are
         // addressed positionally rather than as "everything but the diagonal"
         const PetscInt first = row_slot_offset_d(r);

         const PetscScalar cx = fabs(mu_d(a)) / dx;
         const PetscScalar cy = fabs(eta_d(a)) / dy;

         // Each axis contributes |cosine| / h (psi_here - psi_upwind), whichever
         // way it points - the direction is already baked into which neighbour
         // the discretisation put in the slot
         coo_v_d(diag_slot_d(r)) += cx + cy;
         coo_v_d(first)     += -cx;
         coo_v_d(first + 1) += -cy;
      });

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// RemovalTerm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode RemovalTerm::create(const PhaseSpace &ps, const Discretisation &disc, const PetscScalarKokkosView &sigma_t_d)
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());

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

PetscErrorCode ScatteringTerm::create(const PhaseSpace &ps, const AngularQuadrature &quad, const PetscScalarKokkosView &sigma_s_d)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());

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

   // Now let's integrate the angular flux to get the scalar flux
   PetscCall(UboltAngularIntegral(x, n_angles, w_d_, scalar_flux_d));

   PetscScalarKokkosView y_d;
   PetscCall(VecGetKokkosView(y, &y_d));

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

   // Restore the view
   PetscCall(VecRestoreKokkosView(y, &y_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
