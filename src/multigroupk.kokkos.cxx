#include "ubolt/multigroup.hpp"
#include "petsc_kokkos.hpp"
#include <KokkosBlas.hpp>
#include <utility>

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

   PetscCall(ps.check_decomposed());
   PetscCall(create(ps.n_groups, ps.local_cells));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupXSections::create(PetscInt n_groups, PetscInt n_entries)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(n_groups > 0 && n_entries >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "GroupXSections needs a positive group count and a non-negative size, was given %" PetscInt_FMT \
      " groups over %" PetscInt_FMT " entries", n_groups, n_entries);

   n_groups_ = n_groups;
   local_cells_ = n_entries;

   // Zero initialised, so a group pair that is never set simply does not couple
   sigma_t_d_ = PetscScalar2DRightKokkosView("sigma_t_d", n_groups_, local_cells_);
   sigma_s_d_ = PetscScalar3DRightKokkosView("sigma_s_d", n_groups_, n_groups_, local_cells_);

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

   Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), sigma_t(g), value);

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

   Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), sigma_s(g_from, g_to), value);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Free functions rather than local lambdas: an extended device lambda can't be
// defined inside another lambda

// Cell is the parallel index and the fastest extent of both destinations, so
// consecutive threads write consecutive cells; the group loops run inside
static void FillFromMaterialsKernel(PetscScalar2DRightKokkosView sigma_t_d, \
   PetscScalar3DRightKokkosView sigma_s_d, PetscScalarKokkosView sigma_t_tab_d, \
   PetscScalarKokkosView sigma_s_tab_d, PetscIntKokkosView mat_id_d, PetscInt n_groups, \
   PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

         const PetscInt m = mat_id_d(c);
         for (PetscInt g = 0; g < n_groups; g++) {
            sigma_t_d(g, c) = sigma_t_tab_d(m * n_groups + g);
         }
         for (PetscInt g_from = 0; g_from < n_groups; g_from++) {
            for (PetscInt g_to = 0; g_to < n_groups; g_to++) {
               sigma_s_d(g_from, g_to, c) = sigma_s_tab_d((m * n_groups + g_from) * n_groups + g_to);
            }
         }
      });
}

PetscErrorCode GroupXSections::set_from_materials(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d)
{
   PetscFunctionBeginUser;

   PetscCheck(mats.n_groups() == n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the material spec has %" PetscInt_FMT " groups but the xsections have %" PetscInt_FMT, \
      mats.n_groups(), n_groups_);
   PetscCheck((PetscInt)mat_id_d.extent(0) == local_cells_, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "material ids cover %" PetscInt_FMT " cells but there are %" PetscInt_FMT " local cells", \
      (PetscInt)mat_id_d.extent(0), local_cells_);

   PetscCall(UboltCheckMaterialIds(mats, mat_id_d));

   // The material tables are tiny, so upload them whole
   PetscScalarKokkosView sigma_t_tab_d("sigma_t_tab_d", (PetscInt)mats.sigma_t_host().size());
   PetscScalarKokkosView sigma_s_tab_d("sigma_s_tab_d", (PetscInt)mats.sigma_s_host().size());
   PetscScalarKokkosViewHostUnmanaged sigma_t_tab_h( \
      const_cast<PetscScalar *>(mats.sigma_t_host().data()), mats.sigma_t_host().size());
   PetscScalarKokkosViewHostUnmanaged sigma_s_tab_h( \
      const_cast<PetscScalar *>(mats.sigma_s_host().data()), mats.sigma_s_host().size());
   auto exec = PetscGetKokkosExecutionSpace();
   Kokkos::deep_copy(exec, sigma_t_tab_d, sigma_t_tab_h);
   Kokkos::deep_copy(exec, sigma_s_tab_d, sigma_s_tab_h);
   // The copies are asynchronous and the host tables are the caller's
   exec.fence();

   FillFromMaterialsKernel(sigma_t_d_, sigma_s_d_, sigma_t_tab_d, sigma_s_tab_d, mat_id_d, \
      n_groups_, local_cells_);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GroupTransfer
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::create(const PhaseSpace &ps, const AngularQuadrature &quad, const GroupXSections &xs, \
   const BoundaryInfo &boundary, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck((PetscInt)mat_id_d.extent(0) == ps.local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "material ids cover %" PetscInt_FMT " cells but there are %" PetscInt_FMT " local cells", \
      (PetscInt)mat_id_d.extent(0), ps.local_cells);
   PetscCheck((PetscInt)boundary.is_bc_row_d.extent(0) == ps.local_rows(), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "the boundary info covers %" PetscInt_FMT " rows but there are %" \
      PetscInt_FMT " local rows", (PetscInt)boundary.is_bc_row_d.extent(0), ps.local_rows());
   PetscCall(source_.create(mats, mat_id_d));

   n_angles_ = ps.n_angles;
   n_basis_ = ps.n_basis;
   local_nodes_ = ps.local_nodes();
   sum_weights_ = quad.sum_weights();
   xs_ = &xs;
   w_d_ = quad.w_d();
   is_bc_row_d_ = boundary.is_bc_row_d;
   // Per NODE, (cell, basis) - the scatter's rule: with an orthonormal modal
   // basis the transfer is the same per node as per cell at one dof per cell
   scalar_flux_d_ = PetscScalar2DKokkosView("scalar_flux_d", local_nodes_, 1);

   phi_.resize(ps.n_groups);
   phi_set_.assign(ps.n_groups, PETSC_FALSE);
   for (PetscInt g = 0; g < ps.n_groups; g++) {
      phi_[g] = PetscScalar2DKokkosView("phi_d", local_nodes_, 1);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::add_external(PetscInt g, Vec b) const
{
   PetscFunctionBeginUser;

   PetscCall(source_.add_isotropic(g, n_angles_, n_basis_, sum_weights_, is_bc_row_d_, b));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// cache = fresh, summing |fresh - cache|^2 and |fresh|^2 over the nodes on the
// way. A copy, so the cached values are bit for bit the integral's
static void OverwriteMeasured(PetscScalar2DKokkosView fresh_d, PetscScalar2DKokkosView cache_d, \
   PetscInt local_nodes, PetscReal *change_sq, PetscReal *norm_sq)
{
   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_nodes), \
      KOKKOS_LAMBDA(PetscInt i, PetscReal &change, PetscReal &norm) {

         const PetscScalar diff = fresh_d(i, 0) - cache_d(i, 0);
         change += PetscRealPart(diff * PetscConj(diff));
         norm += PetscRealPart(fresh_d(i, 0) * PetscConj(fresh_d(i, 0)));
         cache_d(i, 0) = fresh_d(i, 0);
      }, *change_sq, *norm_sq);
}

// current(node, d) = sum_a psi(node, a) w_a Omega_a[d]: the angular integral's
// gemm against the (n_angles, dim) first-moment weights instead of w
static PetscErrorCode IntegrateCurrent(Vec psi, PetscInt n_angles, const PetscScalar2DKokkosView &w_omega_d, \
   const PetscScalar2DKokkosView &current_d)
{
   PetscFunctionBeginUser;

   PetscInt local_rows;
   PetscCall(VecGetLocalSize(psi, &local_rows));
   PetscScalarConstKokkosView psi_d;
   PetscCall(VecGetKokkosView(psi, &psi_d));
   // LayoutRight: angle contiguous, as in UboltAngularIntegral
   PetscScalar2DConstKokkosView psi_d_2d(psi_d.data(), local_rows / n_angles, n_angles);
   const PetscScalar alpha = 1.0, beta = 0.0;
   KokkosBlas::gemm(PetscGetKokkosExecutionSpace(), "N", "N", alpha, psi_d_2d, w_omega_d, beta, current_d);
   PetscCall(VecRestoreKokkosView(psi, &psi_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::set_scalar_flux(PetscInt g, Vec psi_g, PetscReal *change_sq, PetscReal *norm_sq)
{
   PetscFunctionBeginUser;

   PetscCheck(g >= 0 && g < (PetscInt)phi_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, (PetscInt)phi_.size());
   PetscCheck((change_sq == nullptr) == (norm_sq == nullptr), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, \
      "set_scalar_flux measures the change and the norm together: pass both or neither");

   // The current first, so the scalar flux below is untouched by it: its own
   // gemm, against w_a Omega_a, rather than one wider gemm against [w | w
   // Omega] that would put phi's column through a different kernel shape and
   // so not keep it bitwise the single-moment integral every term does
   if (n_dim_ > 0) PetscCall(IntegrateCurrent(psi_g, n_angles_, w_omega_d_, current_[g]));

   if (!change_sq) {
      PetscCall(UboltAngularIntegral(psi_g, n_angles_, w_d_, phi_[g]));
   } else {
      // Integrate into the scratch, then overwrite the cache from it. The
      // nodes are this rank's own (the rows of psi_g are owned), so the local
      // sums add up across ranks with nothing counted twice. The reduction
      // into host scalars waits for the kernel
      PetscReal sums[2] = {0.0, 0.0};
      PetscCall(UboltAngularIntegral(psi_g, n_angles_, w_d_, scalar_flux_d_));
      OverwriteMeasured(scalar_flux_d_, phi_[g], local_nodes_, &sums[0], &sums[1]);
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sums, 2, MPIU_REAL, MPIU_SUM, \
         PetscObjectComm((PetscObject)psi_g)));
      *change_sq += sums[0];
      *norm_sq += sums[1];
   }
   phi_set_[g] = PETSC_TRUE;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Take the source group's cached scalar flux, scale it by the transfer xsection
// and spread it isotropically over the target group's angles
// This happens entirely on the device
PetscErrorCode GroupTransfer::add_transfer(PetscInt g_from, PetscInt g_to, Vec b) const
{
   const PetscInt n_angles = n_angles_;
   const PetscInt n_basis = n_basis_;
   const PetscInt local_nodes = local_nodes_;
   const PetscScalar sum_weights = sum_weights_;
   const PetscScalar2DKokkosView scalar_flux_d = scalar_flux_d_;
   const PetscIntKokkosView is_bc_row_d = is_bc_row_d_;

   PetscFunctionBeginUser;

   PetscCheck(g_from >= 0 && g_from < (PetscInt)phi_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g_from, (PetscInt)phi_.size());
   PetscCheck(g_to >= 0 && g_to < (PetscInt)phi_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g_to, (PetscInt)phi_.size());
   PetscCheck(phi_set_[g_from], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
      "no scalar flux cached for group %" PetscInt_FMT " - set_scalar_flux() must be called " \
      "when that group is solved, before anything scatters out of it", g_from);

   // The subview is only taken once the group indices have been checked
   const PetscScalarKokkosView sigma_s_d = xs_->sigma_s(g_from, g_to);
   const PetscScalar2DKokkosView phi_d = phi_[g_from];

   // Scale by the transfer xsection, and by the sum of weights to get the
   // amount going into each angle. Into the scratch, not the cache: the source
   // group scatters into every group below it and each wants its own xsection
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_nodes), KOKKOS_LAMBDA(PetscInt i) {

         // The xsection is per cell
         scalar_flux_d(i, 0) = phi_d(i, 0) * (sigma_s_d(i / n_basis) / sum_weights);
      });

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));

   // Plus the source, unlike the within-group scatter which is on the lhs
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_nodes, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // node
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The P1 tables, on the host once: w_omega(a, d) = w_a Omega_a[d] and
// p1(a, d) = (T^{-1} Omega_a)[d], T = sum_a w_a Omega_a Omega_a^T the dim x dim
// second moment (symmetric positive definite for any set that spans its
// dimension; (W / 3) I for the level-symmetric and Gauss-Legendre sets),
// inverted by Gauss-Jordan. Then the per-group J views
PetscErrorCode GroupTransfer::enable_current(const AngularQuadrature &quad)
{
   PetscFunctionBeginUser;

   PetscCheck(n_angles_ > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "enable_current needs create() first");
   PetscCheck(quad.n_angles() == n_angles_, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the quadrature has %" PetscInt_FMT " angles but the transfer was created with %" PetscInt_FMT, \
      quad.n_angles(), n_angles_);
   if (n_dim_ > 0) PetscFunctionReturn(PETSC_SUCCESS);

   const PetscInt dim = quad.dimension();
   const PetscScalar *omega = quad.omega_host();
   PetscCheck(dim >= 1 && dim <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "the quadrature's dimension is %" PetscInt_FMT, dim);

   auto exec = PetscGetKokkosExecutionSpace();
   auto w_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), quad.w_d());
   Kokkos::deep_copy(exec, w_h, quad.w_d());
   exec.fence();

   // T, and the first moment sum_a w_a Omega_a, which must vanish for the
   // expansion's scalar flux to be phi
   PetscScalar t[9] = {0.0}, tinv[9] = {0.0}, first[3] = {0.0};
   for (PetscInt a = 0; a < n_angles_; a++) {
      for (PetscInt d = 0; d < dim; d++) {
         first[d] += w_h(a, 0) * omega[a * 3 + d];
         for (PetscInt e = 0; e < dim; e++) t[d * 3 + e] += w_h(a, 0) * omega[a * 3 + d] * omega[a * 3 + e];
      }
   }
   for (PetscInt d = 0; d < dim; d++) {
      PetscCheck(PetscAbsScalar(first[d]) <= 1e-12 * PetscAbsScalar(sum_weights_), PETSC_COMM_SELF, \
         PETSC_ERR_ARG_INCOMP, "the P1 expansion needs a quadrature whose weighted directions sum to zero: " \
         "sum_a w_a Omega_a[%" PetscInt_FMT "] is %g", d, (double)PetscRealPart(first[d]));
   }
   for (PetscInt d = 0; d < dim; d++) tinv[d * 3 + d] = 1.0;
   for (PetscInt k = 0; k < dim; k++) {
      PetscInt piv = k;
      for (PetscInt i = k + 1; i < dim; i++) {
         if (PetscAbsScalar(t[i * 3 + k]) > PetscAbsScalar(t[piv * 3 + k])) piv = i;
      }
      PetscCheck(PetscAbsScalar(t[piv * 3 + k]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
         "the quadrature's second moment sum_a w_a Omega_a Omega_a^T is singular");
      for (PetscInt j = 0; j < 3; j++) {
         std::swap(t[k * 3 + j], t[piv * 3 + j]);
         std::swap(tinv[k * 3 + j], tinv[piv * 3 + j]);
      }
      const PetscScalar inv = 1.0 / t[k * 3 + k];
      for (PetscInt j = 0; j < dim; j++) {
         t[k * 3 + j] *= inv;
         tinv[k * 3 + j] *= inv;
      }
      for (PetscInt i = 0; i < dim; i++) {
         if (i == k) continue;
         const PetscScalar f = t[i * 3 + k];
         for (PetscInt j = 0; j < dim; j++) {
            t[i * 3 + j] -= f * t[k * 3 + j];
            tinv[i * 3 + j] -= f * tinv[k * 3 + j];
         }
      }
   }

   w_omega_d_ = PetscScalar2DKokkosView("w_omega_d", n_angles_, dim);
   p1_d_ = PetscScalar2DKokkosView("p1_d", n_angles_, dim);
   auto w_omega_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), w_omega_d_);
   auto p1_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), p1_d_);
   for (PetscInt a = 0; a < n_angles_; a++) {
      for (PetscInt d = 0; d < dim; d++) {
         w_omega_h(a, d) = w_h(a, 0) * omega[a * 3 + d];
         PetscScalar sum = 0.0;
         for (PetscInt e = 0; e < dim; e++) sum += tinv[d * 3 + e] * omega[a * 3 + e];
         p1_h(a, d) = sum;
      }
   }
   Kokkos::deep_copy(exec, w_omega_d_, w_omega_h);
   Kokkos::deep_copy(exec, p1_d_, p1_h);
   // The host mirrors go out of scope at return
   exec.fence();

   // Zero initialised: a group cached before this has no current
   current_.resize(phi_.size());
   for (size_t g = 0; g < current_.size(); g++) current_[g] = PetscScalar2DKokkosView("current_d", local_nodes_, dim);
   n_dim_ = dim;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransfer::expand_p1(PetscInt g, Vec psi) const
{
   const PetscInt n_angles = n_angles_;
   const PetscInt n_dim = n_dim_;
   const PetscScalar sum_weights = sum_weights_;
   const PetscScalar2DKokkosView p1_d = p1_d_;

   PetscFunctionBeginUser;

   PetscCheck(n_dim_ > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
      "expand_p1 needs the current: call enable_current() before the sweep");
   PetscCheck(g >= 0 && g < (PetscInt)phi_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, (PetscInt)phi_.size());
   PetscCheck(phi_set_[g], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
      "no scalar flux cached for group %" PetscInt_FMT " to expand", g);

   PetscInt local_rows = 0;
   PetscCall(VecGetLocalSize(psi, &local_rows));
   PetscCheck(local_rows == local_nodes_ * n_angles_, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "psi has %" PetscInt_FMT " local rows but the transfer covers %" PetscInt_FMT, local_rows, \
      local_nodes_ * n_angles_);

   const PetscScalar2DKokkosView phi_d = phi_[g];
   const PetscScalar2DKokkosView current_d = current_[g];
   PetscScalarKokkosView psi_d;
   PetscCall(VecGetKokkosViewWrite(psi, &psi_d));
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         const PetscInt i = r / n_angles, a = r % n_angles;
         PetscScalar val = phi_d(i, 0) / sum_weights;
         for (PetscInt d = 0; d < n_dim; d++) val += p1_d(a, d) * current_d(i, d);
         psi_d(r) = val;
      });
   PetscCall(VecRestoreKokkosViewWrite(psi, &psi_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
