#include "ubolt/terms.hpp"
#include "ubolt/multigroup.hpp"
#include "petsc_kokkos.hpp"

// The kernels below capture plain values and shallow view copies, never `this`
// - a member access inside a KOKKOS_LAMBDA would dereference a host pointer on
// the device

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// StructuredStreamingTerm and the per-dimension creates
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode StructuredStreamingTerm::create_common(const PhaseSpace &ps, const Discretisation &disc, \
   PetscInt n_dims, const PetscScalar h[], const PetscScalarKokkosView cos_d[])
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());

   n_dims_ = n_dims;
   n_angles_ = ps.n_angles;
   local_rows_ = ps.local_rows();
   for (PetscInt d = 0; d < n_dims; d++) {
      h_[d] = h[d];
      cos_d_[d] = cos_d[d];
   }
   pattern_ = disc.coo_pattern();
   boundary_ = disc.boundary_info();

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StreamingTerm1D::create(const PhaseSpace &ps, const StructuredFD1D &disc, const SNQuadrature &quad)
{
   const PetscScalar h[] = {disc.dx()};
   const PetscScalarKokkosView cos_d[] = {quad.mu_d()};

   PetscFunctionBeginUser;
   PetscCall(create_common(ps, disc, 1, h, cos_d));
   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StreamingTerm2D::create(const PhaseSpace &ps, const StructuredFD2D &disc, const SNQuadrature2D &quad)
{
   const PetscScalar h[] = {disc.dx(), disc.dy()};
   const PetscScalarKokkosView cos_d[] = {quad.mu_d(), quad.eta_d()};

   PetscFunctionBeginUser;
   PetscCall(create_common(ps, disc, 2, h, cos_d));
   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StreamingTerm3D::create(const PhaseSpace &ps, const StructuredFD3D &disc, const SNQuadrature3D &quad)
{
   const PetscScalar h[] = {disc.dx(), disc.dy(), disc.dz()};
   const PetscScalarKokkosView cos_d[] = {quad.mu_d(), quad.eta_d(), quad.xi_d()};

   PetscFunctionBeginUser;
   PetscCall(create_common(ps, disc, 3, h, cos_d));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Per row, c_d = |cos_d(a)| / h_d: the diagonal gets c_0 + c_1 + c_2 (summed
// left to right, the order add_diagonal must match bitwise), upwind slot d
// gets -c_d. diag_only adds just the diagonal into out_d(r) - the add_diagonal
// half. A free function templated on the axis count, so each dimension
// compiles to its own unrolled kernel
template <int DIM>
static void StructuredStreamingKernel(PetscScalarKokkosView out_d, bool diag_only, PetscScalarKokkosView cos0_d, \
   PetscScalarKokkosView cos1_d, PetscScalarKokkosView cos2_d, PetscScalar h0, PetscScalar h1, PetscScalar h2, \
   PetscIntKokkosView row_slot_offset_d, PetscIntKokkosView diag_slot_d, PetscIntKokkosView is_bc_row_d, \
   PetscInt n_angles, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         // BC rows carry only what the assembly puts on them
         if (is_bc_row_d(r)) return;

         const PetscInt a = r % n_angles;
         PetscScalar c[DIM];
         c[0] = PetscAbsScalar(cos0_d(a)) / h0;
         if constexpr (DIM > 1) c[1] = PetscAbsScalar(cos1_d(a)) / h1;
         if constexpr (DIM > 2) c[2] = PetscAbsScalar(cos2_d(a)) / h2;
         PetscScalar diag = c[0];
         for (int d = 1; d < DIM; d++) diag += c[d];

         if (diag_only) {
            out_d(r) += diag;
            return;
         }
         out_d(diag_slot_d(r)) += diag;
         const PetscInt first = row_slot_offset_d(r);
         for (int d = 0; d < DIM; d++) out_d(first + d) += -c[d];
      });
}

static void StructuredStreamingFill(PetscInt n_dims, PetscScalarKokkosView out_d, bool diag_only, \
   const PetscScalarKokkosView cos_d[], const PetscScalar h[], const CooPattern &pattern, \
   const BoundaryInfo &boundary, PetscInt n_angles, PetscInt local_rows)
{
   if (n_dims == 1) StructuredStreamingKernel<1>(out_d, diag_only, cos_d[0], cos_d[1], cos_d[2], h[0], h[1], h[2], \
      pattern.row_slot_offset_d, pattern.diag_slot_d, boundary.is_bc_row_d, n_angles, local_rows);
   else if (n_dims == 2) StructuredStreamingKernel<2>(out_d, diag_only, cos_d[0], cos_d[1], cos_d[2], h[0], h[1], \
      h[2], pattern.row_slot_offset_d, pattern.diag_slot_d, boundary.is_bc_row_d, n_angles, local_rows);
   else StructuredStreamingKernel<3>(out_d, diag_only, cos_d[0], cos_d[1], cos_d[2], h[0], h[1], h[2], \
      pattern.row_slot_offset_d, pattern.diag_slot_d, boundary.is_bc_row_d, n_angles, local_rows);
}

// This happens entirely on the device
PetscErrorCode StructuredStreamingTerm::assemble_add(const PetscScalarKokkosView &coo_v_d) const
{
   PetscFunctionBeginUser;

   StructuredStreamingFill(n_dims_, coo_v_d, false, cos_d_, h_, pattern_, boundary_, n_angles_, local_rows_);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// This happens entirely on the device
PetscErrorCode StructuredStreamingTerm::add_diagonal(Vec d) const
{
   PetscFunctionBeginUser;

   PetscScalarKokkosView d_d;
   PetscCall(VecGetKokkosView(d, &d_d));
   StructuredStreamingFill(n_dims_, d_d, true, cos_d_, h_, pattern_, boundary_, n_angles_, local_rows_);
   PetscCall(VecRestoreKokkosView(d, &d_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// StreamingTermDG0
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode StreamingTermDG0::create(const PhaseSpace &ps, const UnstructuredDG &disc)
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());

   PetscCheck(disc.order() == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "StreamingTermDG0 needs the " \
      "backend at order 0, it was built at order %" PetscInt_FMT " - use StreamingTermDG1", disc.order());
   PetscCheck((PetscInt)disc.inv_volume_d().extent(0) == ps.local_cells, PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "the discretisation covers %" PetscInt_FMT " local cells but the phase " \
      "space %" PetscInt_FMT " - create the backend first", (PetscInt)disc.inv_volume_d().extent(0), \
      ps.local_cells);
   PetscCheck((PetscInt)disc.omega_d().extent(0) == 3 * ps.n_angles, PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "the discretisation was classified with %" PetscInt_FMT " angles, the " \
      "phase space has %" PetscInt_FMT, (PetscInt)disc.omega_d().extent(0) / 3, ps.n_angles);

   n_angles_ = ps.n_angles;
   local_rows_ = ps.local_rows();
   omega_d_ = disc.omega_d();
   cell_face_offset_d_ = disc.cell_face_offset_d();
   face_nA_d_ = disc.face_nA_d();
   inv_volume_d_ = disc.inv_volume_d();
   pattern_ = disc.coo_pattern();
   boundary_ = disc.boundary_info();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Omega_a . nA_k, the upwind test and the face coefficient before 1 / V_c
KOKKOS_INLINE_FUNCTION PetscScalar DG0FaceFlux(const PetscScalarKokkosView &omega_d, \
   const PetscScalarKokkosView &face_nA_d, PetscInt a, PetscInt k)
{
   return omega_d(3 * a) * face_nA_d(3 * k) + omega_d(3 * a + 1) * face_nA_d(3 * k + 1) + \
      omega_d(3 * a + 2) * face_nA_d(3 * k + 2);
}

// A row's diagonal: the outflow fluxes summed, then scaled by 1 / V_c once.
// assemble_add and add_diagonal BOTH add exactly this value, so the composed
// diagonal is bitwise the assembled one. Accumulating s / V_c face by face in
// each kernel instead is not: under -ffp-contract=fast (the CI flags) the
// compiler fuses the per-face multiply-add differently in the two kernels'
// branch structures, and the results differ in the last bit
KOKKOS_INLINE_FUNCTION PetscScalar DG0OutflowDiagonal(const PetscScalarKokkosView &omega_d, \
   const PetscScalarKokkosView &face_nA_d, const PetscIntKokkosView &cell_face_offset_d, \
   const PetscScalarKokkosView &inv_volume_d, PetscInt c, PetscInt a)
{
   PetscScalar outflow = 0.0;
   for (PetscInt k = cell_face_offset_d(c); k < cell_face_offset_d(c + 1); k++) {
      const PetscScalar s = DG0FaceFlux(omega_d, face_nA_d, a, k);
      if (PetscRealPart(s) > 0.0) outflow += s;
   }
   return outflow * inv_volume_d(c);
}

// Add the upwind face fluxes into the shared COO values
// This happens entirely on the device
PetscErrorCode StreamingTermDG0::assemble_add(const PetscScalarKokkosView &coo_v_d) const
{
   const PetscInt n_angles = n_angles_;
   const PetscScalarKokkosView omega_d = omega_d_;
   const PetscIntKokkosView cell_face_offset_d = cell_face_offset_d_;
   const PetscScalarKokkosView face_nA_d = face_nA_d_;
   const PetscScalarKokkosView inv_volume_d = inv_volume_d_;
   const PetscIntKokkosView row_slot_offset_d = pattern_.row_slot_offset_d;
   const PetscIntKokkosView diag_slot_d = pattern_.diag_slot_d;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         // BC rows carry only what the assembly puts on them
         if (is_bc_row_d(r)) return;

         const PetscInt c = r / n_angles;
         const PetscInt a = r % n_angles;
         // Slot order is the discretisation's: one per face in cone order,
         // then the diagonal
         const PetscInt first = row_slot_offset_d(r);
         const PetscInt diag = diag_slot_d(r);
         const PetscInt k0 = cell_face_offset_d(c);

         // Outflow goes on the diagonal; inflow onto the face's slot, which
         // the backend pointed at the upwind neighbour - or nulled, on a
         // boundary face, so the entry is dropped. s == 0 adds nothing
         coo_v_d(diag) += DG0OutflowDiagonal(omega_d, face_nA_d, cell_face_offset_d, inv_volume_d, c, a);
         for (PetscInt k = k0; k < cell_face_offset_d(c + 1); k++) {
            const PetscScalar s = DG0FaceFlux(omega_d, face_nA_d, a, k);
            if (!(PetscRealPart(s) > 0.0)) coo_v_d(first + (k - k0)) += s * inv_volume_d(c);
         }
      });

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The outflow terms this term writes into the diagonal slot above - the same
// DG0OutflowDiagonal value, so a composed diagonal is bitwise the assembled one
// This happens entirely on the device
PetscErrorCode StreamingTermDG0::add_diagonal(Vec d) const
{
   const PetscInt n_angles = n_angles_;
   const PetscScalarKokkosView omega_d = omega_d_;
   const PetscIntKokkosView cell_face_offset_d = cell_face_offset_d_;
   const PetscScalarKokkosView face_nA_d = face_nA_d_;
   const PetscScalarKokkosView inv_volume_d = inv_volume_d_;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   PetscScalarKokkosView d_d;
   PetscCall(VecGetKokkosView(d, &d_d));

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         if (is_bc_row_d(r)) return;

         const PetscInt c = r / n_angles;
         const PetscInt a = r % n_angles;

         d_d(r) += DG0OutflowDiagonal(omega_d, face_nA_d, cell_face_offset_d, inv_volume_d, c, a);
      });

   PetscCall(VecRestoreKokkosView(d, &d_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// StreamingTermDG1
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode StreamingTermDG1::create(const PhaseSpace &ps, const UnstructuredDG &disc)
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());

   PetscCheck(disc.order() == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "StreamingTermDG1 needs the " \
      "backend at order 1, it was built at order %" PetscInt_FMT " - use StreamingTermDG0", disc.order());
   PetscCheck(ps.n_basis == disc.n_basis(), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "the phase space has %" \
      PetscInt_FMT " basis functions per cell, the discretisation %" PetscInt_FMT " - create the backend first", \
      ps.n_basis, disc.n_basis());
   PetscCheck((PetscInt)disc.basis_grad_d().extent(0) == ps.local_cells * ps.n_basis * 3, PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "the discretisation covers %" PetscInt_FMT " local cells but the phase space %" \
      PetscInt_FMT, (PetscInt)disc.basis_grad_d().extent(0) / (3 * ps.n_basis), ps.local_cells);
   PetscCheck((PetscInt)disc.omega_d().extent(0) == 3 * ps.n_angles, PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "the discretisation was classified with %" PetscInt_FMT " angles, the " \
      "phase space has %" PetscInt_FMT, (PetscInt)disc.omega_d().extent(0) / 3, ps.n_angles);

   n_angles_ = ps.n_angles;
   n_basis_ = ps.n_basis;
   local_rows_ = ps.local_rows();
   omega_d_ = disc.omega_d();
   cell_face_offset_d_ = disc.cell_face_offset_d();
   face_nA_d_ = disc.face_nA_d();
   face_own_d_ = disc.face_own_d();
   face_up_d_ = disc.face_up_d();
   basis_grad_d_ = disc.basis_grad_d();
   pattern_ = disc.coo_pattern();
   boundary_ = disc.boundary_info();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A row's diagonal: the outflow faces' own-cell (i, i) entries. assemble_add
// and add_diagonal BOTH add exactly this value - see DG0OutflowDiagonal for why
// it is one function
KOKKOS_INLINE_FUNCTION PetscScalar DG1OutflowDiagonal(const PetscScalarKokkosView &omega_d, \
   const PetscScalarKokkosView &face_nA_d, const PetscIntKokkosView &cell_face_offset_d, \
   const PetscScalarKokkosView &face_own_d, PetscInt nb, PetscInt c, PetscInt i, PetscInt a)
{
   PetscScalar diag = 0.0;
   for (PetscInt k = cell_face_offset_d(c); k < cell_face_offset_d(c + 1); k++) {
      const PetscScalar s = DG0FaceFlux(omega_d, face_nA_d, a, k);
      if (PetscRealPart(s) > 0.0) diag += s * face_own_d((k * nb + i) * nb + i);
   }
   return diag;
}

// Add the DG1 face and volume terms into the shared COO values
// This happens entirely on the device
PetscErrorCode StreamingTermDG1::assemble_add(const PetscScalarKokkosView &coo_v_d) const
{
   const PetscInt n_angles = n_angles_;
   const PetscInt nb = n_basis_;
   const PetscScalarKokkosView omega_d = omega_d_;
   const PetscIntKokkosView cell_face_offset_d = cell_face_offset_d_;
   const PetscScalarKokkosView face_nA_d = face_nA_d_;
   const PetscScalarKokkosView face_own_d = face_own_d_;
   const PetscScalarKokkosView face_up_d = face_up_d_;
   const PetscScalarKokkosView basis_grad_d = basis_grad_d_;
   const PetscIntKokkosView row_slot_offset_d = pattern_.row_slot_offset_d;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         // DG1 has no BC rows, but the contract costs nothing to keep
         if (is_bc_row_d(r)) return;

         const PetscInt node = r / n_angles;
         const PetscInt a = r % n_angles;
         const PetscInt c = node / nb;
         const PetscInt i = node % nb;
         // Slot order is the discretisation's: n_basis per face in cone
         // order, then this cell's own n_basis
         const PetscInt first = row_slot_offset_d(r);
         const PetscInt k0 = cell_face_offset_d(c);
         const PetscInt own = first + (cell_face_offset_d(c + 1) - k0) * nb;

         // Outflow faces couple this cell's own basis functions - everything
         // but the diagonal here, which gets its one shared value below; inflow
         // faces the upwind block the backend pointed the face's slots at
         for (PetscInt k = k0; k < cell_face_offset_d(c + 1); k++) {
            const PetscScalar s = DG0FaceFlux(omega_d, face_nA_d, a, k);
            if (PetscRealPart(s) > 0.0) {
               for (PetscInt j = 0; j < nb; j++) {
                  if (j != i) coo_v_d(own + j) += s * face_own_d((k * nb + i) * nb + j);
               }
            } else {
               for (PetscInt j = 0; j < nb; j++) {
                  coo_v_d(first + (k - k0) * nb + j) += s * face_up_d((k * nb + i) * nb + j);
               }
            }
         }
         // The volume term, -(Omega . grad phi_i) onto basis 0 (zero for i = 0,
         // whose gradient is zero - and whose basis-0 slot is the diagonal)
         if (i > 0) {
            coo_v_d(own) -= omega_d(3 * a) * basis_grad_d((c * nb + i) * 3) + \
               omega_d(3 * a + 1) * basis_grad_d((c * nb + i) * 3 + 1) + \
               omega_d(3 * a + 2) * basis_grad_d((c * nb + i) * 3 + 2);
         }
         coo_v_d(own + i) += DG1OutflowDiagonal(omega_d, face_nA_d, cell_face_offset_d, face_own_d, nb, c, i, a);
      });

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The diagonal this term writes above - the same DG1OutflowDiagonal value
// This happens entirely on the device
PetscErrorCode StreamingTermDG1::add_diagonal(Vec d) const
{
   const PetscInt n_angles = n_angles_;
   const PetscInt nb = n_basis_;
   const PetscScalarKokkosView omega_d = omega_d_;
   const PetscIntKokkosView cell_face_offset_d = cell_face_offset_d_;
   const PetscScalarKokkosView face_nA_d = face_nA_d_;
   const PetscScalarKokkosView face_own_d = face_own_d_;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   PetscScalarKokkosView d_d;
   PetscCall(VecGetKokkosView(d, &d_d));

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         if (is_bc_row_d(r)) return;

         const PetscInt node = r / n_angles;
         d_d(r) += DG1OutflowDiagonal(omega_d, face_nA_d, cell_face_offset_d, face_own_d, nb, node / nb, \
            node % nb, r % n_angles);
      });

   PetscCall(VecRestoreKokkosView(d, &d_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// RemovalTerm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode RemovalTerm::create(const PhaseSpace &ps, const Discretisation &disc, const PetscScalarKokkosView &sigma_t_d)
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());

   rows_per_cell_ = ps.rows_per_cell();
   local_rows_ = ps.local_rows();
   sigma_t_d_ = sigma_t_d;
   pattern_ = disc.coo_pattern();
   boundary_ = disc.boundary_info();

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RemovalTerm::set_group(const GroupXSections &xs, PetscInt g)
{
   PetscFunctionBeginUser;
   set_sigma_t(xs.sigma_t(g));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Add sigma_t onto the diagonal
// This happens entirely on the device
PetscErrorCode RemovalTerm::assemble_add(const PetscScalarKokkosView &coo_v_d) const
{
   const PetscInt rows_per_cell = rows_per_cell_;
   const PetscScalarKokkosView sigma_t_d = sigma_t_d_;
   const PetscIntKokkosView diag_slot_d = pattern_.diag_slot_d;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         // Add nothing to the bcs
         if (is_bc_row_d(r)) return;

         // The xsections are per cell, the rows are per cell, basis and angle
         coo_v_d(diag_slot_d(r)) += sigma_t_d(r / rows_per_cell);
      });

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The matrix-free half of the same term: y += sigma_t psi, row by row. A pure
// diagonal, so unlike the scatter there is nothing to integrate and nothing to
// keep scratch for - it reads sigma_t_d straight through, which is why a group
// sweep in this mode has no per-group work at all
// This happens entirely on the device
PetscErrorCode RemovalTerm::apply_add(Vec x, Vec y) const
{
   const PetscInt rows_per_cell = rows_per_cell_;
   const PetscScalarKokkosView sigma_t_d = sigma_t_d_;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   // Const on the way in so PETSc does not mark x dirty
   PetscScalarConstKokkosView x_d;
   PetscScalarKokkosView y_d;
   PetscCall(VecGetKokkosView(x, &x_d));
   PetscCall(VecGetKokkosView(y, &y_d));

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         // Add nothing to the bcs - the assembly wrote those rows and this
         // would take that straight back off
         if (is_bc_row_d(r)) return;

         // The xsections are per cell, the rows are per cell, basis and angle
         y_d(r) += sigma_t_d(r / rows_per_cell) * x_d(r);
      });

   PetscCall(VecRestoreKokkosView(x, &x_d));
   PetscCall(VecRestoreKokkosView(y, &y_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The same sigma_t this term writes into the diagonal slot above - which IS
// the whole of this term, so assembled or matrix-free the composed diagonal is
// bitwise the assembled one
// This happens entirely on the device
PetscErrorCode RemovalTerm::add_diagonal(Vec d) const
{
   const PetscInt rows_per_cell = rows_per_cell_;
   const PetscScalarKokkosView sigma_t_d = sigma_t_d_;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   PetscScalarKokkosView d_d;
   PetscCall(VecGetKokkosView(d, &d_d));

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_), KOKKOS_LAMBDA(PetscInt r) {

         if (is_bc_row_d(r)) return;

         d_d(r) += sigma_t_d(r / rows_per_cell);
      });

   PetscCall(VecRestoreKokkosView(d, &d_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ScatteringTerm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ScatteringTerm::create(const PhaseSpace &ps, const Discretisation &disc, \
   const AngularQuadrature &quad, const PetscScalarKokkosView &sigma_s_d)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);

   n_angles_ = ps.n_angles;
   n_basis_ = ps.n_basis;
   sum_weights_ = quad.sum_weights();
   sigma_s_d_ = sigma_s_d;
   w_d_ = quad.w_d();
   boundary_ = disc.boundary_info();
   // Persistent scratch for the scalar flux, sized by the local NODES - one
   // per (cell, basis) pair, the cells themselves when n_basis is 1 - the
   // scatter is applied to the local part of the vectors only
   scalar_flux_d_ = PetscScalar2DKokkosView("scalar_flux_d", ps.local_nodes(), 1);

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ScatteringTerm::set_group(const GroupXSections &xs, PetscInt g)
{
   PetscFunctionBeginUser;
   set_sigma_s(xs.sigma_s(g, g));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Integrate the angular flux, scale by sigma_s and take it off y (the
// scattering term is on the lhs)
// This happens entirely on the device
PetscErrorCode ScatteringTerm::apply_add(Vec x, Vec y) const
{
   const PetscInt n_angles = n_angles_;
   const PetscInt n_basis = n_basis_;
   const PetscScalar sum_weights = sum_weights_;
   const PetscScalarKokkosView sigma_s_d = sigma_s_d_;
   const PetscScalar2DKokkosView scalar_flux_d = scalar_flux_d_;
   const PetscIntKokkosView is_bc_row_d = boundary_.is_bc_row_d;

   PetscFunctionBeginUser;

   // All the kernels below operate on the local part of the vectors only -
   // this is how many local nodes (cell, basis pairs) we have. With a modal
   // basis orthonormal on each cell the scatter is the same per node as it is
   // per cell at DG0: the mass matrix is the identity, and the xsection is
   // constant on the cell
   PetscInt local_rows;
   PetscCall(VecGetLocalSize(x, &local_rows));
   const PetscInt local_nodes = local_rows / n_angles;

   // Now let's integrate the angular flux to get the scalar flux
   PetscCall(UboltAngularIntegral(x, n_angles, w_d_, scalar_flux_d));

   PetscScalarKokkosView y_d;
   PetscCall(VecGetKokkosView(y, &y_d));

   // Now let's multiply by the scattering xsection
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_nodes), KOKKOS_LAMBDA(PetscInt i) {

         // We have to divide by the sum of weights to get the amount going
         // into each angle. The xsection is per cell
         scalar_flux_d(i, 0) *= sigma_s_d(i / n_basis) / sum_weights;
      });

   // Now let's put the scatter back into y
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_nodes, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // node
         PetscInt i = t.league_rank();

         // For all the angles
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt j) {

               // Add nothing to the bcs - the assembled part wrote those rows
               // and this would take that back off. Note the scalar flux
               // itself is still integrated over EVERY angle, BC ones
               // included, and for a reflective row that is REQUIRED, not
               // just permitted: the prescribed incoming flux, or the
               // reflected outgoing flux, is a real part of the flux in that
               // cell, so the interior angles scatter off it
               if (is_bc_row_d(i * n_angles + j)) return;

               // Minus the scattering term as it's on the lhs
               y_d(i * n_angles + j) -= scalar_flux_d(i, 0);
         });
   });

   // Restore the view
   PetscCall(VecRestoreKokkosView(y, &y_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
