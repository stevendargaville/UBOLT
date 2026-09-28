#include "ubolt/terms_cg.hpp"
#include "petsc_kokkos.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// SUPGTermCG
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode SUPGTermCG::create(const PhaseSpace &ps, const UnstructuredCG &disc, const PetscScalarKokkosView &sigma_t_e)
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());
   PetscCheck(ps.n_basis == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "the CG backend has one dof per vertex");

   disc_ = &disc;
   n_angles_ = ps.n_angles;
   local_rows_ = ps.local_rows();
   sigma_t_e_ = sigma_t_e;
   pattern_ = disc.coo_pattern();

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SUPGTermCG::set_group(const GroupXSections &xs, PetscInt g)
{
   PetscFunctionBeginUser;
   set_sigma_t(xs.sigma_t(g));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda (see docs/dev/kokkos.md). One
// thread per row: gather over the vertex's star, element vertex by element
// vertex, into the slot each one's column has in this row, then the weak
// boundary faces. diag_only writes only what lands on the diagonal slot, into
// out_d(r) rather than the COO values - the add_diagonal half, the same
// arithmetic in the same order
static void SUPGFillKernel(PetscScalarKokkosView out_d, bool diag_only, PetscScalarKokkosView sigma_t_e, \
   PetscReal zeta, PetscScalarKokkosView omega_d, PetscScalarKokkosView inv_mass_d, \
   PetscIntKokkosView star_offset_d, PetscIntKokkosView star_elem_d, PetscIntKokkosView star_li_d, \
   PetscIntKokkosView star_slot_d, PetscScalarKokkosView mass_d, PetscScalarKokkosView grad_d, \
   PetscScalarKokkosView stiff_d, PetscScalarKokkosView centre_grad_d, PetscIntKokkosView bface_offset_d, \
   PetscScalarKokkosView bface_nA_d, PetscScalarKokkosView bface_mass_d, PetscIntKokkosView bface_slot_d, \
   PetscIntKokkosView row_slot_offset_d, PetscIntKokkosView diag_slot_d, PetscInt nv, PetscInt n_angles, \
   PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         const PetscInt k = r / n_angles;
         const PetscInt a = r % n_angles;
         const PetscInt base = row_slot_offset_d(r);
         const PetscInt diag = diag_slot_d(r) - base;
         const PetscScalar inv_m = inv_mass_d(k);
         const PetscScalar om[3] = {omega_d(3 * a), omega_d(3 * a + 1), omega_d(3 * a + 2)};

         for (PetscInt s = star_offset_d(k); s < star_offset_d(k + 1); s++) {
            const PetscInt e = star_elem_d(s);
            const PetscInt li = star_li_d(s);
            const PetscScalar sigma = sigma_t_e(e);
            const PetscScalar tau = UboltSUPGTau(centre_grad_d, omega_d, e, a, nv, sigma, zeta);
            for (PetscInt j = 0; j < nv; j++) {
               const PetscInt slot = star_slot_d(s * nv + j);
               if (diag_only && slot != diag) continue;
               const PetscInt ij = (e * nv + li) * nv + j;
               const PetscInt ji = (e * nv + j) * nv + li;
               // Galerkin streaming (phi_i, Omega . grad phi_j)
               const PetscScalar og_ij = om[0] * grad_d(3 * ij) + om[1] * grad_d(3 * ij + 1) + om[2] * grad_d(3 * ij + 2);
               // SUPG streaming (Omega . grad phi_i, Omega . grad phi_j)
               PetscScalar oko = 0.0;
               for (PetscInt d = 0; d < 3; d++) {
                  for (PetscInt dd = 0; dd < 3; dd++) oko += om[d] * om[dd] * stiff_d(9 * ij + 3 * d + dd);
               }
               // Removal with the SUPG weight (phi_i + tau Omega . grad phi_i, phi_j)
               const PetscScalar og_ji = om[0] * grad_d(3 * ji) + om[1] * grad_d(3 * ji + 1) + om[2] * grad_d(3 * ji + 2);
               const PetscScalar coef = (og_ij + tau * oko + sigma * (mass_d(ij) + tau * og_ji)) * inv_m;
               if (diag_only) out_d(r) += coef;
               else out_d(base + slot) += coef;
            }
         }

         // The weak inflow, face by face: |s| m^f_i / m_i onto the diagonal,
         // and its negative onto the mirrored angle on a reflective face
         for (PetscInt f = bface_offset_d(k); f < bface_offset_d(k + 1); f++) {
            const PetscScalar s = om[0] * bface_nA_d(3 * f) + om[1] * bface_nA_d(3 * f + 1) + om[2] * bface_nA_d(3 * f + 2);
            if (PetscRealPart(s) >= 0.0) continue;
            const PetscScalar c = -s * bface_mass_d(f);
            if (diag_only) out_d(r) += c;
            else {
               out_d(base + diag) += c;
               if (bface_slot_d(f) >= 0) out_d(base + bface_slot_d(f)) -= c;
            }
         }
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This happens entirely on the device
PetscErrorCode SUPGTermCG::assemble_add(const PetscScalarKokkosView &coo_v_d) const
{
   const UnstructuredCG &d = *disc_;

   PetscFunctionBeginUser;

   PetscCheck((PetscInt)sigma_t_e_.extent(0) == d.n_local_elements(), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "sigma_t covers %" PetscInt_FMT " elements but there are %" PetscInt_FMT " local elements", \
      (PetscInt)sigma_t_e_.extent(0), d.n_local_elements());

   SUPGFillKernel(coo_v_d, false, sigma_t_e_, d.zeta(), d.omega_d(), d.inv_mass_d(), d.star_offset_d(), \
      d.star_elem_d(), d.star_li_d(), d.star_slot_d(), d.mass_d(), d.grad_d(), d.stiff_d(), d.centre_grad_d(), \
      d.bface_offset_d(), d.bface_nA_d(), d.bface_mass_d(), d.bface_slot_d(), pattern_.row_slot_offset_d, \
      pattern_.diag_slot_d, d.n_element_vertices(), n_angles_, local_rows_);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This happens entirely on the device
PetscErrorCode SUPGTermCG::add_diagonal(Vec diag) const
{
   const UnstructuredCG &d = *disc_;

   PetscFunctionBeginUser;

   PetscScalarKokkosView d_d;
   PetscCall(VecGetKokkosView(diag, &d_d));
   SUPGFillKernel(d_d, true, sigma_t_e_, d.zeta(), d.omega_d(), d.inv_mass_d(), d.star_offset_d(), \
      d.star_elem_d(), d.star_li_d(), d.star_slot_d(), d.mass_d(), d.grad_d(), d.stiff_d(), d.centre_grad_d(), \
      d.bface_offset_d(), d.bface_nA_d(), d.bface_mass_d(), d.bface_slot_d(), pattern_.row_slot_offset_d, \
      pattern_.diag_slot_d, d.n_element_vertices(), n_angles_, local_rows_);
   PetscCall(VecRestoreKokkosView(diag, &d_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The nodal scalar flux of the owned vertices into a nodal global Vec (its
// local part is indexed by owned vertex, see UnstructuredCG). Overwrites it all
static PetscErrorCode NodalFromScalarFlux(const PetscScalar2DKokkosView &phi_d, Vec nodal)
{
   PetscFunctionBeginUser;

   PetscScalarKokkosView n_d;
   PetscCall(VecGetKokkosViewWrite(nodal, &n_d));
   const PetscInt n = (PetscInt)n_d.extent(0);
   PetscCheck(n == (PetscInt)phi_d.extent(0), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "a nodal Vec of %" PetscInt_FMT " vertices for %" PetscInt_FMT " owned vertices", n, (PetscInt)phi_d.extent(0));
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(PetscInt i) { n_d(i) = phi_d(i, 0); });
   PetscCall(VecRestoreKokkosViewWrite(nodal, &n_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ScatteringTermCG
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ScatteringTermCG::create(const PhaseSpace &ps, const UnstructuredCG &disc, \
   const AngularQuadrature &quad, const PetscScalarKokkosView &sigma_t_e, const PetscScalarKokkosView &sigma_s_e)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());
   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);

   disc_ = &disc;
   n_angles_ = ps.n_angles;
   sum_weights_ = quad.sum_weights();
   w_d_ = quad.w_d();
   sigma_t_e_ = sigma_t_e;
   sigma_s_e_ = sigma_s_e;
   scalar_flux_d_ = PetscScalar2DKokkosView("scalar_flux_d", ps.local_nodes(), 1);
   PetscCall(disc.create_nodal_vecs(&phi_global_, &phi_local_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ScatteringTermCG::set_group(const GroupXSections &xs, PetscInt g)
{
   PetscFunctionBeginUser;
   set_group(xs.sigma_t(g), xs.sigma_s(g, g));
   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ScatteringTermCG::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(VecDestroy(&phi_global_));
   PetscCall(VecDestroy(&phi_local_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Integrate, bring in the overlap vertices' flux, and take the weighted load
// off y (the scatter is on the lhs)
// This happens entirely on the device, bar the ghost exchange
PetscErrorCode ScatteringTermCG::apply_add(Vec x, Vec y) const
{
   PetscFunctionBeginUser;

   PetscCall(UboltAngularIntegral(x, n_angles_, w_d_, scalar_flux_d_));
   PetscCall(NodalFromScalarFlux(scalar_flux_d_, phi_global_));
   PetscCall(disc_->nodal_global_to_local(phi_global_, phi_local_));
   PetscCall(disc_->add_weighted_load(sigma_t_e_, sigma_s_e_, phi_local_, -1.0 / sum_weights_, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GroupTransferCG
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode GroupTransferCG::create(const PhaseSpace &ps, const UnstructuredCG &disc, \
   const AngularQuadrature &quad, const GroupXSections &xs, const MaterialSpec &mats, \
   const PetscIntKokkosView &mat_id_d)
{
   Vec unused_local = NULL;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());
   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck((PetscInt)mat_id_d.extent(0) == disc.n_local_elements(), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "material ids cover %" PetscInt_FMT " elements but there are %" PetscInt_FMT " local elements", \
      (PetscInt)mat_id_d.extent(0), disc.n_local_elements());

   disc_ = &disc;
   xs_ = &xs;
   n_angles_ = ps.n_angles;
   sum_weights_ = quad.sum_weights();
   w_d_ = quad.w_d();
   PetscCall(source_.create(mats, mat_id_d));
   q_e_ = PetscScalarKokkosView("q_e", disc.n_local_elements());
   scalar_flux_d_ = PetscScalar2DKokkosView("scalar_flux_d", ps.local_nodes(), 1);
   PetscCall(disc.create_nodal_vecs(&phi_global_, &unused_local));
   PetscCall(VecDestroy(&unused_local));
   phi_.assign(ps.n_groups, NULL);
   phi_set_.assign(ps.n_groups, PETSC_FALSE);

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GroupTransferCG::destroy()
{
   PetscFunctionBeginUser;

   for (auto &v : phi_) PetscCall(VecDestroy(&v));
   PetscCall(VecDestroy(&phi_global_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Per group: two kernel launches, into persistent scratch
PetscErrorCode GroupTransferCG::add_external(PetscInt g, Vec b) const
{
   PetscFunctionBeginUser;

   PetscCall(source_.fill_entries(g, q_e_));
   PetscCall(disc_->add_weighted_load(xs_->sigma_t(g), q_e_, NULL, 1.0 / sum_weights_, b));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GroupTransferCG::set_scalar_flux(PetscInt g, Vec psi_g)
{
   Vec unused_global = NULL;

   PetscFunctionBeginUser;

   PetscCheck(g >= 0 && g < (PetscInt)phi_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, (PetscInt)phi_.size());

   if (!phi_[g]) {
      PetscCall(disc_->create_nodal_vecs(&unused_global, &phi_[g]));
      PetscCall(VecDestroy(&unused_global));
   }
   PetscCall(UboltAngularIntegral(psi_g, n_angles_, w_d_, scalar_flux_d_));
   PetscCall(NodalFromScalarFlux(scalar_flux_d_, phi_global_));
   PetscCall(disc_->nodal_global_to_local(phi_global_, phi_[g]));
   phi_set_[g] = PETSC_TRUE;

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GroupTransferCG::add_transfer(PetscInt g_from, PetscInt g_to, Vec b) const
{
   PetscFunctionBeginUser;

   PetscCheck(g_from >= 0 && g_from < (PetscInt)phi_.size() && g_to >= 0 && g_to < (PetscInt)phi_.size(), \
      PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "groups %" PetscInt_FMT " -> %" PetscInt_FMT \
      " out of range, n_groups is %" PetscInt_FMT, g_from, g_to, (PetscInt)phi_.size());
   PetscCheck(phi_set_[g_from], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
      "no scalar flux cached for group %" PetscInt_FMT " - set_scalar_flux() must be called " \
      "when that group is solved, before anything scatters out of it", g_from);

   PetscCall(disc_->add_weighted_load(xs_->sigma_t(g_to), xs_->sigma_s(g_from, g_to), phi_[g_from], \
      1.0 / sum_weights_, b));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The one-shot external source
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UboltFillSourceCG(const UnstructuredCG &disc, const AngularQuadrature &quad, \
   const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, const PetscScalarKokkosView &sigma_t_e, \
   PetscInt g, Vec b)
{
   MaterialSourceTable table;

   PetscFunctionBeginUser;

   PetscCall(table.create(mats, mat_id_d));
   PetscScalarKokkosView q_e("q_e", disc.n_local_elements());
   PetscCall(table.fill_entries(g, q_e));
   PetscCall(disc.add_weighted_load(sigma_t_e, q_e, NULL, 1.0 / quad.sum_weights(), b));

   PetscFunctionReturn(PETSC_SUCCESS);
}
