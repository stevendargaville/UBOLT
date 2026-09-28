#include "dsa_operatork.hpp"
#include "ubolt/unstructured_cg.hpp"
#include <petscdmplex.h>

// The CG-SUPG (UnstructuredCG) backend's DSA operator: one diffusion unknown
// per owned VERTEX, the transport's own numbering over n_angles, so R and P
// are the identity in space. The continuous weak form
//    sum_e int_e D_e grad u . grad v + sigma_a u v  +  sum_{vacuum f} c_f int_f u v
// on the backend's element tables (M consistent), the face mass lumped the way
// the transport's weak boundary is. Every element touching an owned vertex is
// local (FEM overlap), so each owned row is complete on its rank. The
// xsections, and the voids, are per local ELEMENT
class DSACG final : public DSAOperator {
public:
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredCG &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);

   PetscInt n_units() const override { return n_elem_; }
   const PetscInt *owned_units() const override { return elem_owned_.data(); }
   const char *unit_name() const override { return "elements"; }
   PetscBool can_mask() const override { return PETSC_FALSE; }
   PetscErrorCode void_volume_surface(const PetscInt *is_void, PetscReal vs[2]) override;
   PetscErrorCode assemble(const DSAGroup &g) override;

private:
   PetscInt local_vertices_ = 0;
   PetscInt nv_ = 0;
   PetscInt n_elem_ = 0;
   std::vector<PetscInt> elem_owned_;
   // Per local element, the COO entries of its owned vertices' rows run in
   // (element, owned li, j) order; then one diagonal entry per owned vertex
   // carrying its vacuum faces. entry_li_/_j_ say which element-matrix entry
   // each element entry is, entry_elem_ which element
   std::vector<PetscInt> entry_elem_, entry_li_, entry_j_;
   std::vector<PetscScalar> mass_, stiff_, coo_v_;
   // Per owned vertex, sum over its vacuum boundary faces of the face's
   // half-range current times m^f_i
   std::vector<PetscReal> vacuum_weight_;
   // What the SUPG D in an unbridged void needs: the backend's ordinates,
   // centroid gradients and zeta, and the quadrature
   PetscScalarKokkosView omega_d_, centre_grad_d_;
   PetscReal zeta_ = 0.5;
   PetscScalar2DKokkosView w_d_;
   PetscInt n_angles_ = 0;
   PetscScalar sum_weights_ = 0.0;
   // D per element, 9 each: host, filled per group; the device copy only for
   // the SUPG tensor of an unbridged void
   std::vector<PetscScalar> d_e_h_;
   PetscScalarKokkosView d_e_d_;
   // For the voids' mean chord, per OWNED element: its volume, and its faces
   // [face_offset_[k], face_offset_[k + 1]) - the local element across each
   // (-1 on the boundary), the area, and whether a boundary face is vacuum.
   // owned_elem_ is the owned elements' local indices
   std::vector<PetscInt> owned_elem_, face_offset_, face_nb_;
   std::vector<PetscReal> elem_volume_, face_area_;
   std::vector<PetscBool> face_vacuum_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The CG-SUPG operator's own diffusion tensor per element: an isotropic flux
// phi / W through the SUPG term tau Omega Omega : K, summed over the
// ordinates, is (1/W) sum_a w_a tau_a Omega_a Omega_a^T : K phi (the other
// terms are odd in Omega and cancel). On the device so it is UboltSUPGTau's
// own arithmetic. d_e is 9 per element
static void CGSupgDKernel(PetscScalarKokkosView sigma_t_e, PetscScalarKokkosView omega_d, \
   PetscScalarKokkosView centre_grad_d, PetscScalar2DKokkosView w_d, PetscInt n_angles, PetscInt nv, \
   PetscReal zeta, PetscScalar sum_weights, PetscInt n_elem, PetscScalarKokkosView d_e)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n_elem), KOKKOS_LAMBDA(PetscInt e) {

         PetscScalar acc[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
         for (PetscInt a = 0; a < n_angles; a++) {
            const PetscScalar tau = UboltSUPGTau(centre_grad_d, omega_d, e, a, nv, sigma_t_e(e), zeta);
            const PetscScalar wt = w_d(a, 0) * tau / sum_weights;
            for (PetscInt d = 0; d < 3; d++) {
               for (PetscInt dd = 0; dd < 3; dd++) acc[3 * d + dd] += wt * omega_d(3 * a + d) * omega_d(3 * a + dd);
            }
         }
         for (PetscInt i = 0; i < 9; i++) d_e(9 * e + i) = acc[i];
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSACG::create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredCG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscInt any_vac = 0;

   PetscFunctionBeginUser;

   local_vertices_ = ps.local_cells;
   nv_ = disc.n_element_vertices();
   n_elem_ = disc.n_local_elements();
   elem_owned_ = disc.element_owned_host();
   omega_d_ = disc.omega_d();
   centre_grad_d_ = disc.centre_grad_d();
   zeta_ = disc.zeta();
   w_d_ = quad.w_d();
   n_angles_ = quad.n_angles();
   sum_weights_ = quad.sum_weights();
   mass_ = disc.mass_host();
   stiff_ = disc.stiff_host();
   d_e_h_.assign(9 * n_elem_, 0.0);
   const PetscInt nv = nv_;

   const std::vector<PetscInt> &elem_vertex = disc.elem_vertex_host();
   const std::vector<PetscInt> &lv_owned = disc.local_to_owned_host();
   const std::vector<PetscInt> &lv_global = disc.local_vertex_global_host();

   // The element entries: every (element, li, j) whose row vertex li is owned
   std::vector<PetscInt> coo_i, coo_j;
   for (PetscInt e = 0; e < n_elem_; e++) {
      for (PetscInt li = 0; li < nv; li++) {
         const PetscInt lv = elem_vertex[e * nv + li];
         if (lv_owned[lv] < 0) continue;
         for (PetscInt j = 0; j < nv; j++) {
            coo_i.push_back(lv_global[lv]);
            coo_j.push_back(lv_global[elem_vertex[e * nv + j]]);
            entry_elem_.push_back(e);
            entry_li_.push_back(li);
            entry_j_.push_back(j);
         }
      }
   }

   // The vacuum faces, per owned vertex: the face's incoming half-range
   // current (the weak inflow term restricted) times the lumped face mass
   // m^f_i = A_f / n_face_vertices, which the backend stores as m^f_i /
   // (m_i A_f). A reflective face restricts to nothing (its diagonal and
   // mirror terms cancel on an isotropic flux): zero Neumann
   const std::vector<PetscInt> &bf_offset = disc.bface_offset_host();
   const std::vector<PetscScalar> &bf_nA = disc.bface_nA_host();
   const std::vector<PetscScalar> &bf_mass = disc.bface_mass_host();
   const std::vector<PetscInt> &bf_slot = disc.bface_slot_host();
   const std::vector<PetscReal> &lumped = disc.lumped_mass_host();
   std::vector<PetscReal> w_h, cos_h(n_angles_);
   DSAHostWeights(quad, w_h);
   const PetscReal sw = PetscRealPart(quad.sum_weights());
   auto omega_h = DSAHostCopy(disc.omega_d());
   vacuum_weight_.assign(local_vertices_, 0.0);
   for (PetscInt k = 0; k < local_vertices_; k++) {
      for (PetscInt f = bf_offset[k]; f < bf_offset[k + 1]; f++) {
         if (bf_slot[f] >= 0) continue;
         any_vac = 1;
         PetscReal area = 0.0;
         for (PetscInt d = 0; d < 3; d++) area += PetscRealPart(bf_nA[3 * f + d] * bf_nA[3 * f + d]);
         area = PetscSqrtReal(area);
         for (PetscInt a = 0; a < n_angles_; a++) {
            PetscReal dot = 0.0;
            for (PetscInt d = 0; d < 3; d++) dot += PetscRealPart(omega_h(3 * a + d) * bf_nA[3 * f + d]);
            // The INCOMING half-range: the ordinates with Omega . n < 0
            cos_h[a] = -dot / area;
         }
         const PetscReal m = HalfRangeCurrent(w_h, sw, cos_h);
         vacuum_weight_[k] += m * PetscRealPart(bf_mass[f]) * lumped[k] * area;
      }
      coo_i.push_back(lv_global[disc.owned_local_vertex_host()[k]]);
      coo_j.push_back(coo_i.back());
   }
   coo_v_.assign(coo_i.size(), 0.0);

   // The owned elements' faces, for the voids' mean chord. An owned element
   // is inside the one-cell overlap, so a face of it with one local cell is
   // the domain boundary
   {
      DM dm = disc.dm();
      DMLabel face_sets = NULL;
      PetscInt c_start = 0, c_end = 0;
      PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
      PetscCall(DMGetLabel(dm, "Face Sets", &face_sets));
      const std::vector<PetscReal> &volume = disc.element_volume_host();
      face_offset_.assign(1, 0);
      for (PetscInt e = 0; e < n_elem_; e++) {
         if (!elem_owned_[e]) continue;
         owned_elem_.push_back(e);
         elem_volume_.push_back(volume[e]);
         PetscInt n_cone = 0;
         const PetscInt *cone = nullptr;
         PetscCall(DMPlexGetConeSize(dm, c_start + e, &n_cone));
         PetscCall(DMPlexGetCone(dm, c_start + e, &cone));
         for (PetscInt f = 0; f < n_cone; f++) {
            PetscReal area = 0.0;
            PetscInt n_support = 0, label = -1;
            const PetscInt *support = nullptr;
            PetscCall(DMPlexComputeCellGeometryFVM(dm, cone[f], &area, NULL, NULL));
            PetscCall(DMPlexGetSupportSize(dm, cone[f], &n_support));
            PetscCall(DMPlexGetSupport(dm, cone[f], &support));
            PetscInt nb = -1;
            if (n_support == 2) nb = (support[0] == c_start + e ? support[1] : support[0]) - c_start;
            else if (face_sets) PetscCall(DMLabelGetValue(face_sets, cone[f], &label));
            face_nb_.push_back(nb);
            face_area_.push_back(area);
            face_vacuum_.push_back((PetscBool)(nb < 0 && bcs.type(label) == BCType::VACUUM));
         }
         face_offset_.push_back((PetscInt)face_nb_.size());
      }
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &any_vac, 1, MPIU_INT, MPI_MAX, comm));
   any_vacuum = (PetscBool)(any_vac != 0);

   PetscCall(MatCreate(comm, &mat));
   PetscCall(MatSetSizes(mat, local_vertices_, local_vertices_, PETSC_DETERMINE, PETSC_DETERMINE));
   PetscCall(MatSetType(mat, MATAIJKOKKOS));
   PetscCall(MatSetPreallocationCOO(mat, (PetscCount)coo_i.size(), coo_i.data(), coo_j.data()));
   PetscCall(MatSetOption(mat, MAT_SPD, PETSC_TRUE));
   PetscCall(MatCreateVecs(mat, &sol, &rhs));

   // The transport rows are divided by the lumped mass, the diffusion rows
   // are the weak form: R's moment is scaled by m_i to match
   PetscCall(VecDuplicate(rhs, &weight));
   {
      PetscScalar *v = nullptr;
      PetscCall(VecGetArrayWrite(weight, &v));
      for (PetscInt k = 0; k < local_vertices_; k++) v[k] = lumped[k];
      PetscCall(VecRestoreArrayWrite(weight, &v));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Over the owned elements; a neighbour element is local (overlap), so its
// void flag needs no exchange
PetscErrorCode DSACG::void_volume_surface(const PetscInt *is_void, PetscReal vs[2])
{
   PetscFunctionBeginUser;

   for (size_t k = 0; k < owned_elem_.size(); k++) {
      if (!is_void[owned_elem_[k]]) continue;
      vs[0] += elem_volume_[k];
      for (PetscInt f = face_offset_[k]; f < face_offset_[k + 1]; f++) {
         const PetscInt nb = face_nb_[f];
         if (nb >= 0 ? !is_void[nb] : face_vacuum_[f]) vs[1] += face_area_[f];
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Element by element into the owned rows. D per element is the policy's,
// isotropic, except in an unbridged void, which keeps the SUPG operator's
// own tensor - the only D defined there
PetscErrorCode DSACG::assemble(const DSAGroup &g)
{
   const PetscInt nv = nv_;

   PetscFunctionBeginUser;

   if (g.n_void > 0 && !g.bridged) {
      if (d_e_d_.extent(0) == 0) d_e_d_ = PetscScalarKokkosView("dsa_cg_d", 9 * n_elem_);
      CGSupgDKernel(g.sigma_t_d, omega_d_, centre_grad_d_, w_d_, n_angles_, nv, zeta_, sum_weights_, n_elem_, \
         d_e_d_);
      Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), \
         PetscScalarKokkosViewHostUnmanaged(d_e_h_.data(), 9 * n_elem_), d_e_d_);
      PetscGetKokkosExecutionSpace().fence();
   }
   for (PetscInt e = 0; e < n_elem_; e++) {
      if (g.is_void[e] && !g.bridged) continue;
      const PetscReal d = PetscRealPart(g.d[e]);
      for (PetscInt i = 0; i < 9; i++) d_e_h_[9 * e + i] = (i % 4 == 0) ? d : 0.0;
   }

   const PetscInt n_entries = (PetscInt)entry_elem_.size();
   for (PetscInt q = 0; q < n_entries; q++) {
      const PetscInt e = entry_elem_[q];
      const PetscInt ij = (e * nv + entry_li_[q]) * nv + entry_j_[q];
      PetscScalar v = (g.sigma_t_h[e] - g.sigma_s_h[e]) * mass_[ij];
      for (PetscInt dd = 0; dd < 9; dd++) v += d_e_h_[9 * e + dd] * stiff_[9 * ij + dd];
      coo_v_[q] = v;
   }
   for (PetscInt k = 0; k < local_vertices_; k++) coo_v_[n_entries + k] = vacuum_weight_[k];

   PetscCall(MatSetValuesCOO(mat, coo_v_.data(), INSERT_VALUES));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredCG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op)
{
   DSACG *cg = new DSACG;

   PetscFunctionBeginUser;

   *op = cg;
   PetscCall(cg->create(comm, ps, disc, quad, bcs));

   PetscFunctionReturn(PETSC_SUCCESS);
}
