#include "dsa_operatork.hpp"
#include "ubolt/unstructured_dg.hpp"

// The DMPlex (UnstructuredDG) backend's DSA operator, either order: at DG0 the
// cell-centred two-point flux, at DG1 the MIP interior penalty form on the
// backend's modal basis. Both VOLUME-weighted (the rows are V times the
// structured backends' per-unit-volume rows), so R's moment is scaled by V.
// Host values into a COO pattern set once at create - the structured path's
// reasoning for host assembly - and MatSetValuesCOO bumps the matrix state,
// so the inner KSP redoes its setup
class DSAPlex final : public DSAOperator {
public:
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);

   PetscInt n_units() const override { return local_cells_; }
   PetscErrorCode void_volume_surface(const PetscInt *is_void, PetscReal vs[2]) override;
   PetscErrorCode assemble(const DSAGroup &g) override;

protected:
   PetscErrorCode destroy_own() override;

private:
   // Stage a per-cell value and scatter it onto the interior face slots
   PetscErrorCode stage(const PetscScalar *v);
   PetscErrorCode assemble_dg0(const DSAGroup &g, const PetscScalar *d_nb);
   PetscErrorCode assemble_dg1(const DSAGroup &g, const PetscScalar *d_nb);

   PetscInt local_cells_ = 0;
   PetscInt n_basis_ = 1;
   // Per owned cell, its COO entries run [cell_entry_offset_[c],
   // cell_entry_offset_[c + 1]): one per interior face in cone order, then
   // the diagonal - at DG1 each of the cell's n_basis rows in turn, n_basis
   // entries (the column cell's basis j) where DG0 has one
   std::vector<PetscInt> cell_entry_offset_;
   // Per (cell, face) slot, the backend's CSR: the neighbour's slot in d_nb_
   // (-1 on a boundary face), A_f, the two centroid-to-face distances, and
   // whether a boundary face is vacuum
   std::vector<PetscInt> cell_face_offset_;
   std::vector<PetscInt> face_nb_;
   std::vector<PetscReal> face_area_;
   std::vector<PetscReal> face_distance_;
   std::vector<PetscBool> face_vacuum_;
   std::vector<PetscReal> volume_;
   std::vector<PetscScalar> coo_v_;
   // DG0: the consistent D's half-range current per face slot
   PetscBool consistent_d_ = PETSC_TRUE;
   PetscReal consistent_power_ = 1.5;
   std::vector<PetscReal> face_half_range_;
   // DG1: the backend's face matrices int_f phi_i^c phi_j^{c | n} / (V_c A_f),
   // the unit normal per face slot, the basis gradients of this cell and of
   // the neighbour across each interior face slot, and the penalty constant
   std::vector<PetscScalar> face_own_, face_up_, basis_grad_, face_nb_grad_;
   std::vector<PetscReal> face_normal_;
   PetscReal mip_penalty_ = 4.0;
   // D is per cell whatever the order: staged on d_global_ (host, one entry
   // per owned cell), scattered onto d_nb_ (one per interior face slot, owned
   // neighbours included - one path for both)
   Vec d_global_ = NULL, d_nb_ = NULL;
   VecScatter nb_scatter_ = NULL;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// No DMDA twin: the matrix is sized off the backend's owned cells, whose
// global order is the transport rows' (CheckPlexStratumLayout), and preallocated
// from its face CSR. Host work, once
PetscErrorCode DSAPlex::create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscInt cstart = 0, any_vac = 0;
   IS nb_is = NULL;

   PetscFunctionBeginUser;

   PetscCheck(ps.n_basis == disc.n_basis(), comm, PETSC_ERR_ARG_INCOMP, \
      "the phase space has %" PetscInt_FMT " basis functions per cell but the backend %" PetscInt_FMT, \
      ps.n_basis, disc.n_basis());

   local_cells_ = ps.local_cells;
   n_basis_ = ps.n_basis;

   // Owned cells are contiguous in the global numbering, rank by rank, in the
   // transport rows' order - so the global cell of local cell k is cstart + k,
   // and a neighbour's is its row base over the rows per cell
   PetscCallMPI(MPI_Scan(&local_cells_, &cstart, 1, MPIU_INT, MPI_SUM, comm));
   cstart -= local_cells_;
   const PetscInt rows_per_cell = ps.rows_per_cell();

   const std::vector<PetscInt> &offset = disc.cell_face_offset_host();
   const std::vector<PetscScalar> &nA = disc.face_nA_host();
   const std::vector<PetscInt> &nb_row = disc.face_neighbour_row_host();
   const std::vector<PetscInt> &label = disc.face_label_host();
   const PetscInt n_slots = offset[local_cells_];
   const PetscInt nb = n_basis_;

   cell_face_offset_ = offset;
   face_distance_ = disc.face_distance_host();
   volume_ = disc.volume_host();
   face_nb_.assign(n_slots, -1);
   face_area_.assign(n_slots, 0.0);
   face_vacuum_.assign(n_slots, PETSC_FALSE);
   cell_entry_offset_.assign(local_cells_ + 1, 0);

   std::vector<PetscInt> nb_global, coo_i, coo_j;
   for (PetscInt c = 0; c < local_cells_; c++) {
      for (PetscInt k = offset[c]; k < offset[c + 1]; k++) {

         PetscReal area = 0.0;
         for (PetscInt d = 0; d < 3; d++) area += PetscRealPart(nA[3 * k + d] * nA[3 * k + d]);
         face_area_[k] = PetscSqrtReal(area);

         if (nb_row[k] >= 0) {
            face_nb_[k] = (PetscInt)nb_global.size();
            nb_global.push_back(nb_row[k] / rows_per_cell);
         }
         // A boundary face: Marshak if vacuum, zero Neumann if reflective
         else if (bcs.type(label[k]) == BCType::VACUUM) {
            face_vacuum_[k] = PETSC_TRUE;
            any_vac = 1;
         }
      }

      // Row (cell, i): n_basis columns per interior face in cone order (the
      // neighbour's basis j), then the cell's own n_basis
      for (PetscInt i = 0; i < nb; i++) {
         const PetscInt row = (cstart + c) * nb + i;
         for (PetscInt k = offset[c]; k < offset[c + 1]; k++) {
            if (face_nb_[k] < 0) continue;
            for (PetscInt j = 0; j < nb; j++) {
               coo_i.push_back(row);
               coo_j.push_back(nb_global[face_nb_[k]] * nb + j);
            }
         }
         for (PetscInt j = 0; j < nb; j++) {
            coo_i.push_back(row);
            coo_j.push_back((cstart + c) * nb + j);
         }
      }
      cell_entry_offset_[c + 1] = (PetscInt)coo_i.size();
   }
   coo_v_.assign(coo_i.size(), 0.0);

   PetscCall(ConsistentDOptions(comm, &consistent_d_, &consistent_power_));
   if (nb == 1) {

      // DG0's m per face, off the backend's own ordinates so the two cannot
      // disagree
      std::vector<PetscReal> w_h, cos_h(quad.n_angles());
      DSAHostWeights(quad, w_h);
      const PetscReal sw = PetscRealPart(quad.sum_weights());
      auto omega_h = DSAHostCopy(disc.omega_d());
      face_half_range_.assign(n_slots, 0.0);
      for (PetscInt k = 0; k < n_slots; k++) {
         for (PetscInt a = 0; a < quad.n_angles(); a++) {
            PetscReal dot = 0.0;
            for (PetscInt d = 0; d < 3; d++) dot += PetscRealPart(omega_h(3 * a + d) * nA[3 * k + d]);
            cos_h[a] = dot / face_area_[k];
         }
         face_half_range_[k] = HalfRangeCurrent(w_h, sw, cos_h);
      }
   }
   else {

      // DG1: the backend's face matrices and basis gradients, and the unit
      // normals
      face_own_ = disc.face_own_host();
      face_up_ = disc.face_up_host();
      basis_grad_ = disc.basis_grad_host();
      face_nb_grad_ = disc.face_neighbour_grad_host();
      face_normal_.assign(3 * n_slots, 0.0);
      for (PetscInt k = 0; k < n_slots; k++) {
         for (PetscInt d = 0; d < 3; d++) face_normal_[3 * k + d] = PetscRealPart(nA[3 * k + d]) / face_area_[k];
      }
      PetscCall(PetscOptionsGetReal(NULL, "dsa_", "-mip_penalty", &mip_penalty_, NULL));
      PetscCheck(mip_penalty_ > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "-dsa_mip_penalty must be positive, got %g", (double)mip_penalty_);
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &any_vac, 1, MPIU_INT, MPI_MAX, comm));
   any_vacuum = (PetscBool)(any_vac != 0);

   PetscCall(MatCreate(comm, &mat));
   PetscCall(MatSetSizes(mat, local_cells_ * nb, local_cells_ * nb, PETSC_DETERMINE, PETSC_DETERMINE));
   PetscCall(MatSetType(mat, MATAIJKOKKOS));
   // At DG1 a cell's nodes are one block, so GAMG aggregates them together
   PetscCall(MatSetBlockSize(mat, nb));
   PetscCall(MatSetPreallocationCOO(mat, (PetscCount)coo_i.size(), coo_i.data(), coo_j.data()));
   PetscCall(MatSetOption(mat, MAT_SPD, PETSC_TRUE));
   PetscCall(MatCreateVecs(mat, &sol, &rhs));

   PetscCall(VecCreateMPI(comm, local_cells_, PETSC_DETERMINE, &d_global_));
   PetscCall(VecDuplicate(rhs, &weight));
   {
      PetscScalar *v = nullptr;
      PetscCall(VecGetArrayWrite(weight, &v));
      for (PetscInt c = 0; c < local_cells_; c++) {
         for (PetscInt i = 0; i < nb; i++) v[c * nb + i] = volume_[c];
      }
      PetscCall(VecRestoreArrayWrite(weight, &v));
   }

   PetscCall(VecCreateSeq(PETSC_COMM_SELF, (PetscInt)nb_global.size(), &d_nb_));
   PetscCall(ISCreateGeneral(PETSC_COMM_SELF, (PetscInt)nb_global.size(), nb_global.data(), \
      PETSC_COPY_VALUES, &nb_is));
   PetscCall(VecScatterCreate(d_global_, nb_is, d_nb_, NULL, &nb_scatter_));
   PetscCall(ISDestroy(&nb_is));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPlex::destroy_own()
{
   PetscFunctionBeginUser;

   PetscCall(VecScatterDestroy(&nb_scatter_));
   PetscCall(VecDestroy(&d_global_));
   PetscCall(VecDestroy(&d_nb_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPlex::stage(const PetscScalar *v)
{
   PetscScalar *global = nullptr;

   PetscFunctionBeginUser;

   PetscCall(VecGetArrayWrite(d_global_, &global));
   for (PetscInt c = 0; c < local_cells_; c++) global[c] = v[c];
   PetscCall(VecRestoreArrayWrite(d_global_, &global));
   PetscCall(VecScatterBegin(nb_scatter_, d_global_, d_nb_, INSERT_VALUES, SCATTER_FORWARD));
   PetscCall(VecScatterEnd(nb_scatter_, d_global_, d_nb_, INSERT_VALUES, SCATTER_FORWARD));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPlex::void_volume_surface(const PetscInt *is_void, PetscReal vs[2])
{
   const PetscScalar *flag_nb = nullptr;
   std::vector<PetscScalar> flag(local_cells_);

   PetscFunctionBeginUser;

   for (PetscInt c = 0; c < local_cells_; c++) flag[c] = StageVoidFlag(is_void[c]);
   PetscCall(stage(flag.data()));
   PetscCall(VecGetArrayRead(d_nb_, &flag_nb));
   for (PetscInt c = 0; c < local_cells_; c++) {
      if (!is_void[c]) continue;
      vs[0] += volume_[c];
      for (PetscInt k = cell_face_offset_[c]; k < cell_face_offset_[c + 1]; k++) {
         if (face_nb_[k] >= 0 ? !StagedVoidFlag(flag_nb[face_nb_[k]]) : face_vacuum_[k]) vs[1] += face_area_[k];
      }
   }
   PetscCall(VecRestoreArrayRead(d_nb_, &flag_nb));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPlex::assemble(const DSAGroup &g)
{
   const PetscScalar *d_nb = nullptr;
   std::vector<PetscScalar> staged(local_cells_);

   PetscFunctionBeginUser;

   for (PetscInt c = 0; c < local_cells_; c++) staged[c] = StageD(g.d[c], (PetscBool)(g.bridged && g.is_void[c]));
   PetscCall(stage(staged.data()));
   PetscCall(VecGetArrayRead(d_nb_, &d_nb));
   if (n_basis_ == 1) PetscCall(assemble_dg0(g, d_nb));
   else PetscCall(assemble_dg1(g, d_nb));
   PetscCall(VecRestoreArrayRead(d_nb_, &d_nb));
   PetscCall(MatSetValuesCOO(mat, coo_v_.data(), INSERT_VALUES));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Row c: V_c sigma_a phi_c + sum_f T_f (phi_c - phi_n) + sum_vacuum M_f phi_c
PetscErrorCode DSAPlex::assemble_dg0(const DSAGroup &g, const PetscScalar *d_nb)
{
   PetscFunctionBeginUser;

   for (PetscInt c = 0; c < local_cells_; c++) {

      PetscInt e = cell_entry_offset_[c];

      // A masked void: V times the identity (the plex matrix stays V times
      // the structured one), couplings zeroed - the COO values persist
      // across groups, so they are written, not skipped
      if (g.masked[c]) {
         for (; e < cell_entry_offset_[c + 1] - 1; e++) coo_v_[e] = 0.0;
         coo_v_[e] = volume_[c];
         continue;
      }

      const PetscScalar d_c = g.d[c];
      PetscScalar diag = volume_[c] * (g.sigma_t_h[c] - g.sigma_s_h[c]);

      for (PetscInt k = cell_face_offset_[c]; k < cell_face_offset_[c + 1]; k++) {

         const PetscReal d_own = face_distance_[2 * k];
         const PetscScalar d_n = face_nb_[k] >= 0 ? StagedD(d_nb[face_nb_[k]]) : (PetscScalar)0.0;
         if (face_nb_[k] >= 0 && StagedMasked(d_n)) {

            // A face into a masked void: no coupling (not even the blend's
            // m h), and the vacuum boundary face's Marshak term
            coo_v_[e++] = 0.0;
            const PetscScalar d_b = consistent_d_ ? \
               BlendD(d_c, 2.0 * d_own * face_half_range_[k], consistent_power_) : d_c;
            diag += face_area_[k] / (2.0 + d_own / d_b);
         }
         else if (face_nb_[k] >= 0) {

            // Two-point flux: current continuity between the two centroids,
            // a series resistance - the harmonic face D on a uniform grid.
            // The structured blend then goes on the face's effective D over
            // the centroid-to-centroid length
            PetscScalar t_f = face_area_[k] / (d_own / d_c + face_distance_[2 * k + 1] / d_n);
            if (consistent_d_) {
               const PetscReal len = d_own + face_distance_[2 * k + 1];
               t_f = face_area_[k] / len * BlendD(t_f * len / face_area_[k], face_half_range_[k] * len, \
                  consistent_power_);
            }
            coo_v_[e++] = -t_f;
            diag += t_f;
         }
         else if (face_vacuum_[k]) {

            // Marshak: the structured face, times the volume
            const PetscScalar d_b = consistent_d_ ? \
               BlendD(d_c, 2.0 * d_own * face_half_range_[k], consistent_power_) : d_c;
            diag += face_area_[k] / (2.0 + d_own / d_b);
         }
      }
      coo_v_[e++] = diag;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// DG1: the MIP form, row (c, i) tested with phi_i^c. Every face is visited
// from both of its cells and each visit writes only its own cell's rows: v =
// phi_i^c jumps by phi_i^c across the face (n out of c) and averages
// 1/2 D_c grad phi_i^c . n. With F the face matrices times V_c A_f,
//    F_own(i, j) = int_f phi_i^c phi_j^c,  F_up(i, j) = int_f phi_i^c phi_j^n,
// m_i = F_own(i, 0) = int_f phi_i^c, and g_j, gn_j this cell's and the
// neighbour's grad phi_j . n (constant on a cell), an interior face adds
//    own (c, j): kappa F_own(i, j) - 1/2 D_c g_j m_i - 1/2 D_c g_i F_own(0, j)
//    nbr (n, j): -kappa F_up(i, j) - 1/2 D_n gn_j m_i + 1/2 D_c g_i F_up(0, j)
// and a vacuum face the own block with kappa_b (the 1/2 is the MIP boundary
// form's). Symmetric: the neighbour's visit writes the transpose of the nbr
// block. Volume terms V_c (D_c grad phi_i . grad phi_j + sigma_a delta_ij),
// the basis being orthonormal in the volume average
PetscErrorCode DSAPlex::assemble_dg1(const DSAGroup &g, const PetscScalar *d_nb)
{
   const PetscInt nb = n_basis_;
   // n_basis is at most 4 (3D), so fixed-size locals
   PetscScalar own[4][4], gc[4], gn[4];

   PetscFunctionBeginUser;

   for (PetscInt c = 0; c < local_cells_; c++) {

      const PetscReal vol = volume_[c];
      // In row i's run, the q-th interior face (cone order) owns entries
      // [q * nb, (q + 1) * nb), and the own block comes last
      const PetscInt row_len = (cell_entry_offset_[c + 1] - cell_entry_offset_[c]) / nb;
      const PetscInt n_interior = row_len / nb - 1;

      // A masked void: V times the identity block, every coupling zeroed
      if (g.masked[c]) {
         for (PetscInt e = cell_entry_offset_[c]; e < cell_entry_offset_[c + 1]; e++) coo_v_[e] = 0.0;
         for (PetscInt i = 0; i < nb; i++) coo_v_[cell_entry_offset_[c] + i * row_len + n_interior * nb + i] = vol;
         continue;
      }

      const PetscScalar d_c = g.d[c];
      const PetscScalar sigma_a = g.sigma_t_h[c] - g.sigma_s_h[c];

      for (PetscInt i = 0; i < nb; i++) {
         for (PetscInt j = 0; j < nb; j++) {
            PetscScalar grad_dot = 0.0;
            for (PetscInt d = 0; d < 3; d++) grad_dot += basis_grad_[(c * nb + i) * 3 + d] * basis_grad_[(c * nb + j) * 3 + d];
            own[i][j] = vol * (d_c * grad_dot + (i == j ? sigma_a : (PetscScalar)0.0));
         }
      }

      PetscInt q = 0;

      for (PetscInt k = cell_face_offset_[c]; k < cell_face_offset_[c + 1]; k++) {

         // A face into a masked void keeps its COO slots, zeroed, and is a
         // vacuum face below - as if the mesh stopped there
         const PetscBool has_nb = (PetscBool)(face_nb_[k] >= 0);
         const PetscScalar d_n_staged = has_nb ? d_nb[face_nb_[k]] : (PetscScalar)0.0;
         const PetscScalar d_n = StagedD(d_n_staged);
         const PetscBool interior = (PetscBool)(has_nb && !StagedMasked(d_n));
         if (has_nb && !interior) {
            for (PetscInt i = 0; i < nb; i++) {
               for (PetscInt j = 0; j < nb; j++) coo_v_[cell_entry_offset_[c] + i * row_len + q * nb + j] = 0.0;
            }
            q++;
         }
         if (!has_nb && !face_vacuum_[k]) continue;

         const PetscReal scale = vol * face_area_[k];
         const PetscScalar *f_own = &face_own_[k * nb * nb];
         const PetscScalar *f_up = &face_up_[k * nb * nb];
         for (PetscInt j = 0; j < nb; j++) {
            gc[j] = 0.0;
            gn[j] = 0.0;
            for (PetscInt d = 0; d < 3; d++) {
               gc[j] += basis_grad_[(c * nb + j) * 3 + d] * face_normal_[3 * k + d];
               gn[j] += face_nb_grad_[(k * nb + j) * 3 + d] * face_normal_[3 * k + d];
            }
         }

         const PetscReal h_c = 2.0 * face_distance_[2 * k];
         if (interior) {

            const PetscReal h_n = 2.0 * face_distance_[2 * k + 1];
            // A face touching a bridged void takes the weighted interior
            // penalty: both sides' D replaced by the face's harmonic D, in the
            // averages and in kappa (both cells compute the same D_h, so it
            // stays symmetric); plain MIP everywhere else
            PetscScalar dc_f = d_c, dn_f = d_n;
            if (g.bridged && (g.is_void[c] || StagedBridged(d_n_staged))) {
               dc_f = 2.0 * d_c * d_n / (d_c + d_n);
               dn_f = dc_f;
            }
            const PetscReal kappa = PetscMax(0.5 * mip_penalty_ * PetscRealPart(dc_f / h_c + dn_f / h_n), 0.25);

            for (PetscInt i = 0; i < nb; i++) {
               const PetscScalar m_i = scale * f_own[i * nb];
               PetscScalar *nbr = &coo_v_[cell_entry_offset_[c] + i * row_len + q * nb];
               for (PetscInt j = 0; j < nb; j++) {
                  own[i][j] += kappa * scale * f_own[i * nb + j] - 0.5 * dc_f * gc[j] * m_i \
                     - 0.5 * dc_f * gc[i] * scale * f_own[j];
                  nbr[j] = -kappa * scale * f_up[i * nb + j] - 0.5 * dn_f * gn[j] * m_i \
                     + 0.5 * dc_f * gc[i] * scale * f_up[j];
               }
            }
            q++;
         }
         else {

            const PetscReal kappa = PetscMax(mip_penalty_ * PetscRealPart(d_c) / h_c, 0.25);
            for (PetscInt i = 0; i < nb; i++) {
               const PetscScalar m_i = scale * f_own[i * nb];
               for (PetscInt j = 0; j < nb; j++) {
                  own[i][j] += kappa * scale * f_own[i * nb + j] - 0.5 * d_c * gc[j] * m_i \
                     - 0.5 * d_c * gc[i] * scale * f_own[j];
               }
            }
         }
      }

      for (PetscInt i = 0; i < nb; i++) {
         for (PetscInt j = 0; j < nb; j++) coo_v_[cell_entry_offset_[c] + i * row_len + n_interior * nb + j] = own[i][j];
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op)
{
   DSAPlex *p = new DSAPlex;

   PetscFunctionBeginUser;

   *op = p;
   PetscCall(p->create(comm, ps, disc, quad, bcs));

   PetscFunctionReturn(PETSC_SUCCESS);
}
