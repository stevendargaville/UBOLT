#include "ubolt/dsa.hpp"
#include "petsc_kokkos.hpp"
#include <petscdmda.h>

// The kernels below are file-static free functions taking value copies of the
// views, never `this` - a member access inside a KOKKOS_LAMBDA would
// dereference a host pointer on the device (see docs/dev/kokkos.md)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// R_angle: the 0th angular moment of the residual, masked on the BC rows
//
// Deliberately NOT UboltAngularIntegral: that is a gemm over every row, and
// this has to leave the BC rows out of the sum. A BC row carries the boundary
// condition (the identity, or the identity minus its mirrored angle), not a
// physical residual, so including it would feed the diffusion solve something
// that is not a neutron balance - and the correction has to come back off those
// rows again in the prolongation, so the mask has to be the same in both halves
//
// Per (cell, basis) node: the cell at n_basis 1, and at DG1 every basis
// function's moment, which is what the DG1 diffusion operator's rows test with
static void RestrictKernel(PetscScalarConstKokkosView x_d, PetscScalarKokkosView rhs_d, \
   PetscScalar2DKokkosView w_d, PetscIntKokkosView is_bc_row_d, PetscInt n_angles, PetscInt n_nodes)
{
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), n_nodes, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // node
         const PetscInt node = t.league_rank();

         PetscScalar moment = 0.0;
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt a, PetscScalar &acc) {

               const PetscInt r = node * n_angles + a;
               if (!is_bc_row_d(r)) acc += w_d(a, 0) * x_d(r);
            }, moment);

         Kokkos::single(Kokkos::PerTeam(t), [&]() { rhs_d(node) = moment; });
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// P_angle: broadcast the scalar correction isotropically over the ordinates
//
// The / sum_weights is the consistency scaling (see the header). Every row is
// written, so y needs no zeroing beforehand; the BC rows get an explicit zero,
// which is what the BC contract owes them - the assembled operator already
// holds the boundary condition on those rows and a correction there would
// pollute it. node_d is the inner solution, one entry per (cell, basis) node
static void ProlongKernel(PetscScalarConstKokkosView node_d, PetscScalarKokkosView y_d, \
   PetscIntKokkosView is_bc_row_d, PetscInt n_angles, PetscScalar sum_weights, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         y_d(r) = is_bc_row_d(r) ? (PetscScalar)0.0 : node_d(r / n_angles) / sum_weights;
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The half-range current of an isotropic unit scalar flux through a face of
// unit normal n: sum over the ordinates leaving through it of w_a (Omega_a . n),
// over sum_weights. 1/4 in the continuum, the quadrature's own value here.
// cos_h(a) is Omega_a . n, one per ordinate, on the host
static PetscErrorCode HalfRangeCurrent(const AngularQuadrature &quad, const std::vector<PetscReal> &cos_h, \
   PetscReal *m)
{
   PetscFunctionBeginUser;

   auto w_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), quad.w_d());
   PetscReal sum = 0.0;
   for (PetscInt a = 0; a < quad.n_angles(); a++) {
      if (cos_h[a] > 0.0) sum += PetscRealPart(w_h(a, 0)) * cos_h[a];
   }
   *m = sum / PetscRealPart(quad.sum_weights());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// The face D with the upwind numerical diffusion M blended in: the p-norm of
// (D, M), which at p = 1 is plain addition
static inline PetscScalar BlendD(PetscScalar d, PetscReal m, PetscReal p)
{
   if (p == 1.0) return d + m;
   return PetscPowReal(PetscPowReal(PetscRealPart(d), p) + PetscPowReal(m, p), 1.0 / p);
}

// -dsa_consistent_d and its blend power, under the inner solve's prefix like
// -dsa_mip_penalty. p < 1 would add MORE than the sum, which nothing supports
static PetscErrorCode ConsistentDOptions(MPI_Comm comm, PetscBool *on, PetscReal *power)
{
   PetscFunctionBeginUser;

   PetscCall(PetscOptionsGetBool(NULL, "dsa_", "-consistent_d", on, NULL));
   PetscCall(PetscOptionsGetReal(NULL, "dsa_", "-consistent_d_power", power, NULL));
   PetscCheck(*power >= 1.0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
      "-dsa_consistent_d_power must be at least 1, got %g", (double)*power);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// Per axis of a structured backend: the ordinates' cosine on that axis
static PetscErrorCode AxisHalfRangeCurrent(const AngularQuadrature &quad, const PetscScalar *cos, PetscReal *m)
{
   std::vector<PetscReal> cos_h(quad.n_angles());

   PetscFunctionBeginUser;

   for (PetscInt a = 0; a < quad.n_angles(); a++) cos_h[a] = PetscRealPart(cos[a]);
   PetscCall(HalfRangeCurrent(quad, cos_h, m));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// create
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 1D: one axis of geometry, and the two faces of the slab
PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD1D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;

   dim_ = 1;
   h_[0] = disc.dx();
   vacuum_lo_[0] = (PetscBool)(bcs.type(StructuredFD1D::FACE_LEFT) == BCType::VACUUM);
   vacuum_hi_[0] = (PetscBool)(bcs.type(StructuredFD1D::FACE_RIGHT) == BCType::VACUUM);

   const SNQuadrature *sn = dynamic_cast<const SNQuadrature *>(&quad);
   PetscCheck(sn, comm, PETSC_ERR_ARG_WRONG, "the 1D DSA needs the 1D SNQuadrature");
   PetscCall(AxisHalfRangeCurrent(quad, sn->mu_host(), &half_range_[0]));

   PetscCall(create_common(comm, ps, disc, quad));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 2D: two axes of geometry, and the four faces of the box. CAREFUL with the
// face names - in 2D bottom/top are the Y faces, where in 3D they are the Z
// ones (PETSc's box-mesh convention)
PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD2D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;

   dim_ = 2;
   h_[0] = disc.dx();
   h_[1] = disc.dy();
   vacuum_lo_[0] = (PetscBool)(bcs.type(StructuredFD2D::FACE_LEFT) == BCType::VACUUM);
   vacuum_hi_[0] = (PetscBool)(bcs.type(StructuredFD2D::FACE_RIGHT) == BCType::VACUUM);
   vacuum_lo_[1] = (PetscBool)(bcs.type(StructuredFD2D::FACE_BOTTOM) == BCType::VACUUM);
   vacuum_hi_[1] = (PetscBool)(bcs.type(StructuredFD2D::FACE_TOP) == BCType::VACUUM);

   const SNQuadrature2D *sn = dynamic_cast<const SNQuadrature2D *>(&quad);
   PetscCheck(sn, comm, PETSC_ERR_ARG_WRONG, "the 2D DSA needs the SNQuadrature2D");
   PetscCall(AxisHalfRangeCurrent(quad, sn->mu_host(), &half_range_[0]));
   PetscCall(AxisHalfRangeCurrent(quad, sn->eta_host(), &half_range_[1]));

   PetscCall(create_common(comm, ps, disc, quad));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 3D: three axes of geometry, and the six faces of the box. CAREFUL with the
// face names - PETSc's box convention puts bottom/top on Z here, so the Y faces
// are front (y-min) and back (y-max), which is NOT what 2D calls them
PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD3D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscFunctionBeginUser;

   dim_ = 3;
   h_[0] = disc.dx();
   h_[1] = disc.dy();
   h_[2] = disc.dz();
   vacuum_lo_[0] = (PetscBool)(bcs.type(StructuredFD3D::FACE_LEFT) == BCType::VACUUM);
   vacuum_hi_[0] = (PetscBool)(bcs.type(StructuredFD3D::FACE_RIGHT) == BCType::VACUUM);
   vacuum_lo_[1] = (PetscBool)(bcs.type(StructuredFD3D::FACE_FRONT) == BCType::VACUUM);
   vacuum_hi_[1] = (PetscBool)(bcs.type(StructuredFD3D::FACE_BACK) == BCType::VACUUM);
   vacuum_lo_[2] = (PetscBool)(bcs.type(StructuredFD3D::FACE_BOTTOM) == BCType::VACUUM);
   vacuum_hi_[2] = (PetscBool)(bcs.type(StructuredFD3D::FACE_TOP) == BCType::VACUUM);

   const SNQuadrature3D *sn = dynamic_cast<const SNQuadrature3D *>(&quad);
   PetscCheck(sn, comm, PETSC_ERR_ARG_WRONG, "the 3D DSA needs the SNQuadrature3D");
   PetscCall(AxisHalfRangeCurrent(quad, sn->mu_host(), &half_range_[0]));
   PetscCall(AxisHalfRangeCurrent(quad, sn->eta_host(), &half_range_[1]));
   PetscCall(AxisHalfRangeCurrent(quad, sn->xi_host(), &half_range_[2]));

   PetscCall(create_common(comm, ps, disc, quad));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DMPlex backend: no DMDA twin, so the matrix and its layout come straight
// off the backend's owned cells and its face CSR. Everything here is host
// work done once; set_group() refills values only
PetscErrorCode DSAPrecon::create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs)
{
   PetscInt cstart = 0, any_vacuum = 0;
   IS nb_is = NULL;

   PetscFunctionBeginUser;

   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck(ps.n_basis == disc.n_basis(), comm, PETSC_ERR_ARG_INCOMP, \
      "the phase space has %" PetscInt_FMT " basis functions per cell but the backend %" PetscInt_FMT, \
      ps.n_basis, disc.n_basis());

   comm_ = comm;
   plex_ = PETSC_TRUE;
   dim_ = disc.dimension();
   n_angles_ = ps.n_angles;
   n_basis_ = ps.n_basis;
   local_cells_ = ps.local_cells;
   sum_weights_ = quad.sum_weights();
   w_d_ = quad.w_d();
   is_bc_row_d_ = disc.boundary_info().is_bc_row_d;

   // Owned cells are contiguous in the global numbering, rank by rank, in the
   // same order as the transport rows (CheckPlexLayout) - so the global cell of
   // local cell k is cstart + k, and a neighbour's is its row base over the
   // rows per cell
   PetscCallMPI(MPI_Scan(&local_cells_, &cstart, 1, MPIU_INT, MPI_SUM, comm_));
   cstart -= local_cells_;
   const PetscInt rows_per_cell = ps.rows_per_cell();

   const std::vector<PetscInt> &offset = disc.cell_face_offset_host();
   const std::vector<PetscScalar> &nA = disc.face_nA_host();
   const std::vector<PetscInt> &nb_row = disc.face_neighbour_row_host();
   const std::vector<PetscInt> &label = disc.face_label_host();
   const std::vector<PetscReal> &distance = disc.face_distance_host();
   const PetscInt n_slots = offset[local_cells_];
   const PetscInt nb = n_basis_;

   cell_face_offset_ = offset;
   face_distance_ = distance;
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
         // A boundary face: Marshak if vacuum, zero Neumann if reflective -
         // the same families the structured faces take
         else if (bcs.type(label[k]) == BCType::VACUUM) {
            face_vacuum_[k] = PETSC_TRUE;
            any_vacuum = 1;
         }
      }

      // Row (cell, i): n_basis columns per interior face in cone order (the
      // neighbour's basis j), then the cell's own n_basis. At DG0 that is the
      // face graph plus the diagonal
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

   // The upwind numerical diffusion per face (see assemble_plex), off the
   // backend's own ordinates so the two cannot disagree. DG0 only - DG1's
   // interior penalty already floors at the same half-range current
   PetscCall(ConsistentDOptions(comm, &consistent_d_, &consistent_power_));
   if (nb == 1) {
      auto omega_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), disc.omega_d());
      std::vector<PetscReal> cos_h(n_angles_);
      face_half_range_.assign(n_slots, 0.0);
      for (PetscInt k = 0; k < n_slots; k++) {
         for (PetscInt a = 0; a < n_angles_; a++) {
            PetscReal dot = 0.0;
            for (PetscInt d = 0; d < 3; d++) dot += PetscRealPart(omega_h(3 * a + d) * nA[3 * k + d]);
            cos_h[a] = dot / face_area_[k];
         }
         PetscCall(HalfRangeCurrent(quad, cos_h, &face_half_range_[k]));
      }
   }

   if (nb > 1) {

      // DG1: the backend's face matrices and basis gradients, and the unit
      // normals. The penalty constant is a runtime knob under the inner
      // solve's prefix
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
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &any_vacuum, 1, MPIU_INT, MPI_MAX, comm_));
   any_vacuum_ = (PetscBool)(any_vacuum != 0);

   // The COO pattern is the face graph plus the diagonal (in n_basis blocks at
   // DG1), set once
   PetscCall(MatCreate(comm_, &diff_mat_));
   PetscCall(MatSetSizes(diff_mat_, local_cells_ * nb, local_cells_ * nb, PETSC_DETERMINE, PETSC_DETERMINE));
   PetscCall(MatSetType(diff_mat_, MATAIJKOKKOS));
   // At DG1 a cell's n_basis nodes are one block, which is what lets GAMG
   // aggregate them together (measured Sep 2026, one V-cycle: tets 16 -> 9,
   // triangles 11 -> 8 on the diffusive problems, and no worse anywhere)
   PetscCall(MatSetBlockSize(diff_mat_, nb));
   PetscCall(MatSetPreallocationCOO(diff_mat_, (PetscCount)coo_i.size(), coo_i.data(), coo_j.data()));
   PetscCall(MatSetOption(diff_mat_, MAT_SPD, PETSC_TRUE));

   PetscCall(MatCreateVecs(diff_mat_, &sol_, &rhs_));
   // D is per cell whatever the order, so it is staged on a cell-sized Vec
   PetscCall(VecCreateMPI(comm_, local_cells_, PETSC_DETERMINE, &d_global_));
   PetscCall(VecDuplicate(rhs_, &volume_vec_));
   {
      PetscScalar *v = nullptr;
      PetscCall(VecGetArrayWrite(volume_vec_, &v));
      for (PetscInt c = 0; c < local_cells_; c++) {
         for (PetscInt i = 0; i < nb; i++) v[c * nb + i] = volume_[c];
      }
      PetscCall(VecRestoreArrayWrite(volume_vec_, &v));
   }

   // The neighbours' D, one seq entry per interior face slot, owned or not
   PetscCall(VecCreateSeq(PETSC_COMM_SELF, (PetscInt)nb_global.size(), &d_nb_));
   PetscCall(ISCreateGeneral(PETSC_COMM_SELF, (PetscInt)nb_global.size(), nb_global.data(), \
      PETSC_COPY_VALUES, &nb_is));
   PetscCall(VecScatterCreate(d_global_, nb_is, d_nb_, NULL, &nb_scatter_));
   PetscCall(ISDestroy(&nb_is));

   PetscCall(create_ksp());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPrecon::create_common(MPI_Comm comm, const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad)
{
   PetscInt da_dim = 0;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   // Either vacuum treatment is fine here. Under ghost-flux the vacuum inflow
   // rows are not BC rows, so the restriction sums them and the prolongation
   // corrects them like any interior row; under Dirichlet-cell they are masked
   // out of both halves as before

   comm_ = comm;
   n_angles_ = ps.n_angles;
   n_basis_ = ps.n_basis;
   local_cells_ = ps.local_cells;
   sum_weights_ = quad.sum_weights();
   w_d_ = quad.w_d();
   is_bc_row_d_ = disc.boundary_info().is_bc_row_d;
   PetscCall(ConsistentDOptions(comm, &consistent_d_, &consistent_power_));

   // A dof-1 twin of the backend's DMDA: same grid, same decomposition, one
   // unknown per cell. Its global vector holds the owned patch contiguously in
   // the patch's own lexicographic order - the local cell order the backends'
   // CheckDALayout asserts and the flux writer already relies on - so the
   // restriction and prolongation index it straight by local cell
   PetscCall(DMDACreateCompatibleDMDA(disc.dm(), 1, &da_));
   // pflare and the Kokkos kernels below both dispatch on the type, and unlike
   // the transport objects these DO come through the DM
   PetscCall(DMSetMatType(da_, MATAIJKOKKOS));
   PetscCall(DMSetVecType(da_, VECKOKKOS));

   PetscCall(DMDAGetInfo(da_, &da_dim, &n_cells_[0], &n_cells_[1], &n_cells_[2], NULL, NULL, \
      NULL, NULL, NULL, NULL, NULL, NULL, NULL));
   PetscCheck(da_dim == dim_, comm_, PETSC_ERR_ARG_INCOMP, \
      "the discretisation's DM is %" PetscInt_FMT "D but this DSAPrecon was created for %" \
      PetscInt_FMT "D", da_dim, dim_);

   // DMCreateMatrix here, unlike the transport matrix, and deliberately: the
   // objection there is upwind-specific (a DMDA star stencil preallocates every
   // point it reaches, where the upwind operator touches one neighbour per
   // axis), and the diffusion operator IS that star - the 3/5/7-point stencil
   // in 1D/2D/3D. So the DMDA's own preallocation is exactly right
   PetscCall(DMCreateMatrix(da_, &diff_mat_));
   PetscCall(MatSetOption(diff_mat_, MAT_SPD, PETSC_TRUE));

   // Persistent work vectors - never allocate inside an apply. d_local_ is
   // ghosted so the harmonic face means can read a neighbour rank's cell
   PetscCall(DMCreateGlobalVector(da_, &rhs_));
   PetscCall(VecDuplicate(rhs_, &sol_));
   PetscCall(VecDuplicate(rhs_, &d_global_));
   PetscCall(DMCreateLocalVector(da_, &d_local_));

   for (PetscInt d = 0; d < dim_; d++) {
      if (vacuum_lo_[d] || vacuum_hi_[d]) any_vacuum_ = PETSC_TRUE;
   }

   PetscCall(create_ksp());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPrecon::create_ksp()
{
   PC pc = NULL;

   PetscFunctionBeginUser;

   // The inner diffusion solve. The paper uses one BoomerAMG V-cycle; there is
   // no hypre in this build or the CI images, so the default is one PCGAMG
   // application. KSPSetFromOptions last, so -dsa_ksp_type, -dsa_pc_type hypre
   // and friends select anything at runtime
   PetscCall(KSPCreate(comm_, &ksp_));
   PetscCall(KSPSetOptionsPrefix(ksp_, "dsa_"));
   PetscCall(KSPSetType(ksp_, KSPPREONLY));
   PetscCall(KSPGetPC(ksp_, &pc));
   PetscCall(PCSetType(pc, PCGAMG));
   PetscCall(KSPSetOperators(ksp_, diff_mat_, diff_mat_));
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

// Refill the diffusion matrix for the current group
//
// Host assembly through MatSetValuesStencil, on purpose: this matrix is
// n_angles times smaller than the transport one and is refilled once per group,
// so its assembly cost is noise, and MatSetValuesStencil already knows the
// DMDA's patch-lexicographic global numbering and its ghost columns. A device
// COO path would need slot-map machinery of its own to buy nothing
PetscErrorCode DSAPrecon::assemble()
{
   PetscReal min_sigma_t = PETSC_MAX_REAL, max_sigma_a = 0.0;

   PetscFunctionBeginUser;

   PetscCheck(sigma_t_d_.extent(0) == (size_t)local_cells_ && \
      sigma_s_d_.extent(0) == (size_t)local_cells_, comm_, PETSC_ERR_ARG_INCOMP, \
      "the group xsections cover %" PetscInt_FMT " and %" PetscInt_FMT " cells but there are %" \
      PetscInt_FMT " local cells", (PetscInt)sigma_t_d_.extent(0), \
      (PetscInt)sigma_s_d_.extent(0), local_cells_);

   // Two cell-sized mirrors - the assembly is a host loop
   auto sigma_t_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sigma_t_d_);
   auto sigma_s_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sigma_s_d_);

   for (PetscInt c = 0; c < local_cells_; c++) {
      const PetscReal sigma_t = PetscRealPart(sigma_t_h(c));
      min_sigma_t = PetscMin(min_sigma_t, sigma_t);
      max_sigma_a = PetscMax(max_sigma_a, sigma_t - PetscRealPart(sigma_s_h(c)));
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &min_sigma_t, 1, MPIU_REAL, MPIU_MIN, comm_));
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &max_sigma_a, 1, MPIU_REAL, MPIU_MAX, comm_));

   // The void guard. D = 1/(3 sigma_t) is not defined in a void, and the fix is
   // a region-masked diffusion operator (future work, as in the paper) rather
   // than a fudged coefficient - so say so rather than produce a silent inf
   PetscCheck(min_sigma_t > 0.0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "the DSA preconditioner needs Sigma_t > 0 in every cell - D = 1/(3 Sigma_t) is not " \
      "defined in a void, and the smallest Sigma_t in this group is %g. Run without " \
      "-precon_dsa, or give the void a small Sigma_t", (double)min_sigma_t);

   // The singularity guard: with every face reflective the diffusion operator
   // is pure Neumann, so only the absorption keeps it nonsingular. That is the
   // same constraint the transport operator itself carries (see
   // docs/dev/testing.md) - all-reflect with a scattering ratio of exactly 1
   // has the constants in its kernel
   PetscCheck(any_vacuum_ || max_sigma_a > 0.0, comm_, PETSC_ERR_ARG_WRONGSTATE, \
      "the DSA diffusion operator is singular: every face is reflective (pure Neumann) and " \
      "Sigma_a = Sigma_t - Sigma_s is zero everywhere in this group. Leave one face vacuum, " \
      "or give the material absorption");

   if (plex_) PetscCall(assemble_plex(sigma_t_h.data(), sigma_s_h.data()));
   else PetscCall(assemble_structured(sigma_t_h.data(), sigma_s_h.data()));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DMDA star
PetscErrorCode DSAPrecon::assemble_structured(const PetscScalar *sigma_t_h, const PetscScalar *sigma_s_h)
{
   PetscInt xs = 0, ys = 0, zs = 0, xm = 1, ym = 1, zm = 1;
   PetscInt gxs = 0, gys = 0, gzs = 0, gxm = 1, gym = 1, gzm = 1;
   PetscScalar *d_global_a = nullptr;
   const PetscScalar *d_local_a = nullptr;

   PetscFunctionBeginUser;

   // D per cell, in local cell order, then ghosted so the harmonic means below
   // can reach a neighbour rank's cell
   PetscCall(VecGetArray(d_global_, &d_global_a));
   for (PetscInt c = 0; c < local_cells_; c++) d_global_a[c] = 1.0 / (3.0 * sigma_t_h[c]);
   PetscCall(VecRestoreArray(d_global_, &d_global_a));
   PetscCall(DMGlobalToLocal(da_, d_global_, INSERT_VALUES, d_local_));

   PetscCall(MatZeroEntries(diff_mat_));

   PetscCall(DMDAGetCorners(da_, &xs, &ys, &zs, &xm, &ym, &zm));
   PetscCall(DMDAGetGhostCorners(da_, &gxs, &gys, &gzs, &gxm, &gym, &gzm));
   PetscCall(VecGetArrayRead(d_local_, &d_local_a));

   // ONE loop for every dimension: an unused axis comes back from
   // DMDAGetCorners as start 0 and width 1, so the triple loop degenerates and
   // the per-axis face work below only runs for d < dim_. A dof-1 DMDA local
   // vector is lexicographic over the ghosted patch, which is what makes the
   // neighbour lookup plain index arithmetic
   for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
         for (PetscInt i = xs; i < xs + xm; i++) {

            const PetscInt ijk[3] = {i, j, k};
            // The local cell index the per-cell xsections are indexed by
            const PetscInt c = ((k - zs) * ym + (j - ys)) * xm + (i - xs);
            const PetscInt gc = ((k - gzs) * gym + (j - gys)) * gxm + (i - gxs);
            const PetscScalar d_c = d_local_a[gc];

            MatStencil row, col[7];
            PetscScalar v[7];
            PetscInt n_col = 0;

            row.i = i; row.j = j; row.k = k; row.c = 0;

            // The diagonal is built up as the faces are visited and written
            // last, so the entry count is known without a second pass
            PetscScalar diag = sigma_t_h[c] - sigma_s_h[c];

            for (PetscInt d = 0; d < dim_; d++) {
               const PetscScalar h = h_[d];

               for (PetscInt side = -1; side <= 1; side += 2) {
                  const PetscInt n = ijk[d] + side;

                  if (n >= 0 && n < n_cells_[d]) {

                     // Interior face: harmonic mean of the two cells' D, which
                     // is the flux-continuous face value and the reason a
                     // material interface does not need special-casing
                     PetscInt nijk[3] = {i, j, k};
                     nijk[d] = n;
                     const PetscInt gn = ((nijk[2] - gzs) * gym + (nijk[1] - gys)) * gxm \
                        + (nijk[0] - gxs);
                     const PetscScalar d_n = d_local_a[gn];
                     PetscScalar d_f = 2.0 * d_c * d_n / (d_c + d_n);
                     // The upwind scheme's numerical diffusion, m h, blended
                     // in (see the header) - a property of the face, so it
                     // goes on after the harmonic mean, not per cell
                     if (consistent_d_) d_f = BlendD(d_f, half_range_[d] * h, consistent_power_);

                     col[n_col].i = nijk[0]; col[n_col].j = nijk[1]; col[n_col].k = nijk[2];
                     col[n_col].c = 0;
                     v[n_col] = -d_f / (h * h);
                     n_col++;
                     diag += d_f / (h * h);
                  }
                  else if (side < 0 ? vacuum_lo_[d] : vacuum_hi_[d]) {

                     // Marshak (Robin) vacuum face: eliminating the face value
                     // from J = D_c (phi_c - phi_f) / (h/2) and J = phi_f / 2
                     // leaves an outgoing current proportional to phi_c alone
                     // with the same consistent D as an interior face
                     const PetscScalar d_b = consistent_d_ ? \
                        BlendD(d_c, half_range_[d] * h, consistent_power_) : d_c;
                     diag += 1.0 / (h * (2.0 + h / (2.0 * d_b)));
                  }
                  // A reflective face is zero Neumann: no current through it,
                  // so it contributes nothing at all
               }
            }

            col[n_col] = row;
            v[n_col] = diag;
            n_col++;

            PetscCall(MatSetValuesStencil(diff_mat_, 1, &row, n_col, col, v, INSERT_VALUES));
         }
      }
   }

   PetscCall(VecRestoreArrayRead(d_local_, &d_local_a));

   // The inner KSP re-does its setup on its own: KSPSolve sees the matrix state
   // change, the same mechanism the per-group PCAIR refill relies on
   PetscCall(MatAssemblyBegin(diff_mat_, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(diff_mat_, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The plex faces, volume-weighted (see the header): row c is
//    V_c sigma_a phi_c + sum_f T_f (phi_c - phi_n) + sum_vacuum M_f phi_c
// Host values into the COO pattern set at create - the same reasoning as the
// structured path's host stencil assembly, and MatSetValuesCOO bumps the
// matrix state so the inner KSP re-does its setup
PetscErrorCode DSAPrecon::assemble_plex(const PetscScalar *sigma_t_h, const PetscScalar *sigma_s_h)
{
   PetscScalar *d_global_a = nullptr;
   const PetscScalar *d_nb_a = nullptr;

   PetscFunctionBeginUser;

   PetscCall(VecGetArrayWrite(d_global_, &d_global_a));
   for (PetscInt c = 0; c < local_cells_; c++) d_global_a[c] = 1.0 / (3.0 * sigma_t_h[c]);
   PetscCall(VecRestoreArrayWrite(d_global_, &d_global_a));
   PetscCall(VecScatterBegin(nb_scatter_, d_global_, d_nb_, INSERT_VALUES, SCATTER_FORWARD));
   PetscCall(VecScatterEnd(nb_scatter_, d_global_, d_nb_, INSERT_VALUES, SCATTER_FORWARD));
   if (n_basis_ > 1) {
      PetscCall(assemble_plex_dg1(sigma_t_h, sigma_s_h));
      PetscFunctionReturn(PETSC_SUCCESS);
   }
   PetscCall(VecGetArrayRead(d_nb_, &d_nb_a));

   for (PetscInt c = 0; c < local_cells_; c++) {

      const PetscScalar d_c = 1.0 / (3.0 * sigma_t_h[c]);
      PetscScalar diag = volume_[c] * (sigma_t_h[c] - sigma_s_h[c]);
      PetscInt e = cell_entry_offset_[c];

      for (PetscInt k = cell_face_offset_[c]; k < cell_face_offset_[c + 1]; k++) {

         const PetscReal d_own = face_distance_[2 * k];
         if (face_nb_[k] >= 0) {

            // Two-point flux: continuity of the current across the face
            // between the two centroids gives the series resistance - the
            // harmonic face D on a uniform grid
            const PetscScalar d_n = d_nb_a[face_nb_[k]];
            PetscScalar t_f = face_area_[k] / (d_own / d_c + face_distance_[2 * k + 1] / d_n);
            if (consistent_d_) {
               // The structured blend, on the face's effective D over the
               // centroid-to-centroid length (on a box, the harmonic D and h)
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

   PetscCall(VecRestoreArrayRead(d_nb_, &d_nb_a));
   PetscCall(MatSetValuesCOO(diff_mat_, coo_v_.data(), INSERT_VALUES));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// DG1: the MIP form (see the header), row (c, i) tested with phi_i^c. Every
// face is visited from both of its cells and each visit writes only its own
// cell's rows, which is all a face contributes to them: v = phi_i^c has jump
// phi_i^c across the face (normal n out of c) and average gradient
// 1/2 D_c grad phi_i^c . n. With F the face matrices times V_c A_f,
//    F_own(i, j) = int_f phi_i^c phi_j^c,  F_up(i, j) = int_f phi_i^c phi_j^n,
// m_i = F_own(i, 0) = int_f phi_i^c, and g_j, gn_j this cell's and the
// neighbour's grad phi_j . n (constant on a cell), an interior face adds
//    own (c, j): kappa F_own(i, j) - 1/2 D_c g_j m_i - 1/2 D_c g_i F_own(0, j)
//    nbr (n, j): -kappa F_up(i, j) - 1/2 D_n gn_j m_i + 1/2 D_c g_i F_up(0, j)
// and a vacuum face the own block with kappa_b - the 1/2 is the MIP boundary
// form's, not the average's. The matrix is symmetric: the neighbour's visit
// writes the transpose of the nbr block (both normals and jumps flip). Volume
// terms: V_c (D_c grad phi_i . grad phi_j + sigma_a delta_ij), the basis being
// orthonormal in the volume average
PetscErrorCode DSAPrecon::assemble_plex_dg1(const PetscScalar *sigma_t_h, const PetscScalar *sigma_s_h)
{
   const PetscScalar *d_nb_a = nullptr;
   const PetscInt nb = n_basis_;
   // n_basis is at most 4 (3D), so fixed-size locals
   PetscScalar own[4][4], g[4], gn[4];

   PetscFunctionBeginUser;

   PetscCall(VecGetArrayRead(d_nb_, &d_nb_a));

   for (PetscInt c = 0; c < local_cells_; c++) {

      const PetscScalar d_c = 1.0 / (3.0 * sigma_t_h[c]);
      const PetscScalar sigma_a = sigma_t_h[c] - sigma_s_h[c];
      const PetscReal vol = volume_[c];

      for (PetscInt i = 0; i < nb; i++) {
         for (PetscInt j = 0; j < nb; j++) {
            PetscScalar grad_dot = 0.0;
            for (PetscInt d = 0; d < 3; d++) grad_dot += basis_grad_[(c * nb + i) * 3 + d] * basis_grad_[(c * nb + j) * 3 + d];
            own[i][j] = vol * (d_c * grad_dot + (i == j ? sigma_a : (PetscScalar)0.0));
         }
      }

      // The neighbour blocks are written straight into their COO entries: in
      // row i's run, the q-th interior face of this cell (cone order) owns
      // entries [q * nb, (q + 1) * nb), and the own block comes last
      const PetscInt row_len = (cell_entry_offset_[c + 1] - cell_entry_offset_[c]) / nb;
      const PetscInt n_interior = row_len / nb - 1;
      PetscInt q = 0;

      for (PetscInt k = cell_face_offset_[c]; k < cell_face_offset_[c + 1]; k++) {

         const PetscBool interior = (PetscBool)(face_nb_[k] >= 0);
         if (!interior && !face_vacuum_[k]) continue;

         const PetscReal scale = vol * face_area_[k];
         const PetscScalar *f_own = &face_own_[k * nb * nb];
         const PetscScalar *f_up = &face_up_[k * nb * nb];
         for (PetscInt j = 0; j < nb; j++) {
            g[j] = 0.0;
            gn[j] = 0.0;
            for (PetscInt d = 0; d < 3; d++) {
               g[j] += basis_grad_[(c * nb + j) * 3 + d] * face_normal_[3 * k + d];
               gn[j] += face_nb_grad_[(k * nb + j) * 3 + d] * face_normal_[3 * k + d];
            }
         }

         const PetscReal h_c = 2.0 * face_distance_[2 * k];
         if (interior) {

            const PetscScalar d_n = d_nb_a[face_nb_[k]];
            const PetscReal h_n = 2.0 * face_distance_[2 * k + 1];
            const PetscReal kappa = PetscMax(0.5 * mip_penalty_ * PetscRealPart(d_c / h_c + d_n / h_n), 0.25);

            for (PetscInt i = 0; i < nb; i++) {
               const PetscScalar m_i = scale * f_own[i * nb];
               PetscScalar *nbr = &coo_v_[cell_entry_offset_[c] + i * row_len + q * nb];
               for (PetscInt j = 0; j < nb; j++) {
                  own[i][j] += kappa * scale * f_own[i * nb + j] - 0.5 * d_c * g[j] * m_i \
                     - 0.5 * d_c * g[i] * scale * f_own[j];
                  nbr[j] = -kappa * scale * f_up[i * nb + j] - 0.5 * d_n * gn[j] * m_i \
                     + 0.5 * d_c * g[i] * scale * f_up[j];
               }
            }
            q++;
         }
         else {

            const PetscReal kappa = PetscMax(mip_penalty_ * PetscRealPart(d_c) / h_c, 0.25);
            for (PetscInt i = 0; i < nb; i++) {
               const PetscScalar m_i = scale * f_own[i * nb];
               for (PetscInt j = 0; j < nb; j++) {
                  own[i][j] += kappa * scale * f_own[i * nb + j] - 0.5 * d_c * g[j] * m_i \
                     - 0.5 * d_c * g[i] * scale * f_own[j];
               }
            }
         }
      }

      for (PetscInt i = 0; i < nb; i++) {
         for (PetscInt j = 0; j < nb; j++) coo_v_[cell_entry_offset_[c] + i * row_len + n_interior * nb + j] = own[i][j];
      }
   }

   PetscCall(VecRestoreArrayRead(d_nb_, &d_nb_a));
   PetscCall(MatSetValuesCOO(diff_mat_, coo_v_.data(), INSERT_VALUES));

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

   // Restrict: the masked 0th moment of the residual onto the cells
   {
      PetscScalarConstKokkosView x_d;
      PetscScalarKokkosView rhs_d;
      PetscCall(VecGetKokkosView(x, &x_d));
      PetscCall(VecGetKokkosViewWrite(rhs_, &rhs_d));
      RestrictKernel(x_d, rhs_d, w_d_, is_bc_row_d_, n_angles_, local_cells_ * n_basis_);
      PetscCall(VecRestoreKokkosViewWrite(rhs_, &rhs_d));
      PetscCall(VecRestoreKokkosView(x, &x_d));
   }
   // The plex matrix is volume-weighted, so its rhs is too
   if (volume_vec_) PetscCall(VecPointwiseMult(rhs_, rhs_, volume_vec_));

   // Invert the diffusion operator, inexactly
   PetscCall(KSPSolve(ksp_, rhs_, sol_));

   // Prolong: the scalar correction back onto every ordinate
   {
      PetscScalarConstKokkosView sol_d;
      PetscScalarKokkosView y_d;
      PetscCall(VecGetKokkosView(sol_, &sol_d));
      PetscCall(VecGetKokkosViewWrite(y, &y_d));
      ProlongKernel(sol_d, y_d, is_bc_row_d_, n_angles_, sum_weights_, local_rows);
      PetscCall(VecRestoreKokkosViewWrite(y, &y_d));
      PetscCall(VecRestoreKokkosView(sol_, &sol_d));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Destroying a NULL PETSc handle is a no-op, so this is safe on a never-created
// object too. The xsection views are not ours to drop
PetscErrorCode DSAPrecon::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(KSPDestroy(&ksp_));
   PetscCall(MatDestroy(&diff_mat_));
   PetscCall(VecDestroy(&rhs_));
   PetscCall(VecDestroy(&sol_));
   PetscCall(VecDestroy(&d_global_));
   PetscCall(VecDestroy(&d_local_));
   PetscCall(VecDestroy(&volume_vec_));
   PetscCall(VecDestroy(&d_nb_));
   PetscCall(VecScatterDestroy(&nb_scatter_));
   PetscCall(DMDestroy(&da_));

   PetscFunctionReturn(PETSC_SUCCESS);
}
