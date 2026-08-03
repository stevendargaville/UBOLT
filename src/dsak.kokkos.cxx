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
static void RestrictKernel(PetscScalarConstKokkosView x_d, PetscScalarKokkosView rhs_d, \
   PetscScalar2DKokkosView w_d, PetscIntKokkosView is_bc_row_d, PetscInt n_angles, \
   PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_cells, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // cell
         const PetscInt c = t.league_rank();

         PetscScalar moment = 0.0;
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt a, PetscScalar &acc) {

               const PetscInt r = c * n_angles + a;
               if (!is_bc_row_d(r)) acc += w_d(a, 0) * x_d(r);
            }, moment);

         Kokkos::single(Kokkos::PerTeam(t), [&]() { rhs_d(c) = moment; });
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// P_angle: broadcast the scalar correction isotropically over the ordinates
//
// The / sum_weights is the consistency scaling (see the header). Every row is
// written, so y needs no zeroing beforehand; the BC rows get an explicit zero,
// which is what the BC contract owes them - the assembled operator already
// holds the boundary condition on those rows and a correction there would
// pollute it
static void ProlongKernel(PetscScalarConstKokkosView sol_d, PetscScalarKokkosView y_d, \
   PetscIntKokkosView is_bc_row_d, PetscInt n_angles, PetscScalar sum_weights, \
   PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         y_d(r) = is_bc_row_d(r) ? (PetscScalar)0.0 : sol_d(r / n_angles) / sum_weights;
      });
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

   PetscCall(create_common(comm, ps, disc, quad));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAPrecon::create_common(MPI_Comm comm, const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad)
{
   PC pc = NULL;
   PetscInt da_dim = 0;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);

   comm_ = comm;
   n_angles_ = ps.n_angles;
   local_cells_ = ps.local_cells;
   sum_weights_ = quad.sum_weights();
   w_d_ = quad.w_d();
   is_bc_row_d_ = disc.boundary_info().is_bc_row_d;

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
   PetscInt xs = 0, ys = 0, zs = 0, xm = 1, ym = 1, zm = 1;
   PetscInt gxs = 0, gys = 0, gzs = 0, gxm = 1, gym = 1, gzm = 1;
   PetscScalar *d_global_a = nullptr;
   const PetscScalar *d_local_a = nullptr;
   PetscReal min_sigma_t = PETSC_MAX_REAL, max_sigma_a = 0.0;
   PetscBool any_vacuum = PETSC_FALSE;

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

   for (PetscInt d = 0; d < dim_; d++) {
      if (vacuum_lo_[d] || vacuum_hi_[d]) any_vacuum = PETSC_TRUE;
   }
   // The singularity guard: with every face reflective the diffusion operator
   // is pure Neumann, so only the absorption keeps it nonsingular. That is the
   // same constraint the transport operator itself carries (see
   // docs/dev/testing.md) - all-reflect with a scattering ratio of exactly 1
   // has the constants in its kernel
   PetscCheck(any_vacuum || max_sigma_a > 0.0, comm_, PETSC_ERR_ARG_WRONGSTATE, \
      "the DSA diffusion operator is singular: every face is reflective (pure Neumann) and " \
      "Sigma_a = Sigma_t - Sigma_s is zero everywhere in this group. Leave one face vacuum, " \
      "or give the material absorption");

   // D per cell, in local cell order, then ghosted so the harmonic means below
   // can reach a neighbour rank's cell
   PetscCall(VecGetArray(d_global_, &d_global_a));
   for (PetscInt c = 0; c < local_cells_; c++) d_global_a[c] = 1.0 / (3.0 * sigma_t_h(c));
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
            PetscScalar diag = sigma_t_h(c) - sigma_s_h(c);

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
                     const PetscScalar d_f = 2.0 * d_c * d_n / (d_c + d_n);

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
                     diag += 1.0 / (h * (2.0 + h / (2.0 * d_c)));
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

// y = P D_diff^-1 R x
PetscErrorCode DSAPrecon::apply(Vec x, Vec y)
{
   PetscInt local_rows = 0;

   PetscFunctionBeginUser;

   PetscCall(VecGetLocalSize(x, &local_rows));
   PetscCheck(local_rows == local_cells_ * n_angles_, comm_, PETSC_ERR_ARG_INCOMP, \
      "x has %" PetscInt_FMT " local rows but this DSAPrecon covers %" PetscInt_FMT, \
      local_rows, local_cells_ * n_angles_);

   // Restrict: the masked 0th moment of the residual onto the cells
   {
      PetscScalarConstKokkosView x_d;
      PetscScalarKokkosView rhs_d;
      PetscCall(VecGetKokkosView(x, &x_d));
      PetscCall(VecGetKokkosViewWrite(rhs_, &rhs_d));
      RestrictKernel(x_d, rhs_d, w_d_, is_bc_row_d_, n_angles_, local_cells_);
      PetscCall(VecRestoreKokkosViewWrite(rhs_, &rhs_d));
      PetscCall(VecRestoreKokkosView(x, &x_d));
   }

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
   PetscCall(DMDestroy(&da_));

   PetscFunctionReturn(PETSC_SUCCESS);
}
