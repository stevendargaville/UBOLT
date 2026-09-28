#include "dsa_operatork.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"
#include <petscdmda.h>

// The structured backends' DSA operator: the cell-centred star on a dof-1
// twin of the backend's DMDA, written once for 1/2/3D with the unused axes
// degenerate (start 0, width 1, spacing 0)
//
// Host assembly through MatSetValuesStencil, on purpose: this matrix is
// n_angles times smaller than the transport one and refilled once per group,
// and MatSetValuesStencil already knows the DMDA's patch-lexicographic global
// numbering and its ghost columns. A device COO path would buy nothing
class DSAStructured final : public DSAOperator {
public:
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc);

   PetscInt n_units() const override { return local_cells_; }
   PetscErrorCode void_volume_surface(const PetscInt *is_void, PetscReal vs[2]) override;
   PetscErrorCode assemble(const DSAGroup &g) override;

   // Filled by the per-dimension builders before create(): the spacings, the
   // vacuum (Marshak) faces per axis - a reflective face is zero Neumann, no
   // contribution at all - and the quadrature's half-range current per axis
   PetscInt dim = 0;
   PetscScalar h[3] = {0.0, 0.0, 0.0};
   PetscBool vacuum_lo[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
   PetscBool vacuum_hi[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
   PetscReal half_range[3] = {0.0, 0.0, 0.0};

protected:
   PetscErrorCode destroy_own() override;

private:
   // Stage a per-cell value and ghost it, so a face can read a neighbour
   // rank's cell; returns the ghosted array (restore with VecRestoreArrayRead)
   PetscErrorCode stage(const PetscScalar *v, const PetscScalar **local);

   MPI_Comm comm_ = MPI_COMM_NULL;
   PetscInt local_cells_ = 0;
   PetscInt n_cells_[3] = {1, 1, 1};
   DM da_ = NULL;
   // Host Vecs (created before the DM's vec type is set): the staged values
   // are produced and consumed on the host, so the ghost exchange runs there
   Vec d_global_ = NULL, d_local_ = NULL;
   PetscBool consistent_d_ = PETSC_TRUE;
   PetscReal consistent_power_ = 1.5;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAStructured::create(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc)
{
   PetscInt da_dim = 0;

   PetscFunctionBeginUser;

   comm_ = comm;
   local_cells_ = ps.local_cells;
   PetscCall(ConsistentDOptions(comm, &consistent_d_, &consistent_power_));

   // Same grid, same decomposition, one unknown per cell. Its global vector
   // holds the owned patch in the patch's lexicographic order - the local cell
   // order the backends' CheckDALayout asserts - so R and P index it by cell
   PetscCall(DMDACreateCompatibleDMDA(disc.dm(), 1, &da_));
   PetscCall(DMCreateGlobalVector(da_, &d_global_));
   PetscCall(DMCreateLocalVector(da_, &d_local_));
   // pflare and the Kokkos kernels dispatch on the type, and unlike the
   // transport objects the solver's DO come through this DM
   PetscCall(DMSetMatType(da_, MATAIJKOKKOS));
   PetscCall(DMSetVecType(da_, VECKOKKOS));

   PetscCall(DMDAGetInfo(da_, &da_dim, &n_cells_[0], &n_cells_[1], &n_cells_[2], NULL, NULL, \
      NULL, NULL, NULL, NULL, NULL, NULL, NULL));
   PetscCheck(da_dim == dim, comm_, PETSC_ERR_ARG_INCOMP, \
      "the discretisation's DM is %" PetscInt_FMT "D but this DSAPrecon was created for %" \
      PetscInt_FMT "D", da_dim, dim);

   // DMCreateMatrix here, unlike the transport matrix: the objection there is
   // upwind-specific (the star preallocates more than one upwind neighbour per
   // axis), and the diffusion operator IS the 3/5/7-point star
   PetscCall(DMCreateMatrix(da_, &mat));
   PetscCall(MatSetOption(mat, MAT_SPD, PETSC_TRUE));
   PetscCall(DMCreateGlobalVector(da_, &rhs));
   PetscCall(VecDuplicate(rhs, &sol));

   for (PetscInt d = 0; d < dim; d++) {
      if (vacuum_lo[d] || vacuum_hi[d]) any_vacuum = PETSC_TRUE;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAStructured::destroy_own()
{
   PetscFunctionBeginUser;

   PetscCall(VecDestroy(&d_global_));
   PetscCall(VecDestroy(&d_local_));
   PetscCall(DMDestroy(&da_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAStructured::stage(const PetscScalar *v, const PetscScalar **local)
{
   PetscScalar *global = nullptr;

   PetscFunctionBeginUser;

   PetscCall(VecGetArrayWrite(d_global_, &global));
   for (PetscInt c = 0; c < local_cells_; c++) global[c] = v[c];
   PetscCall(VecRestoreArrayWrite(d_global_, &global));
   PetscCall(DMGlobalToLocal(da_, d_global_, INSERT_VALUES, d_local_));
   PetscCall(VecGetArrayRead(d_local_, local));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode DSAStructured::void_volume_surface(const PetscInt *is_void, PetscReal vs[2])
{
   PetscInt xs = 0, ys = 0, zs = 0, xm = 1, ym = 1, zm = 1;
   PetscInt gxs = 0, gys = 0, gzs = 0, gxm = 1, gym = 1, gzm = 1;
   const PetscScalar *flag_local = nullptr;
   std::vector<PetscScalar> flag(local_cells_);

   PetscFunctionBeginUser;

   for (PetscInt c = 0; c < local_cells_; c++) flag[c] = StageVoidFlag(is_void[c]);
   PetscCall(stage(flag.data(), &flag_local));
   PetscCall(DMDAGetCorners(da_, &xs, &ys, &zs, &xm, &ym, &zm));
   PetscCall(DMDAGetGhostCorners(da_, &gxs, &gys, &gzs, &gxm, &gym, &gzm));

   // The cell volume, and each axis' face area, over the dim axes only
   PetscReal vol = 1.0, area[3] = {1.0, 1.0, 1.0};
   for (PetscInt d = 0; d < dim; d++) {
      vol *= PetscRealPart(h[d]);
      for (PetscInt e = 0; e < dim; e++) {
         if (e != d) area[d] *= PetscRealPart(h[e]);
      }
   }

   for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
         for (PetscInt i = xs; i < xs + xm; i++) {
            const PetscInt c = ((k - zs) * ym + (j - ys)) * xm + (i - xs);
            if (!is_void[c]) continue;
            vs[0] += vol;
            const PetscInt ijk[3] = {i, j, k};
            for (PetscInt d = 0; d < dim; d++) {
               for (PetscInt side = -1; side <= 1; side += 2) {
                  const PetscInt n = ijk[d] + side;
                  PetscBool ends = PETSC_FALSE;
                  if (n >= 0 && n < n_cells_[d]) {
                     PetscInt nijk[3] = {i, j, k};
                     nijk[d] = n;
                     const PetscInt gn = ((nijk[2] - gzs) * gym + (nijk[1] - gys)) * gxm + (nijk[0] - gxs);
                     ends = (PetscBool)!StagedVoidFlag(flag_local[gn]);
                  }
                  else ends = side < 0 ? vacuum_lo[d] : vacuum_hi[d];
                  if (ends) vs[1] += area[d];
               }
            }
         }
      }
   }
   PetscCall(VecRestoreArrayRead(d_local_, &flag_local));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DMDA star, one loop for every dimension: an unused axis comes back from
// DMDAGetCorners as start 0, width 1, and the face work only runs for d < dim
PetscErrorCode DSAStructured::assemble(const DSAGroup &g)
{
   PetscInt xs = 0, ys = 0, zs = 0, xm = 1, ym = 1, zm = 1;
   PetscInt gxs = 0, gys = 0, gzs = 0, gxm = 1, gym = 1, gzm = 1;
   const PetscScalar *d_local_a = nullptr;
   std::vector<PetscScalar> staged(local_cells_);

   PetscFunctionBeginUser;

   for (PetscInt c = 0; c < local_cells_; c++) staged[c] = StageD(g.d[c], (PetscBool)(g.bridged && g.is_void[c]));
   PetscCall(stage(staged.data(), &d_local_a));

   PetscCall(MatZeroEntries(mat));
   PetscCall(DMDAGetCorners(da_, &xs, &ys, &zs, &xm, &ym, &zm));
   PetscCall(DMDAGetGhostCorners(da_, &gxs, &gys, &gzs, &gxm, &gym, &gzm));

   for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
         for (PetscInt i = xs; i < xs + xm; i++) {

            const PetscInt ijk[3] = {i, j, k};
            // The local cell the per-cell xsections are indexed by, and its
            // place in the ghosted patch
            const PetscInt c = ((k - zs) * ym + (j - ys)) * xm + (i - xs);
            const PetscInt gc = ((k - gzs) * gym + (j - gys)) * gxm + (i - gxs);
            const PetscScalar d_c = StagedD(d_local_a[gc]);

            MatStencil row, col[7];
            PetscScalar v[7];
            PetscInt n_col = 0;

            row.i = i; row.j = j; row.k = k; row.c = 0;

            // A masked void is an identity row, decoupled both ways (its
            // neighbours skip it too), so the matrix stays symmetric
            if (g.masked[c]) {
               v[0] = 1.0;
               PetscCall(MatSetValuesStencil(mat, 1, &row, 1, &row, v, INSERT_VALUES));
               continue;
            }

            // Built up as the faces are visited and written last
            PetscScalar diag = g.sigma_t_h[c] - g.sigma_s_h[c];

            for (PetscInt d = 0; d < dim; d++) {
               const PetscScalar hd = h[d];

               for (PetscInt side = -1; side <= 1; side += 2) {
                  const PetscInt n = ijk[d] + side;
                  const PetscBool in_grid = (PetscBool)(n >= 0 && n < n_cells_[d]);
                  PetscInt nijk[3] = {i, j, k};
                  PetscScalar d_n = 0.0;
                  if (in_grid) {
                     nijk[d] = n;
                     const PetscInt gn = ((nijk[2] - gzs) * gym + (nijk[1] - gys)) * gxm \
                        + (nijk[0] - gxs);
                     d_n = d_local_a[gn];
                  }

                  if (in_grid && !StagedMasked(d_n)) {

                     // Interior face: the harmonic mean, which is flux
                     // continuous, so a material interface needs nothing
                     // special; then the face's m h blended in
                     d_n = StagedD(d_n);
                     PetscScalar d_f = 2.0 * d_c * d_n / (d_c + d_n);
                     if (consistent_d_) d_f = BlendD(d_f, half_range[d] * hd, consistent_power_);

                     col[n_col].i = nijk[0]; col[n_col].j = nijk[1]; col[n_col].k = nijk[2];
                     col[n_col].c = 0;
                     v[n_col] = -d_f / (hd * hd);
                     n_col++;
                     diag += d_f / (hd * hd);
                  }
                  else if (in_grid || (side < 0 ? vacuum_lo[d] : vacuum_hi[d])) {

                     // Marshak: eliminating the face value from J = D (phi_c -
                     // phi_f) / (h/2) and J = phi_f / 2. A face into a masked
                     // void takes it too, blend included, and no coupling
                     const PetscScalar d_b = consistent_d_ ? \
                        BlendD(d_c, half_range[d] * hd, consistent_power_) : d_c;
                     diag += 1.0 / (hd * (2.0 + hd / (2.0 * d_b)));
                  }
               }
            }

            col[n_col] = row;
            v[n_col] = diag;
            n_col++;

            PetscCall(MatSetValuesStencil(mat, 1, &row, n_col, col, v, INSERT_VALUES));
         }
      }
   }

   PetscCall(VecRestoreArrayRead(d_local_, &d_local_a));

   // The state change makes the inner KSP redo its setup at the next solve
   PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The per-dimension builders: the spacings, the face families and the
// half-range currents off the concrete backend and SN set
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscReal AxisHalfRangeCurrent(const std::vector<PetscReal> &w_h, PetscReal sum_weights, \
   const PetscScalar *cos, PetscInt n_angles)
{
   std::vector<PetscReal> cos_h(n_angles);
   for (PetscInt a = 0; a < n_angles; a++) cos_h[a] = PetscRealPart(cos[a]);
   return HalfRangeCurrent(w_h, sum_weights, cos_h);
}

static inline PetscBool IsVacuum(const BCSpec &bcs, PetscInt face)
{
   return (PetscBool)(bcs.type(face) == BCType::VACUUM);
}

PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD1D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op)
{
   DSAStructured *s = new DSAStructured;
   std::vector<PetscReal> w_h;

   PetscFunctionBeginUser;

   *op = s;
   s->dim = 1;
   s->h[0] = disc.dx();
   s->vacuum_lo[0] = IsVacuum(bcs, StructuredFD1D::FACE_LEFT);
   s->vacuum_hi[0] = IsVacuum(bcs, StructuredFD1D::FACE_RIGHT);

   const SNQuadrature *sn = dynamic_cast<const SNQuadrature *>(&quad);
   PetscCheck(sn, comm, PETSC_ERR_ARG_WRONG, "the 1D DSA needs the 1D SNQuadrature");
   DSAHostWeights(quad, w_h);
   const PetscReal sw = PetscRealPart(quad.sum_weights());
   s->half_range[0] = AxisHalfRangeCurrent(w_h, sw, sn->mu_host(), quad.n_angles());

   PetscCall(s->create(comm, ps, disc));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// CAREFUL: in 2D bottom/top are the Y faces
PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD2D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op)
{
   DSAStructured *s = new DSAStructured;
   std::vector<PetscReal> w_h;

   PetscFunctionBeginUser;

   *op = s;
   s->dim = 2;
   s->h[0] = disc.dx();
   s->h[1] = disc.dy();
   s->vacuum_lo[0] = IsVacuum(bcs, StructuredFD2D::FACE_LEFT);
   s->vacuum_hi[0] = IsVacuum(bcs, StructuredFD2D::FACE_RIGHT);
   s->vacuum_lo[1] = IsVacuum(bcs, StructuredFD2D::FACE_BOTTOM);
   s->vacuum_hi[1] = IsVacuum(bcs, StructuredFD2D::FACE_TOP);

   const SNQuadrature2D *sn = dynamic_cast<const SNQuadrature2D *>(&quad);
   PetscCheck(sn, comm, PETSC_ERR_ARG_WRONG, "the 2D DSA needs the SNQuadrature2D");
   DSAHostWeights(quad, w_h);
   const PetscReal sw = PetscRealPart(quad.sum_weights());
   s->half_range[0] = AxisHalfRangeCurrent(w_h, sw, sn->mu_host(), quad.n_angles());
   s->half_range[1] = AxisHalfRangeCurrent(w_h, sw, sn->eta_host(), quad.n_angles());

   PetscCall(s->create(comm, ps, disc));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// CAREFUL: PETSc's box convention puts bottom/top on Z in 3D, so the Y faces
// are front (y-min) and back (y-max) - NOT what 2D calls them
PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD3D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op)
{
   DSAStructured *s = new DSAStructured;
   std::vector<PetscReal> w_h;

   PetscFunctionBeginUser;

   *op = s;
   s->dim = 3;
   s->h[0] = disc.dx();
   s->h[1] = disc.dy();
   s->h[2] = disc.dz();
   s->vacuum_lo[0] = IsVacuum(bcs, StructuredFD3D::FACE_LEFT);
   s->vacuum_hi[0] = IsVacuum(bcs, StructuredFD3D::FACE_RIGHT);
   s->vacuum_lo[1] = IsVacuum(bcs, StructuredFD3D::FACE_FRONT);
   s->vacuum_hi[1] = IsVacuum(bcs, StructuredFD3D::FACE_BACK);
   s->vacuum_lo[2] = IsVacuum(bcs, StructuredFD3D::FACE_BOTTOM);
   s->vacuum_hi[2] = IsVacuum(bcs, StructuredFD3D::FACE_TOP);

   const SNQuadrature3D *sn = dynamic_cast<const SNQuadrature3D *>(&quad);
   PetscCheck(sn, comm, PETSC_ERR_ARG_WRONG, "the 3D DSA needs the SNQuadrature3D");
   DSAHostWeights(quad, w_h);
   const PetscReal sw = PetscRealPart(quad.sum_weights());
   s->half_range[0] = AxisHalfRangeCurrent(w_h, sw, sn->mu_host(), quad.n_angles());
   s->half_range[1] = AxisHalfRangeCurrent(w_h, sw, sn->eta_host(), quad.n_angles());
   s->half_range[2] = AxisHalfRangeCurrent(w_h, sw, sn->xi_host(), quad.n_angles());

   PetscCall(s->create(comm, ps, disc));

   PetscFunctionReturn(PETSC_SUCCESS);
}
