#include "ubolt/block_inverse.hpp"
#include <petscmat_kokkos.hpp>
#include <type_traits>

// The kernels below capture plain values and shallow view copies, never `this`
// - a member access inside a KOKKOS_LAMBDA would dereference a host pointer on
// the device. Free functions rather than lambdas in the methods for the same
// reason the rest of the library keeps them free: an extended device lambda
// can't be defined inside another lambda

static constexpr PetscInt MAXB = ElementBlockInverse::MAX_BLOCK;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A MATSEQAIJKOKKOS matrix's CSR on the device: row offsets and columns
// (unmanaged, PETSc owns them) and the number of rows
struct DeviceCSR {
   PetscIntConstKokkosViewUnmanaged i;
   PetscIntConstKokkosViewUnmanaged j;
   PetscInt n_rows = 0;
   PetscInt nnz = 0;
};

static PetscErrorCode GetDeviceCSR(Mat A, DeviceCSR &csr)
{
   const PetscInt *i = nullptr, *j = nullptr;
   PetscMemType mtype;
   PetscInt n_cols = 0;

   PetscFunctionBeginUser;

   PetscCall(MatGetSize(A, &csr.n_rows, &n_cols));
   PetscCall(MatSeqAIJGetCSRAndMemType(A, &i, &j, NULL, &mtype));
   // A host-only memory type means this is not a Kokkos matrix, and the
   // kernels below would read host pointers on the device
   PetscCheck(PetscMemTypeDevice(mtype) || (std::is_same<DefaultMemorySpace, Kokkos::HostSpace>::value), \
      PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ElementBlockInverse needs the CSR on the device");
   csr.i = PetscIntConstKokkosViewUnmanaged(i, csr.n_rows + 1);
   // The last row offset is the nonzero count - read it back (one value)
   Kokkos::View<PetscInt, Kokkos::HostSpace> nnz_h("nnz_h");
   Kokkos::deep_copy(nnz_h, Kokkos::subview(csr.i, csr.n_rows));
   csr.nnz = nnz_h();
   csr.j = PetscIntConstKokkosViewUnmanaged(j, csr.nnz);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The diagonal (owned) block of A, and the off-process one if A is MPI (NULL
// otherwise). Every element block lies in the diagonal part
static PetscErrorCode SplitLocal(Mat A, Mat *Ad, Mat *Ao)
{
   PetscBool mpi = PETSC_FALSE, seq = PETSC_FALSE;

   PetscFunctionBeginUser;

   PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJKOKKOS, &mpi));
   PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJKOKKOS, &seq));
   PetscCheck(mpi || seq, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, \
      "ElementBlockInverse reads the blocks off a MATAIJKOKKOS matrix");
   if (mpi) PetscCall(MatMPIAIJGetSeqAIJ(A, Ad, Ao, NULL));
   else {
      *Ad = A;
      *Ao = NULL;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Read block (c, a) of the owned CSR and invert it into inv_d by Gauss-Jordan
// with partial pivoting. The owned block's columns are local indices over the
// same layout as the rows, so a column is block (c, a)'s basis j exactly when
// col = (c * nb + j) * n_angles + a. Anything else in the row is a coupling to
// another block and is not read here. With has_diag, the block's diagonal is
// taken from diag_d instead of the matrix (see setup). Returns the number of
// singular blocks
static PetscInt InvertBlocksKernel(PetscScalarKokkosView inv_d, PetscIntConstKokkosViewUnmanaged ad_i, \
   PetscIntConstKokkosViewUnmanaged ad_j, PetscScalarMatConstKokkosView ad_a, PetscScalarConstKokkosView diag_d, \
   PetscBool has_diag, PetscInt n_angles, PetscInt nb, PetscInt n_blocks)
{
   PetscInt n_singular = 0;

   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(0, n_blocks), KOKKOS_LAMBDA(PetscInt b, PetscInt &singular) {

         const PetscInt c = b / n_angles;
         const PetscInt a = b % n_angles;

         // [m | w] = [B | I], reduced to [I | B^{-1}]
         PetscScalar m[MAXB][MAXB], w[MAXB][MAXB];
         for (PetscInt i = 0; i < nb; i++) {
            for (PetscInt j = 0; j < nb; j++) {
               m[i][j] = 0.0;
               w[i][j] = (i == j) ? 1.0 : 0.0;
            }
            const PetscInt r = (c * nb + i) * n_angles + a;
            for (PetscInt p = ad_i(r); p < ad_i(r + 1); p++) {
               const PetscInt col = ad_j(p);
               if (col % n_angles != a || col / (nb * n_angles) != c) continue;
               m[i][(col / n_angles) % nb] = ad_a(p);
            }
            if (has_diag) m[i][i] = diag_d(r);
         }

         PetscBool ok = PETSC_TRUE;
         for (PetscInt k = 0; k < nb && ok; k++) {
            PetscInt piv = k;
            for (PetscInt i = k + 1; i < nb; i++) {
               if (PetscAbsScalar(m[i][k]) > PetscAbsScalar(m[piv][k])) piv = i;
            }
            if (PetscAbsScalar(m[piv][k]) == 0.0) {
               ok = PETSC_FALSE;
               break;
            }
            if (piv != k) {
               for (PetscInt j = 0; j < nb; j++) {
                  PetscScalar t = m[k][j]; m[k][j] = m[piv][j]; m[piv][j] = t;
                  t = w[k][j]; w[k][j] = w[piv][j]; w[piv][j] = t;
               }
            }
            const PetscScalar inv_piv = 1.0 / m[k][k];
            for (PetscInt j = 0; j < nb; j++) {
               m[k][j] *= inv_piv;
               w[k][j] *= inv_piv;
            }
            for (PetscInt i = 0; i < nb; i++) {
               if (i == k) continue;
               const PetscScalar f = m[i][k];
               if (f == 0.0) continue;
               for (PetscInt j = 0; j < nb; j++) {
                  m[i][j] -= f * m[k][j];
                  w[i][j] -= f * w[k][j];
               }
            }
         }
         if (!ok) singular++;

         for (PetscInt i = 0; i < nb; i++) {
            for (PetscInt j = 0; j < nb; j++) inv_d((b * nb + i) * nb + j) = ok ? w[i][j] : (PetscScalar)0.0;
         }
      }, n_singular);

   return n_singular;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// out(r_i, :) = sum_j inv(i, j) in(r_j, :) over one CSR (the owned or the
// off-process part), block by block. Relies on every row of a block holding
// the same columns in the same order - checked here, per block, before
// anything is written: a block whose rows disagree is left alone and counted,
// and the caller errors on a non-zero count
static PetscInt ScaleRowsKernel(PetscScalarMatKokkosView out_a, PetscIntConstKokkosViewUnmanaged csr_i, \
   PetscIntConstKokkosViewUnmanaged csr_j, PetscScalarMatConstKokkosView in_a, PetscScalarKokkosView inv_d, \
   PetscInt n_angles, PetscInt nb, PetscInt n_blocks)
{
   PetscInt n_mismatch = 0;

   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(0, n_blocks), KOKKOS_LAMBDA(PetscInt b, PetscInt &mismatch) {

         const PetscInt c = b / n_angles;
         const PetscInt a = b % n_angles;
         PetscInt start[MAXB];
         for (PetscInt i = 0; i < nb; i++) start[i] = csr_i((c * nb + i) * n_angles + a);
         const PetscInt r0 = c * nb * n_angles + a;
         const PetscInt len = csr_i(r0 + 1) - csr_i(r0);

         for (PetscInt i = 1; i < nb; i++) {
            const PetscInt r = (c * nb + i) * n_angles + a;
            if (csr_i(r + 1) - csr_i(r) != len) {
               mismatch++;
               return;
            }
            for (PetscInt p = 0; p < len; p++) {
               if (csr_j(start[i] + p) != csr_j(start[0] + p)) {
                  mismatch++;
                  return;
               }
            }
         }

         for (PetscInt p = 0; p < len; p++) {
            PetscScalar v[MAXB];
            for (PetscInt j = 0; j < nb; j++) v[j] = in_a(start[j] + p);
            for (PetscInt i = 0; i < nb; i++) {
               PetscScalar sum = 0.0;
               for (PetscInt j = 0; j < nb; j++) sum += inv_d((b * nb + i) * nb + j) * v[j];
               out_a(start[i] + p) = sum;
            }
         }
      }, n_mismatch);

   return n_mismatch;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// y(r_i) = sum_j inv(i, j) x(r_j), block by block
static void ApplyKernel(PetscScalarKokkosView y_d, PetscScalarConstKokkosView x_d, PetscScalarKokkosView inv_d, \
   PetscInt n_angles, PetscInt nb, PetscInt n_blocks)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, n_blocks), KOKKOS_LAMBDA(PetscInt b) {

         const PetscInt c = b / n_angles;
         const PetscInt a = b % n_angles;
         PetscScalar v[MAXB];
         for (PetscInt j = 0; j < nb; j++) v[j] = x_d((c * nb + j) * n_angles + a);
         for (PetscInt i = 0; i < nb; i++) {
            PetscScalar sum = 0.0;
            for (PetscInt j = 0; j < nb; j++) sum += inv_d((b * nb + i) * nb + j) * v[j];
            y_d((c * nb + i) * n_angles + a) = sum;
         }
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ElementBlockInverse::create(const PhaseSpace &ps)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());
   PetscCheck(ps.n_basis >= 1 && ps.n_basis <= MAX_BLOCK, PETSC_COMM_SELF, PETSC_ERR_SUP, \
      "ElementBlockInverse inverts blocks of up to %" PetscInt_FMT " basis functions, the phase space has %" \
      PetscInt_FMT, MAX_BLOCK, ps.n_basis);

   n_angles_ = ps.n_angles;
   nb_ = ps.n_basis;
   local_cells_ = ps.local_cells;
   n_blocks_ = local_cells_ * n_angles_;
   inv_d_ = PetscScalarKokkosView("block_inv_d", n_blocks_ * nb_ * nb_);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ElementBlockInverse::setup(Mat A, Vec diag)
{
   Mat Ad = NULL, Ao = NULL;
   DeviceCSR csr;
   PetscScalarMatConstKokkosView ad_a;
   PetscScalarConstKokkosView diag_d;

   PetscFunctionBeginUser;

   PetscCall(SplitLocal(A, &Ad, &Ao));
   PetscCall(GetDeviceCSR(Ad, csr));
   PetscCheck(csr.n_rows == n_blocks_ * nb_, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "the matrix has %" \
      PetscInt_FMT " local rows, the phase space %" PetscInt_FMT, csr.n_rows, n_blocks_ * nb_);

   if (diag) {
      PetscCall(VecGetKokkosView(diag, &diag_d));
      PetscCheck((PetscInt)diag_d.extent(0) == csr.n_rows, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "the diagonal " \
         "has %" PetscInt_FMT " local rows, the matrix %" PetscInt_FMT, (PetscInt)diag_d.extent(0), csr.n_rows);
   }
   PetscCall(MatSeqAIJGetKokkosView(Ad, &ad_a));
   const PetscInt n_singular = InvertBlocksKernel(inv_d_, csr.i, csr.j, ad_a, diag_d, diag ? PETSC_TRUE : \
      PETSC_FALSE, n_angles_, nb_, n_blocks_);
   PetscCall(MatSeqAIJRestoreKokkosView(Ad, &ad_a));
   if (diag) PetscCall(VecRestoreKokkosView(diag, &diag_d));
   PetscCheck(n_singular == 0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "%" PetscInt_FMT " of %" \
      PetscInt_FMT " element blocks are singular", n_singular, n_blocks_);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ElementBlockInverse::apply(Vec x, Vec y) const
{
   PetscScalarConstKokkosView x_d;
   PetscScalarKokkosView y_d;

   PetscFunctionBeginUser;

   PetscCall(VecGetKokkosView(x, &x_d));
   PetscCall(VecGetKokkosViewWrite(y, &y_d));
   PetscCheck((PetscInt)x_d.extent(0) == n_blocks_ * nb_, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, \
      "the vector has %" PetscInt_FMT " local rows, the phase space %" PetscInt_FMT, (PetscInt)x_d.extent(0), \
      n_blocks_ * nb_);
   ApplyKernel(y_d, x_d, inv_d_, n_angles_, nb_, n_blocks_);
   PetscCall(VecRestoreKokkosViewWrite(y, &y_d));
   PetscCall(VecRestoreKokkosView(x, &x_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ElementBlockInverse::scale(Mat A, MatReuse reuse, Mat *scaled) const
{
   Mat parts_in[2] = {NULL, NULL}, parts_out[2] = {NULL, NULL};

   PetscFunctionBeginUser;

   PetscCheck(reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX, PETSC_COMM_SELF, \
      PETSC_ERR_ARG_OUTOFRANGE, "ElementBlockInverse::scale takes MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX");
   if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, scaled));

   PetscCall(SplitLocal(A, &parts_in[0], &parts_in[1]));
   PetscCall(SplitLocal(*scaled, &parts_out[0], &parts_out[1]));

   // The owned part, then the off-process one: an element block's rows share
   // their off-process columns too, so the same row combination applies there
   for (PetscInt part = 0; part < 2; part++) {
      if (!parts_in[part]) continue;
      DeviceCSR csr;
      PetscInt out_nnz = 0;
      PetscScalarMatConstKokkosView in_a;
      PetscScalarMatKokkosView out_a;

      PetscCall(GetDeviceCSR(parts_in[part], csr));
      PetscCheck(csr.n_rows == n_blocks_ * nb_, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "the matrix has %" \
         PetscInt_FMT " local rows, the phase space %" PetscInt_FMT, csr.n_rows, n_blocks_ * nb_);

      PetscCall(MatSeqAIJGetKokkosView(parts_in[part], &in_a));
      PetscCall(MatSeqAIJGetKokkosViewWrite(parts_out[part], &out_a));
      out_nnz = (PetscInt)out_a.extent(0);
      PetscInt n_mismatch = 0;
      // The output is a copy of A's pattern: same row offsets, same columns
      if (out_nnz == csr.nnz) n_mismatch = ScaleRowsKernel(out_a, csr.i, csr.j, in_a, inv_d_, n_angles_, nb_, \
         n_blocks_);
      PetscCall(MatSeqAIJRestoreKokkosViewWrite(parts_out[part], &out_a));
      PetscCall(MatSeqAIJRestoreKokkosView(parts_in[part], &in_a));

      PetscCheck(out_nnz == csr.nnz, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "the scaled matrix has %" \
         PetscInt_FMT " nonzeros where the matrix it scales has %" PetscInt_FMT " - it must come from an " \
         "earlier scale() of the same matrix", out_nnz, csr.nnz);
      PetscCheck(n_mismatch == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "%" PetscInt_FMT " element blocks " \
         "have rows with different sparsity, so D^{-1} A would not keep A's pattern", n_mismatch);
   }

   // The restores above mark the seq parts modified (and bump THEIR state),
   // but a PC built on an MPI matrix compares the wrapper's state, which
   // nothing has touched. A final assembly is the public way to bump it: no
   // values are stashed and the pattern is unchanged, so it rebuilds nothing
   PetscCall(MatAssemblyBegin(*scaled, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*scaled, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}
