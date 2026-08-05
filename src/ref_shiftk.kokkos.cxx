#include "ubolt/ref_shift.hpp"
#include "petsc_kokkos.hpp"
#include <algorithm>
#include <numeric>

// The kernel below captures plain values and shallow view copies, never `this`
// - a member access inside a KOKKOS_LAMBDA would dereference a host pointer on
// the device (see docs/dev/kokkos.md)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The masked shift: alpha * D_ref on the interior rows, ZERO on every row the
// boundary condition owns
//
// The BC rows of the streaming matrix are the exact identity (Dirichlet) or the
// identity minus the mirrored angle (reflective) - the same rows the transport
// operator carries, because the pmat is a copy of it. Shifting them would make
// the preconditioner's boundary condition a different one from the operator's,
// which is the same contract every term already has with the mask
static void ShiftFillKernel(PetscScalarKokkosView shift_d, \
   PetscScalarKokkosView sigma_t_ref_d, PetscIntKokkosView is_bc_row_d, \
   PetscScalar alpha, PetscInt n_angles, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         shift_d(r) = is_bc_row_d(r) ? (PetscScalar)0.0 : alpha * sigma_t_ref_d(r / n_angles);
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// alpha_g: the log-mean over the mesh of Sigma_t(g) / Sigma_t(0)
//
// The geometric mean rather than the arithmetic one because the quantity that
// governs the preconditioner's quality is a RATIO - a group twice the reference
// and a group half of it are equally mismatched, and only the log-mean says so.
// It also makes the identity exact in the case that matters: when every
// material is a density-scaled copy of one material, Sigma_t(g)/Sigma_t(0) is
// the same number in every cell, and its log-mean is that number
//
// A host loop over one cell-sized mirror per group, once, at setup - the same
// shape as DSAPrecon::assemble's guard pass, and for the same reason: there is
// nothing here worth a device reduction and the arithmetic is easier to read
PetscErrorCode RefShiftPmats::compute_alphas(const GroupXSections &xs)
{
   PetscReal min_sigma_t = PETSC_MAX_REAL;

   PetscFunctionBeginUser;

   alpha_.assign(n_groups_, 1.0);

   auto sigma_t_ref_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xs.sigma_t(0));

   // The void guard first, over every group: log(0) is not a mismatch ratio,
   // it is a missing material. Reduced over the ranks before it is checked, so
   // every rank fails the same way
   for (PetscInt g = 0; g < n_groups_; g++) {
      auto sigma_t_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xs.sigma_t(g));
      for (PetscInt c = 0; c < ps_.local_cells; c++) {
         min_sigma_t = PetscMin(min_sigma_t, PetscRealPart(sigma_t_h(c)));
      }
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &min_sigma_t, 1, MPIU_REAL, MPIU_MIN, comm_));
   PetscCheck(min_sigma_t > 0.0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "the reference-shifted pmat needs Sigma_t > 0 in every cell and every group - the " \
      "shift is a RATIO to the reference removal and a void has none, and the smallest " \
      "Sigma_t anywhere is %g. Run without -precon_ref_shift, or give the void a small " \
      "Sigma_t", (double)min_sigma_t);

   // Group 0 IS the reference, so its ratio is 1 by construction and there is
   // nothing to sum - taking the log of sigma_t/sigma_t would only add rounding
   for (PetscInt g = 1; g < n_groups_; g++) {

      PetscReal log_sum = 0.0;

      auto sigma_t_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xs.sigma_t(g));
      for (PetscInt c = 0; c < ps_.local_cells; c++) {
         log_sum += PetscLogReal(PetscRealPart(sigma_t_h(c)) / PetscRealPart(sigma_t_ref_h(c)));
      }
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &log_sum, 1, MPIU_REAL, MPIU_SUM, comm_));

      alpha_[g] = PetscExpReal(log_sum / (PetscReal)ps_.n_cells);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// How many bins a greedy left-to-right pass over the sorted log-alphas needs if
// no bin may span more than `spread` - the standard fact that greedy is optimal
// for "fewest intervals of a bounded width", which is what makes the search
// below exact rather than a heuristic
static PetscInt BinsNeeded(const std::vector<PetscReal> &sorted_log, PetscReal spread)
{
   PetscInt bins = 0;
   size_t i = 0;

   while (i < sorted_log.size()) {
      const PetscReal start = sorted_log[i];
      while (i < sorted_log.size() && sorted_log[i] - start <= spread) i++;
      bins++;
   }
   return bins;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Cluster the groups into n_bins contiguous runs of the sorted log-alphas, so
// that the widest bin is as narrow as it can be
//
// Log-spaced because the mismatch that costs iterations is a RATIO, so equal
// cost is equal distance in log alpha. The narrowest-widest-bin partition
// because it is the one whose worst case is the thing being minimised, and
// because it degenerates correctly at both ends: one bin is the single shared
// pmat, and n_bins at or above the number of DISTINCT alphas admits a spread of
// zero, so every bin holds one value and the shift is EXACT per group. One knob
// from end to end, not two code paths
//
// The optimal spread is found by searching the candidate widths - every
// pairwise difference of log-alphas, since the answer is one of them - for the
// smallest that BinsNeeded can manage in n_bins. n_groups is small (this is a
// group structure, not a mesh) and it runs once at setup
//
// Each bin's representative then sits at the log-MIDPOINT of the alphas it
// holds, which is the choice that minimises the worst mismatch inside the bin:
// a bin of spread s leaves every group in it within sqrt(s) of the alpha its
// pmat was built with, and that mismatch is what the default rule reads
void RefShiftPmats::bin_alphas(PetscInt n_bins)
{
   std::vector<PetscInt> order(n_groups_);
   std::vector<PetscReal> log_alpha(n_groups_), sorted_log(n_groups_), candidates;

   for (PetscInt g = 0; g < n_groups_; g++) log_alpha[g] = PetscLogReal(alpha_[g]);
   std::iota(order.begin(), order.end(), 0);
   // Stable, so equal alphas keep group order and the binning is deterministic:
   // it has to be, because every rank computes it redundantly
   std::stable_sort(order.begin(), order.end(), \
      [&](PetscInt a, PetscInt b) { return log_alpha[a] < log_alpha[b]; });
   for (PetscInt i = 0; i < n_groups_; i++) sorted_log[i] = log_alpha[order[i]];

   for (PetscInt i = 0; i < n_groups_; i++) {
      for (PetscInt j = i; j < n_groups_; j++) candidates.push_back(sorted_log[j] - sorted_log[i]);
   }
   std::sort(candidates.begin(), candidates.end());
   PetscReal spread = candidates.back();
   for (size_t c = 0; c < candidates.size(); c++) {
      if (BinsNeeded(sorted_log, candidates[c]) <= n_bins) { spread = candidates[c]; break; }
   }

   bin_of_group_.assign(n_groups_, 0);
   bin_alpha_.clear();

   PetscInt bin_start = 0;
   for (PetscInt i = 0; i < n_groups_; i++) {

      // A new bin starts as soon as this alpha is further than `spread` above
      // the one that opened the current bin - the same greedy pass BinsNeeded
      // counted, so it lands in exactly n_bins bins or fewer
      if (sorted_log[i] - sorted_log[bin_start] > spread) {
         bin_alpha_.push_back(PetscExpReal( \
            0.5 * (sorted_log[bin_start] + sorted_log[i - 1])));
         bin_start = i;
      }
      bin_of_group_[order[i]] = (PetscInt)bin_alpha_.size();
   }
   bin_alpha_.push_back(PetscExpReal( \
      0.5 * (sorted_log[bin_start] + sorted_log[n_groups_ - 1])));

   worst_mismatch_ = 1.0;
   for (PetscInt g = 0; g < n_groups_; g++) {
      const PetscReal ratio = alpha_[g] / bin_alpha_[bin_of_group_[g]];
      worst_mismatch_ = PetscMax(worst_mismatch_, PetscMax(ratio, 1.0 / ratio));
   }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode RefShiftPmats::create(MPI_Comm comm, const PhaseSpace &ps, \
   const Discretisation &disc, const GroupXSections &xs, Mat streaming_mat, PetscInt n_bins)
{
   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());

   comm_ = comm;
   ps_ = ps;
   n_groups_ = ps.n_groups;

   PetscCall(compute_alphas(xs));

   // How many bins. A value the caller asked for is taken as given (clamped to
   // something that exists); otherwise the default rule - see the header for
   // where the 3s come from
   if (n_bins > 0) n_bins = PetscMin(n_bins, n_groups_);
   else {

      const PetscInt cap = PetscMin(default_max_bins, n_groups_);
      std::vector<PetscReal> mismatch(cap + 1, 0.0);
      PetscReal best = PETSC_MAX_REAL;

      n_bins = 0;
      for (PetscInt k = 1; k <= cap; k++) {
         bin_alphas(k);
         mismatch[k] = worst_mismatch_;
         best = PetscMin(best, worst_mismatch_);
         if (!n_bins && worst_mismatch_ <= default_max_mismatch) n_bins = k;
      }
      // Nothing within the cap met the target, so take the fewest bins that do
      // as well as the cap can. The tolerance is there because two different
      // partitions can hit the same worst case in different floating order
      for (PetscInt k = 1; k <= cap && !n_bins; k++) {
         if (mismatch[k] <= best * (1.0 + 1e-12)) n_bins = k;
      }
   }
   bin_alphas(n_bins);
   // What the binning actually needed, which can be fewer than what was asked
   // for: an exact partition of fewer distinct alphas than bins is still exact,
   // and a hierarchy per duplicate would buy nothing
   n_bins = (PetscInt)bin_alpha_.size();

   // One pmat per bin: a copy of the streaming operator with that bin's
   // representative removal added onto the interior diagonal
   const BoundaryInfo &boundary = disc.boundary_info();
   PetscScalarKokkosView sigma_t_ref_d = xs.sigma_t(0);

   pmats_.assign(n_bins, NULL);
   for (PetscInt k = 0; k < n_bins; k++) {

      Vec shift_vec;
      PetscScalarKokkosView shift_d;

      PetscCall(MatDuplicate(streaming_mat, MAT_COPY_VALUES, &pmats_[k]));
      PetscCall(MatCreateVecs(pmats_[k], &shift_vec, PETSC_NULLPTR));

      PetscCall(VecGetKokkosViewWrite(shift_vec, &shift_d));
      ShiftFillKernel(shift_d, sigma_t_ref_d, boundary.is_bc_row_d, \
         (PetscScalar)bin_alpha_[k], ps_.n_angles, ps_.local_rows());
      PetscCall(VecRestoreKokkosViewWrite(shift_vec, &shift_d));

      PetscCall(MatDiagonalSet(pmats_[k], shift_vec, ADD_VALUES));
      PetscCall(VecDestroy(&shift_vec));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode RefShiftPmats::destroy()
{
   PetscFunctionBeginUser;

   for (size_t k = 0; k < pmats_.size(); k++) PetscCall(MatDestroy(&pmats_[k]));
   pmats_.clear();

   PetscFunctionReturn(PETSC_SUCCESS);
}
