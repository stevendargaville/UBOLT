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
   PetscScalar alpha, PetscInt rows_per_cell, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         shift_d(r) = is_bc_row_d(r) ? (PetscScalar)0.0 : alpha * sigma_t_ref_d(r / rows_per_cell);
      });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// alpha_g: the log-mean over the mesh of Sigma_t(g) / Sigma_t(ref)
//
// The geometric mean rather than the arithmetic one because the quantity that
// governs the preconditioner's quality is a RATIO - a group twice the reference
// and a group half of it are equally mismatched, and only the log-mean says so.
// It also makes the identity exact in the case that matters: when every
// material is a density-scaled copy of one material, Sigma_t(g)/Sigma_t(ref) is
// the same number in every cell, and its log-mean is that number
//
// The reference is the FIRST group whose removal is positive in every cell -
// group 0 whenever group 0 qualifies, so the alphas (and the pins) are what
// they always were on any problem the old group-0 convention accepted. A group
// whose removal is identically ZERO is a streaming-only group: no ratio to any
// reference exists, but none is needed, because the exactly-right pmat for it
// is the bare streaming matrix itself - it gets alpha = 0 and bin_alphas gives
// those groups a dedicated unshifted bin. What stays a hard error is a group
// that is zero in SOME cells: a single ratio cannot represent a field that is
// removal here and void there (and a negative Sigma_t is nonsense anywhere)
//
// A host loop over one cell-sized mirror per group, once, at setup - the same
// shape as DSAPrecon::assemble's guard pass, and for the same reason: there is
// nothing here worth a device reduction and the arithmetic is easier to read
PetscErrorCode RefShiftPmats::compute_alphas(const GroupXSections &xs)
{
   PetscFunctionBeginUser;

   alpha_.assign(n_groups_, 1.0);
   ref_group_ = -1;

   // Classify every group by its global min and max: all-positive (a ratio
   // exists), identically zero (streaming-only), or mixed (the error). One
   // reduction for all groups, so every rank classifies - and fails - the
   // same way
   std::vector<PetscReal> min_max(2 * n_groups_);
   for (PetscInt g = 0; g < n_groups_; g++) {
      PetscReal mn = PETSC_MAX_REAL, mx = -PETSC_MAX_REAL;
      auto sigma_t_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xs.sigma_t(g));
      for (PetscInt c = 0; c < ps_.local_cells; c++) {
         mn = PetscMin(mn, PetscRealPart(sigma_t_h(c)));
         mx = PetscMax(mx, PetscRealPart(sigma_t_h(c)));
      }
      // One buffer, min negated so a single MAX reduction serves both
      min_max[2 * g] = -mn;
      min_max[2 * g + 1] = mx;
   }
   PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, min_max.data(), 2 * n_groups_, MPIU_REAL, \
      MPIU_MAX, comm_));

   is_streaming_group_.assign(n_groups_, false);
   for (PetscInt g = 0; g < n_groups_; g++) {
      const PetscReal mn = -min_max[2 * g], mx = min_max[2 * g + 1];
      PetscCheck(mn >= 0.0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
         "group %" PetscInt_FMT " has a negative Sigma_t (%g) somewhere", g, (double)mn);
      if (mx == 0.0) { is_streaming_group_[g] = true; alpha_[g] = 0.0; }
      else {
         PetscCheck(mn > 0.0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
            "group %" PetscInt_FMT "'s Sigma_t is zero in some cells and positive in others " \
            "(min %g, max %g) - the reference shift is a single RATIO per group, and no " \
            "ratio represents a field that is removal here and void there. A group that is " \
            "zero EVERYWHERE is fine (it is preconditioned with the streaming matrix " \
            "itself); for a void region, run without -precon_ref_shift or give the void a " \
            "small Sigma_t", g, (double)mn, (double)mx);
         if (ref_group_ < 0) ref_group_ = g;
      }
   }
   // Every group streaming-only: no reference, no ratios, and nothing to do -
   // bin_alphas puts them all in the one unshifted bin
   if (ref_group_ < 0) PetscFunctionReturn(PETSC_SUCCESS);

   auto sigma_t_ref_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), \
      xs.sigma_t(ref_group_));

   // The reference's own ratio is 1 by construction and there is nothing to
   // sum - taking the log of sigma_t/sigma_t would only add rounding
   for (PetscInt g = 0; g < n_groups_; g++) {
      if (g == ref_group_ || is_streaming_group_[g]) continue;

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
// below exact rather than a heuristic. `tol` is the tie tolerance - see
// bin_alphas
static PetscInt BinsNeeded(const std::vector<PetscReal> &sorted_log, PetscReal spread, \
   PetscReal tol)
{
   PetscInt bins = 0;
   size_t i = 0;

   while (i < sorted_log.size()) {
      const PetscReal start = sorted_log[i];
      while (i < sorted_log.size() && sorted_log[i] - start <= spread + tol) i++;
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
// Every width comparison carries a tie tolerance. The alphas are log-means
// reduced over the mesh, so their last bits depend on the MPI sum order, and
// when two candidate partitions tie in exact arithmetic - equally spaced
// log-alphas, as in the decades files, where any adjacent pair could share the
// one merged bin at k = 3 - rounding alone would pick the winner, and serial
// and parallel runs would build different pmats. Differences within `tol` are
// treated as equal, which makes the greedy pass decide, deterministically
//
// The greedy pass at the optimal width can use FEWER bins than were asked for:
// exactly, when there are fewer distinct alphas than bins, and also at a tie
// (0, 1, 2, 3 in 3 bins is {0,1}, {2,3} at width 1). The spare bins are then
// spent from the TOP, splitting the highest distinct alpha off the highest bin
// that still holds more than one, until they run out or every bin is exact.
// That never widens a bin, so the worst mismatch stands, and the top is where
// the spare hierarchy pays: the high-alpha groups are the thick ones, where a
// mismatch costs the most iterations (box_decades4 at k = 3: 15, 29 on the two
// thick groups with them exact, against 27, 51 with them sharing a bin)
//
// Each bin's representative then sits at the log-MIDPOINT of the alphas it
// holds, which is the choice that minimises the worst mismatch inside the bin:
// a bin of spread s leaves every group in it within sqrt(s) of the alpha its
// pmat was built with, and that mismatch is what the default rule reads
void RefShiftPmats::bin_alphas(PetscInt n_bins)
{
   // Only the groups that HAVE a ratio are clustered. A streaming-only group
   // (alpha 0) sits at minus infinity in log space and needs no clustering
   // anyway: all such groups share one dedicated unshifted bin, appended after
   // the shifted ones - its pmat is the streaming matrix itself, which is
   // EXACT for them, so they never contribute mismatch
   std::vector<PetscInt> pos;
   for (PetscInt g = 0; g < n_groups_; g++) {
      if (!is_streaming_group_[g]) pos.push_back(g);
   }
   const PetscInt n_pos = (PetscInt)pos.size();

   bin_of_group_.assign(n_groups_, 0);
   bin_alpha_.clear();
   worst_mismatch_ = 1.0;

   if (n_pos > 0) {

      std::vector<PetscInt> order(n_pos);
      std::vector<PetscReal> log_alpha(n_pos), sorted_log(n_pos), candidates;

      for (PetscInt i = 0; i < n_pos; i++) log_alpha[i] = PetscLogReal(alpha_[pos[i]]);
      std::iota(order.begin(), order.end(), 0);
      // Stable, so equal alphas keep group order and the binning is
      // deterministic: it has to be, because every rank computes it redundantly
      std::stable_sort(order.begin(), order.end(), \
         [&](PetscInt a, PetscInt b) { return log_alpha[a] < log_alpha[b]; });
      for (PetscInt i = 0; i < n_pos; i++) sorted_log[i] = log_alpha[order[i]];

      for (PetscInt i = 0; i < n_pos; i++) {
         for (PetscInt j = i; j < n_pos; j++) candidates.push_back(sorted_log[j] - sorted_log[i]);
      }
      std::sort(candidates.begin(), candidates.end());

      // Far above the rounding in a mesh-sized log-mean, far below any ratio
      // worth a hierarchy of its own
      const PetscReal tol = 1e-10 * PetscMax((PetscReal)1.0, sorted_log[n_pos - 1] - sorted_log[0]);

      PetscReal spread = candidates.back();
      for (size_t c = 0; c < candidates.size(); c++) {
         if (BinsNeeded(sorted_log, candidates[c], tol) <= n_bins) { spread = candidates[c]; break; }
      }

      // The bins as the index each one starts at in sorted_log. A new bin
      // starts as soon as an alpha is further than `spread` above the one that
      // opened the current bin - the same greedy pass BinsNeeded counted, so it
      // lands in n_bins bins or fewer
      std::vector<PetscInt> starts = {0};
      for (PetscInt i = 1; i < n_pos; i++) {
         if (sorted_log[i] - sorted_log[starts.back()] > spread + tol) starts.push_back(i);
      }

      // Spend any spare bins from the top: split off the highest distinct
      // alpha of the highest bin that has more than one
      while ((PetscInt)starts.size() < n_bins) {
         PetscInt split = -1;
         for (PetscInt b = (PetscInt)starts.size() - 1; b >= 0 && split < 0; b--) {
            const PetscInt end = b + 1 < (PetscInt)starts.size() ? starts[b + 1] : n_pos;
            for (PetscInt j = end - 1; j > starts[b]; j--) {
               if (sorted_log[j] - sorted_log[j - 1] > tol) { split = j; break; }
            }
         }
         if (split < 0) break;
         starts.insert(std::upper_bound(starts.begin(), starts.end(), split), split);
      }

      for (size_t b = 0; b < starts.size(); b++) {
         const PetscInt end = b + 1 < starts.size() ? starts[b + 1] : n_pos;
         for (PetscInt i = starts[b]; i < end; i++) bin_of_group_[pos[order[i]]] = (PetscInt)b;
         bin_alpha_.push_back(PetscExpReal(0.5 * (sorted_log[starts[b]] + sorted_log[end - 1])));
      }

      for (PetscInt i = 0; i < n_pos; i++) {
         const PetscReal ratio = alpha_[pos[i]] / bin_alpha_[bin_of_group_[pos[i]]];
         worst_mismatch_ = PetscMax(worst_mismatch_, PetscMax(ratio, 1.0 / ratio));
      }
   }

   if (n_pos < n_groups_) {
      const PetscInt streaming_bin = (PetscInt)bin_alpha_.size();
      bin_alpha_.push_back(0.0);
      for (PetscInt g = 0; g < n_groups_; g++) {
         if (is_streaming_group_[g]) bin_of_group_[g] = streaming_bin;
      }
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

   // How many bins - counting the SHIFTED ones only: the unshifted bin any
   // streaming-only groups share is appended on top of the count, because it
   // is exact for them and its matrix is not even a copy (below). A value the
   // caller asked for is taken as given (clamped to something that exists);
   // otherwise the default rule - see the header for where the 3s come from
   PetscInt n_pos = 0;
   for (PetscInt g = 0; g < n_groups_; g++) {
      if (!is_streaming_group_[g]) n_pos++;
   }
   if (n_pos == 0) n_bins = 0;
   else if (n_bins > 0) n_bins = PetscMin(n_bins, n_pos);
   else {

      const PetscInt cap = PetscMin(default_max_bins, n_pos);
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
   // The binning takes what it actually needs, which can be fewer bins than
   // were asked for: an exact partition of fewer distinct alphas than bins is
   // still exact, and a hierarchy per duplicate would buy nothing. From here
   // bin_alpha_ is the authority on how many bins exist (n_bins() reads it),
   // shifted ones plus the unshifted one if any group needs it
   bin_alphas(n_bins);

   // One pmat per bin: a copy of the streaming operator with that bin's
   // representative removal added onto the interior diagonal. The unshifted
   // bin (streaming-only groups, bin_alpha 0) is not even a copy - it IS the
   // streaming matrix, reference-counted so destroy() can treat every bin the
   // same way. Its groups' operator is exactly L, so their pmat is exact for
   // free
   const BoundaryInfo &boundary = disc.boundary_info();
   PetscScalarKokkosView sigma_t_ref_d = xs.sigma_t(ref_group_ < 0 ? 0 : ref_group_);

   pmats_.assign((PetscInt)bin_alpha_.size(), NULL);
   for (PetscInt k = 0; k < (PetscInt)pmats_.size(); k++) {

      if (bin_alpha_[k] == 0.0) {
         PetscCall(PetscObjectReference((PetscObject)streaming_mat));
         pmats_[k] = streaming_mat;
         continue;
      }

      Vec shift_vec;
      PetscScalarKokkosView shift_d;

      PetscCall(MatDuplicate(streaming_mat, MAT_COPY_VALUES, &pmats_[k]));
      PetscCall(MatCreateVecs(pmats_[k], &shift_vec, PETSC_NULLPTR));

      PetscCall(VecGetKokkosViewWrite(shift_vec, &shift_d));
      ShiftFillKernel(shift_d, sigma_t_ref_d, boundary.is_bc_row_d, \
         (PetscScalar)bin_alpha_[k], ps_.rows_per_cell(), ps_.local_rows());
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
