#ifndef UBOLT_REF_SHIFT_HPP
#define UBOLT_REF_SHIFT_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/multigroup.hpp"
#include "ubolt/phase_space.hpp"
#include <petscmat.h>
#include <vector>

// Reference-shifted streaming preconditioning matrices: k copies of the
// streaming operator, each carrying a REPRESENTATIVE removal, shared by the
// groups whose own removal is close enough to it
//
// The problem this solves. Under -matfree_removal the assembled matrix carries
// STREAMING ONLY - that is the whole point of the mode, one matrix for the
// whole sweep - and the pmat defaults to it. PCAIR on a bare streaming operator
// is fine while the removal is weak and diverges once it is strong (from
// roughly one mean free path per cell), because the operator it is set up on is
// then nothing like the one being solved. A pmat of L + alpha * D_ref puts the
// removal back:
//   - D_ref is the REFERENCE removal, group 0's per-cell Sigma_t. Per cell, not
//     a scalar: on a heterogeneous problem a scalar shift is wrong everywhere
//     except one material, while a per-cell reference tracks the geometry
//   - alpha_g is what relates group g's removal to that reference. If every
//     material shares the same group-to-group ratio structure - a
//     density-scaled copy of one material, the common case - then
//     alpha_g * D_ref IS Sigma_t(g) exactly, cell by cell, and the pmat is the
//     full one. Otherwise it is the best per-group rank-0 fit to it
//
// Why k of them rather than one per group. One pmat per group is exact and
// costs a PCAIR hierarchy per group; one pmat for the whole sweep costs one
// hierarchy and is exact only if every group has the same removal. The measured
// middle (Phase 5 campaign, Aug 2026): iteration quality degrades gently with
// the MISMATCH RATIO between a group's alpha and the one its pmat was built
// with - a factor of 3 costs roughly 1.3-1.6x, and the whole useful window is
// about a factor of 10 - and that window is independent of the mesh and the
// angular order. So a handful of hierarchies, log-spaced over the range the
// groups occupy, covers a realistic multigroup problem
//
// Which is what this class is: the alphas, the binning, and the k matrices. The
// group loop stays with the caller, which owns one solver per bin and asks
// bin_of_group() which one this group belongs to
class PETSC_VISIBILITY_PUBLIC RefShiftPmats {
public:
   // The default binning rule: the fewest bins that keep every group's
   // mismatch - its own alpha over the one its pmat was built with - within
   // default_max_mismatch, and if default_max_bins cannot manage that, the
   // fewest that do as well as that cap allows (asking for a bin that does not
   // improve the worst case buys a hierarchy and nothing else)
   //
   // 3 is where the campaign measured a mismatch of that size costing roughly
   // 1.3-1.6x the exact-ratio iteration count, with the whole useful window
   // about a factor of 10 wide - so it is the far end of "still cheap" rather
   // than the edge of "still works". 3 bins covers a factor of 9 apiece, i.e.
   // an alpha range of a few hundred, which is more than a realistic group
   // structure spans; past that a PCAIR hierarchy per bin is the thing that
   // stops being free, so the default stops and -precon_ref_k is how a caller
   // buys more deliberately
   static constexpr PetscReal default_max_mismatch = 3.0;
   static constexpr PetscInt  default_max_bins     = 3;

   // streaming_mat is the assembled streaming operator (under -matfree_removal,
   // the transport operator's own assembled matrix). It is only READ: each bin
   // gets a MatDuplicate of it plus its own shift, so the caller keeps using it
   // as it was
   //
   // n_bins <= 0 asks for the default rule above. A positive value is taken as
   // given, clamped to [1, n_groups] - n_groups bins is one per group, which is
   // exact whatever the alphas are
   //
   // ps must already carry the decomposition (this reads local_cells), and xs
   // must already be filled - the alphas are computed here, once
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
      const GroupXSections &xs, Mat streaming_mat, PetscInt n_bins);
   PetscErrorCode destroy();

   PetscInt n_bins() const { return (PetscInt)pmats_.size(); }
   // Which bin's pmat group g is preconditioned with
   PetscInt bin_of_group(PetscInt g) const { return bin_of_group_[g]; }
   // The pmat of a bin: L + bin_alpha(bin) * D_ref on the non-BC rows, and the
   // streaming matrix's exact boundary rows everywhere else. Not owned by the
   // caller - destroy() frees them
   Mat pmat(PetscInt bin) const { return pmats_[bin]; }

   // Group g's own ratio to the reference removal (alpha_0 is 1 by definition)
   PetscReal alpha(PetscInt g) const { return alpha_[g]; }
   // The ratio the bin's pmat was actually built with
   PetscReal bin_alpha(PetscInt bin) const { return bin_alpha_[bin]; }
   // The worst mismatch any group is preconditioned at, as a ratio >= 1
   // (max over groups of alpha_g / bin_alpha and its reciprocal). Exactly 1.0
   // means every group is covered exactly, so the k pmats are the k distinct
   // full pmats and nothing has been approximated
   PetscReal worst_mismatch() const { return worst_mismatch_; }

private:
   // The log-mean of Sigma_t(g) / Sigma_t(0) over every cell, per group
   PetscErrorCode compute_alphas(const GroupXSections &xs);
   // Partition the sorted log-alphas into contiguous bins minimising the
   // widest bin - see the .cxx for why this beats cutting at the largest gaps
   void bin_alphas(PetscInt n_bins);

   MPI_Comm comm_ = MPI_COMM_NULL;
   PhaseSpace ps_;
   PetscInt n_groups_ = 0;
   std::vector<PetscReal> alpha_;
   std::vector<PetscReal> bin_alpha_;
   std::vector<PetscInt> bin_of_group_;
   std::vector<Mat> pmats_;
   PetscReal worst_mismatch_ = 1.0;
};

#endif
