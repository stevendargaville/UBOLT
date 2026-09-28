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
//   - D_ref is the REFERENCE removal: a group's per-cell Sigma_t. Per cell,
//     not a scalar: on a heterogeneous problem a scalar shift is wrong
//     everywhere except one material, while a per-cell reference tracks the
//     geometry
//   - VOIDS are part of that geometry. Each group's SUPPORT - the cells where
//     its Sigma_t is positive - sorts the groups into support classes, and
//     every class has its own reference, its first group (group 0 whenever
//     group 0 qualifies). The reference is zero exactly where the class is
//     void, so the shift is too: a void cell's pmat row is the bare streaming
//     row, which is the true operator's row there (a void has no removal to
//     approximate). The ratios are taken over the support only, where both
//     fields are positive. On a void-free problem there is one class, the
//     whole mesh, and nothing about this is new. Groups with different
//     supports (a material void in some groups only) never share a bin: the
//     pmat of one would carry removal where the other has none
//   - a group whose removal is identically ZERO (streaming-only) is the class
//     with an empty support: no ratio to anything, and none needed, since its
//     operator is exactly L - so all such groups share one unshifted bin whose
//     pmat IS the streaming matrix (reference-counted, not copied), exact by
//     construction
//   - alpha_g is what relates group g's removal to that reference. If every
//     material shares the same group-to-group ratio structure - a
//     density-scaled copy of one material, the common case - then
//     alpha_g * D_ref IS Sigma_t(g) exactly, cell by cell, and the pmat is the
//     full one - void cells included. Otherwise it is the best per-group
//     rank-0 fit to it
//
// Why k of them rather than one per group. One pmat per group is exact and
// costs a PCAIR hierarchy per group; one pmat for the whole sweep costs one
// hierarchy and is exact only if every group has the same removal. Iteration
// quality degrades gently with the MISMATCH RATIO between a group's alpha and
// the one its pmat was built with, independently of the mesh and the angular
// order (measurements: TODO.md Phase 5), so a handful of hierarchies,
// log-spaced over the range the groups occupy, covers a realistic multigroup
// problem
//
// Which is what this class is: the alphas, the binning, and the k matrices. The
// group loop stays with the caller, which owns one solver per bin and asks
// bin_of_group() which one this group belongs to. The binning is per support
// class (one class on any problem whose groups are void in the same cells, or
// in none), so k is the number of shifted bins PER CLASS
class PETSC_VISIBILITY_PUBLIC RefShiftPmats {
public:
   // The default binning rule: the fewest bins that keep every group's
   // mismatch - its own alpha over the one its pmat was built with - within
   // default_max_mismatch, and if default_max_bins cannot manage that, the
   // fewest that do as well as that cap allows (asking for a bin that does not
   // improve the worst case buys a hierarchy and nothing else)
   //
   // A mismatch of 3 is the far end of "still cheap" rather than the edge of
   // "still works" (TODO.md Phase 5). 3 bins of a factor of 9 apiece span an
   // alpha range of a few hundred, more than a realistic group structure;
   // past that a hierarchy per bin stops being free, so -precon_ref_k is how
   // a caller buys more deliberately
   static constexpr PetscReal default_max_mismatch = 3.0;
   static constexpr PetscInt  default_max_bins     = 3;

   // streaming_mat is the assembled streaming operator (under -matfree_removal,
   // the transport operator's own assembled matrix). It is only READ: each bin
   // gets a MatDuplicate of it plus its own shift, so the caller keeps using it
   // as it was
   //
   // n_bins <= 0 asks for the default rule above. A positive value is taken as
   // given, per support class, clamped to [1, groups in the class] - one bin
   // per group is exact whatever the alphas are
   //
   // A negative Sigma_t anywhere is an error
   //
   // ps must already carry the decomposition (this reads local_cells), and xs
   // must already be filled - the alphas are computed here, once
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
      const GroupXSections &xs, Mat streaming_mat, PetscInt n_bins);
   PetscErrorCode destroy();

   PetscInt n_bins() const { return (PetscInt)pmats_.size(); }
   // Which bin's pmat group g is preconditioned with
   PetscInt bin_of_group(PetscInt g) const { return bin_of_group_[g]; }
   // The pmat of a bin: L + bin_alpha(bin) * D_ref on the non-BC rows (D_ref
   // being the bin's class reference, zero in its voids), and the streaming
   // matrix's exact boundary rows everywhere else. Not owned by the caller -
   // destroy() frees them
   Mat pmat(PetscInt bin) const { return pmats_[bin]; }

   // Group g's own ratio to its class's reference removal (a reference's is 1
   // by definition; a streaming-only group's is 0)
   PetscReal alpha(PetscInt g) const { return alpha_[g]; }
   // How many support classes carry removal - 1 on any problem whose groups
   // are void in the same cells or in none; the streaming-only groups, if
   // any, are not counted (they have no reference)
   PetscInt n_classes() const { return (PetscInt)class_ref_.size(); }
   // The group whose per-cell Sigma_t is the bin's D_ref - the first group of
   // its support class; -1 for the unshifted bin, whose pmat is the streaming
   // matrix itself
   PetscInt bin_ref_group(PetscInt bin) const { return bin_ref_group_[bin]; }
   // The ratio the bin's pmat was actually built with
   PetscReal bin_alpha(PetscInt bin) const { return bin_alpha_[bin]; }
   // The worst mismatch any group is preconditioned at, as a ratio >= 1
   // (max over groups of alpha_g / bin_alpha and its reciprocal). Exactly 1.0
   // means every group is covered exactly, so the k pmats are the k distinct
   // full pmats and nothing has been approximated
   PetscReal worst_mismatch() const { return worst_mismatch_; }

private:
   // Classify the groups by support (streaming-only, or one of the removal
   // classes), and the log-mean of Sigma_t(g) / Sigma_t(ref) over the class's
   // support, per group
   PetscErrorCode compute_alphas(const GroupXSections &xs);

   MPI_Comm comm_ = MPI_COMM_NULL;
   PhaseSpace ps_;
   PetscInt n_groups_ = 0;
   std::vector<bool> is_streaming_group_;
   // Per group: its support class, -1 for a streaming-only group
   std::vector<PetscInt> class_of_group_;
   // Per class: its reference group
   std::vector<PetscInt> class_ref_;
   std::vector<PetscReal> alpha_;
   std::vector<PetscReal> bin_alpha_;
   std::vector<PetscInt> bin_of_group_;
   std::vector<PetscInt> bin_ref_group_;
   std::vector<Mat> pmats_;
   PetscReal worst_mismatch_ = 1.0;
};

#endif
