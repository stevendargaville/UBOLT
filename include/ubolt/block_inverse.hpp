#ifndef UBOLT_BLOCK_INVERSE_HPP
#define UBOLT_BLOCK_INVERSE_HPP

#include "ubolt/types.hpp"
#include "ubolt/phase_space.hpp"
#include <petscmat.h>

// The inverse of each element block of a matrix over the phase space, and the
// left scaling by it: D^{-1} x and D^{-1} A, all on the device
//
// An element block is the n_basis x n_basis coupling of one (cell, angle)
// between its own basis functions - for DG1, the upwind cell-local block, the
// thing that has to be inverted for a DG discretisation to look like DG0 to an
// algebraic multigrid; at n_basis 1 it is just the diagonal entry, so a
// one-dof-per-cell backend gets point-diagonal scaling out of the same code.
// Under the library's layout A, rows (cell * n_basis + i) * n_angles + a, the
// rows of block (c, a) are STRIDED by n_angles - this is why PETSc's
// contiguous-block MatInvertBlockDiagonal does not fit, and why the blocks are
// read here straight out of the matrix's CSR rather than through PETSc's
// variable-block interface
//
// The solver uses it twice: the removal stage (composite index 0) IS D_op^{-1},
// the operator's blocks, on every backend - point Jacobi at n_basis 1 - and the
// block-scaled streaming stage builds PCAIR on D^{-1} pmat
//
// Why the matrix and not the terms: the matrix the blocks come from is
// whatever pmat the solver was handed - the assembled operator, a
// streaming-only copy, a reference-shifted one, or one -diag_scale has already
// scaled - and reading it is the only way every one of those gets the blocks of
// the matrix it actually carries. The blocks are cell-local, so they sit in
// the diagonal (owned) part of an MPIAIJ matrix and nothing is communicated
//
// No destroy(): this owns device views only, no PETSc handle. It must be gone
// (out of scope, or deleted) before PetscFinalize takes Kokkos down
class PETSC_VISIBILITY_PUBLIC ElementBlockInverse {
public:
   // Blocks of n_basis x n_basis, local_cells * n_angles of them. The block
   // size is capped at MAX_BLOCK (DG1 in 3D is 4) - the inversion is a
   // per-thread dense Gauss-Jordan on stack arrays of that size
   static constexpr PetscInt MAX_BLOCK = 4;
   PetscErrorCode create(const PhaseSpace &ps);

   // Read every element block of A and invert it (partial pivoting). A must be
   // a MATAIJKOKKOS matrix (seq or MPI) over the phase space's rows, square,
   // with the phase space's parallel layout. A missing block entry reads as
   // zero; a singular block is an error
   //
   // diag, if given, replaces the blocks' diagonal entries (a Vec over A's
   // rows): the blocks are A's off-diagonals around diag. That is how the
   // removal stage gets the operator's blocks when a diagonal-carrying term
   // is applied matrix-free, so A is missing it - the removal is diagonal, so
   // the assembled off-diagonals are the whole story and the composed
   // diagonal supplies the rest
   PetscErrorCode setup(Mat A, Vec diag = NULL);

   // y = D^{-1} x over the local rows. x and y may not alias
   PetscErrorCode apply(Vec x, Vec y) const;

   // scaled = D^{-1} A, with the SAME nonzero pattern as A. That holds because
   // every row of a block carries the same columns - true of every UBOLT
   // backend, whose slot maps give a node's rows one pattern per (cell, angle)
   // - and it is checked, on the device, every call. MAT_INITIAL_MATRIX
   // duplicates A's pattern into *scaled; MAT_REUSE_MATRIX writes the values
   // of an *scaled from an earlier call, and bumps its state (the MPI wrapper's
   // too, not only its seq parts) so a PC built on it sees the new values.
   // Uses the blocks from the last setup(), which must have been off this same A
   PetscErrorCode scale(Mat A, MatReuse reuse, Mat *scaled) const;

   PetscInt block_size() const { return nb_; }
   PetscInt n_blocks() const { return n_blocks_; }
   // The inverted blocks, n_blocks * nb * nb flat: ((c * n_angles + a) * nb
   // + i) * nb + j, row i column j of block (c, a) - for the tests
   const PetscScalarKokkosView &inverse_d() const { return inv_d_; }

private:
   PetscInt n_angles_ = 0;
   PetscInt nb_ = 1;
   PetscInt local_cells_ = 0;
   PetscInt n_blocks_ = 0;
   PetscScalarKokkosView inv_d_;
};

#endif
