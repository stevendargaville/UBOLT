#ifndef UBOLT_STRUCTURED_FD_1D_HPP
#define UBOLT_STRUCTURED_FD_1D_HPP

#include "ubolt/types.hpp"
#include "ubolt/coo_pattern.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include <petscmat.h>
#include <vector>

// Uniform-grid upwinded finite difference discretisation of a 1D slab
//
// Owns the mesh and the COO sparsity: two entries per row (the upwind neighbour
// and the diagonal, in that order), with Dirichlet rows keeping only their
// diagonal. Hands terms a CooPattern + BoundaryInfo and knows nothing about the
// physics that fills it
class PETSC_VISIBILITY_PUBLIC StructuredFD1D {
public:
   // The quadrature decides which neighbour is upwind for each angle
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, PetscReal length, const SNQuadrature &quad);

   // A MATAIJKOKKOS matrix with this sparsity preallocated for COO assembly.
   // Can be called more than once (e.g. for a streaming-only pmat)
   PetscErrorCode create_matrix(Mat *mat) const;

   const CooPattern &coo_pattern() const { return pattern_; }
   const BoundaryInfo &boundary_info() const { return boundary_; }

   // Uniform grid so every cell has the same width
   PetscScalar dx() const { return dx_; }

private:
   MPI_Comm comm_ = MPI_COMM_NULL;
   PhaseSpace ps_;
   PetscScalar dx_ = 0.0;
   PetscInt global_row_start_ = 0;
   // COO coordinates, kept so each matrix built from this discretisation can be
   // preallocated from them
   std::vector<PetscInt> oor_;
   std::vector<PetscInt> ooc_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

#endif
