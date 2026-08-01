#ifndef UBOLT_STRUCTURED_FD_1D_HPP
#define UBOLT_STRUCTURED_FD_1D_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include <petscdmda.h>

// Uniform-grid upwinded finite difference discretisation of a 1D slab
//
// Owns the mesh and the COO sparsity: two entries per row (the upwind neighbour
// and the diagonal, in that order), with Dirichlet rows keeping only their
// diagonal. Everything it hands terms - the CooPattern, the BoundaryInfo, the
// preallocated matrices - is on the Discretisation base; what is 1D about it is
// the mesh it builds and dx()
class PETSC_VISIBILITY_PUBLIC StructuredFD1D : public Discretisation {
public:
   // The quadrature decides which neighbour is upwind for each angle.
   //
   // The layout comes from a 1D DMDA, which also decides the parallel
   // decomposition: `ps` is filled in (ps.local_cells) rather than read, so
   // this must be created before anything sized off the phase space
   PetscErrorCode create(MPI_Comm comm, PhaseSpace &ps, PetscReal length, const SNQuadrature &quad);

   // Uniform grid so every cell has the same width
   PetscScalar dx() const { return dx_; }

private:
   PetscScalar dx_ = 0.0;
};

#endif
