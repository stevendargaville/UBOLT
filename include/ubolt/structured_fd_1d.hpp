#ifndef UBOLT_STRUCTURED_FD_1D_HPP
#define UBOLT_STRUCTURED_FD_1D_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include <petscdmda.h>

// Uniform-grid upwinded finite difference discretisation of a 1D slab
//
// Owns the mesh and the COO sparsity: two entries per row (the upwind neighbour
// and the diagonal, in that order). A BC row loses its upwind entry - a
// Dirichlet row keeps only the diagonal, a reflective row repurposes the slot
// for the -1 coupling to the mirrored angle in the same cell. Everything it
// hands terms - the CooPattern, the BoundaryInfo, the preallocated matrices -
// is on the Discretisation base; what is 1D about it is the mesh it builds and
// dx()
class PETSC_VISIBILITY_PUBLIC StructuredFD1D : public Discretisation {
public:
   // The boundary label ids this backend hands to the BCSpec - the ids
   // PETSc's DMPlexCreateBoxMesh "Face Sets" convention gives a 1D interval,
   // so the future unstructured backend agrees with these
   static constexpr PetscInt FACE_LEFT = 1;
   static constexpr PetscInt FACE_RIGHT = 2;

   // The quadrature decides which neighbour is upwind for each angle, and on a
   // reflective face which angle mirrors which. A default bcs is vacuum
   // (prescribed inflow) on both faces
   //
   // The layout comes from a 1D DMDA, which also decides the parallel
   // decomposition: `ps` is filled in (ps.local_cells) rather than read, so
   // this must be created before anything sized off the phase space
   PetscErrorCode create(MPI_Comm comm, PhaseSpace &ps, PetscReal length, const SNQuadrature &quad, \
      const BCSpec &bcs = BCSpec());

   // Uniform grid so every cell has the same width
   PetscScalar dx() const { return dx_; }

private:
   PetscScalar dx_ = 0.0;
};

#endif
