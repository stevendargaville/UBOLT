#ifndef UBOLT_DISCRETISATION_HPP
#define UBOLT_DISCRETISATION_HPP

#include "ubolt/types.hpp"
#include "ubolt/coo_pattern.hpp"
#include "ubolt/phase_space.hpp"
#include <petscmat.h>
#include <petscdm.h>
#include <vector>

// What every discretisation backend owes the rest of the library - and only the
// parts of it that do not depend on the dimension
//
// A backend owns a DM, and through it the mesh, the layout and the parallel
// decomposition; it writes local_cells back into the PhaseSpace and fixes the
// COO sparsity once. What it hands out is a CooPattern (slot maps) plus a
// BoundaryInfo (Dirichlet row mask), and matrices preallocated for that
// sparsity. Terms and TransportOperator address entries through the slot maps
// and never see a raw index, which is exactly why all of that is
// dimension-independent and can live here
//
// GEOMETRY DELIBERATELY DOES NOT. dx()/dy() stay on the concrete backends: the
// only thing that wants them is the streaming term, which owns the upwind slot
// convention and therefore needs a per-dimension sibling regardless. So a
// dimension-specific term takes the concrete class and this base never has to
// grow a geometry interface. create() is per-dimension for the same reason -
// it takes the mesh sizes
class PETSC_VISIBILITY_PUBLIC Discretisation {
public:
   virtual ~Discretisation() = default;

   // A MATAIJKOKKOS matrix with this backend's sparsity preallocated for COO
   // assembly. Can be called more than once (e.g. for a streaming-only pmat)
   //
   // NOT DMCreateMatrix: a DMDA preallocates from its stencil (every point the
   // stencil reaches, times dof), while the upwind operator touches one
   // neighbour per axis. A different sparsity is a different matrix and PCAIR
   // would see it. The DM supplies ownership and indices, not the matrix
   PetscErrorCode create_matrix(Mat *mat) const;

   // The DM is the only PETSc handle a backend owns
   PetscErrorCode destroy();

   const CooPattern &coo_pattern() const { return pattern_; }
   const BoundaryInfo &boundary_info() const { return boundary_; }

   // The DM the layout, the mesh and the decomposition came from
   DM dm() const { return dm_; }

protected:
   // Backends are built through their own create(), never by instantiating this
   Discretisation() = default;

   // Build the slot maps for a backend whose rows all carry the same number of
   // COO entries, with the diagonal LAST (1D: upwind, diagonal; 2D: upwind-x,
   // upwind-y, diagonal). Uploads the Dirichlet mask at the same time
   //
   // ps_ must already carry the decomposition, and oor_/ooc_ must already be
   // filled - this only turns them into the device-side maps terms read
   PetscErrorCode set_uniform_pattern(PetscInt slots_per_row, const std::vector<PetscInt> &is_dirichlet_row);

   MPI_Comm comm_ = MPI_COMM_NULL;
   // Taken once the DM has said what the decomposition is
   PhaseSpace ps_;
   DM dm_ = NULL;
   // COO coordinates, kept so each matrix built from this discretisation can be
   // preallocated from them
   std::vector<PetscInt> oor_;
   std::vector<PetscInt> ooc_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

#endif
