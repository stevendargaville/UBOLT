#ifndef UBOLT_DISCRETISATION_HPP
#define UBOLT_DISCRETISATION_HPP

#include "ubolt/types.hpp"
#include "ubolt/coo_pattern.hpp"
#include "ubolt/phase_space.hpp"
#include <petscmat.h>
#include <petscdm.h>
#include <vector>

// The per-row boundary data a backend builds on the host before set_pattern
// uploads it into the BoundaryInfo (see there for what each entry means).
// reset() gives every row the "no boundary condition" defaults
struct PETSC_VISIBILITY_PUBLIC BoundaryRows {
   std::vector<PetscInt> is_bc_row;          // 0
   std::vector<PetscInt> reflect_slot;       // -1
   std::vector<PetscScalar> dirichlet_value; // 0.0
   std::vector<PetscScalar> ghost_inflow;    // 0.0, summed from 0.0 face by face
   PetscBool ghost_flux = PETSC_FALSE;       // the BCSpec's VacuumTreatment is GHOST_FLUX

   void reset(PetscInt local_rows, PetscBool ghost_flux_in)
   {
      is_bc_row.assign(local_rows, 0);
      reflect_slot.assign(local_rows, -1);
      dirichlet_value.assign(local_rows, 0.0);
      ghost_inflow.assign(local_rows, 0.0);
      ghost_flux = ghost_flux_in;
   }
};

// What every discretisation backend owes the rest of the library - and only the
// parts of it that do not depend on the dimension
//
// A backend owns a DM, and through it the mesh, the layout and the parallel
// decomposition; it writes local_cells back into the PhaseSpace and fixes the
// COO sparsity once. What it hands out is a CooPattern (slot maps) plus a
// BoundaryInfo (BC row mask + reflect slots), and matrices preallocated for
// that sparsity. Terms and TransportOperator address entries through the slot
// maps and never see a raw index, which is why all of that lives here
//
// Geometry does not: the only thing that wants it is the streaming term, which
// owns the upwind slot convention and so has a sibling per backend anyway. A
// geometry-dependent term takes the concrete class
//
// Writing a backend - its create() must:
//  1. set comm_ and build dm_ (the DM decides the decomposition),
//  2. write ps.local_cells (and ps.n_basis if not 1) and copy ps into ps_,
//  3. fill oor_/ooc_ in slot order, -1/-1 for a nulled slot (the slot still
//     exists in the values array, terms may write it, PETSc drops it),
//  4. fill a BoundaryRows and call set_uniform_pattern or set_pattern,
//  5. override destroy() if it owns a PETSc handle beyond dm_, and
//     n_material_entries() if cross sections are not per local cell
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

   // Virtual because a backend may own a second handle (UnstructuredCG's
   // vertex twin) and the driver destroys through this base
   virtual PetscErrorCode destroy();

   const CooPattern &coo_pattern() const { return pattern_; }
   const BoundaryInfo &boundary_info() const { return boundary_; }

   // The DM the layout, the mesh and the decomposition came from
   DM dm() const { return dm_; }
   MPI_Comm comm() const { return comm_; }
   // The phase space as create() left it: local_cells and n_basis filled in
   const PhaseSpace &phase_space() const { return ps_; }

   // How many entries a per-cell material table (material ids, cross
   // sections, sources) has on this rank: the local cells, or whatever
   // spatial unit the backend's materials live on (every local element,
   // overlap included, on UnstructuredCG)
   virtual PetscInt n_material_entries() const { return ps_.local_cells; }

protected:
   // Backends are built through their own create(), never by instantiating this
   Discretisation() = default;

   // Slot maps for a backend whose rows all carry the same number of COO
   // entries, the diagonal LAST (1D: upwind, diagonal; 2D: upwind-x,
   // upwind-y, diagonal). A wrapper over set_pattern
   PetscErrorCode set_uniform_pattern(PetscInt slots_per_row, const BoundaryRows &rows);

   // Slot maps for a backend whose rows carry DIFFERENT numbers of COO entries.
   // row_slot_offset is CSR-shaped (local_rows + 1, last entry == oor_.size()),
   // diag_slot is per row and must lie inside that row's range. Uploads the
   // boundary rows too. ps_ and oor_/ooc_ must already be filled
   PetscErrorCode set_pattern(const std::vector<PetscInt> &row_slot_offset, const std::vector<PetscInt> &diag_slot, \
      const BoundaryRows &rows);

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
