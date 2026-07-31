#ifndef UBOLT_PHASE_SPACE_HPP
#define UBOLT_PHASE_SPACE_HPP

#include <petscsys.h>

// The discrete phase space: how many spatial cells and how many angles, and how
// they are split over the ranks
//
// The parallel decomposition is decided in CELLS. All the angles on a cell live
// on the same rank, so a row split can never land mid-cell (with PETSC_DECIDE
// over rows it can, e.g. 1000 cells x 4 angles on 3 ranks)
//
// Dof ordering is angle-fastest: row = cell * n_angles + angle
struct PhaseSpace {
   PetscInt n_cells     = 0;  // global number of spatial cells
   PetscInt n_angles    = 0;
   PetscInt local_cells = 0;  // cells owned by this rank

   PetscInt global_rows() const { return n_cells * n_angles; }
   PetscInt local_rows() const { return local_cells * n_angles; }

   PetscErrorCode create(MPI_Comm comm, PetscInt n_cells_in, PetscInt n_angles_in)
   {
      PetscFunctionBeginUser;

      PetscCheck(n_cells_in > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "n_cells must be positive, was given %" PetscInt_FMT, n_cells_in);
      PetscCheck(n_angles_in > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "n_angles must be positive, was given %" PetscInt_FMT, n_angles_in);

      n_cells  = n_cells_in;
      n_angles = n_angles_in;

      local_cells = PETSC_DECIDE;
      PetscCall(PetscSplitOwnership(comm, &local_cells, &n_cells));

      PetscFunctionReturn(PETSC_SUCCESS);
   }
};

#endif
