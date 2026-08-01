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
// This struct does NOT decide the decomposition - the discretisation's DM does,
// and StructuredFD1D::create writes local_cells back here (Phase 3b). Until
// then local_cells is PETSC_DECIDE, so anything sized off it must be built
// AFTER the discretisation and calls check_decomposed() to say so
//
// Dof ordering is angle-fastest: row = cell * n_angles + angle
//
// Energy groups are NOT part of the row count. Groups are solved one at a time
// (one Vec and one system per group, swept in a group Gauss-Seidel), so the
// row counts below - and therefore the sparsity, the COO pattern and the
// assembled matrix - are the same for every group
struct PhaseSpace {
   PetscInt n_cells     = 0;  // global number of spatial cells
   PetscInt n_angles    = 0;
   PetscInt n_groups    = 1;  // energy groups, ordered high energy to low
   // Cells owned by this rank. PETSC_DECIDE until the discretisation fills it
   PetscInt local_cells = PETSC_DECIDE;

   // Per group - see the note above
   PetscInt global_rows() const { return n_cells * n_angles; }
   PetscInt local_rows() const { return local_cells * n_angles; }

   // Has the discretisation handed us its decomposition yet? Everything sized
   // off local_cells calls this, because the alternative to a clear error here
   // is a Kokkos view allocated with a negative extent somewhere downstream
   PetscErrorCode check_decomposed() const
   {
      PetscFunctionBeginUser;

      PetscCheck(local_cells >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
         "PhaseSpace has no decomposition yet - the discretisation (StructuredFD1D::create) " \
         "decides it, so build this after the discretisation");

      PetscFunctionReturn(PETSC_SUCCESS);
   }

   PetscErrorCode create(MPI_Comm comm, PetscInt n_cells_in, PetscInt n_angles_in, PetscInt n_groups_in = 1)
   {
      PetscFunctionBeginUser;

      PetscCheck(n_cells_in > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "n_cells must be positive, was given %" PetscInt_FMT, n_cells_in);
      PetscCheck(n_angles_in > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "n_angles must be positive, was given %" PetscInt_FMT, n_angles_in);
      PetscCheck(n_groups_in > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
         "n_groups must be positive, was given %" PetscInt_FMT, n_groups_in);

      n_cells  = n_cells_in;
      n_angles = n_angles_in;
      n_groups = n_groups_in;
      // Left for the discretisation's DM to fill - see the note above
      local_cells = PETSC_DECIDE;

      PetscFunctionReturn(PETSC_SUCCESS);
   }
};

#endif
