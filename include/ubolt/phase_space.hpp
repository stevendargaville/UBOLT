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
// and the backend's create() writes local_cells back here. Until that create
// runs, local_cells is PETSC_DECIDE, so anything sized off it must be built
// AFTER the discretisation and calls check_decomposed() to say so
//
// Dof ordering is angle-fastest: row = (cell * n_basis + basis) * n_angles +
// angle. n_basis is the number of SPATIAL dofs per cell - 1 for the finite
// difference and DG0 backends, where this is row = cell * n_angles + angle -
// and like local_cells it is the discretisation's to decide: its create()
// writes it here (UnstructuredDG at order 1 has dimension + 1). A (cell,
// basis) pair is a "node", and every node owns a contiguous run of n_angles
// rows, which is what lets the dimension-independent terms work on nodes and
// never learn what a basis function is: the angular integral is per node, the
// xsections are per CELL, node / n_basis
//
// Energy groups are NOT part of the row count. Groups are solved one at a time
// (one Vec and one system per group, swept in a group Gauss-Seidel), so the
// row counts below - and therefore the sparsity, the COO pattern and the
// assembled matrix - are the same for every group
struct PETSC_VISIBILITY_PUBLIC PhaseSpace {
   PetscInt n_cells     = 0;  // global number of spatial cells
   PetscInt n_angles    = 0;
   PetscInt n_groups    = 1;  // energy groups, ordered high energy to low
   // Spatial dofs per cell - see above. Written by the discretisation's create()
   PetscInt n_basis     = 1;
   // Cells owned by this rank. PETSC_DECIDE until the discretisation fills it
   PetscInt local_cells = PETSC_DECIDE;

   // Per group - see the note above
   PetscInt global_rows() const { return n_cells * n_basis * n_angles; }
   PetscInt local_rows() const { return local_cells * n_basis * n_angles; }
   // (cell, basis) pairs on this rank, each a contiguous run of n_angles rows -
   // what the angular integral and the scalar flux are sized on
   PetscInt local_nodes() const { return local_cells * n_basis; }
   // Rows per cell: row / rows_per_cell() is the local cell of a local row
   PetscInt rows_per_cell() const { return n_basis * n_angles; }

   // Has the discretisation handed us its decomposition yet? Everything sized
   // off local_cells calls this, because the alternative to a clear error here
   // is a Kokkos view allocated with a negative extent somewhere downstream
   PetscErrorCode check_decomposed() const
   {
      PetscFunctionBeginUser;

      PetscCheck(local_cells >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
         "PhaseSpace has no decomposition yet - the discretisation's create() " \
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
      // Left for the discretisation's DM to fill - see the note above. One
      // spatial dof per cell until a backend says otherwise
      local_cells = PETSC_DECIDE;
      n_basis = 1;

      PetscFunctionReturn(PETSC_SUCCESS);
   }
};

#endif
