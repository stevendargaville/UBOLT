#ifndef UBOLT_MATSHELL_HPP
#define UBOLT_MATSHELL_HPP

// types.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/types.hpp"
#include <petscmat.h>

// Create the matshell which we use to apply streaming/removal + scattering
// The device views handed in are stored by pointer in the shell context, so
// they have to outlive the solve
PETSC_EXTERN PetscErrorCode create_matshell(PetscInt n_angles, Mat A_stream_removal, Mat *A, \
   PetscScalarKokkosView *sigma_s_d, \
   PetscScalar2DKokkosView *scalar_flux_d, \
   PetscScalar2DKokkosView *w_d, \
   PetscScalar sum_weights);

// Free the shell context - call before MatDestroy
PETSC_EXTERN PetscErrorCode destroy_matshell_ctx(Mat A);

#endif
