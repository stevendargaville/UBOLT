#ifndef UBOLT_STREAMING_HPP
#define UBOLT_STREAMING_HPP

// types.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/types.hpp"
#include <petscmat.h>

// Create the preallocation structure for the assembled streaming/removal
// operator for a single group - happens on the host, only called once
PETSC_EXTERN PetscErrorCode preallocate_streaming_removal(PetscInt n_angles, const PetscScalar *mu, Mat A_stream_removal);

// Insert the streaming term for a single group - happens entirely on the device
// dx is the (uniform) cell width
PETSC_EXTERN PetscErrorCode insert_streaming(PetscInt n_angles, PetscScalar dx, PetscScalarKokkosView &mu_d, \
   PetscScalarKokkosView &coo_v_d, \
   Mat A_stream_removal);

// Add the removal term for a single group - happens entirely on the device
PETSC_EXTERN PetscErrorCode add_removal(PetscInt n_angles, PetscScalarKokkosView &mu_d, \
   PetscScalarKokkosView &sigma_t_d, \
   PetscScalarKokkosView &coo_v_d, \
   Mat A_stream_removal);

#endif
