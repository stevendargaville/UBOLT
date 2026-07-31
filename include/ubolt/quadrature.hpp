#ifndef UBOLT_QUADRATURE_HPP
#define UBOLT_QUADRATURE_HPP

#include <petscsys.h>

// SN quadrature (Gauss-Legendre for n_angles = 2 or 4)
// mu and w must have room for n_angles entries
PETSC_EXTERN PetscErrorCode create_sn_quadrature(PetscInt n_angles, PetscScalar *mu, PetscScalar *w, PetscScalar *sum_weights);

#endif
