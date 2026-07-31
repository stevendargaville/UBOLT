#ifndef UBOLT_PRECON_HPP
#define UBOLT_PRECON_HPP

#include <petscksp.h>

// Create the PETSc preconditioner: a multiplicative composite of a shell
// preconditioner for the removal term and PCAIR for the streaming term
PETSC_EXTERN PetscErrorCode create_pc(PC pc, Mat A_stream_removal);

#endif
