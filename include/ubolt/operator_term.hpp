#ifndef UBOLT_OPERATOR_TERM_HPP
#define UBOLT_OPERATOR_TERM_HPP

#include "ubolt/types.hpp"
#include <petscmat.h>

// One physical term of the transport operator
//
// A term contributes either an assembled block (adding into the shared COO
// values through the discretisation's slot maps) or a matrix-free apply, or
// both. New physics = a new subclass; nothing else has to change
//
// Contract: assemble_add must contribute NOTHING to rows flagged in the
// BoundaryInfo Dirichlet mask - the assembly step puts the identity there
class PETSC_VISIBILITY_PUBLIC OperatorTerm {
public:
   virtual ~OperatorTerm() = default;

   // Does this term contribute to the assembled matrix / to the matrix-free apply
   virtual PetscBool assembled() const { return PETSC_FALSE; }
   virtual PetscBool matrix_free() const { return PETSC_FALSE; }

   // Add this term's contribution into the shared COO values (device)
   virtual PetscErrorCode assemble_add(PetscScalarKokkosView &) const { return PETSC_SUCCESS; }
   // Add this term's action to y: y += term * x
   virtual PetscErrorCode apply_add(Vec, Vec) const { return PETSC_SUCCESS; }
};

#endif
