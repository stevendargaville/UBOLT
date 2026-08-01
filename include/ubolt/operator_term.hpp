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
// Contract: a term must contribute NOTHING to rows flagged in the BoundaryInfo
// Dirichlet mask - not from assemble_add, and not from apply_add either. The
// assembly step puts the identity on those rows, and a matrix-free term adding
// to them afterwards would take it straight back off, leaving the operator's
// boundary rows something other than the boundary condition
//
// This used to be stated for assemble_add alone, and ScatteringTerm::apply_add
// did in fact write to the Dirichlet rows (fixed Aug 2026, its own commit and
// baseline re-capture - see TODO.md). A term whose apply needs the mask takes a
// Discretisation in its create, as the assembled terms do
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
