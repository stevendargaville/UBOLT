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
// BC row mask - not from assemble_add, and not from apply_add either. The
// assembly step writes those rows itself - the identity, plus the -1.0
// reflection coupling on reflective rows - and a matrix-free term adding to
// them afterwards would take that straight back off, leaving the operator's
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
   //
   // Both are read by TransportOperator at ASSEMBLE time, not when the term is
   // added, so a term that can be either (see RemovalTerm::set_matrix_free) can
   // be switched any time before the first TransportOperator::assemble()
   virtual PetscBool assembled() const { return PETSC_FALSE; }
   virtual PetscBool matrix_free() const { return PETSC_FALSE; }

   // Add this term's contribution into the shared COO values (device)
   virtual PetscErrorCode assemble_add(PetscScalarKokkosView &) const
   {
      PetscFunctionBeginUser;
      PetscFunctionReturn(PETSC_SUCCESS);
   }
   // Add this term's action to y: y += term * x
   virtual PetscErrorCode apply_add(Vec, Vec) const
   {
      PetscFunctionBeginUser;
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // Does this term sit on the streaming/removal operator's diagonal, and if so
   // add it into d. TransportOperator::diagonal() composes the whole diagonal
   // that way, for when a term carrying one is applied matrix-free and is
   // therefore missing from the assembled matrix - MatGetDiagonal would then
   // hand the Jacobi stage of the preconditioner a silently incomplete diagonal
   //
   // "The diagonal" is the ASSEMBLED operator's: streaming plus removal.
   // ScatteringTerm deliberately does not override this. It has never been in
   // the assembled matrix, so nothing in the preconditioner has ever seen its
   // diagonal, and composing it in would be a different preconditioner rather
   // than the same one written another way
   //
   // A term that does override it owes the composition the SAME arithmetic its
   // assemble_add writes into the diagonal slot, so the composed value matches
   // the assembled one bitwise - transportk's -check_matfree pins exactly that.
   // BC rows are not a term's business here either: the composition writes them
   // itself, the 1.0 the assembly puts on them
   virtual PetscBool has_diagonal() const { return PETSC_FALSE; }
   virtual PetscErrorCode add_diagonal(Vec) const
   {
      PetscFunctionBeginUser;
      PetscFunctionReturn(PETSC_SUCCESS);
   }
};

#endif
