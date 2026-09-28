#ifndef UBOLT_OPERATOR_TERM_HPP
#define UBOLT_OPERATOR_TERM_HPP

#include "ubolt/types.hpp"
#include <petscmat.h>

class GroupXSections;

// One physical term of the transport operator
//
// A term contributes either an assembled block (adding into the shared COO
// values through the discretisation's slot maps) or a matrix-free apply. New
// physics = a new subclass; nothing else has to change
//
// Contract: a term must contribute NOTHING to rows flagged in the BoundaryInfo
// BC row mask - not from assemble_add, and not from apply_add either. The
// assembly step writes those rows itself - the identity, plus the -1.0
// reflection coupling on reflective rows - and a matrix-free term adding to
// them afterwards would take that straight back off. A term whose apply needs
// the mask takes a Discretisation in its create, as the assembled terms do
class PETSC_VISIBILITY_PUBLIC OperatorTerm {
public:
   virtual ~OperatorTerm() = default;

   // Does this term contribute to the assembled matrix / to the matrix-free
   // apply. At most one of the two at a time: TransportOperator uses both
   // answers as given, so a term answering yes to both is applied twice
   //
   // Both are read by TransportOperator at ASSEMBLE time, not when the term is
   // added, so a term that can be either (see RemovalTerm::set_matrix_free) can
   // be switched any time before the first TransportOperator::assemble()
   virtual PetscBool assembled() const { return PETSC_FALSE; }
   virtual PetscBool matrix_free() const { return PETSC_FALSE; }

   // Point the term at energy group g's cross sections. A pointer swap, no
   // device work: an assembled term's values change at the next
   // TransportOperator::assemble(), a matrix-free one reads them straight
   // through. The default is for a term that does not depend on the group
   virtual PetscErrorCode set_group(const GroupXSections &, PetscInt)
   {
      PetscFunctionBeginUser;
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // Add this term's contribution into the shared COO values (device)
   virtual PetscErrorCode assemble_add(const PetscScalarKokkosView &) const
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
   // that way, for when a term carrying one is applied matrix-free and so is
   // missing from the assembled matrix
   //
   // "The diagonal" is the ASSEMBLED operator's: streaming plus removal. The
   // scatter is never in it - composing it in would be a different
   // preconditioner, not the same one written another way
   //
   // An override owes the SAME arithmetic its assemble_add writes into the
   // diagonal slot, so the composed value matches the assembled one bitwise
   // (transportk's -check_matfree pins it). BC rows are the composition's
   virtual PetscBool has_diagonal() const { return PETSC_FALSE; }
   virtual PetscErrorCode add_diagonal(Vec) const
   {
      PetscFunctionBeginUser;
      PetscFunctionReturn(PETSC_SUCCESS);
   }
};

#endif
