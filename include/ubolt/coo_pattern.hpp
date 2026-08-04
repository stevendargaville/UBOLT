#ifndef UBOLT_COO_PATTERN_HPP
#define UBOLT_COO_PATTERN_HPP

#include "ubolt/types.hpp"

// Where each row's entries live in the shared COO values array
//
// A discretisation backend fixes the COO ordering once (in its preallocation)
// and hands out these slot maps; assembled terms address entries through them
// and never touch raw COO positions. See docs/dev/kokkos.md
struct PETSC_VISIBILITY_PUBLIC CooPattern {
   // CSR-shaped: row r owns slots [row_slot_offset_d(r), row_slot_offset_d(r+1))
   // Sized local_rows + 1
   PetscIntKokkosView row_slot_offset_d;
   // Which of those slots is the diagonal. Sized local_rows
   PetscIntKokkosView diag_slot_d;
   // Total local COO entries (== row_slot_offset_d(local_rows))
   PetscCount n_slots = 0;
};

// Which rows carry a boundary condition, and what kind
//
// Contract: a term must contribute NOTHING to a flagged row - not from its
// assembled contribution and not from its matrix-free apply. The assembly step
// writes the boundary rows afterwards, so a term never has to know what the
// boundary condition is. Two kinds exist:
// - Dirichlet (prescribed inflow, the "vacuum" family): the row is the
//   identity, and the rhs there carries the incoming flux value - which is
//   per row, and is dirichlet_value_d below
// - reflective: psi_r - psi_partner = 0, the identity plus a -1.0 in one
//   repurposed slot whose column the backend pointed at (same cell, mirrored
//   angle) at preallocation time. The rhs there must be zero
struct PETSC_VISIBILITY_PUBLIC BoundaryInfo {
   // 1 on rows the assembly owns (Dirichlet AND reflective), 0 elsewhere.
   // Sized local_rows
   PetscIntKokkosView is_bc_row_d;
   // The COO values-array slot carrying the -1.0 reflection coupling, -1 if
   // the row is not reflective - so >= 0 doubles as "is a reflect row".
   // Sized local_rows
   PetscIntKokkosView reflect_slot_d;
   // The PER-ANGLE value each Dirichlet row's rhs carries - the winning
   // face's angle-integrated inflow already divided by the quadrature's
   // sum_weights, zeroed outside that face's window - and 0.0 on
   // every row that is not Dirichlet (interior and reflective rows alike).
   // Computed once by the backend at create time. When a row is incoming
   // through more than one vacuum face (a corner), the winning face is the
   // FIRST vacuum incoming face in axis order x, y, z - the order ClassifyRow
   // checks - and the window test applies on that face. Sized local_rows
   PetscScalarKokkosView dirichlet_value_d;
};

// Write the per-row inflow onto the Dirichlet rows of b (other rows are
// untouched): call on a zeroed b, before UboltFillSource
PETSC_EXTERN PetscErrorCode UboltFillInflow(const BoundaryInfo &boundary, Vec b);

// Zero b on the reflective rows: their equation is psi_r - psi_partner = 0, so
// whatever the driver filled b with (source, inflow value) must not stand
// there. Call after filling b and before any diagonal scaling
PETSC_EXTERN PetscErrorCode UboltZeroReflectRows(const BoundaryInfo &boundary, Vec b);

#endif
