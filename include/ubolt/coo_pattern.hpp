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
   //
   // Under VacuumTreatment::GHOST_FLUX there are no Dirichlet rows, so this is
   // zero everywhere and ghost_inflow_d below carries the boundary value instead
   PetscScalarKokkosView dirichlet_value_d;
   // The GHOST_FLUX rhs contribution: sum over the row's vacuum inflow FACES of
   // |Omega_axis| / h_axis times that face's per-angle inflow, windowed. Unlike
   // dirichlet_value_d this SUMS over faces - a corner cell is fed through both
   // of them - and it is ADDED to the rhs rather than written over it, because
   // a ghost row also carries the external source. Zero everywhere under the
   // default treatment. Sized local_rows
   PetscScalarKokkosView ghost_inflow_d;
   // Is any row in this discretisation a ghost-flux row? Host-side only, so
   // the default treatment can skip the extra rhs kernel entirely and stay
   // bitwise what it was
   PetscBool ghost_flux_vacuum = PETSC_FALSE;
};

// Put the boundary inflow into b: the per-row value written ONTO each
// Dirichlet row (other rows untouched) under the default vacuum treatment, and
// ADDED to each ghost-flux row under VacuumTreatment::GHOST_FLUX. Call on a
// zeroed b, before UboltFillSource
PETSC_EXTERN PetscErrorCode UboltFillInflow(const BoundaryInfo &boundary, Vec b);

// Zero b on the reflective rows: their equation is psi_r - psi_partner = 0, so
// whatever the driver filled b with (source, inflow value) must not stand
// there. Call after filling b and before any diagonal scaling
PETSC_EXTERN PetscErrorCode UboltZeroReflectRows(const BoundaryInfo &boundary, Vec b);

#endif
