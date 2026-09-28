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
// Under VacuumTreatment::GHOST_FLUX (the default) no row is flagged: every
// boundary value is a face flux, and only ghost_inflow_d below is non-trivial.
// Under DIRICHLET_CELL two kinds of row exist:
// - Dirichlet (prescribed inflow, the "vacuum" family): the row is the
//   identity, and the rhs there carries the incoming flux, dirichlet_value_d
// - reflective: psi_r - psi_partner = 0, the identity plus a -1.0 in one
//   repurposed slot whose column the backend pointed at (same cell, mirrored
//   angle) at preallocation time. The rhs there must be zero
//
// Contract: a term must contribute NOTHING to a flagged row - not from its
// assembled contribution and not from its matrix-free apply. The assembly step
// writes the boundary rows afterwards, so a term never has to know what the
// boundary condition is
struct PETSC_VISIBILITY_PUBLIC BoundaryInfo {
   // 1 on rows the assembly owns (Dirichlet AND reflective), 0 elsewhere.
   // Sized local_rows
   PetscIntKokkosView is_bc_row_d;
   // The COO values-array slot carrying the -1.0 reflection coupling, -1 if
   // the row is not reflective - so >= 0 doubles as "is a reflect row".
   // Sized local_rows
   PetscIntKokkosView reflect_slot_d;
   // The PER-ANGLE value each Dirichlet row's rhs carries - the winning
   // face's angle-integrated inflow divided by the quadrature's sum_weights,
   // zeroed outside that face's window - and 0.0 on every other row. At a
   // corner the winning face is the backend's to pick (the first vacuum
   // inflow face in axis order x, y, z on the structured ones). Sized
   // local_rows
   PetscScalarKokkosView dirichlet_value_d;
   // The GHOST_FLUX rhs contribution: summed over the row's vacuum inflow
   // faces, the upwind face coefficient times that face's per-angle inflow,
   // windowed per face. ADDED to the rhs, because a ghost row also carries
   // the external source. Allocated only under GHOST_FLUX. Sized local_rows
   PetscScalarKokkosView ghost_inflow_d;
   // Host-side flags, so the rhs helpers launch only the kernels that can
   // write something: the treatment is ghost-flux, and whether this rank has
   // any Dirichlet / reflective row
   PetscBool ghost_flux_vacuum = PETSC_FALSE;
   PetscBool has_dirichlet_rows = PETSC_FALSE;
   PetscBool has_reflect_rows = PETSC_FALSE;
};

// Put the boundary inflow into b: the per-row value written ONTO each
// Dirichlet row (other rows untouched) under VacuumTreatment::DIRICHLET_CELL,
// and ADDED onto the rows under the default GHOST_FLUX. Call on a zeroed b,
// before UboltFillSource
PETSC_EXTERN PetscErrorCode UboltFillInflow(const BoundaryInfo &boundary, Vec b);

// Zero b on the reflective rows: their equation is psi_r - psi_partner = 0, so
// whatever the driver filled b with (source, inflow value) must not stand
// there. Call after filling b and before any diagonal scaling
PETSC_EXTERN PetscErrorCode UboltZeroReflectRows(const BoundaryInfo &boundary, Vec b);

#endif
