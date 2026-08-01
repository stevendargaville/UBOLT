#ifndef UBOLT_COO_PATTERN_HPP
#define UBOLT_COO_PATTERN_HPP

#include "ubolt/types.hpp"

// Where each row's entries live in the shared COO values array
//
// A discretisation backend fixes the COO ordering once (in its preallocation)
// and hands out these slot maps; assembled terms address entries through them
// and never touch raw COO positions. See docs/dev/kokkos.md
struct CooPattern {
   // CSR-shaped: row r owns slots [row_slot_offset_d(r), row_slot_offset_d(r+1))
   // Sized local_rows + 1
   PetscIntKokkosView row_slot_offset_d;
   // Which of those slots is the diagonal. Sized local_rows
   PetscIntKokkosView diag_slot_d;
   // Total local COO entries (== row_slot_offset_d(local_rows))
   PetscCount n_slots = 0;
};

// Which rows carry a Dirichlet condition
//
// Contract: a term must contribute NOTHING to a flagged row - not from its
// assembled contribution and not from its matrix-free apply. The assembly step
// puts the identity on them afterwards, so a term never has to know what the
// boundary condition is
struct BoundaryInfo {
   // 1 on Dirichlet rows, 0 elsewhere. Sized local_rows
   PetscIntKokkosView is_dirichlet_row_d;
};

#endif
