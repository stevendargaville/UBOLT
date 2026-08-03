#ifndef UBOLT_UBOLT_HPP
#define UBOLT_UBOLT_HPP

// Umbrella header for libubolt
// types.hpp comes first: it pulls in petscvec_kokkos.hpp, which must precede any
// other PETSc header in a C++ file (see docs/dev/kokkos.md)

#include "ubolt/types.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/material_spec.hpp"
#include "ubolt/coo_pattern.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"
#include "ubolt/problem_spec.hpp"
#include "ubolt/operator_term.hpp"
#include "ubolt/terms.hpp"
#include "ubolt/multigroup.hpp"
#include "ubolt/transport_operator.hpp"
#include "ubolt/dsa.hpp"
#include "ubolt/transport_solver.hpp"
#include "ubolt/flux_output.hpp"

#endif
