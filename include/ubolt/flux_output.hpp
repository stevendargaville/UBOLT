#ifndef UBOLT_FLUX_OUTPUT_HPP
#define UBOLT_FLUX_OUTPUT_HPP

#include "ubolt/types.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/sn_quadrature.hpp"
#include <petscvec.h>

// Write the scalar flux of an angular flux Vec to a VTK file, for looking at
// the solution in ParaView/VisIt
//
// The angular integral is UboltAngularIntegral - the same one the terms use -
// and the result goes onto a dof-1 twin of the discretisation's DMDA
// (DMDACreateCompatibleDMDA, which also carries over the mesh coordinates the
// backend set), so PETSc's own VTK viewer does the parallel gather and the
// writing. That makes the filename extension pick the format, and a DMDA is a
// structured grid: .vts or .vtr. PETSc reserves .vtu for unstructured
// (DMPlex) meshes, so it is not accepted here
//
// Collective on psi's communicator. Allocates its own device scratch, so this
// is for after a solve, not inside one
PETSC_EXTERN PetscErrorCode UboltWriteScalarFluxVTK(const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad, Vec psi, const char *filename);

#endif
