#ifndef UBOLT_FLUX_OUTPUT_HPP
#define UBOLT_FLUX_OUTPUT_HPP

#include "ubolt/types.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/sn_quadrature.hpp"
#include <petscvec.h>

// One extra field to write alongside the scalar flux: a name and one value per
// material entry - the local cells in the backends' local cell order, the
// local elements on CG-SUPG - i.e. what GroupXSections::sigma_t and
// UboltFillCellSource hand back. The name is what the field is labelled with
// in the file
struct UboltCellField {
   const char *name;
   PetscScalarKokkosView values;
};

// Write the scalar flux of an angular flux Vec to a VTK file, for looking at
// the solution in ParaView/VisIt, plus any number of extra per-cell fields on
// the same mesh - the data that went INTO the solve (sigma_t, the external
// source), so a file can be read without going back to the problem definition
// to find out what was solved
//
// With more than one spatial dof per cell (DG1) the scalar_flux field is the
// CELL AVERAGE - basis 0 of the modal basis. The in-cell slope is not a
// PhaseSpace concept, so a caller that wants it passes it as extra fields
// (UnstructuredDG::scalar_flux_gradient, which the driver writes)
//
// The angular integral is UboltAngularIntegral - the same one the terms use -
// and the result goes onto a dof-1 twin of the discretisation's DM, so PETSc's
// own VTK viewer does the parallel gather and the writing. That makes the
// filename extension pick the format, and the format is the mesh's:
// - a DMDA backend (the structured ones) is a structured grid: .vts or .vtr.
//   The twin is DMDACreateCompatibleDMDA, which also carries over the mesh
//   coordinates the backend set
// - a DMPlex backend is an unstructured grid: .vtu. The twin is a DMClone
//   with one dof per cell, and only OWNED cells are written (PETSc's VTU
//   writer would otherwise write the overlap twice). On CG-SUPG the unknowns
//   live on vertices, so the flux is POINT data, "scalar_flux.nodal" (PETSc
//   appends a component name to a point array), and the extras are cell data
// The wrong extension for the backend is an error saying which one to use
//
// Collective on psi's communicator. Allocates its own device scratch, so this
// is for after a solve, not inside one
PETSC_EXTERN PetscErrorCode UboltWriteScalarFluxVTK(const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad, Vec psi, PetscInt n_extra, \
   const UboltCellField *extra, const char *filename);

// The same file from a scalar flux already integrated: scalar_flux_d is
// (local_nodes, 1), per node in local node order - what UboltAngularIntegral
// writes and GroupTransfer::scalar_flux caches. For a caller that keeps the
// moments rather than the angular flux (tests/transportk -outer_store
// moments); UboltWriteScalarFluxVTK is this after the integral. Collective on
// the discretisation's communicator
PETSC_EXTERN PetscErrorCode UboltWriteScalarFluxVTKFromPhi(const PhaseSpace &ps, \
   const Discretisation &disc, const PetscScalar2DKokkosView &scalar_flux_d, PetscInt n_extra, \
   const UboltCellField *extra, const char *filename);

#endif
