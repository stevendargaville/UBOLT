#include "ubolt/flux_output.hpp"
#include <petscdmda.h>
#include <petscviewer.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UboltWriteScalarFluxVTK(const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad, Vec psi, const char *filename)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)psi);
   DM flux_da = NULL;
   Vec phi = NULL;
   PetscViewer viewer = NULL;
   PetscBool is_vts = PETSC_FALSE, is_vtr = PETSC_FALSE;
   PetscInt local_rows = 0;
   PetscScalar *phi_a = nullptr;

   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCall(VecGetLocalSize(psi, &local_rows));
   PetscCheck(local_rows == ps.local_rows(), comm, PETSC_ERR_ARG_INCOMP, \
      "psi has %" PetscInt_FMT " local rows but the phase space says %" PetscInt_FMT, \
      local_rows, ps.local_rows());

   // The extension is what tells PETSc's VTK viewer which format to write, and
   // a DMDA is a structured grid - fail up front with the reason rather than
   // letting the viewer refuse the file at destroy time
   PetscCall(PetscStrendswith(filename, ".vts", &is_vts));
   PetscCall(PetscStrendswith(filename, ".vtr", &is_vtr));
   PetscCheck(is_vts || is_vtr, comm, PETSC_ERR_ARG_WRONG, \
      "'%s': a scalar flux on a DMDA writes the VTK structured formats, so the " \
      "filename must end in .vts or .vtr (.vtu is PETSc's unstructured/DMPlex format)", filename);

   // The shared angular integral, so what gets written is bit-identical with
   // what the terms integrate
   PetscScalar2DKokkosView scalar_flux_d("scalar_flux_d", ps.local_cells, 1);
   PetscCall(UboltAngularIntegral(psi, ps.n_angles, quad.w_d(), scalar_flux_d));

   // A dof-1 twin of the backend's DMDA: same grid, same decomposition, one
   // value per cell. DMDACreateCompatibleDMDA carries the coordinates over from
   // the backend's DM, so the file gets the real mesh positions
   PetscCall(DMDACreateCompatibleDMDA(disc.dm(), 1, &flux_da));

   // A dof-1 DMDA global vector holds the owned patch contiguously in the
   // patch's own lexicographic order - exactly the local cell order the
   // backends' CheckDALayout asserts - so this is a straight copy. The
   // object name is what the viewer labels the field with in the file
   PetscCall(DMCreateGlobalVector(flux_da, &phi));
   PetscCall(PetscObjectSetName((PetscObject)phi, "scalar_flux"));
   auto flux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar_flux_d);
   PetscCall(VecGetArray(phi, &phi_a));
   for (PetscInt c = 0; c < ps.local_cells; c++) phi_a[c] = flux_h(c, 0);
   PetscCall(VecRestoreArray(phi, &phi_a));

   // The VTK viewer queues the Vec and writes the file when it is destroyed,
   // so the viewer goes first
   PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &viewer));
   PetscCall(VecView(phi, viewer));
   PetscCall(PetscViewerDestroy(&viewer));
   PetscCall(VecDestroy(&phi));
   PetscCall(DMDestroy(&flux_da));

   PetscFunctionReturn(PETSC_SUCCESS);
}
