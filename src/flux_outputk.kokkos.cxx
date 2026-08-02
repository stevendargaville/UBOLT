#include "ubolt/flux_output.hpp"
#include <petscdmda.h>
#include <petscviewer.h>
#include <vector>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// One per-cell field onto a dof-1 DMDA global vector and into the viewer's
// queue. A dof-1 DMDA global vector holds the owned patch contiguously in the
// patch's own lexicographic order - exactly the local cell order the backends'
// CheckDALayout asserts - so this is a straight copy. The object name is what
// the viewer labels the field with in the file
//
// The Vec comes back out rather than being destroyed here: the VTK viewer
// writes at destroy time, so the caller keeps every field alive until the file
// is on disk
static PetscErrorCode QueueCellFieldVTK(DM flux_da, PetscInt local_cells, const char *name, \
   const PetscScalar *values_h, PetscViewer viewer, Vec *field)
{
   PetscScalar *field_a = nullptr;

   PetscFunctionBeginUser;

   PetscCall(DMCreateGlobalVector(flux_da, field));
   PetscCall(PetscObjectSetName((PetscObject)*field, name));
   PetscCall(VecGetArray(*field, &field_a));
   for (PetscInt c = 0; c < local_cells; c++) field_a[c] = values_h[c];
   PetscCall(VecRestoreArray(*field, &field_a));
   PetscCall(VecView(*field, viewer));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UboltWriteScalarFluxVTK(const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad, Vec psi, PetscInt n_extra, \
   const UboltCellField *extra, const char *filename)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)psi);
   DM flux_da = NULL;
   PetscViewer viewer = NULL;
   PetscBool is_vts = PETSC_FALSE, is_vtr = PETSC_FALSE;
   PetscInt local_rows = 0;
   std::vector<Vec> fields(1 + n_extra, NULL);

   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCall(VecGetLocalSize(psi, &local_rows));
   PetscCheck(local_rows == ps.local_rows(), comm, PETSC_ERR_ARG_INCOMP, \
      "psi has %" PetscInt_FMT " local rows but the phase space says %" PetscInt_FMT, \
      local_rows, ps.local_rows());
   PetscCheck(n_extra == 0 || extra, comm, PETSC_ERR_ARG_NULL, \
      "%" PetscInt_FMT " extra fields asked for but none given", n_extra);
   for (PetscInt f = 0; f < n_extra; f++) {
      PetscCheck((PetscInt)extra[f].values.extent(0) == ps.local_cells, comm, PETSC_ERR_ARG_INCOMP, \
         "extra field '%s' covers %" PetscInt_FMT " cells but there are %" PetscInt_FMT \
         " local cells", extra[f].name, (PetscInt)extra[f].values.extent(0), ps.local_cells);
   }

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

   // The VTK viewer queues each Vec and writes them all as fields of the one
   // file when it is destroyed, so the viewer goes first and the Vecs after
   PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &viewer));

   // The angular integral writes a (cells, 1) gemm output, the extra fields are
   // plain per-cell views; both are contiguous in cell on either backend, so
   // the mirror's data() is the field in local cell order in both cases
   auto flux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar_flux_d);
   PetscCall(QueueCellFieldVTK(flux_da, ps.local_cells, "scalar_flux", flux_h.data(), viewer, \
      &fields[0]));
   for (PetscInt f = 0; f < n_extra; f++) {
      auto extra_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), extra[f].values);
      PetscCall(QueueCellFieldVTK(flux_da, ps.local_cells, extra[f].name, extra_h.data(), viewer, \
         &fields[1 + f]));
   }

   PetscCall(PetscViewerDestroy(&viewer));
   for (auto &field : fields) PetscCall(VecDestroy(&field));
   PetscCall(DMDestroy(&flux_da));

   PetscFunctionReturn(PETSC_SUCCESS);
}
