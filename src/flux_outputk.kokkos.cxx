#include "ubolt/flux_output.hpp"
#include <petscdmda.h>
#include <petscdmplex.h>
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

// The DMPlex path: one dof-1 twin of the backend's mesh, every field a global
// Vec on it, written as a .vtu
//
// Two things differ from the DMDA path. The backend's mesh carries a one-cell
// overlap, and PETSc's VTU writer writes EVERY cell of each rank's local mesh
// unless a "vtk" label marks the ones to write - so the twin labels its owned
// cells, and each cell appears in the file exactly once. And the vector index
// of a cell is read from the twin's own global section (offset - rstart)
// rather than assumed: the backend's local cell k is the k-th owned cell in
// point order (its CheckPlexLayout asserts the global numbering follows that),
// and this map is what ties the two together here
static PetscErrorCode WritePlexVTU(DM dm, PetscInt local_cells, PetscInt n_fields, \
   const char *const names[], const PetscScalar *const values[], const char *filename)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)dm);
   DM flux_dm = NULL;
   PetscSection sec = NULL, gsec = NULL;
   DMLabel vtk_label = NULL;
   PetscViewer viewer = NULL;
   Vec probe = NULL;
   PetscInt p_start = 0, p_end = 0, c_start = 0, c_end = 0, rstart = 0, k = 0;
   std::vector<PetscInt> vec_index(local_cells, -1);
   std::vector<Vec> fields(n_fields, NULL);

   PetscFunctionBeginUser;

   // A clone shares the topology and copies the labels and coordinates, so
   // the twin's own section and "vtk" label never touch the backend's DM
   PetscCall(DMClone(dm, &flux_dm));
   PetscCall(DMPlexGetChart(flux_dm, &p_start, &p_end));
   PetscCall(DMPlexGetHeightStratum(flux_dm, 0, &c_start, &c_end));
   PetscCall(PetscSectionCreate(comm, &sec));
   // One field with an EMPTY name: the writer names each array vec name +
   // field name, and on a section with no fields at all it prints a null
   // field name, so the arrays came out as "scalar_flux(null)"
   PetscCall(PetscSectionSetNumFields(sec, 1));
   PetscCall(PetscSectionSetFieldName(sec, 0, ""));
   PetscCall(PetscSectionSetChart(sec, p_start, p_end));
   for (PetscInt c = c_start; c < c_end; c++) {
      PetscCall(PetscSectionSetDof(sec, c, 1));
      PetscCall(PetscSectionSetFieldDof(sec, c, 0, 1));
   }
   PetscCall(PetscSectionSetUp(sec));
   PetscCall(DMSetLocalSection(flux_dm, sec));
   PetscCall(PetscSectionDestroy(&sec));
   PetscCall(DMGetGlobalSection(flux_dm, &gsec));

   PetscCall(DMCreateGlobalVector(flux_dm, &probe));
   PetscCall(VecGetOwnershipRange(probe, &rstart, NULL));
   PetscCall(VecDestroy(&probe));

   PetscCall(DMCreateLabel(flux_dm, "vtk"));
   PetscCall(DMGetLabel(flux_dm, "vtk", &vtk_label));
   for (PetscInt c = c_start; c < c_end; c++) {
      PetscInt g = 0;
      PetscCall(PetscSectionGetOffset(gsec, c, &g));
      // Overlap ghosts are encoded negative - someone else writes them
      if (g < 0) continue;
      PetscCheck(k < local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
         "the mesh owns more cells than the phase space's %" PetscInt_FMT, local_cells);
      vec_index[k++] = g - rstart;
      PetscCall(DMLabelSetValue(vtk_label, c, 1));
   }
   PetscCheck(k == local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "the mesh owns %" PetscInt_FMT \
      " cells but the phase space has %" PetscInt_FMT, k, local_cells);

   // The VTK viewer queues each Vec and writes them all as fields of the one
   // file when it is destroyed, so the viewer goes first and the Vecs after
   PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &viewer));
   for (PetscInt f = 0; f < n_fields; f++) {
      PetscScalar *field_a = nullptr;
      PetscCall(DMCreateGlobalVector(flux_dm, &fields[f]));
      PetscCall(PetscObjectSetName((PetscObject)fields[f], names[f]));
      PetscCall(VecGetArray(fields[f], &field_a));
      for (PetscInt c = 0; c < local_cells; c++) field_a[vec_index[c]] = values[f][c];
      PetscCall(VecRestoreArray(fields[f], &field_a));
      PetscCall(VecView(fields[f], viewer));
   }

   PetscCall(PetscViewerDestroy(&viewer));
   for (auto &field : fields) PetscCall(VecDestroy(&field));
   PetscCall(DMDestroy(&flux_dm));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DMDA path: a dof-1 twin of the backend's DMDA (DMDACreateCompatibleDMDA,
// which carries the coordinates over, so the file gets the real mesh
// positions), written as .vts or .vtr
static PetscErrorCode WriteDAVTK(DM da, PetscInt local_cells, PetscInt n_fields, \
   const char *const names[], const PetscScalar *const values[], const char *filename)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)da);
   DM flux_da = NULL;
   PetscViewer viewer = NULL;
   std::vector<Vec> fields(n_fields, NULL);

   PetscFunctionBeginUser;

   PetscCall(DMDACreateCompatibleDMDA(da, 1, &flux_da));

   // The VTK viewer queues each Vec and writes them all as fields of the one
   // file when it is destroyed, so the viewer goes first and the Vecs after
   PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &viewer));
   for (PetscInt f = 0; f < n_fields; f++) {
      PetscCall(QueueCellFieldVTK(flux_da, local_cells, names[f], values[f], viewer, &fields[f]));
   }

   PetscCall(PetscViewerDestroy(&viewer));
   for (auto &field : fields) PetscCall(VecDestroy(&field));
   PetscCall(DMDestroy(&flux_da));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UboltWriteScalarFluxVTK(const PhaseSpace &ps, \
   const Discretisation &disc, const AngularQuadrature &quad, Vec psi, PetscInt n_extra, \
   const UboltCellField *extra, const char *filename)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)psi);
   PetscBool is_plex = PETSC_FALSE, is_vts = PETSC_FALSE, is_vtr = PETSC_FALSE, is_vtu = PETSC_FALSE;
   PetscInt local_rows = 0;

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
   // the format is the mesh's: fail up front with the reason rather than
   // letting the viewer refuse the file at destroy time
   PetscCall(PetscObjectTypeCompare((PetscObject)disc.dm(), DMPLEX, &is_plex));
   PetscCall(PetscStrendswith(filename, ".vts", &is_vts));
   PetscCall(PetscStrendswith(filename, ".vtr", &is_vtr));
   PetscCall(PetscStrendswith(filename, ".vtu", &is_vtu));
   if (is_plex) {
      PetscCheck(is_vtu, comm, PETSC_ERR_ARG_WRONG, \
         "'%s': a scalar flux on the unstructured (DMPlex) backend writes the VTK unstructured " \
         "format, so the filename must end in .vtu (.vts/.vtr are for the structured backends)", filename);
   } else {
      PetscCheck(is_vts || is_vtr, comm, PETSC_ERR_ARG_WRONG, \
         "'%s': a scalar flux on a DMDA writes the VTK structured formats, so the " \
         "filename must end in .vts or .vtr (.vtu is for the unstructured DMPlex backend)", filename);
   }

   // The shared angular integral, so what gets written is bit-identical with
   // what the terms integrate. Per node - (cell, basis) - and what is written
   // is basis 0 of each cell: the cell itself at one dof per cell, the cell
   // AVERAGE with a modal basis whose basis 0 is the constant (DG1)
   PetscScalar2DKokkosView scalar_flux_d("scalar_flux_d", ps.local_nodes(), 1);
   PetscCall(UboltAngularIntegral(psi, ps.n_angles, quad.w_d(), scalar_flux_d));

   // The angular integral writes a (nodes, 1) gemm output, the extra fields are
   // plain per-cell views; both are contiguous on either backend, so the
   // mirror's data() is the field in local node (cell) order in both cases
   auto flux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar_flux_d);
   std::vector<PetscScalar> cell_flux(ps.local_cells);
   for (PetscInt c = 0; c < ps.local_cells; c++) cell_flux[c] = flux_h.data()[c * ps.n_basis];
   std::vector<decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), extra[0].values))> extra_h;
   std::vector<const char *> names(1 + n_extra);
   std::vector<const PetscScalar *> values(1 + n_extra);
   names[0] = "scalar_flux";
   values[0] = cell_flux.data();
   for (PetscInt f = 0; f < n_extra; f++) {
      extra_h.push_back(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), extra[f].values));
      names[1 + f] = extra[f].name;
      values[1 + f] = extra_h.back().data();
   }

   // Every field rides the same twin DM, which is what lets them share one file
   if (is_plex) PetscCall(WritePlexVTU(disc.dm(), ps.local_cells, 1 + n_extra, names.data(), values.data(), filename));
   else PetscCall(WriteDAVTK(disc.dm(), ps.local_cells, 1 + n_extra, names.data(), values.data(), filename));

   PetscFunctionReturn(PETSC_SUCCESS);
}
