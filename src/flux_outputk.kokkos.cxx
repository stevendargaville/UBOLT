#include "ubolt/flux_output.hpp"
#include "ubolt/unstructured_cg.hpp"
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

// A dof-1 twin of a plex backend's mesh: a clone (it shares the topology and
// copies the labels and coordinates, so nothing here touches the backend's DM)
// with one dof on every point of the stratum [start, end) - cells, or the
// vertices for the CG flux - under one field with an EMPTY name: the writer
// names each array vec name + field name, and on a section with no fields at
// all it prints a null field name ("scalar_flux(null)"). A POINT array is
// named vec name + field name + "." + component name whatever the component
// count, so `component` gives it a real one
static PetscErrorCode CreateStratumTwin(DM dm, PetscInt start, PetscInt end, const char *component, DM *twin)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)dm);
   PetscSection sec = NULL;
   PetscInt p_start = 0, p_end = 0;

   PetscFunctionBeginUser;

   PetscCall(DMClone(dm, twin));
   PetscCall(DMPlexGetChart(*twin, &p_start, &p_end));
   PetscCall(PetscSectionCreate(comm, &sec));
   PetscCall(PetscSectionSetNumFields(sec, 1));
   PetscCall(PetscSectionSetFieldName(sec, 0, ""));
   if (component) PetscCall(PetscSectionSetComponentName(sec, 0, 0, component));
   PetscCall(PetscSectionSetChart(sec, p_start, p_end));
   for (PetscInt p = start; p < end; p++) {
      PetscCall(PetscSectionSetDof(sec, p, 1));
      PetscCall(PetscSectionSetFieldDof(sec, p, 0, 1));
   }
   PetscCall(PetscSectionSetUp(sec));
   PetscCall(DMSetLocalSection(*twin, sec));
   PetscCall(PetscSectionDestroy(&sec));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// PETSc's VTU writer writes EVERY cell of each rank's local mesh unless a "vtk"
// label marks the ones to write, and the plex backends' meshes carry an
// overlap - so the twin labels the cells this rank owns (owned[c - c_start])
// and each cell appears in the file exactly once
static PetscErrorCode LabelOwnedCells(DM twin, PetscInt c_start, const std::vector<PetscInt> &owned)
{
   DMLabel vtk_label = NULL;

   PetscFunctionBeginUser;

   PetscCall(DMCreateLabel(twin, "vtk"));
   PetscCall(DMGetLabel(twin, "vtk", &vtk_label));
   for (size_t k = 0; k < owned.size(); k++) {
      if (owned[k]) PetscCall(DMLabelSetValue(vtk_label, c_start + (PetscInt)k, 1));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DG path: every field a global Vec on the cell twin, written as a .vtu.
// A cell's vector index is read from the twin's own global section (offset -
// rstart) rather than assumed: the backend's local cell k is the k-th owned
// cell in point order (its CheckPlexLayout asserts the global numbering
// follows that), and this map is what ties the two together here
static PetscErrorCode WritePlexVTU(DM dm, PetscInt local_cells, PetscInt n_fields, \
   const char *const names[], const PetscScalar *const values[], const char *filename)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)dm);
   DM flux_dm = NULL;
   PetscSection gsec = NULL;
   PetscViewer viewer = NULL;
   Vec probe = NULL;
   PetscInt c_start = 0, c_end = 0, rstart = 0, k = 0;
   std::vector<PetscInt> vec_index(local_cells, -1);
   std::vector<Vec> fields(n_fields, NULL);

   PetscFunctionBeginUser;

   PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
   PetscCall(CreateStratumTwin(dm, c_start, c_end, NULL, &flux_dm));
   PetscCall(DMGetGlobalSection(flux_dm, &gsec));

   PetscCall(DMCreateGlobalVector(flux_dm, &probe));
   PetscCall(VecGetOwnershipRange(probe, &rstart, NULL));
   PetscCall(VecDestroy(&probe));

   std::vector<PetscInt> owned(c_end - c_start, 0);
   for (PetscInt c = c_start; c < c_end; c++) {
      PetscInt g = 0;
      PetscCall(PetscSectionGetOffset(gsec, c, &g));
      // Overlap ghosts are encoded negative - someone else writes them
      if (g < 0) continue;
      PetscCheck(k < local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
         "the mesh owns more cells than the phase space's %" PetscInt_FMT, local_cells);
      vec_index[k++] = g - rstart;
      owned[c - c_start] = 1;
   }
   PetscCheck(k == local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "the mesh owns %" PetscInt_FMT \
      " cells but the phase space has %" PetscInt_FMT, k, local_cells);
   PetscCall(LabelOwnedCells(flux_dm, c_start, owned));

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

// The CG path: the flux lives on VERTICES, so it is POINT data on a vertex
// twin ("scalar_flux.nodal"), and the per-element extras CELL data on a cell
// twin. Each rank writes the cells it owns and their vertices, so the overlap
// vertices of those cells need their owners' values: the owned flux goes into
// the vertex twin's global Vec by its own global section and out to its local
// Vec. Fields from two grids in one file are queued through
// PetscViewerVTKAddField directly with checkdm off, since VecView would refuse
// the second grid; the writer reads each field through its own Vec's DM, and
// the viewer takes over the local Vecs
static PetscErrorCode WriteCGVTU(const PhaseSpace &ps, const UnstructuredCG &disc, const PetscScalar *flux, \
   PetscInt n_extra, const UboltCellField *extra, const char *filename)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)disc.dm());
   PetscInt v_start = 0, v_end = 0, c_start = 0, c_end = 0, rstart = 0;
   DM vertex_twin = NULL, cell_twin = NULL;
   PetscSection gsec = NULL, lsec = NULL;
   PetscViewer viewer = NULL;
   Vec flux_global = NULL, flux_local = NULL;

   PetscFunctionBeginUser;

   PetscCall(DMPlexGetDepthStratum(disc.dm(), 0, &v_start, &v_end));
   PetscCall(DMPlexGetHeightStratum(disc.dm(), 0, &c_start, &c_end));
   const std::vector<PetscInt> &cell_owned = disc.element_owned_host();

   PetscCall(CreateStratumTwin(disc.dm(), v_start, v_end, "nodal", &vertex_twin));
   PetscCall(LabelOwnedCells(vertex_twin, c_start, cell_owned));
   PetscCall(DMGetGlobalSection(vertex_twin, &gsec));
   PetscCall(DMCreateGlobalVector(vertex_twin, &flux_global));
   PetscCall(VecGetOwnershipRange(flux_global, &rstart, NULL));
   {
      PetscScalar *a = nullptr;
      const std::vector<PetscInt> &owned_lv = disc.owned_local_vertex_host();
      PetscCall(VecGetArray(flux_global, &a));
      for (PetscInt k = 0; k < ps.local_cells; k++) {
         PetscInt g = 0;
         PetscCall(PetscSectionGetOffset(gsec, v_start + owned_lv[k], &g));
         PetscCheck(g >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "owned vertex %" PetscInt_FMT " has no global offset", k);
         a[g - rstart] = flux[k];
      }
      PetscCall(VecRestoreArray(flux_global, &a));
   }
   PetscCall(DMCreateLocalVector(vertex_twin, &flux_local));
   PetscCall(DMGlobalToLocal(vertex_twin, flux_global, INSERT_VALUES, flux_local));
   PetscCall(VecDestroy(&flux_global));
   PetscCall(PetscObjectSetName((PetscObject)flux_local, "scalar_flux"));

   PetscCall(PetscViewerVTKOpen(comm, filename, FILE_MODE_WRITE, &viewer));
   PetscCall(PetscViewerVTKAddField(viewer, (PetscObject)vertex_twin, DMPlexVTKWriteAll, 0, PETSC_VTK_POINT_FIELD, \
      PETSC_FALSE, (PetscObject)flux_local));

   if (n_extra > 0) {
      PetscCall(CreateStratumTwin(disc.dm(), c_start, c_end, NULL, &cell_twin));
      PetscCall(LabelOwnedCells(cell_twin, c_start, cell_owned));
      PetscCall(DMGetLocalSection(cell_twin, &lsec));
      for (PetscInt f = 0; f < n_extra; f++) {
         Vec loc = NULL;
         PetscScalar *a = nullptr;
         auto values_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), extra[f].values);
         PetscCall(DMCreateLocalVector(cell_twin, &loc));
         PetscCall(PetscObjectSetName((PetscObject)loc, extra[f].name));
         PetscCall(VecGetArray(loc, &a));
         for (PetscInt c = c_start; c < c_end; c++) {
            PetscInt off = 0;
            PetscCall(PetscSectionGetOffset(lsec, c, &off));
            a[off] = values_h(c - c_start);
         }
         PetscCall(VecRestoreArray(loc, &a));
         PetscCall(PetscViewerVTKAddField(viewer, (PetscObject)vertex_twin, DMPlexVTKWriteAll, 0, \
            PETSC_VTK_CELL_FIELD, PETSC_FALSE, (PetscObject)loc));
      }
   }

   PetscCall(PetscViewerDestroy(&viewer));
   PetscCall(DMDestroy(&cell_twin));
   PetscCall(DMDestroy(&vertex_twin));

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
   // Host side, output time: the one backend whose unknowns are not per cell
   const UnstructuredCG *cg = dynamic_cast<const UnstructuredCG *>(&disc);

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
      PetscCheck((PetscInt)extra[f].values.extent(0) == disc.n_material_entries(), comm, PETSC_ERR_ARG_INCOMP, \
         "extra field '%s' covers %" PetscInt_FMT " entries but the backend has %" PetscInt_FMT \
         " (local cells, local elements on CG-SUPG)", extra[f].name, (PetscInt)extra[f].values.extent(0), \
         disc.n_material_entries());
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
         "'%s': a scalar flux on the unstructured (DMPlex) backends writes the VTK unstructured " \
         "format, so the filename must end in .vtu (.vts/.vtr are for the structured backends)", filename);
   } else {
      PetscCheck(is_vts || is_vtr, comm, PETSC_ERR_ARG_WRONG, \
         "'%s': a scalar flux on a DMDA writes the VTK structured formats, so the " \
         "filename must end in .vts or .vtr (.vtu is for the unstructured DMPlex backends)", filename);
   }

   // The shared angular integral, so what gets written is bit-identical with
   // what the terms integrate. Per node - (cell, basis), or vertex on CG - and
   // the (nodes, 1) gemm output is contiguous, so the mirror's data() is the
   // field in local node order
   PetscScalar2DKokkosView scalar_flux_d("scalar_flux_d", ps.local_nodes(), 1);
   PetscCall(UboltAngularIntegral(psi, ps.n_angles, quad.w_d(), scalar_flux_d));
   auto flux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar_flux_d);

   if (cg) {
      PetscCall(WriteCGVTU(ps, *cg, flux_h.data(), n_extra, extra, filename));
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // Per cell from here: basis 0 of each cell is the cell itself at one dof per
   // cell, the cell AVERAGE with a modal basis whose basis 0 is the constant
   // (DG1). The extra fields are plain per-cell views
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
