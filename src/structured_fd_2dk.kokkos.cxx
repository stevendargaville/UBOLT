#include "ubolt/structured_fd_2d.hpp"
#include "structured_fd_commonk.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build the DMDA, the COO sparsity and the slot maps - on the host, once
PetscErrorCode StructuredFD2D::create(MPI_Comm comm, PhaseSpace &ps, PetscInt n_cells_x, PetscInt n_cells_y, \
   PetscReal length_x, PetscReal length_y, const SNQuadrature2D &quad, const BCSpec &bcs)
{
   const PetscInt *ltog = nullptr;
   ISLocalToGlobalMapping ltog_map = NULL;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(quad.n_angles() == ps.n_angles, comm, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck(n_cells_x > 0 && n_cells_y > 0 && n_cells_x * n_cells_y == ps.n_cells, comm, \
      PETSC_ERR_ARG_INCOMP, "a %" PetscInt_FMT " x %" PetscInt_FMT " grid is not the phase space's %" \
      PetscInt_FMT " cells", n_cells_x, n_cells_y, ps.n_cells);

   comm_ = comm;
   n_cells_x_ = n_cells_x;
   n_cells_y_ = n_cells_y;
   // Uniform grid so every cell has the same width
   dx_ = length_x / n_cells_x;
   dy_ = length_y / n_cells_y;

   // One dof per angle, star stencil of width 1 (one neighbour per axis). No
   // DMSetFromOptions: -da_grid_x/y could resize the mesh out from under the
   // PhaseSpace. The DM chooses the processor decomposition and the
   // PhaseSpace is told what it decided
   PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, \
      n_cells_x, n_cells_y, PETSC_DECIDE, PETSC_DECIDE, ps.n_angles, 1, NULL, NULL, &dm_));
   PetscCall(DMSetUp(dm_));

   const PetscInt n[2] = {n_cells_x, n_cells_y};
   DAPatch<2> patch;
   PetscCall(GetDAPatch<2>(dm_, n, &patch));
   cell_start_x_ = patch.start[0];
   cell_start_y_ = patch.start[1];
   local_cells_x_ = patch.m[0];
   local_cells_y_ = patch.m[1];
   ps.local_cells = patch.local_cells();

   // Node (i, j) sits at (i dx, j dy), for output only (the VTK writer). A
   // single-node direction has no spacing (PETSc would divide by n - 1), so
   // that degenerate grid keeps no coordinates
   if (n_cells_x > 1 && n_cells_y > 1) PetscCall(DMDASetUniformCoordinates(dm_, \
      0.0, length_x - length_x / n_cells_x, 0.0, length_y - length_y / n_cells_y, 0.0, 0.0));

   PetscCall(DMGetLocalToGlobalMapping(dm_, &ltog_map));
   PetscCall(ISLocalToGlobalMappingGetIndices(ltog_map, &ltog));
   PetscCall(CheckDALayout<2>(dm_, ps, ltog, patch));

   ps_ = ps;

   // Slot order upwind-x, upwind-y, diagonal
   const PetscInt face_id[2][2] = {{FACE_LEFT, FACE_RIGHT}, {FACE_BOTTOM, FACE_TOP}};
   AxisBC axis_bc[2];
   PetscCall(BuildAxisBC<2>(comm_, bcs, face_id, quad.sum_weights(), axis_bc));
   const PetscScalar h[2] = {dx_, dy_};
   const PetscScalar *const cosine[2] = {quad.mu_host(), quad.eta_host()};
   const PetscInt *const reflect[2] = {quad.reflect_mu_host(), quad.reflect_eta_host()};
   BoundaryRows rows;
   PetscCall(BuildStructuredCOO<2>(comm_, patch, ltog, ps.n_angles, h, cosine, reflect, axis_bc, \
      bcs.ghost_flux_vacuum(), oor_, ooc_, rows));

   PetscCall(ISLocalToGlobalMappingRestoreIndices(ltog_map, &ltog));

   PetscCall(set_uniform_pattern(3, rows));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode StructuredFD2D::paint_boxes(PetscInt background_material, \
   const std::vector<MaterialBox2D> &boxes, PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());
   const PetscInt start[3] = {cell_start_x_, cell_start_y_, 0};
   const PetscInt m[3] = {local_cells_x_, local_cells_y_, 1};
   const PetscScalar h[3] = {dx_, dy_, 0.0};
   PetscCall(PaintStructuredBoxes(comm_, background_material, boxes, start, m, h, ps_.local_cells, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
