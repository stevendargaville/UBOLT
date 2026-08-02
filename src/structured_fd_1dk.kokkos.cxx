#include "ubolt/structured_fd_1d.hpp"
#include "petsc_kokkos.hpp"
#include <petscdmda.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build a 1D DMDA over the cells: one dof per angle, stencil width 1 because
// the upwind operator reaches exactly one neighbour
//
// We deliberately do NOT call DMSetFromOptions - it would expose -da_grid_x,
// which could resize the mesh out from under the PhaseSpace that every other
// object has already been sized from. The sizes come from the PhaseSpace, which
// the drivers already expose as -n_cells/-n_angles
//
// The DM chooses the distribution and the PhaseSpace is told what it decided.
// PETSc's default 1D DMDA split is M/size + ((M % size) > rank), which is the
// same formula PetscSplitOwnership uses, so this is the same decomposition
// UBOLT had before the DM owned it (Phase 3a verified that bitwise)
static PetscErrorCode CreateDA(MPI_Comm comm, const PhaseSpace &ps, DM *da)
{
   PetscFunctionBeginUser;

   PetscCall(DMDACreate1d(comm, DM_BOUNDARY_NONE, ps.n_cells, ps.n_angles, 1, NULL, da));
   PetscCall(DMSetUp(*da));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check the DM's layout is the one every COO index below is written against:
// angle-fastest dof ordering (row = cell * n_angles + angle), contiguous
// ownership, and a decomposition matching the PhaseSpace's
//
// 1D DMDA ordering coincides with the natural ordering, so this passes - but it
// is load-bearing for every index we write, and a silent disagreement would
// show up as a wrong answer rather than an error, so check it once at setup
static PetscErrorCode CheckDALayout(DM da, const PhaseSpace &ps, PetscInt cell_start)
{
   const PetscInt local_cells = ps.local_cells;
   PetscInt dm_cells = 0, dm_dof = 0, rstart = 0, rend = 0;
   Vec gv;

   PetscFunctionBeginUser;

   PetscCall(DMDAGetInfo(da, NULL, &dm_cells, NULL, NULL, NULL, NULL, NULL, &dm_dof, \
      NULL, NULL, NULL, NULL, NULL));
   PetscCheck(dm_cells == ps.n_cells && dm_dof == ps.n_angles, PetscObjectComm((PetscObject)da), \
      PETSC_ERR_PLIB, "DMDA is %" PetscInt_FMT " cells x %" PetscInt_FMT " dof, phase space is %" \
      PetscInt_FMT " x %" PetscInt_FMT, dm_cells, dm_dof, ps.n_cells, ps.n_angles);

   // Angle-fastest and contiguous: the rows the DM's global vector owns must be
   // exactly [cell_start * n_angles, (cell_start + local_cells) * n_angles)
   PetscCall(DMCreateGlobalVector(da, &gv));
   PetscCall(VecGetOwnershipRange(gv, &rstart, &rend));
   PetscCheck(rstart == cell_start * ps.n_angles && rend - rstart == local_cells * ps.n_angles, \
      PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, \
      "DMDA global rows [%" PetscInt_FMT ", %" PetscInt_FMT ") are not the angle-fastest rows [%" \
      PetscInt_FMT ", %" PetscInt_FMT ")", rstart, rend, cell_start * ps.n_angles, \
      (cell_start + local_cells) * ps.n_angles);
   PetscCall(VecDestroy(&gv));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build the COO sparsity and the slot maps
// This happens on the host but we only need to do it once
PetscErrorCode StructuredFD1D::create(MPI_Comm comm, PhaseSpace &ps, PetscReal length, const SNQuadrature &quad, \
   const BCSpec &bcs)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   comm_ = comm;

   PetscCheck(quad.n_angles() == ps.n_angles, comm_, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);

   // Uniform grid so every cell has the same width
   dx_ = length / ps.n_cells;

   const PetscInt n_angles = ps.n_angles;
   const PetscScalar *mu = quad.mu_host();

   // The DM decides the parallel decomposition, so building it has to come
   // before anything that reads local_cells - starting with the PhaseSpace,
   // which we fill in rather than read
   PetscInt cell_start = 0, local_cells = 0;
   PetscCall(CreateDA(comm, ps, &dm_));
   // DMDAGetCorners divides the dof back out, so these come out in cells
   PetscCall(DMDAGetCorners(dm_, &cell_start, NULL, NULL, &local_cells, NULL, NULL));
   cell_start_x_ = cell_start;
   ps.local_cells = local_cells;
   PetscCall(CheckDALayout(dm_, ps, cell_start));

   // The DM owns the mesh, so it carries the node positions too: node i sits at
   // i dx, so the last one is at length - dx. Nothing in the solve reads these -
   // they are for output (the scalar flux VTK writer picks them up). A
   // single-cell slab has no spacing to define and PETSc would divide by
   // n_cells - 1, so it keeps no coordinates
   if (ps.n_cells > 1) PetscCall(DMDASetUniformCoordinates(dm_, 0.0, \
      length - length / ps.n_cells, 0.0, 0.0, 0.0, 0.0));

   ps_ = ps;
   const PetscInt local_rows = ps.local_rows();
   const PetscInt global_row_start = cell_start * n_angles;

   const PetscBool on_left_boundary  = (PetscBool)(cell_start == 0);
   const PetscBool on_right_boundary = (PetscBool)(cell_start + local_cells == ps.n_cells);

   // ~~~~~~~~~~
   // COO coordinates - two entries per row
   // ~~~~~~~~~~
   oor_.assign(2 * local_rows, 0);
   ooc_.assign(2 * local_rows, 0);

   // Loop over local cells, which have n_angles unknowns on them
   // We add two non-zeros per unknown. On boundary rows we still include two,
   // and make the preallocation ignore one of them, so that the value fills
   // don't have to change depending on the boundary condition
   PetscCount counter = 0;
   for (PetscInt i = 0; i < local_cells; i++)
   {
      for (PetscInt a = 0; a < n_angles; a++) {

         // Upwinded uniform grid finite difference operator
         // The upwind neighbour is on the left for positive velocities and on
         // the right for negative ones. The diagonal is always written last so
         // the value fills don't have to detect the direction
         const PetscInt upwind_cell = (mu[a] > 0) ? i - 1 : i + 1;

         oor_[counter] = i * n_angles + a + global_row_start;
         ooc_[counter] = upwind_cell * n_angles + a + global_row_start;

         oor_[counter + 1] = i * n_angles + a + global_row_start;
         ooc_[counter + 1] = i * n_angles + a + global_row_start;

         counter = counter + 2;
      }
   }

   // ~~~~~~~~~~
   // Boundary conditions on the left and right boundaries
   // ~~~~~~~~~~
   // Incoming directions get their equation replaced by the boundary
   // condition. On a vacuum face that is the identity: a -1 row/col index in
   // the COO format says just ignore this entry, so the row keeps only its
   // diagonal and the rhs supplies the incoming flux. On a reflective face the
   // row is psi(cell, a) - psi(cell, a') = 0 with a' the mirrored angle, so
   // the upwind slot is repurposed to point at the partner - same cell, so the
   // column is always rank-local - and the assembly writes the -1.0 there
   std::vector<PetscInt> is_bc_row(local_rows, 0);
   std::vector<PetscInt> reflect_slot(local_rows, -1);
   const PetscInt *reflect_mu = quad.reflect_mu_host();

   // All positive angles are incoming on the left boundary
   if (on_left_boundary)
   {
      const BCType left_bc = bcs.type(FACE_LEFT);
      for (PetscInt a = 0; a < n_angles; a++) {
         if (mu[a] > 0)
         {
            is_bc_row[a] = 1;
            if (left_bc == BCType::REFLECT)
            {
               // The mirrored angle is outgoing here, so its row is a real
               // unknown - unless the slab is a single cell and the other
               // face makes it a BC row too
               PetscCheck(ps.n_cells > 1, comm_, PETSC_ERR_SUP, \
                  "a reflective face on a single-cell slab couples two boundary rows");
               oor_[a * 2] = a + global_row_start;
               ooc_[a * 2] = reflect_mu[a] + global_row_start;
               reflect_slot[a] = a * 2;
            }
            else
            {
               oor_[a * 2] = -1;
               ooc_[a * 2] = -1;
            }
         }
      }
   }
   // All negative angles are incoming on the right boundary
   if (on_right_boundary)
   {
      const BCType right_bc = bcs.type(FACE_RIGHT);
      for (PetscInt a = 0; a < n_angles; a++) {
         if (mu[a] < 0)
         {
            const PetscInt r = (local_cells - 1) * n_angles + a;
            is_bc_row[r] = 1;
            if (right_bc == BCType::REFLECT)
            {
               PetscCheck(ps.n_cells > 1, comm_, PETSC_ERR_SUP, \
                  "a reflective face on a single-cell slab couples two boundary rows");
               oor_[r * 2] = r + global_row_start;
               ooc_[r * 2] = (local_cells - 1) * n_angles + reflect_mu[a] + global_row_start;
               reflect_slot[r] = r * 2;
            }
            else
            {
               oor_[r * 2] = -1;
               ooc_[r * 2] = -1;
            }
         }
      }
   }

   // Slot maps - row r owns COO slots 2r (upwind neighbour) and 2r + 1 (diagonal)
   PetscCall(set_uniform_pattern(2, is_bc_row, reflect_slot));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda: an extended device lambda can't
// be defined inside another lambda. The interval list rides along as flat
// views - coordinates as (x0, x1) pairs - because a view of structs is not
// one of the typedefs types.hpp allows
static void PaintIntervalsKernel(PetscIntKokkosView mat_id_d, PetscInt background_material, \
   PetscScalarKokkosView interval_x_d, PetscIntKokkosView interval_material_d, PetscInt n_intervals, \
   PetscInt cell_start_x, PetscScalar dx, PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

         // The centre of local cell c - node i sits at i dx
         const PetscScalar xc = ((PetscScalar)(cell_start_x + c) + 0.5) * dx;

         PetscInt m = background_material;
         for (PetscInt s = 0; s < n_intervals; s++) {
            if (xc >= interval_x_d(2 * s) && xc <= interval_x_d(2 * s + 1)) m = interval_material_d(s);
         }
         mat_id_d(c) = m;
      });
}

// See the declaration for the painting rules (background, later intervals win,
// membership by cell centre)
PetscErrorCode StructuredFD1D::paint_intervals(PetscInt background_material, \
   const std::vector<MaterialInterval1D> &intervals, PetscIntKokkosView &mat_id_d) const
{
   const PetscInt n_intervals = (PetscInt)intervals.size();

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps_.check_decomposed());

   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);
   for (PetscInt s = 0; s < n_intervals; s++) {
      PetscCheck(intervals[s].material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
         "interval %" PetscInt_FMT "'s material index %" PetscInt_FMT " is negative", s, intervals[s].material);
      PetscCheck(intervals[s].x0 <= intervals[s].x1, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
         "interval %" PetscInt_FMT " is inside out - give it as x0,x1 with x0 <= x1", s);
   }

   // Flatten the intervals for the device
   std::vector<PetscScalar> interval_x(2 * n_intervals);
   std::vector<PetscInt> interval_material(n_intervals);
   for (PetscInt s = 0; s < n_intervals; s++) {
      interval_x[2 * s]     = intervals[s].x0;
      interval_x[2 * s + 1] = intervals[s].x1;
      interval_material[s] = intervals[s].material;
   }

   PetscScalarKokkosView interval_x_d("interval_x_d", 2 * n_intervals);
   PetscIntKokkosView interval_material_d("interval_material_d", n_intervals);
   if (n_intervals > 0) {
      PetscScalarKokkosViewHostUnmanaged interval_x_h(interval_x.data(), 2 * n_intervals);
      PetscIntKokkosViewHostUnmanaged interval_material_h(interval_material.data(), n_intervals);
      Kokkos::deep_copy(interval_x_d, interval_x_h);
      Kokkos::deep_copy(interval_material_d, interval_material_h);
   }

   mat_id_d = PetscIntKokkosView("mat_id_d", ps_.local_cells);
   PaintIntervalsKernel(mat_id_d, background_material, interval_x_d, interval_material_d, \
      n_intervals, cell_start_x_, dx_, ps_.local_cells);

   PetscFunctionReturn(PETSC_SUCCESS);
}
