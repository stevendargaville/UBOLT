#ifndef UBOLT_STRUCTURED_FD_2D_HPP
#define UBOLT_STRUCTURED_FD_2D_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/material_regions.hpp"
#include <vector>

// Uniform-grid upwinded finite difference discretisation of a 2D (XY) box
//
// THREE COO entries per row - upwind-x, upwind-y, diagonal - not the five the
// DMDA star stencil would preallocate (hence no DMCreateMatrix, see the
// Discretisation base). The upwind neighbour on each axis is fixed per angle
// at preallocation from the cosine's sign, so the value fills never branch; a
// zero cosine nulls its slot
//
// Under VacuumTreatment::GHOST_FLUX (the default) there are no BC rows: each
// slot pointing out of the box takes its own face's ghost value - nulled with
// |cosine| / h times the face's inflow summed into
// BoundaryInfo::ghost_inflow_d on a vacuum face, pointed at the mirrored angle
// in the same cell on a reflective one - so a mixed corner takes both, with no
// precedence rule. Under DIRICHLET_CELL an inflow row is a BC row: Dirichlet
// if any face it comes in through is vacuum (the x face's value and window at
// a corner), else identity minus the angle mirrored over every inflow axis.
// The rules live with the shared code in src/structured_fd_commonk.hpp; the
// long form is docs/architecture.md
class PETSC_VISIBILITY_PUBLIC StructuredFD2D : public Discretisation {
public:
   // The boundary label ids this backend hands to the BCSpec - PETSc's
   // DMPlexCreateBoxMesh "Face Sets" ids for a 2D box, which UnstructuredDG/CG
   // share. Any other id in a BCSpec is an error
   static constexpr PetscInt FACE_BOTTOM = 1;
   static constexpr PetscInt FACE_RIGHT = 2;
   static constexpr PetscInt FACE_TOP = 3;
   static constexpr PetscInt FACE_LEFT = 4;

   // The quadrature decides which neighbours are upwind for each angle, and on
   // a reflective face which angle mirrors which. A default bcs is vacuum
   // (prescribed inflow) on all four faces
   //
   // ps.n_cells must be n_cells_x * n_cells_y. The layout comes from a 2D
   // DMDA, which also decides the parallel decomposition: `ps` is filled in
   // (ps.local_cells) rather than read, so this must be created before
   // anything sized off the phase space
   PetscErrorCode create(MPI_Comm comm, PhaseSpace &ps, PetscInt n_cells_x, PetscInt n_cells_y, \
      PetscReal length_x, PetscReal length_y, const SNQuadrature2D &quad, const BCSpec &bcs = BCSpec());

   // Uniform grid so every cell has the same width
   PetscScalar dx() const { return dx_; }
   PetscScalar dy() const { return dy_; }

   PetscInt n_cells_x() const { return n_cells_x_; }
   PetscInt n_cells_y() const { return n_cells_y_; }

   // This rank's patch of the grid. Local cell index c, which is what the
   // per-cell xsection views are indexed by, is the patch's own lexicographic
   // ordering: c = (j - cell_start_y) * local_cells_x + (i - cell_start_x)
   PetscInt cell_start_x() const { return cell_start_x_; }
   PetscInt cell_start_y() const { return cell_start_y_; }
   PetscInt local_cells_x() const { return local_cells_x_; }
   PetscInt local_cells_y() const { return local_cells_y_; }

   // Paint material indices onto this rank's cells: the background everywhere,
   // then the boxes in order with the later ones winning, membership decided by
   // the cell CENTRE, inclusive. Allocates mat_id_d sized local_cells - the
   // per-cell material index view MaterialSpec's fill steps consume
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox2D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;

private:
   PetscScalar dx_ = 0.0;
   PetscScalar dy_ = 0.0;
   PetscInt n_cells_x_ = 0;
   PetscInt n_cells_y_ = 0;
   PetscInt cell_start_x_ = 0;
   PetscInt cell_start_y_ = 0;
   PetscInt local_cells_x_ = 0;
   PetscInt local_cells_y_ = 0;
};

#endif
