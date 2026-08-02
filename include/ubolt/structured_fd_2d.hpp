#ifndef UBOLT_STRUCTURED_FD_2D_HPP
#define UBOLT_STRUCTURED_FD_2D_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include <vector>

// An axis-aligned box of one material, for StructuredFD2D::paint_boxes
struct PETSC_VISIBILITY_PUBLIC MaterialBox2D {
   PetscScalar x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0;
   PetscInt material = 0;
};

// Uniform-grid upwinded finite difference discretisation of a 2D (XY) box
//
// THREE entries per row, not five: upwinding mu dpsi/dx and eta dpsi/dy
// separately reaches one neighbour per axis, plus the diagonal - the direct
// generalisation of 1D's two. Five is what the DMDA star stencil would
// preallocate, which is why create_matrix builds the sparsity by hand rather
// than calling DMCreateMatrix (see the Discretisation base)
//
// Slot order extends 1D's "off-diagonals first, diagonal last": upwind-x,
// upwind-y, diagonal. Which x and which y neighbour is the upwind one is fixed
// per angle at preallocation from sign(mu)/sign(eta), so the value fills never
// branch on direction. A row whose mu or eta is exactly zero takes a -1 in that
// slot, the same trick the Dirichlet rows use
//
// BC rows are the inflow ones: a node on a boundary face that its own
// direction points in through. In 2D that means any node whose x-upwind OR
// y-upwind neighbour is outside the box, corners included. What the row then
// holds depends on the face's BC family: vacuum makes it a Dirichlet row (both
// upwind slots nulled, the rhs carries the incoming flux), reflective
// repurposes the first slot for the -1 coupling to the mirrored angle in the
// same cell. A direction incoming through BOTH faces of a corner reflects in
// both axes if both faces are reflective, and is Dirichlet if either is
// vacuum - vacuum wins at mixed corners
class PETSC_VISIBILITY_PUBLIC StructuredFD2D : public Discretisation {
public:
   // The boundary label ids this backend hands to the BCSpec - the ids
   // PETSc's DMPlexCreateBoxMesh "Face Sets" convention gives a 2D box, so
   // the future unstructured backend agrees with these
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
   // the cell CENTRE. Allocates mat_id_d sized local_cells - the per-cell
   // material index view MaterialSpec's fill steps consume. The painting is
   // the geometric half of the material split (see MaterialSpec), which is why
   // it lives on the concrete backend
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
