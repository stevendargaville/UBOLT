#ifndef UBOLT_STRUCTURED_FD_3D_HPP
#define UBOLT_STRUCTURED_FD_3D_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include <vector>

// An axis-aligned box of one material, for StructuredFD3D::paint_boxes
struct PETSC_VISIBILITY_PUBLIC MaterialBox3D {
   PetscScalar x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0, z0 = 0.0, z1 = 0.0;
   PetscInt material = 0;
};

// Uniform-grid upwinded finite difference discretisation of a 3D box
//
// FOUR entries per row, not seven: upwinding mu dpsi/dx, eta dpsi/dy and
// xi dpsi/dz separately reaches one neighbour per axis, plus the diagonal -
// the direct generalisation of 2D's three. Seven is what the DMDA star stencil
// would preallocate, which is why create_matrix builds the sparsity by hand
// rather than calling DMCreateMatrix (see the Discretisation base)
//
// Slot order extends the 1D/2D "off-diagonals first, diagonal last": upwind-x,
// upwind-y, upwind-z, diagonal. Which neighbour on each axis is the upwind one
// is fixed per angle at preallocation from the sign of its cosine, so the
// value fills never branch. A row whose cosine is exactly zero on an axis
// takes a -1 in that slot, the same trick the Dirichlet rows use
//
// BC rows are the inflow ones: a node on a boundary face that its own
// direction points in through - any node with an upwind neighbour outside the
// box on any axis, edges and corners included. What the row then holds depends
// on the face's BC family: vacuum makes it a Dirichlet row (every upwind slot
// nulled, the rhs carries the incoming flux - that face's angle-integrated
// inflow shared over the ordinates, zero outside its window), reflective
// repurposes the first slot for the -1 coupling to the mirrored angle in the
// same cell. A direction incoming through several faces of an edge or corner
// reflects in every reflective axis - up to a triple flip where three
// reflective faces meet - and is Dirichlet if any of those faces is vacuum:
// vacuum wins, and the inflow value and window such a row takes are the first
// vacuum incoming face's in axis order x, y, z
class PETSC_VISIBILITY_PUBLIC StructuredFD3D : public Discretisation {
public:
   // The boundary label ids this backend hands to the BCSpec - the ids
   // PETSc's DMPlexCreateBoxMesh "Face Sets" convention gives a 3D box, so
   // the future unstructured backend agrees with these. CAREFUL: PETSc's 3D
   // convention puts bottom/top on the Z axis, where 2D's bottom/top are the
   // Y faces - in 3D the y faces are front (y-min) and back (y-max)
   static constexpr PetscInt FACE_BOTTOM = 1; // z-min
   static constexpr PetscInt FACE_TOP = 2;    // z-max
   static constexpr PetscInt FACE_FRONT = 3;  // y-min
   static constexpr PetscInt FACE_BACK = 4;   // y-max
   static constexpr PetscInt FACE_RIGHT = 5;  // x-max
   static constexpr PetscInt FACE_LEFT = 6;   // x-min

   // The quadrature decides which neighbours are upwind for each angle, and on
   // a reflective face which angle mirrors which. A default bcs is vacuum
   // (prescribed inflow) on all six faces
   //
   // ps.n_cells must be n_cells_x * n_cells_y * n_cells_z. The layout comes
   // from a 3D DMDA, which also decides the parallel decomposition: `ps` is
   // filled in (ps.local_cells) rather than read, so this must be created
   // before anything sized off the phase space
   PetscErrorCode create(MPI_Comm comm, PhaseSpace &ps, PetscInt n_cells_x, PetscInt n_cells_y, \
      PetscInt n_cells_z, PetscReal length_x, PetscReal length_y, PetscReal length_z, \
      const SNQuadrature3D &quad, const BCSpec &bcs = BCSpec());

   // Uniform grid so every cell has the same width
   PetscScalar dx() const { return dx_; }
   PetscScalar dy() const { return dy_; }
   PetscScalar dz() const { return dz_; }

   PetscInt n_cells_x() const { return n_cells_x_; }
   PetscInt n_cells_y() const { return n_cells_y_; }
   PetscInt n_cells_z() const { return n_cells_z_; }

   // This rank's patch of the grid. Local cell index c, which is what the
   // per-cell xsection views are indexed by, is the patch's own lexicographic
   // ordering: c = ((k - cell_start_z) * local_cells_y + (j - cell_start_y))
   // * local_cells_x + (i - cell_start_x)
   PetscInt cell_start_x() const { return cell_start_x_; }
   PetscInt cell_start_y() const { return cell_start_y_; }
   PetscInt cell_start_z() const { return cell_start_z_; }
   PetscInt local_cells_x() const { return local_cells_x_; }
   PetscInt local_cells_y() const { return local_cells_y_; }
   PetscInt local_cells_z() const { return local_cells_z_; }

   // Paint material indices onto this rank's cells: the background everywhere,
   // then the boxes in order with the later ones winning, membership decided by
   // the cell CENTRE. Allocates mat_id_d sized local_cells - the per-cell
   // material index view MaterialSpec's fill steps consume. The painting is
   // the geometric half of the material split (see MaterialSpec), which is why
   // it lives on the concrete backend
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox3D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;

private:
   PetscScalar dx_ = 0.0;
   PetscScalar dy_ = 0.0;
   PetscScalar dz_ = 0.0;
   PetscInt n_cells_x_ = 0;
   PetscInt n_cells_y_ = 0;
   PetscInt n_cells_z_ = 0;
   PetscInt cell_start_x_ = 0;
   PetscInt cell_start_y_ = 0;
   PetscInt cell_start_z_ = 0;
   PetscInt local_cells_x_ = 0;
   PetscInt local_cells_y_ = 0;
   PetscInt local_cells_z_ = 0;
};

#endif
