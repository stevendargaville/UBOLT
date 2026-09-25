#ifndef UBOLT_UNSTRUCTURED_DG_HPP
#define UBOLT_UNSTRUCTURED_DG_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"
#include <map>
#include <string>
#include <vector>

// How to get the mesh: a box built in code, or a file PETSc can read (Gmsh
// .msh, ...)
struct PETSC_VISIBILITY_PUBLIC PlexMeshSpec {
   PetscInt dimension = 0;                 // 2 or 3
   PetscInt n_cells[3] = {0, 0, 0};        // box: cells per axis (ignored with a file)
   PetscReal lengths[3] = {0.0, 0.0, 0.0}; // box: extents from the origin
   PetscBool simplex = PETSC_FALSE;        // box: triangles/tets instead of quads/hexes
   std::string file;                       // non-empty: DMPlexCreateFromFile, interpolated
};

// Upwind discontinuous Galerkin on a DMPlex, 2D or 3D, any cell shape, at
// order 0 (DG0, one spatial dof per cell, the default) or order 1 (DG1, linear:
// dimension + 1 dofs per cell). Most of what follows describes DG0; DG1 is the
// section after it, and everything about the mesh, the distribution, the
// painting and the "Face Sets" ids is shared
//
// ~~~~~~~~~~ DG0 ~~~~~~~~~~
//
// The cell-integrated balance divided by the cell volume, with the upwind flux
// on every face: for cell c, ordinate Omega_a and outward area-weighted face
// normal nA_f, s_f = Omega_a . nA_f, and
//   sum_{s_f > 0} (s_f / V_c) psi(c) + sum_{s_f < 0} (s_f / V_c) psi(c'(f)) + ...
// On a uniform quad/hex grid that is exactly the structured upwind FD stencil
// (|mu| dy / (dx dy) = |mu| / dx), to rounding - tests/verify_plexk checks it
// against StructuredFD2D/3D matrix for matrix. DG0 keeps row = cell * n_angles
// + angle, which is why every dimension-independent piece of the library
// (removal, scatter, group transfer, the operator and the solver) works on it
// unchanged
//
// Slot layout: n_faces(c) + 1 COO slots per row of cell c - one per face in
// CONE order, then the diagonal LAST, the structured backends' convention with
// "one per axis" replaced by "one per face". The upwind selection is done the
// way the structured backends do it, at preallocation: a face slot is live only
// if the face is interior and INFLOW for that angle (s_f < 0); every other face
// slot carries a -1 row/col, so the sparsity is one neighbour per inflow face
// and the value fill only branches on the sign of s (diagonal vs slot). Because
// the slot count varies with the cell shape, this backend uses the general
// Discretisation::set_pattern rather than set_uniform_pattern
//
// BC rows under the opt-in VacuumTreatment::DIRICHLET_CELL are the library's
// Dirichlet-cell rows: a row is a BC row
// iff some BOUNDARY face (support size 1) of its cell has s_f < 0. If any such
// incoming face is vacuum the row is Dirichlet (identity, rhs = the winning
// face's inflow / sum_weights inside its window); else every incoming face is
// reflective and the row is identity minus the partner angle in the same cell,
// the -1.0 in the slot of the first incoming reflective face in cone order.
// The partner composes the quadrature's per-axis reflection maps over the axes
// of the incoming reflective faces, so a reflective face must be AXIS-ALIGNED
// (the mirror of an ordinate in a general plane is not an ordinate): anything
// else is PETSC_ERR_SUP. The partner's own row may be Dirichlet - a cell where
// a reflective plane meets a slanted vacuum face - but not itself reflective
// (a single-cell-wide direction between two reflective faces), which is also
// PETSC_ERR_SUP. A label the BCSpec names that no boundary face carries is an
// error too, so a mistyped "Face Sets" id cannot silently leave a face cold.
// The winning face at a Dirichlet corner is the incoming vacuum face with the
// lowest DOMINANT axis of its normal (x < y < z), ties by face point number -
// on a box that is the structured "first vacuum incoming face in axis order
// x, y, z". A window is tested on the face CENTROID's
// coordinates along the non-dominant axes in ascending axis order, inclusive -
// the boundary cell's centre on a quad/hex box, as in the structured backends
//
// Under VacuumTreatment::GHOST_FLUX - the default, and the natural DG0 vacuum
// condition - a row that comes in only through vacuum faces is NOT a BC row: it
// keeps its physical row (full diagonal, the boundary inflow faces' slots
// nulled) and
// BoundaryInfo::ghost_inflow_d carries sum_f |Omega . nA_f| / V_c times each
// incoming vacuum face's per-angle inflow, windowed per face by its centroid.
// A row that comes in through any reflective face stays reflective (reflect
// wins), mirrored over the reflective axes and the axes of its axis-aligned
// incoming vacuum faces - the structured backends' rule, so a box still
// matches its FD twin
//
// The opposite-ordinate symmetry under GHOST_FLUX (what a half-quadrature
// preconditioner applied transposed on the other half rests on): with P
// swapping (cell, Omega) and (cell, -Omega) and V the cell volumes expanded to
// rows, the VOLUME-WEIGHTED operator S = V A satisfies S^T = P S P - exactly
// off the diagonal, to rounding on it (a cell's outward nA_f sum to zero only
// to rounding). The rows here are per unit volume, so A itself satisfies only
// the similarity A^T = V (P A P) V^{-1}: on a mesh with unequal volumes the
// transposed apply of a hierarchy built for one half has to be wrapped in the
// V scalings, y = V^{-1} M^{-T} (V x) (V commutes with P). On a uniform box V
// is a multiple of the identity and A^T = P A P as in the structured backends.
// volume_host() is V
//
// ~~~~~~~~~~ DG1 ~~~~~~~~~~
//
// Linear DG: on each cell the MODAL basis phi_0 = 1, phi_{1..dim} linear,
// orthonormal in the cell's volume-averaged inner product (1/V) int_c phi_i
// phi_j = delta_ij (the linear ones are x - x_c, x_c the centroid,
// orthonormalised through the Cholesky factor of the cell's second moments).
// Rows are (cell, basis, angle), angle fastest - row = (cell * n_basis + i) *
// n_angles + a, ps.n_basis = dimension + 1 - so every (cell, basis) "node" is a
// contiguous run of n_angles rows and the dimension-independent pieces work per
// node unchanged. The equation for row (c, i, a) is the weak form tested with
// phi_i and divided by V_c, which is what makes the mass matrix the identity:
//   -(Omega . grad phi_i) psi_(c,0) + sum_f (Omega . n_f) (1/V) int_f psi^up phi_i
//   + sigma_t psi_(c,i) - (scatter of node (c, i)) = q delta_i0
// (int_c phi_j = V delta_j0 is what collapses the volume term onto basis 0).
// psi^up on a face is this cell's trace on an outflow face (s > 0), the
// neighbour's on an interior inflow face, the MIRRORED ANGLE's trace in this
// same cell on a reflective inflow face, and the prescribed inflow on a vacuum
// inflow face - which is the ghost-flux condition, the only one DG1 has
// (VacuumTreatment::DIRICHLET_CELL is PETSC_ERR_SUP: a cell with several dofs
// has no one row to replace). So under DG1 there are NO boundary-condition rows
// at all: the BC mask is empty, a reflective face is a face coupling like any
// other (each face mirrors over its own axis only, so a corner needs no
// composed partner and nothing is ever single-cell-wide), and a vacuum face's
// inflow goes to BoundaryInfo::ghost_inflow_d, -(s / V) phi_i(x_f) times the
// per-angle inflow. Reflective faces must still be axis-aligned
//
// Slot layout: (n_faces(c) + 1) * n_basis COO slots per row - n_basis per face
// in CONE order (the upwind cell's basis j = 0 .. n_basis - 1: the neighbour's
// on an interior inflow face, this cell's at the mirrored angle on a
// reflective inflow face, nulled otherwise), then this cell's own n_basis
// (same angle, j = 0 .. n_basis - 1) LAST, the diagonal being own slot i. The
// face integrals are exact (every product is quadratic): cell and face moments
// come from a fan of simplices off the centroids, exact for any cell with
// planar faces. Per (cell, face) the backend precomputes the two n_basis x
// n_basis face matrices (1 / (V_c A_f)) int_f phi_i phi_j^{own | upwind} that
// StreamingTermDG1 scales by s = Omega . nA_f, and per cell the basis
// gradients for the volume term
//
// The opposite-ordinate identity (V A)^T = P (V A) P holds at DG1 too - to
// rounding, reflective faces included (the face couplings pair up across the
// face, and the own-cell block is antisymmetric in Omega up to the divergence
// theorem on the cell). The scalar flux a caller reads per cell is the node of
// basis 0: the cell AVERAGE
//
// Construction is two-stage, because the MESH decides the global cell count:
// create_mesh, then PhaseSpace::create off n_global_cells(), then create
class PETSC_VISIBILITY_PUBLIC UnstructuredDG : public Discretisation {
public:
   // "Face Sets" ids are PETSc's box convention - the ones StructuredFD2D and
   // StructuredFD3D define as FACE_*, so a BCSpec means the same thing on
   // either backend. 2D: 1 bottom, 2 right, 3 top, 4 left. 3D: 1 bottom (z-),
   // 2 top (z+), 3 front (y-), 4 back (y+), 5 right (x+), 6 left (x-). Use the
   // structured backends' constants rather than redefining them here. A
   // boundary face with no label takes BCSpec::face(-1): cold vacuum by default

   // Stage 1: build and distribute the mesh. The mesh decides the GLOBAL cell
   // count, so this comes before PhaseSpace::create - the caller sizes the
   // phase space off n_global_cells(). Errors if the DM's dimension is not
   // mesh.dimension (2 or 3)
   //
   // No DMSetFromOptions, the same rule the DMDA backends keep: the mesh must
   // not be resized out from under the phase space. The distribution is
   // face-adjacency with a one-cell overlap (the face neighbours, which is all
   // DG0 upwinding reads), partitioned by PETSc's "simple" partitioner - which
   // is deterministic across machines and needs no external package -
   // overridable with -petscpartitioner_type
   PetscErrorCode create_mesh(MPI_Comm comm, const PlexMeshSpec &mesh);
   PetscInt n_global_cells() const { return n_global_cells_; }
   PetscInt dimension() const { return dim_; }

   // Stage 2: the layout, geometry, BC classification and COO pattern, at DG
   // order 0 or 1. Fills ps.local_cells like every backend, and ps.n_basis (1
   // at order 0, dimension + 1 at order 1). ps.n_cells must equal
   // n_global_cells(), and the quadrature's dimension the mesh's
   PetscErrorCode create(PhaseSpace &ps, const SNQuadrature2D &quad, const BCSpec &bcs = BCSpec(), \
      PetscInt order = 0);
   PetscErrorCode create(PhaseSpace &ps, const SNQuadrature3D &quad, const BCSpec &bcs = BCSpec(), \
      PetscInt order = 0);

   PetscInt order() const { return order_; }
   PetscInt n_basis() const { return n_basis_; }

   // Device geometry the streaming term reads. All FLAT rank-1 views (no
   // layout trap):
   //  omega_d:            3 * n_angles, (a * 3 + d); xi = 0 in 2D
   //  cell_face_offset_d: local_cells + 1, CSR over the faces of each local cell
   //                      (in cone order, the order of the row's face slots)
   //  face_nA_d:          3 * n_cell_faces, outward area-weighted normal per
   //                      (cell, face), z = 0 in 2D
   //  inv_volume_d:       local_cells
   const PetscScalarKokkosView &omega_d() const { return omega_d_; }
   const PetscIntKokkosView &cell_face_offset_d() const { return cell_face_offset_d_; }
   const PetscScalarKokkosView &face_nA_d() const { return face_nA_d_; }
   const PetscScalarKokkosView &inv_volume_d() const { return inv_volume_d_; }
   // DG1 only (empty at order 0), flat like the rest, nb = n_basis():
   //  face_own_d:  n_cell_faces * nb * nb, ((k * nb + i) * nb + j) for face
   //               slot k: (1 / (V_c A_f)) int_f phi_i^c phi_j^c
   //  face_up_d:   the same shape: (1 / (V_c A_f)) int_f phi_i^c phi_j^up, the
   //               upwind basis being the neighbour's across an interior face,
   //               this cell's own across a reflective one (the mirrored angle
   //               is in the column, not here), zero on a vacuum face
   //  basis_grad_d: local_cells * nb * 3, ((c * nb + i) * 3 + d), grad phi_i
   //               (zero for i = 0; z = 0 in 2D)
   const PetscScalarKokkosView &face_own_d() const { return face_own_d_; }
   const PetscScalarKokkosView &face_up_d() const { return face_up_d_; }
   const PetscScalarKokkosView &basis_grad_d() const { return basis_grad_d_; }
   // Cell centroids, 3 * local_cells flat (z = 0 in 2D), host and device
   // (painting, output, tests), and the cell volumes (areas in 2D). At DG1 both
   // come from the same fan of simplices the basis is built on (the same
   // numbers as the FVM ones on any cell with planar faces, to rounding)
   const std::vector<PetscReal> &centroid_host() const { return centroid_h_; }
   const PetscScalarKokkosView &centroid_d() const { return centroid_d_; }
   const std::vector<PetscReal> &volume_host() const { return volume_h_; }

   // Local cell k is the k-th OWNED cell in DMPlex point order - the order
   // CheckPlexLayout asserts the global numbering follows. This is its point
   const std::vector<PetscInt> &cell_point_host() const { return cell_of_local_; }

   // DG1: the gradient of the scalar flux in each owned cell, axis by axis -
   // grad[d](c) = sum_i phi(c, i) grad phi_i[d], phi the angular integral of
   // psi per node. With the cell average (what UboltWriteScalarFluxVTK writes)
   // and the centroid it is the whole linear solution:
   // phi(x) = average + grad . (x - x_c). Allocates dimension() views sized
   // local_cells. Errors at order 0, where there is no slope
   PetscErrorCode scalar_flux_gradient(Vec psi, const AngularQuadrature &quad, \
      std::vector<PetscScalarKokkosView> &grad) const;

   // Painting, by cell CENTROID, same semantics as the structured paint_boxes:
   // the background everywhere, then the boxes in order with the later ones
   // winning, inclusive at both ends. Allocates mat_id_d sized local_cells.
   // The 2D overload requires dimension 2, the 3D one dimension 3
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox2D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox3D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;
   // "Cell Sets" label value -> material index. Allocates and fills mat_id_d:
   // the background everywhere, then each owned cell whose label value is in
   // the map takes that material (unlisted values and unlabelled cells keep the
   // background). Errors if the map is non-empty and the mesh has no "Cell
   // Sets" label
   PetscErrorCode paint_cell_sets(PetscInt background_material, const std::map<PetscInt, PetscInt> &label_to_material, \
      PetscIntKokkosView &mat_id_d) const;
   // Paint boxes OVER an existing mat_id_d (from paint_cell_sets), later boxes
   // winning - how the driver layers "Cell Sets" then "paint".
   // paint_boxes(bg, boxes, out) is allocate + background + paint_boxes_over
   PetscErrorCode paint_boxes_over(const std::vector<MaterialBox2D> &boxes, PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_boxes_over(const std::vector<MaterialBox3D> &boxes, PetscIntKokkosView &mat_id_d) const;

private:
   // Both create overloads land here: the quadrature is reduced to its host
   // cosines and per-axis reflection maps (xi and its map are null in 2D)
   PetscErrorCode create_common(PhaseSpace &ps, PetscInt quad_dim, PetscInt n_angles, PetscScalar sum_weights, \
      const PetscScalar *mu, const PetscScalar *eta, const PetscScalar *xi, \
      const PetscInt *reflect_mu, const PetscInt *reflect_eta, const PetscInt *reflect_xi, \
      const BCSpec &bcs, PetscInt order);
   // DG1: the modal basis on every cell of the local mesh (overlap ghosts
   // included - a face matrix reads its neighbour's basis), the fan volumes and
   // centroids of the owned cells, and the per-(cell, face) face matrices
   PetscErrorCode build_dg1_geometry(std::vector<PetscScalar> &face_own, std::vector<PetscScalar> &face_up, \
      std::vector<PetscScalar> &basis_grad, std::vector<PetscReal> &face_basis_value, const BCSpec &bcs);
   // The shared body of both paint_boxes_over overloads: boxes flattened to
   // 2 * dim (lo, hi) pairs per box, axis by axis
   PetscErrorCode paint_flat_boxes(PetscInt n_boxes, const std::vector<PetscScalar> &box_lohi, \
      const std::vector<PetscInt> &box_material, PetscIntKokkosView &mat_id_d) const;

   PetscInt dim_ = 0;
   PetscInt order_ = 0;
   PetscInt n_basis_ = 1;
   PetscInt n_global_cells_ = 0;
   // Height-0 point range of the local (overlapped) mesh
   PetscInt c_start_ = 0;
   PetscInt c_end_ = 0;
   // Owned cells, local index <-> point; local_of_cell_ is -1 on overlap
   // (ghost) cells
   std::vector<PetscInt> cell_of_local_;
   std::vector<PetscInt> local_of_cell_;

   // Host geometry and the per-(cell, face) boundary data: for face slot k of
   // the CSR, the neighbour's global row base (-1 on a boundary face) and the
   // boundary face's "Face Sets" value (-1 if unlabelled or interior). Kept on
   // the host because the classification (and a future ghost-flux vacuum BC)
   // is a host-side, create-time step
   std::vector<PetscReal> centroid_h_;
   std::vector<PetscReal> volume_h_;
   std::vector<PetscInt> cell_face_offset_h_;
   std::vector<PetscScalar> face_nA_h_;
   std::vector<PetscInt> face_neighbour_row_h_;
   std::vector<PetscInt> face_label_h_;

   PetscScalarKokkosView omega_d_;
   PetscIntKokkosView cell_face_offset_d_;
   PetscScalarKokkosView face_nA_d_;
   PetscScalarKokkosView inv_volume_d_;
   PetscScalarKokkosView centroid_d_;
   PetscScalarKokkosView face_own_d_;
   PetscScalarKokkosView face_up_d_;
   PetscScalarKokkosView basis_grad_d_;
};

#endif
