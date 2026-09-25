#ifndef UBOLT_UNSTRUCTURED_DG0_HPP
#define UBOLT_UNSTRUCTURED_DG0_HPP

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

// Cell-centred DG0 (one spatial dof per cell) upwind discretisation on a DMPlex,
// 2D or 3D, any cell shape
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
// Construction is two-stage, because the MESH decides the global cell count:
// create_mesh, then PhaseSpace::create off n_global_cells(), then create
class PETSC_VISIBILITY_PUBLIC UnstructuredDG0 : public Discretisation {
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

   // Stage 2: the layout, geometry, BC classification and COO pattern. Fills
   // ps.local_cells like every backend. ps.n_cells must equal n_global_cells(),
   // and the quadrature's dimension the mesh's
   PetscErrorCode create(PhaseSpace &ps, const SNQuadrature2D &quad, const BCSpec &bcs = BCSpec());
   PetscErrorCode create(PhaseSpace &ps, const SNQuadrature3D &quad, const BCSpec &bcs = BCSpec());

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
   // Cell centroids, 3 * local_cells flat (z = 0 in 2D), host and device
   // (painting, output, tests), and the cell volumes (areas in 2D)
   const std::vector<PetscReal> &centroid_host() const { return centroid_h_; }
   const PetscScalarKokkosView &centroid_d() const { return centroid_d_; }
   const std::vector<PetscReal> &volume_host() const { return volume_h_; }

   // Local cell k is the k-th OWNED cell in DMPlex point order - the order
   // CheckPlexLayout asserts the global numbering follows. This is its point
   const std::vector<PetscInt> &cell_point_host() const { return cell_of_local_; }

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
      const BCSpec &bcs);
   // The shared body of both paint_boxes_over overloads: boxes flattened to
   // 2 * dim (lo, hi) pairs per box, axis by axis
   PetscErrorCode paint_flat_boxes(PetscInt n_boxes, const std::vector<PetscScalar> &box_lohi, \
      const std::vector<PetscInt> &box_material, PetscIntKokkosView &mat_id_d) const;

   PetscInt dim_ = 0;
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
};

#endif
