#ifndef UBOLT_UNSTRUCTURED_DG_HPP
#define UBOLT_UNSTRUCTURED_DG_HPP

#include "ubolt/types.hpp"
#include "ubolt/plex_discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include <vector>

// Upwind discontinuous Galerkin on a DMPlex, 2D or 3D, any cell shape, at
// order 0 (DG0, one spatial dof per cell, the default) or 1 (DG1, linear,
// dimension + 1 dofs per cell). The long form - the Dirichlet-cell corner
// rules and the (V A)^T = P (V A) P derivation - is docs/architecture.md
//
// DG0: the cell-integrated balance divided by the cell volume, upwind flux on
// every face: with s_f = Omega_a . nA_f (outward area-weighted normal),
//   sum_{s_f > 0} (s_f / V_c) psi(c) + sum_{s_f < 0} (s_f / V_c) psi(c'(f)) + ...
// On a uniform quad/hex box that is the structured upwind FD stencil to
// rounding (verify_plexk checks it matrix for matrix). Rows are
// cell * n_angles + angle. Slots: n_faces(c) + 1 per row, one per face in CONE
// order then the diagonal LAST, through Discretisation::set_pattern; a face
// slot is live only on an interior INFLOW face (s_f < 0), -1 otherwise
//
// Boundary conditions. Under GHOST_FLUX (the default) there are no BC rows: a
// vacuum inflow face's slot is nulled and |s_f| / V_c times its per-angle
// inflow, windowed by the face CENTROID, goes to BoundaryInfo::ghost_inflow_d;
// a reflective inflow face's slot points at the angle mirrored over that
// face's own axis in the same cell. Under DIRICHLET_CELL (DG0 only) an inflow
// row is Dirichlet if any inflow boundary face is vacuum (the winner is the
// lowest dominant normal axis, then the lowest face point), else identity
// minus the partner mirrored over every inflow axis. A reflective face must be
// AXIS-ALIGNED (PETSC_ERR_SUP otherwise), and a BCSpec label no boundary face
// carries is an error
//
// DG1: a MODAL basis per cell, phi_0 = 1 and phi_1..dim linear, orthonormal in
// (1/V) int_c phi_i phi_j (so the mass matrix is the identity and removal,
// scatter and source stay per node). Rows (cell, basis, angle), angle fastest,
// ps.n_basis = dimension + 1. Row (c, i, a) is the weak form tested with phi_i
// over V_c:
//   -(Omega . grad phi_i) psi_(c,0) + sum_f (Omega . n_f) (1/V) int_f psi^up phi_i
//   + sigma_t psi_(c,i) - (scatter of node (c, i)) = q delta_i0
// psi^up: this cell's trace on an outflow face, the neighbour's on an interior
// inflow face, the mirrored angle's trace in this cell on a reflective inflow
// face, the prescribed inflow on a vacuum one - ghost-flux only
// (DIRICHLET_CELL is PETSC_ERR_SUP) and no BC rows. Slots: n_basis per face in
// cone order, then this cell's own n_basis LAST (the diagonal is own slot i).
// Face integrals are exact: moments come from a fan of simplices (planar
// faces). The scalar flux a caller reads per cell is the basis-0 node, the
// cell AVERAGE
//
// Under ghost-flux the volume-weighted operator satisfies (V A)^T = P (V A) P
// (P swaps Omega and -Omega; volume_host() is V), at both orders
//
// Construction is two-stage, because the MESH decides the global cell count:
// create_mesh, then PhaseSpace::create off n_global_cells(), then create
class PETSC_VISIBILITY_PUBLIC UnstructuredDG : public PlexDiscretisation {
public:
   // Stage 1: build and distribute the mesh - face adjacency with a one-cell
   // overlap (the face neighbours, all the upwind flux reads)
   PetscErrorCode create_mesh(MPI_Comm comm, const PlexMeshSpec &mesh);
   PetscInt n_global_cells() const { return n_global_cells_; }

   // Stage 2: the layout, geometry, BC classification and COO pattern, at DG
   // order 0 or 1. Fills ps.local_cells and ps.n_basis. ps.n_cells must equal
   // n_global_cells(), and the quadrature's dimension the mesh's
   PetscErrorCode create(PhaseSpace &ps, const AngularQuadrature &quad, const BCSpec &bcs = BCSpec(), \
      PetscInt order = 0);

   PetscInt order() const { return order_; }
   PetscInt n_basis() const { return n_basis_; }

   // Device geometry the streaming term reads. All FLAT rank-1 views:
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
   // Cell centroids, 3 * local_cells flat (z = 0 in 2D), and the cell volumes
   // (areas in 2D). At DG1 both come from the fan the basis is built on (the
   // FVM numbers, to rounding, on any cell with planar faces)
   const std::vector<PetscReal> &centroid_host() const { return centroid_h_; }
   const std::vector<PetscReal> &volume_host() const { return volume_h_; }

   // The host-side face data, per (cell, face) slot of the same CSR as
   // cell_face_offset_d (what DSAPrecon's diffusion operator is built from):
   //  face_nA_host:            3 per slot, as face_nA_d
   //  face_neighbour_row_host: the neighbour's GLOBAL row base, (global cell) *
   //                           rows_per_cell, or -1 on a boundary face
   //  face_label_host:         a boundary face's "Face Sets" value, -1 if
   //                           unlabelled or interior
   //  face_distance_host:      2 per slot, (own, neighbour): each centroid's
   //                           distance to the face's plane along its normal,
   //                           the neighbour's 0 on a boundary face. Taken from
   //                           the FVM centroids at both orders
   const std::vector<PetscInt> &cell_face_offset_host() const { return cell_face_offset_h_; }
   const std::vector<PetscScalar> &face_nA_host() const { return face_nA_h_; }
   const std::vector<PetscInt> &face_neighbour_row_host() const { return face_neighbour_row_h_; }
   const std::vector<PetscInt> &face_label_host() const { return face_label_h_; }
   const std::vector<PetscReal> &face_distance_host() const { return face_distance_h_; }
   // DG1 only (empty at order 0), host copies of the device face matrices and
   // gradients above in the same flat layouts, plus
   //  face_neighbour_grad_host: n_cell_faces * nb * 3, ((k * nb + j) * 3 + d),
   //                            grad phi_j of the NEIGHBOUR across interior face
   //                            slot k (an overlap ghost's too), zero on a
   //                            boundary face
   // (what DSAPrecon's DG1 interior penalty operator is built from)
   const std::vector<PetscScalar> &face_own_host() const { return face_own_h_; }
   const std::vector<PetscScalar> &face_up_host() const { return face_up_h_; }
   const std::vector<PetscScalar> &basis_grad_host() const { return basis_grad_h_; }
   const std::vector<PetscScalar> &face_neighbour_grad_host() const { return face_nb_grad_h_; }

   // Local cell k is the k-th OWNED cell in DMPlex point order - the order the
   // layout check asserts the global numbering follows. This is its point
   const std::vector<PetscInt> &cell_point_host() const { return paint_point_; }

   // DG1: the gradient of the scalar flux in each owned cell, axis by axis -
   // grad[d](c) = sum_i phi(c, i) grad phi_i[d], phi the angular integral of
   // psi per node. With the cell average (what UboltWriteScalarFluxVTK writes)
   // and the centroid it is the whole linear solution:
   // phi(x) = average + grad . (x - x_c). Output-time only: allocates a
   // scratch and dimension() views sized local_cells. Errors at order 0
   PetscErrorCode scalar_flux_gradient(Vec psi, const AngularQuadrature &quad, \
      std::vector<PetscScalarKokkosView> &grad) const;

private:
   // DG1: the modal basis on every cell of the local mesh (overlap ghosts
   // included - a face matrix reads its neighbour's basis), the fan volumes and
   // centroids of the owned cells, and the per-(cell, face) face matrices. A
   // non-planar face is recorded in failed/message (see RecordFailure)
   PetscErrorCode build_dg1_geometry(std::vector<PetscScalar> &face_own, std::vector<PetscScalar> &face_up, \
      std::vector<PetscScalar> &basis_grad, std::vector<PetscReal> &face_basis_value, const BCSpec &bcs, \
      PetscBool *failed, char *message, size_t len);

   PetscInt order_ = 0;
   PetscInt n_basis_ = 1;
   PetscInt n_global_cells_ = 0;

   // Host geometry and the per-(cell, face) boundary data (see the accessors)
   std::vector<PetscReal> centroid_h_;
   std::vector<PetscReal> volume_h_;
   std::vector<PetscInt> cell_face_offset_h_;
   std::vector<PetscScalar> face_nA_h_;
   std::vector<PetscInt> face_neighbour_row_h_;
   std::vector<PetscInt> face_label_h_;
   std::vector<PetscReal> face_distance_h_;
   // DG1 only
   std::vector<PetscScalar> face_own_h_;
   std::vector<PetscScalar> face_up_h_;
   std::vector<PetscScalar> basis_grad_h_;
   std::vector<PetscScalar> face_nb_grad_h_;

   PetscScalarKokkosView omega_d_;
   PetscIntKokkosView cell_face_offset_d_;
   PetscScalarKokkosView face_nA_d_;
   PetscScalarKokkosView inv_volume_d_;
   PetscScalarKokkosView face_own_d_;
   PetscScalarKokkosView face_up_d_;
   PetscScalarKokkosView basis_grad_d_;
};

#endif
