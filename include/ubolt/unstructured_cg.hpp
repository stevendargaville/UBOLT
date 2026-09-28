#ifndef UBOLT_UNSTRUCTURED_CG_HPP
#define UBOLT_UNSTRUCTURED_CG_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/unstructured_dg.hpp"
#include <map>
#include <vector>

// Continuous Galerkin with consistent SUPG stabilisation on a DMPlex, 2D or 3D:
// P1 on triangles/tets, Q1 on quads/hexes, one nodal value per VERTEX
//
// ~~~~~~~~~~ The unknowns ~~~~~~~~~~
//
// Rows are (vertex, angle), angle fastest: row = vertex * n_angles + angle,
// ps.n_basis = 1. The PhaseSpace's "cell" is the rows' spatial unit, so here it
// is a VERTEX: ps.n_cells is the global vertex count and ps.local_cells the
// vertices this rank owns (in point order - CheckCGLayout asserts the global
// numbering follows it). Cross sections stay per mesh ELEMENT, the only place a
// material is defined - every LOCAL element, the overlap included, because a
// row reads every element of its vertex's star. So GroupXSections is sized
// n_local_elements() on this backend (its create(n_groups, n_entries)
// overload), and the painting methods paint every local element
//
// ~~~~~~~~~~ The scheme ~~~~~~~~~~
//
// Consistent SUPG: the transport residual R = Omega . grad psi + sigma_t psi -
// sigma_s phi / W - q / W is tested with v + tau Omega . grad v on EVERY term,
// so the exact solution satisfies the discrete equations for any tau (a
// constant is reproduced to rounding - the infinite-medium check holds). It is
// the same method as Wang's SAAF-tau (NSE 176, 2014) written from the
// first-order side. Per element e, with
//   M_ij = int phi_i phi_j,  G^d_ij = int phi_i d_d phi_j,
//   K^{dd'}_ij = int d_d phi_i d_d' phi_j,
// row (i, a) is divided by the lumped mass m_i = sum_e int_e phi_i (so rows are
// per unit volume and the removal sits on the diagonal as ~ sigma_t, like the
// other backends) and reads
//   sum_e sum_j [ Omega.G_ij + tau Omega Omega : K_ij + sigma_t (M_ij + tau Omega.G_ji) ] psi_j
//   + sum over inflow boundary faces (weak BC, below)
// and the scatter, the source and the group transfer carry the same weight
// (M_ij + tau Omega.G_ji) against their nodal field
//
// tau = min(1 / sigma_t, h_Omega / zeta) per (element, angle, group): the SAAF
// value 1 / sigma_t in an optically thick cell (the correct diffusion limit),
// and h_Omega / zeta in a thin cell or a void (sigma_t = 0 is allowed).
// h_Omega = 2 / sum_j |Omega . grad phi_j(x_c)| is the element's length along
// Omega (dx / |mu| on a strip; the Tezduyar/Shakib element length). zeta is a
// discretisation parameter - it changes the solution - so it is a create()
// argument, 0.5 by default (Rattlesnake's). tau depends on the group, so the
// whole assembled operator is refilled per group: one preallocated matrix, a
// values-only refill, as everywhere else
//
// ~~~~~~~~~~ Boundary conditions ~~~~~~~~~~
//
// Weak, with a LUMPED face mass, and never a BC row: for a boundary face f
// with outward area normal nA_f, s = Omega . nA_f / A_f < 0 (inflow), and
// m^f_i = int_f phi_i (= A_f / n_face_vertices, exact on affine faces)
//   diagonal of row (i, a)  += |s| m^f_i / m_i
//   vacuum:     BoundaryInfo::ghost_inflow_d += |s| m^f_i / m_i * inflow / W,
//               windowed by the face CENTROID, as the DG backend does
//   reflective: the slot at (i, mirror_f(a)) gets -|s| m^f_i / m_i - the
//               mirror over that face's own axis, so a reflective face must be
//               AXIS-ALIGNED (PETSC_ERR_SUP otherwise), DG1's rule
// So the BC row mask is empty and VacuumTreatment::DIRICHLET_CELL is
// PETSC_ERR_SUP. SUPG is not adjoint-consistent: there is no A^T = P A P
// identity on this backend
//
// ~~~~~~~~~~ Slots ~~~~~~~~~~
//
// Row (i, a): the vertex's star neighbours at the same angle (every vertex
// sharing an element with i, i excluded, in point order), then one slot per
// reflective AXIS touching i (the mirrored angle at i itself, nulled for an
// angle that comes in through no reflective face on that axis), then the
// diagonal LAST - through Discretisation::set_pattern. The fill is a
// row-parallel GATHER over the star (each (star element, element vertex) pair
// knows its slot inside the row), no atomics, so it is deterministic
//
// ~~~~~~~~~~ Parallel ~~~~~~~~~~
//
// Distributed with PETSc's FEM (closure) adjacency and a one-cell overlap, so
// every element touching an owned vertex is local: a row, its lumped mass and
// its boundary faces never need another rank. The nodal fields the scatter,
// the transfer and the source weigh (and a boundary vertex's faces) are read
// from the LOCAL vertices, owned and overlap, through a dof-1 vertex twin of
// the DM - nodal_global_to_local() refreshes the ghosts
//
// Construction is two-stage like UnstructuredDG, because the MESH decides the
// global vertex count: create_mesh, then PhaseSpace::create off
// n_global_vertices(), then create
class PETSC_VISIBILITY_PUBLIC UnstructuredCG : public Discretisation {
public:
   // Stage 1: build and distribute the mesh (the same box / file choices as
   // UnstructuredDG, the same "Face Sets" ids)
   PetscErrorCode create_mesh(MPI_Comm comm, const PlexMeshSpec &mesh);
   PetscInt n_global_vertices() const { return n_global_vertices_; }
   PetscInt dimension() const { return dim_; }

   // Stage 2: the layout, the element tables, the BC classification and the
   // COO pattern. ps.n_cells must equal n_global_vertices(). zeta > 0 is the
   // thin-cell SUPG parameter (see above)
   PetscErrorCode create(PhaseSpace &ps, const SNQuadrature2D &quad, const BCSpec &bcs = BCSpec(), \
      PetscReal zeta = 0.5);
   PetscErrorCode create(PhaseSpace &ps, const SNQuadrature3D &quad, const BCSpec &bcs = BCSpec(), \
      PetscReal zeta = 0.5);

   PetscReal zeta() const { return zeta_; }
   // Every element of the local mesh, overlap included - what cross sections
   // and material ids are sized on here
   PetscInt n_local_elements() const { return c_end_ - c_start_; }
   // Vertices per element (3/4 in 2D, 4/8 in 3D) - one cell type per mesh
   PetscInt n_element_vertices() const { return nv_; }

   // Device data the CG terms read. All FLAT rank-1 views; nv =
   // n_element_vertices(), n_elem = n_local_elements():
   //  omega_d:       3 * n_angles, (a * 3 + d)
   //  inv_mass_d:    local_cells, 1 / m_i per owned vertex
   //  star_offset_d: local_cells + 1, CSR over each owned vertex's star
   //  star_elem_d:   the star's local element index (c - c_start)
   //  star_li_d:     the owned vertex's position (basis index) in that element
   //  star_slot_d:   nv per star entry, the slot INSIDE THE ROW of element
   //                 vertex j's column (the diagonal's for j == li)
   //  elem_vertex_d: n_elem * nv, the element's vertices as LOCAL vertex
   //                 indices (v - v_start: owned and overlap), basis order
   //  mass_d:        n_elem * nv * nv, M_ij           ((e * nv + i) * nv + j)
   //  grad_d:        n_elem * nv * nv * 3, G^d_ij     (((e * nv + i) * nv + j) * 3 + d)
   //  stiff_d:       n_elem * nv * nv * 9, K^{dd'}_ij (((e * nv + i) * nv + j) * 9 + 3 d + d')
   //  centre_grad_d: n_elem * nv * 3, grad phi_j at the element centroid
   //  bface_offset_d: local_cells + 1, CSR over each owned vertex's boundary
   //                 faces
   //  bface_nA_d:    3 per entry, the face's outward area-weighted normal
   //  bface_mass_d:  per entry, m^f_i / (m_i A_f): |Omega . nA_f| times this
   //                 is the row's diagonal (or mirror) coefficient
   //  bface_slot_d:  per entry, the mirror slot inside the row on a
   //                 reflective face, -1 on a vacuum one
   const PetscScalarKokkosView &omega_d() const { return omega_d_; }
   const PetscScalarKokkosView &inv_mass_d() const { return inv_mass_d_; }
   const PetscIntKokkosView &star_offset_d() const { return star_offset_d_; }
   const PetscIntKokkosView &star_elem_d() const { return star_elem_d_; }
   const PetscIntKokkosView &star_li_d() const { return star_li_d_; }
   const PetscIntKokkosView &star_slot_d() const { return star_slot_d_; }
   const PetscIntKokkosView &elem_vertex_d() const { return elem_vertex_d_; }
   const PetscScalarKokkosView &mass_d() const { return mass_d_; }
   const PetscScalarKokkosView &grad_d() const { return grad_d_; }
   const PetscScalarKokkosView &stiff_d() const { return stiff_d_; }
   const PetscScalarKokkosView &centre_grad_d() const { return centre_grad_d_; }
   const PetscIntKokkosView &bface_offset_d() const { return bface_offset_d_; }
   const PetscScalarKokkosView &bface_nA_d() const { return bface_nA_d_; }
   const PetscScalarKokkosView &bface_mass_d() const { return bface_mass_d_; }
   const PetscIntKokkosView &bface_slot_d() const { return bface_slot_d_; }

   // Host copies, for the tests: the element tables above in the same flat
   // layouts, the element volumes, the lumped masses m_i (owned vertices),
   // the owned vertices' coordinates (3 per vertex) and the local elements'
   // centroids (3 per element)
   const std::vector<PetscScalar> &mass_host() const { return mass_h_; }
   const std::vector<PetscScalar> &grad_host() const { return grad_h_; }
   const std::vector<PetscScalar> &stiff_host() const { return stiff_h_; }
   const std::vector<PetscInt> &elem_vertex_host() const { return elem_vertex_h_; }
   const std::vector<PetscReal> &element_volume_host() const { return elem_volume_h_; }
   const std::vector<PetscReal> &lumped_mass_host() const { return lumped_mass_h_; }
   const std::vector<PetscReal> &vertex_coord_host() const { return vertex_coord_h_; }
   const std::vector<PetscReal> &local_vertex_coord_host() const { return local_vertex_coord_h_; }
   const std::vector<PetscReal> &element_centroid_host() const { return elem_centroid_h_; }
   // Owned vertex k -> local vertex index (v - v_start) and back (-1 on an
   // overlap vertex)
   const std::vector<PetscInt> &owned_local_vertex_host() const { return owned_lv_; }
   const std::vector<PetscInt> &local_to_owned_host() const { return lv_owned_; }
   // Which boundary vertices (owned) touch any boundary face: 1/0 per owned vertex
   const std::vector<PetscInt> &on_boundary_host() const { return on_boundary_h_; }
   // Every local vertex's (owned and overlap, index v - v_start) global
   // vertex index - the vertex twin's, which is the transport rows' over
   // n_angles
   const std::vector<PetscInt> &local_vertex_global_host() const { return local_vertex_global_h_; }
   // Host copies of the boundary-face CSR above (bface_*_d), same layouts
   const std::vector<PetscInt> &bface_offset_host() const { return bface_offset_h_; }
   const std::vector<PetscScalar> &bface_nA_host() const { return bface_nA_h_; }
   const std::vector<PetscScalar> &bface_mass_host() const { return bface_mass_h_; }
   const std::vector<PetscInt> &bface_slot_host() const { return bface_slot_h_; }

   // Nodal fields: one value per OWNED vertex in (a Vec of) the dof-1 vertex
   // twin's global layout, and the same over every LOCAL vertex, overlap
   // included, in the local one (index v - v_start). create_nodal_vecs hands
   // out a matching Kokkos pair for the caller to keep (never allocate inside
   // an apply); nodal_global_to_local refreshes the overlap values
   PetscErrorCode create_nodal_vecs(Vec *global, Vec *local) const;
   PetscErrorCode nodal_global_to_local(Vec global, Vec local) const;
   DM vertex_dm() const { return vertex_dm_; }

   // The weighted load every isotropic right-hand-side-like term shares:
   //   y(i, a) += scale / m_i * sum_e coeff_e sum_j (M_ij + tau_{e,a} Omega_a . G_ji) f_j
   // for every owned vertex i and angle a (every local row), tau from
   // sigma_t_e (per local element: the group being solved's sigma_t). f is a
   // nodal field over the LOCAL vertices (a local Vec of the vertex twin); a
   // NULL f means f = 1 (a constant per element: the external source). Used
   // by ScatteringTermCG (scale -1/W, sigma_s), UboltFillSourceCG (1/W, q)
   // and GroupTransferCG (1/W, sigma_s(g' -> g)). y is a row-layout Vec
   PetscErrorCode add_weighted_load(const PetscScalarKokkosView &sigma_t_e, const PetscScalarKokkosView &coeff_e, \
      Vec f_local, PetscScalar scale, Vec y) const;

   // Painting, same semantics as UnstructuredDG's, but over EVERY LOCAL
   // ELEMENT (overlap included), by element centroid. Allocates mat_id_d sized
   // n_local_elements()
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox2D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_boxes(PetscInt background_material, const std::vector<MaterialBox3D> &boxes, \
      PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_cell_sets(PetscInt background_material, const std::map<PetscInt, PetscInt> &label_to_material, \
      PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_boxes_over(const std::vector<MaterialBox2D> &boxes, PetscIntKokkosView &mat_id_d) const;
   PetscErrorCode paint_boxes_over(const std::vector<MaterialBox3D> &boxes, PetscIntKokkosView &mat_id_d) const;

   // Which local elements this rank owns (1/0, n_local_elements()) - output
   // writes each element once
   const std::vector<PetscInt> &element_owned_host() const { return elem_owned_h_; }

   // The DM and the vertex twin
   PetscErrorCode destroy() override;

private:
   PetscErrorCode create_common(PhaseSpace &ps, PetscInt quad_dim, PetscInt n_angles, PetscScalar sum_weights, \
      const PetscScalar *mu, const PetscScalar *eta, const PetscScalar *xi, \
      const PetscInt *reflect_mu, const PetscInt *reflect_eta, const PetscInt *reflect_xi, \
      const BCSpec &bcs, PetscReal zeta);
   // The element tables, host then device
   PetscErrorCode build_element_tables();

   PetscInt dim_ = 0;
   PetscInt nv_ = 0;
   PetscReal zeta_ = 0.5;
   PetscInt n_global_vertices_ = 0;
   PetscInt c_start_ = 0, c_end_ = 0;
   PetscInt v_start_ = 0, v_end_ = 0;
   DM vertex_dm_ = NULL;

   std::vector<PetscInt> owned_lv_;
   std::vector<PetscInt> lv_owned_;
   std::vector<PetscInt> elem_owned_h_;
   std::vector<PetscInt> on_boundary_h_;
   std::vector<PetscInt> local_vertex_global_h_;
   std::vector<PetscInt> bface_offset_h_, bface_slot_h_;
   std::vector<PetscScalar> bface_nA_h_, bface_mass_h_;

   std::vector<PetscInt> elem_vertex_h_;
   std::vector<PetscScalar> mass_h_, grad_h_, stiff_h_, centre_grad_h_;
   std::vector<PetscReal> elem_volume_h_, elem_centroid_h_;
   std::vector<PetscReal> lumped_mass_h_, vertex_coord_h_, local_vertex_coord_h_;

   PetscScalarKokkosView omega_d_, inv_mass_d_;
   PetscIntKokkosView star_offset_d_, star_elem_d_, star_li_d_, star_slot_d_, elem_vertex_d_;
   PetscScalarKokkosView mass_d_, grad_d_, stiff_d_, centre_grad_d_, elem_centroid_d_;
   PetscIntKokkosView bface_offset_d_, bface_slot_d_;
   PetscScalarKokkosView bface_nA_d_, bface_mass_d_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The SUPG tau for one (element, angle): Omega . grad phi_j at the element
// centroid summed in absolute value is 2 / h_Omega, and
// tau = min(1 / sigma_t, h_Omega / zeta), h_Omega / zeta in a void. Inline so
// the assembled term and every weighted load compute it with the same
// arithmetic in the same order
KOKKOS_INLINE_FUNCTION PetscScalar UboltSUPGTau(const PetscScalarKokkosView &centre_grad_d, \
   const PetscScalarKokkosView &omega_d, PetscInt e, PetscInt a, PetscInt nv, PetscScalar sigma_t, PetscReal zeta)
{
   PetscReal sum = 0.0;
   for (PetscInt j = 0; j < nv; j++) {
      const PetscScalar s = omega_d(3 * a) * centre_grad_d((e * nv + j) * 3) + \
         omega_d(3 * a + 1) * centre_grad_d((e * nv + j) * 3 + 1) + \
         omega_d(3 * a + 2) * centre_grad_d((e * nv + j) * 3 + 2);
      sum += PetscAbsScalar(s);
   }
   // h_Omega / zeta = 2 / (zeta * sum)
   const PetscReal thin = 2.0 / (zeta * sum);
   if (PetscRealPart(sigma_t) > 0.0) {
      const PetscReal thick = 1.0 / PetscRealPart(sigma_t);
      return (thick < thin) ? thick : thin;
   }
   return thin;
}

#endif
