#include "ubolt/unstructured_cg.hpp"
#include "plex_commonk.hpp"
#include "petsc_kokkos.hpp"
#include <petscdmplex.h>
#include <petscfe.h>
#include <petscsf.h>
#include <algorithm>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Build (or read) the mesh and distribute it with a one-cell overlap under
// PETSc's FEM (closure) adjacency: every element touching an owned vertex is
// then local - see the header
PetscErrorCode UnstructuredCG::create_mesh(MPI_Comm comm, const PlexMeshSpec &mesh)
{
   PetscInt n_owned = 0;
   std::vector<PetscInt> owned;

   PetscFunctionBeginUser;

   PetscCheck(!dm_, comm, PETSC_ERR_ARG_WRONGSTATE, "create_mesh has already built this backend's mesh");

   comm_ = comm;

   PetscCall(UboltCreatePlexMesh(comm, mesh, PETSC_FALSE, PETSC_TRUE, &dm_));
   dim_ = mesh.dimension;

   // The mesh decides the global VERTEX count - the rows' spatial unit
   PetscCall(DMPlexGetDepthStratum(dm_, 0, &v_start_, &v_end_));
   PetscCall(DMPlexGetHeightStratum(dm_, 0, &c_start_, &c_end_));
   PetscCall(MarkOwnedPoints(dm_, v_start_, v_end_, owned));
   for (const PetscInt o : owned) n_owned += o;
   PetscCallMPI(MPI_Allreduce(&n_owned, &n_global_vertices_, 1, MPIU_INT, MPI_SUM, comm_));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredCG::destroy()
{
   PetscFunctionBeginUser;

   PetscCall(DMDestroy(&vertex_dm_));
   PetscCall(Discretisation::destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The DG backend's CheckPlexLayout on the vertex stratum: every vertex carries
// dof_per_vertex dof, the global section and the point SF agree on ownership,
// and the owned vertices in increasing POINT order sit at rstart + k * dof -
// contiguous. That last one is what makes "local vertex k = the k-th owned
// vertex in point order" the indexing every per-row kernel uses
static PetscErrorCode CheckCGLayout(DM dm, PetscSection gsec, PetscInt dof_per_vertex, PetscInt v_start, \
   PetscInt v_end, const std::vector<PetscInt> &owned, PetscInt n_global_expected)
{
   MPI_Comm comm = PetscObjectComm((PetscObject)dm);
   PetscInt rstart = 0, rend = 0, n_global = 0, k = 0;
   Vec gv = NULL;

   PetscFunctionBeginUser;

   // A transient global vector, only to read the ownership range
   PetscCall(DMCreateGlobalVector(dm, &gv));
   PetscCall(VecGetOwnershipRange(gv, &rstart, &rend));
   PetscCall(VecDestroy(&gv));

   for (PetscInt v = v_start; v < v_end; v++) {
      PetscInt g = 0, dof = 0;
      PetscBool is_owned = PETSC_FALSE;
      PetscCall(GlobalPointOffset(gsec, v, &g, &is_owned));
      PetscCall(PetscSectionGetDof(gsec, v, &dof));
      if (dof < 0) dof = -(dof + 1);
      PetscCheck(dof == dof_per_vertex, PETSC_COMM_SELF, PETSC_ERR_PLIB, "vertex %" PetscInt_FMT \
         " carries %" PetscInt_FMT " dof, not %" PetscInt_FMT, v, dof, dof_per_vertex);
      PetscCheck(is_owned == (PetscBool)(owned[v - v_start] != 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, \
         "the global section and the point SF disagree about who owns vertex %" PetscInt_FMT, v);
      if (!is_owned) continue;
      PetscCheck(g == rstart + k * dof_per_vertex, PETSC_COMM_SELF, PETSC_ERR_PLIB, "owned vertex %" \
         PetscInt_FMT " (local vertex %" PetscInt_FMT ") starts at global row %" PetscInt_FMT ", not the " \
         "point-ordered %" PetscInt_FMT, v, k, g, rstart + k * dof_per_vertex);
      k++;
   }

   PetscCheck(rend - rstart == k * dof_per_vertex, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
      "the DM owns %" PetscInt_FMT " rows but %" PetscInt_FMT " owned vertices x %" PetscInt_FMT " dof", \
      rend - rstart, k, dof_per_vertex);
   PetscCallMPI(MPI_Allreduce(&k, &n_global, 1, MPIU_INT, MPI_SUM, comm));
   PetscCheck(n_global == n_global_expected, comm, PETSC_ERR_PLIB, "the mesh has %" PetscInt_FMT \
      " vertices, the phase space %" PetscInt_FMT, n_global, n_global_expected);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredCG::create(PhaseSpace &ps, const SNQuadrature2D &quad, const BCSpec &bcs, PetscReal zeta)
{
   PetscFunctionBeginUser;

   PetscCall(create_common(ps, 2, quad.n_angles(), quad.sum_weights(), quad.mu_host(), quad.eta_host(), NULL, \
      quad.reflect_mu_host(), quad.reflect_eta_host(), NULL, bcs, zeta));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UnstructuredCG::create(PhaseSpace &ps, const SNQuadrature3D &quad, const BCSpec &bcs, PetscReal zeta)
{
   PetscFunctionBeginUser;

   PetscCall(create_common(ps, 3, quad.n_angles(), quad.sum_weights(), quad.mu_host(), quad.eta_host(), \
      quad.xi_host(), quad.reflect_mu_host(), quad.reflect_eta_host(), quad.reflect_xi_host(), bcs, zeta));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The element tables, on the host once: which local vertex each basis
// function belongs to, and M, G, K, the volume, the centroid and the centroid
// gradients per local element (overlap included). PetscFE gives the reference
// basis and the quadrature, DMPlexComputeCellGeometryFEM the map
//
// Basis function j is matched to its vertex by where its node lands, not by
// assuming the FE's node order is the plex closure order: the dual space's
// node j is mapped from the reference cell through the cell's own geometry and
// the nearest closure vertex is the one (asserted to be a vertex to rounding)
PetscErrorCode UnstructuredCG::build_element_tables()
{
   const PetscInt dim = dim_;
   const PetscInt n_elem = c_end_ - c_start_;
   PetscFE fe = NULL;
   PetscQuadrature quad = NULL, centre_quad = NULL;
   PetscDualSpace sp = NULL;
   PetscTabulation T = NULL, Tc = NULL;
   PetscSection csec = NULL;
   Vec coords_vec = NULL;
   const PetscScalar *coords = nullptr;
   DMPolytopeType ct;

   PetscFunctionBeginUser;

   // One cell type per mesh: the tables are nv x nv per element
   PetscCall(DMPlexGetCellType(dm_, c_start_, &ct));
   PetscBool simplex = PETSC_FALSE;
   if (ct == DM_POLYTOPE_TRIANGLE || ct == DM_POLYTOPE_TETRAHEDRON) simplex = PETSC_TRUE;
   else PetscCheck(ct == DM_POLYTOPE_QUADRILATERAL || ct == DM_POLYTOPE_HEXAHEDRON, PETSC_COMM_SELF, \
      PETSC_ERR_SUP, "the CG backend takes triangles, quadrilaterals, tetrahedra or hexahedra, not %s", \
      DMPolytopeTypes[ct]);
   for (PetscInt c = c_start_; c < c_end_; c++) {
      DMPolytopeType cc;
      PetscCall(DMPlexGetCellType(dm_, c, &cc));
      PetscCheck(cc == ct, PETSC_COMM_SELF, PETSC_ERR_SUP, "the CG backend takes one cell type per mesh, " \
         "found %s and %s", DMPolytopeTypes[ct], DMPolytopeTypes[cc]);
   }

   // Degree 1 Lagrange, a quadrature exact for the mass matrix on any affine
   // cell (and close on a curved Q1 one)
   PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, simplex, 1, 3, &fe));
   PetscCall(PetscFEGetQuadrature(fe, &quad));
   PetscCall(PetscFEGetDualSpace(fe, &sp));
   PetscInt nb = 0;
   PetscCall(PetscDualSpaceGetDimension(sp, &nb));
   PetscCheck(nb == DMPolytopeTypeGetNumVertices(ct), PETSC_COMM_SELF, PETSC_ERR_PLIB, \
      "a degree-1 Lagrange space with %" PetscInt_FMT " functions on a %" PetscInt_FMT "-vertex cell", \
      nb, (PetscInt)DMPolytopeTypeGetNumVertices(ct));
   nv_ = nb;
   const PetscInt nv = nv_;

   PetscInt q_dim = 0, q_nc = 0, nq = 0;
   const PetscReal *q_pts = nullptr, *q_wts = nullptr;
   PetscCall(PetscQuadratureGetData(quad, &q_dim, &q_nc, &nq, &q_pts, &q_wts));
   PetscCall(PetscFEGetCellTabulation(fe, 1, &T));

   // The reference nodes, and their mean - the reference centroid (the
   // vertices' mean is the centroid of a simplex and of a box alike)
   std::vector<PetscReal> ref_node(nv * dim), ref_centre(dim, 0.0);
   for (PetscInt j = 0; j < nv; j++) {
      PetscQuadrature fq = NULL;
      PetscInt n_fp = 0;
      const PetscReal *fp = nullptr;
      PetscCall(PetscDualSpaceGetFunctional(sp, j, &fq));
      PetscCall(PetscQuadratureGetData(fq, NULL, NULL, &n_fp, &fp, NULL));
      PetscCheck(n_fp == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "a degree-1 Lagrange functional is a point evaluation");
      for (PetscInt d = 0; d < dim; d++) {
         ref_node[j * dim + d] = fp[d];
         ref_centre[d] += fp[d] / nv;
      }
   }
   PetscReal one = 1.0;
   PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &centre_quad));
   {
      PetscReal *pts = nullptr, *wts = nullptr;
      PetscCall(PetscMalloc1(dim, &pts));
      PetscCall(PetscMalloc1(1, &wts));
      for (PetscInt d = 0; d < dim; d++) pts[d] = ref_centre[d];
      wts[0] = one;
      PetscCall(PetscQuadratureSetData(centre_quad, dim, 1, 1, pts, wts));
   }
   PetscCall(PetscFECreateTabulation(fe, 1, 1, ref_centre.data(), 1, &Tc));

   PetscCall(DMGetCoordinateSection(dm_, &csec));
   PetscCall(DMGetCoordinatesLocal(dm_, &coords_vec));
   PetscCall(VecGetArrayRead(coords_vec, &coords));

   elem_vertex_h_.assign(n_elem * nv, -1);
   mass_h_.assign(n_elem * nv * nv, 0.0);
   grad_h_.assign(n_elem * nv * nv * 3, 0.0);
   stiff_h_.assign(n_elem * nv * nv * 9, 0.0);
   centre_grad_h_.assign(n_elem * nv * 3, 0.0);
   elem_volume_h_.assign(n_elem, 0.0);
   elem_centroid_h_.assign(3 * n_elem, 0.0);

   std::vector<PetscReal> v(nq * dim), J(nq * dim * dim), invJ(nq * dim * dim), detJ(nq);
   std::vector<PetscReal> vc(dim), Jc(dim * dim), invJc(dim * dim), gphi(nq * nv * 3);
   PetscReal detJc = 0.0;

   for (PetscInt c = c_start_; c < c_end_; c++) {

      const PetscInt e = c - c_start_;

      // This cell's vertices, and their coordinates
      PetscInt n_cl = 0, *cl = nullptr;
      std::vector<PetscInt> cell_vertices;
      PetscCall(DMPlexGetTransitiveClosure(dm_, c, PETSC_TRUE, &n_cl, &cl));
      for (PetscInt p = 0; p < n_cl; p++) {
         if (cl[2 * p] >= v_start_ && cl[2 * p] < v_end_) cell_vertices.push_back(cl[2 * p]);
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm_, c, PETSC_TRUE, &n_cl, &cl));
      PetscCheck((PetscInt)cell_vertices.size() == nv, PETSC_COMM_SELF, PETSC_ERR_PLIB, "cell %" PetscInt_FMT \
         " has %" PetscInt_FMT " vertices, not %" PetscInt_FMT, c, (PetscInt)cell_vertices.size(), nv);

      // Each basis node onto its vertex
      PetscReal h2 = 0.0;
      {
         PetscInt off0 = 0, off1 = 0;
         PetscCall(PetscSectionGetOffset(csec, cell_vertices[0], &off0));
         for (PetscInt w = 1; w < nv; w++) {
            PetscReal dist = 0.0;
            PetscCall(PetscSectionGetOffset(csec, cell_vertices[w], &off1));
            for (PetscInt d = 0; d < dim; d++) {
               const PetscReal dx = PetscRealPart(coords[off1 + d] - coords[off0 + d]);
               dist += dx * dx;
            }
            h2 = PetscMax(h2, dist);
         }
      }
      std::vector<PetscInt> taken(nv, 0);
      for (PetscInt j = 0; j < nv; j++) {
         PetscReal x[3] = {0.0, 0.0, 0.0};
         PetscCall(DMPlexReferenceToCoordinates(dm_, c, 1, &ref_node[j * dim], x));
         PetscInt best = -1;
         PetscReal best_dist = PETSC_MAX_REAL;
         for (PetscInt w = 0; w < nv; w++) {
            PetscInt off = 0;
            PetscReal dist = 0.0;
            PetscCall(PetscSectionGetOffset(csec, cell_vertices[w], &off));
            for (PetscInt d = 0; d < dim; d++) {
               const PetscReal dx = x[d] - PetscRealPart(coords[off + d]);
               dist += dx * dx;
            }
            if (dist < best_dist) {
               best_dist = dist;
               best = w;
            }
         }
         PetscCheck(best >= 0 && best_dist <= 1e-20 * h2 && !taken[best], PETSC_COMM_SELF, PETSC_ERR_PLIB, \
            "basis node %" PetscInt_FMT " of cell %" PetscInt_FMT " does not land on a vertex of its own", j, c);
         taken[best] = 1;
         elem_vertex_h_[e * nv + j] = cell_vertices[best] - v_start_;
      }

      // The map at the quadrature points: physical gradients and weights
      PetscCall(DMPlexComputeCellGeometryFEM(dm_, c, quad, v.data(), J.data(), invJ.data(), detJ.data()));
      PetscReal vol = 0.0, cen[3] = {0.0, 0.0, 0.0};
      for (PetscInt q = 0; q < nq; q++) {
         PetscCheck(PetscAbsReal(detJ[q]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "cell %" PetscInt_FMT \
            " is degenerate (det J = %g)", c, (double)detJ[q]);
         detJ[q] = PetscAbsReal(detJ[q]);
         for (PetscInt j = 0; j < nv; j++) {
            for (PetscInt d = 0; d < 3; d++) {
               PetscReal g = 0.0;
               if (d < dim) {
                  for (PetscInt i = 0; i < dim; i++) g += T->T[1][(q * nv + j) * dim + i] * invJ[(q * dim + i) * dim + d];
               }
               gphi[(q * nv + j) * 3 + d] = g;
            }
         }
      }
      for (PetscInt q = 0; q < nq; q++) {
         const PetscReal wq = q_wts[q] * detJ[q];
         vol += wq;
         for (PetscInt d = 0; d < dim; d++) cen[d] += wq * v[q * dim + d];
         for (PetscInt i = 0; i < nv; i++) {
            const PetscReal phi_i = T->T[0][q * nv + i];
            for (PetscInt j = 0; j < nv; j++) {
               const PetscReal phi_j = T->T[0][q * nv + j];
               const PetscInt ij = (e * nv + i) * nv + j;
               mass_h_[ij] += wq * phi_i * phi_j;
               for (PetscInt d = 0; d < 3; d++) {
                  grad_h_[ij * 3 + d] += wq * phi_i * gphi[(q * nv + j) * 3 + d];
                  for (PetscInt dd = 0; dd < 3; dd++) {
                     stiff_h_[ij * 9 + 3 * d + dd] += wq * gphi[(q * nv + i) * 3 + d] * gphi[(q * nv + j) * 3 + dd];
                  }
               }
            }
         }
      }
      elem_volume_h_[e] = vol;
      for (PetscInt d = 0; d < dim; d++) elem_centroid_h_[3 * e + d] = cen[d] / vol;

      // And at the reference centroid, for h_Omega
      PetscCall(DMPlexComputeCellGeometryFEM(dm_, c, centre_quad, vc.data(), Jc.data(), invJc.data(), &detJc));
      for (PetscInt j = 0; j < nv; j++) {
         for (PetscInt d = 0; d < dim; d++) {
            PetscReal g = 0.0;
            for (PetscInt i = 0; i < dim; i++) g += Tc->T[1][j * dim + i] * invJc[i * dim + d];
            centre_grad_h_[(e * nv + j) * 3 + d] = g;
         }
      }
   }

   PetscCall(VecRestoreArrayRead(coords_vec, &coords));
   PetscCall(PetscTabulationDestroy(&Tc));
   PetscCall(PetscQuadratureDestroy(&centre_quad));
   PetscCall(PetscFEDestroy(&fe));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// One boundary face as seen from one owned vertex
struct CGBoundaryFace {
   PetscScalar nA[3] = {0.0, 0.0, 0.0};
   PetscReal centroid[3] = {0.0, 0.0, 0.0};
   PetscReal area = 0.0;
   PetscInt n_face_vertices = 0;
   PetscInt label = -1;
   PetscInt axis = 0;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The layout, the element tables, the boundary faces and the COO sparsity.
// This happens on the host but we only need to do it once
PetscErrorCode UnstructuredCG::create_common(PhaseSpace &ps, PetscInt quad_dim, PetscInt n_angles, \
   PetscScalar sum_weights, const PetscScalar *mu, const PetscScalar *eta, const PetscScalar *xi, \
   const PetscInt *reflect_mu, const PetscInt *reflect_eta, const PetscInt *reflect_xi, const BCSpec &bcs, \
   PetscReal zeta)
{
   PetscSection sec = NULL, gsec = NULL;
   DMLabel face_sets = NULL;
   std::vector<PetscInt> owned, elem_owned;
   PetscBool failed = PETSC_FALSE;
   char message[512] = "";

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCheck(dm_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
      "UnstructuredCG::create_mesh has to come first - the mesh decides the vertex count");
   PetscCheck(quad_dim == dim_, comm_, PETSC_ERR_ARG_INCOMP, "a %" PetscInt_FMT "D quadrature was " \
      "handed to a %" PetscInt_FMT "D mesh", quad_dim, dim_);
   PetscCheck(n_angles == ps.n_angles, comm_, PETSC_ERR_ARG_INCOMP, \
      "quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, n_angles, ps.n_angles);
   PetscCheck(ps.n_cells == n_global_vertices_, comm_, PETSC_ERR_ARG_INCOMP, "the mesh has %" PetscInt_FMT \
      " vertices but the phase space %" PetscInt_FMT " cells - size it off n_global_vertices()", \
      n_global_vertices_, ps.n_cells);
   PetscCheck(bcs.ghost_flux_vacuum(), comm_, PETSC_ERR_SUP, "the CG-SUPG backend imposes every boundary " \
      "condition weakly - there is no Dirichlet-cell row, so drop \"vacuum_treatment\": \"dirichlet_cell\"");
   PetscCheck(zeta > 0.0, comm_, PETSC_ERR_ARG_OUTOFRANGE, "the SUPG zeta must be positive, was given %g", \
      (double)zeta);

   const PetscInt dim = dim_;
   const PetscInt *reflect[3] = {reflect_mu, reflect_eta, reflect_xi};
   zeta_ = zeta;
   ps.n_basis = 1;

   // ~~~~~~~~~~
   // The layout: n_angles dof on every vertex (overlap ones included - the
   // global section says where a ghost column lives), nothing anywhere else
   // ~~~~~~~~~~
   PetscInt p_start = 0, p_end = 0;
   PetscCall(DMPlexGetChart(dm_, &p_start, &p_end));
   PetscCall(PetscSectionCreate(comm_, &sec));
   PetscCall(PetscSectionSetChart(sec, p_start, p_end));
   for (PetscInt vtx = v_start_; vtx < v_end_; vtx++) PetscCall(PetscSectionSetDof(sec, vtx, n_angles));
   PetscCall(PetscSectionSetUp(sec));
   PetscCall(DMSetLocalSection(dm_, sec));
   PetscCall(PetscSectionDestroy(&sec));
   PetscCall(DMGetGlobalSection(dm_, &gsec));

   PetscCall(MarkOwnedPoints(dm_, v_start_, v_end_, owned));
   PetscCall(CheckCGLayout(dm_, gsec, n_angles, v_start_, v_end_, owned, ps.n_cells));

   const PetscInt n_lv = v_end_ - v_start_;
   owned_lv_.clear();
   lv_owned_.assign(n_lv, -1);
   for (PetscInt lv = 0; lv < n_lv; lv++) {
      if (!owned[lv]) continue;
      lv_owned_[lv] = (PetscInt)owned_lv_.size();
      owned_lv_.push_back(lv);
   }
   const PetscInt local_vertices = (PetscInt)owned_lv_.size();

   // The DM decided the decomposition - the PhaseSpace is told
   ps.local_cells = local_vertices;
   ps_ = ps;
   const PetscInt local_rows = ps.local_rows();

   PetscCall(MarkOwnedPoints(dm_, c_start_, c_end_, elem_owned_h_));

   // ~~~~~~~~~~
   // The element tables, and the vertex coordinates
   // ~~~~~~~~~~
   PetscCall(build_element_tables());
   const PetscInt nv = nv_;
   const PetscInt n_elem = c_end_ - c_start_;
   {
      PetscSection csec = NULL;
      Vec coords_vec = NULL;
      const PetscScalar *coords = nullptr;
      PetscCall(DMGetCoordinateSection(dm_, &csec));
      PetscCall(DMGetCoordinatesLocal(dm_, &coords_vec));
      PetscCall(VecGetArrayRead(coords_vec, &coords));
      local_vertex_coord_h_.assign(3 * n_lv, 0.0);
      for (PetscInt lv = 0; lv < n_lv; lv++) {
         PetscInt off = 0;
         PetscCall(PetscSectionGetOffset(csec, v_start_ + lv, &off));
         for (PetscInt d = 0; d < dim; d++) local_vertex_coord_h_[3 * lv + d] = PetscRealPart(coords[off + d]);
      }
      PetscCall(VecRestoreArrayRead(coords_vec, &coords));
   }
   vertex_coord_h_.assign(3 * local_vertices, 0.0);
   for (PetscInt k = 0; k < local_vertices; k++) {
      for (PetscInt d = 0; d < 3; d++) vertex_coord_h_[3 * k + d] = local_vertex_coord_h_[3 * owned_lv_[k] + d];
   }

   // ~~~~~~~~~~
   // Per owned vertex: its star (elements and its basis index in each), its
   // neighbours, its lumped mass and its boundary faces
   // ~~~~~~~~~~
   PetscCall(DMGetLabel(dm_, "Face Sets", &face_sets));

   std::vector<PetscInt> star_offset(local_vertices + 1, 0), star_elem, star_li;
   std::vector<std::vector<PetscInt>> neighbours(local_vertices);
   std::vector<std::vector<CGBoundaryFace>> bfaces(local_vertices);
   std::vector<PetscInt> labels_seen;
   lumped_mass_h_.assign(local_vertices, 0.0);
   on_boundary_h_.assign(local_vertices, 0);

   for (PetscInt k = 0; k < local_vertices; k++) {

      const PetscInt lv = owned_lv_[k];
      const PetscInt vtx = v_start_ + lv;
      PetscInt n_st = 0, *st = nullptr;
      PetscCall(DMPlexGetTransitiveClosure(dm_, vtx, PETSC_FALSE, &n_st, &st));

      for (PetscInt p = 0; p < n_st; p++) {

         const PetscInt pt = st[2 * p];

         // An element of the star
         if (pt >= c_start_ && pt < c_end_) {
            const PetscInt e = pt - c_start_;
            PetscInt li = -1;
            for (PetscInt j = 0; j < nv; j++) {
               if (elem_vertex_h_[e * nv + j] == lv) li = j;
               else neighbours[k].push_back(elem_vertex_h_[e * nv + j]);
            }
            PetscCheck(li >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "vertex %" PetscInt_FMT \
               " is in the star of cell %" PetscInt_FMT " but not in its closure", vtx, pt);
            star_elem.push_back(e);
            star_li.push_back(li);
            for (PetscInt j = 0; j < nv; j++) lumped_mass_h_[k] += PetscRealPart(mass_h_[(e * nv + li) * nv + j]);
            continue;
         }

         // A boundary face of the star: a face (height 1) with one cell. The
         // FEM overlap put every cell touching this vertex on this rank, so a
         // face through it with one local cell really is the domain boundary
         PetscInt height = 0;
         PetscCall(DMPlexGetPointHeight(dm_, pt, &height));
         if (height != 1) continue;
         PetscInt n_support = 0;
         const PetscInt *support = nullptr;
         PetscCall(DMPlexGetSupportSize(dm_, pt, &n_support));
         if (n_support != 1) continue;
         PetscCall(DMPlexGetSupport(dm_, pt, &support));

         CGBoundaryFace bf;
         PetscReal normal[3] = {0.0, 0.0, 0.0};
         PetscCall(DMPlexComputeCellGeometryFVM(dm_, pt, &bf.area, bf.centroid, normal));
         PetscCheck(bf.area > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "face %" PetscInt_FMT \
            " has non-positive area %g", pt, (double)bf.area);
         PetscReal norm = 0.0, outward = 0.0;
         const PetscInt e_sup = support[0] - c_start_;
         for (PetscInt d = 0; d < dim; d++) {
            norm += normal[d] * normal[d];
            outward += normal[d] * (bf.centroid[d] - elem_centroid_h_[3 * e_sup + d]);
         }
         norm = PetscSqrtReal(norm);
         const PetscReal sign = (outward < 0.0) ? -1.0 : 1.0;
         for (PetscInt d = 0; d < dim; d++) bf.nA[d] = sign * bf.area * normal[d] / norm;
         bf.axis = DominantAxis(bf.nA, dim);

         PetscInt n_fcl = 0, *fcl = nullptr;
         PetscCall(DMPlexGetTransitiveClosure(dm_, pt, PETSC_TRUE, &n_fcl, &fcl));
         for (PetscInt q = 0; q < n_fcl; q++) {
            if (fcl[2 * q] >= v_start_ && fcl[2 * q] < v_end_) bf.n_face_vertices++;
         }
         PetscCall(DMPlexRestoreTransitiveClosure(dm_, pt, PETSC_TRUE, &n_fcl, &fcl));

         if (face_sets) PetscCall(DMLabelGetValue(face_sets, pt, &bf.label));
         labels_seen.push_back(bf.label);

         const BCFace bc = bcs.face(bf.label);
         if (!failed && bc.n_window_pairs != 0 && bc.n_window_pairs != dim - 1) {
            failed = PETSC_TRUE;
            PetscCall(PetscSNPrintf(message, sizeof(message), "a %" PetscInt_FMT "D face takes %" \
               PetscInt_FMT " tangential [lo, hi] window pairs, face set %" PetscInt_FMT " was given %" \
               PetscInt_FMT, dim, dim - 1, bf.label, bc.n_window_pairs));
         }
         if (!failed && bc.type == BCType::REFLECT && \
             PetscAbsScalar(bf.nA[bf.axis]) < (1.0 - 1e-10) * bf.area) {
            failed = PETSC_TRUE;
            PetscCall(PetscSNPrintf(message, sizeof(message), "face %" PetscInt_FMT " (face set %" \
               PetscInt_FMT ") is reflective but not axis-aligned - reflection maps an ordinate onto " \
               "an ordinate only across a face normal to x, y or z", pt, bf.label));
         }
         bfaces[k].push_back(bf);
         on_boundary_h_[k] = 1;
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm_, vtx, PETSC_FALSE, &n_st, &st));

      star_offset[k + 1] = (PetscInt)star_elem.size();
      std::sort(neighbours[k].begin(), neighbours[k].end());
      neighbours[k].erase(std::unique(neighbours[k].begin(), neighbours[k].end()), neighbours[k].end());
      PetscCheck(lumped_mass_h_[k] > 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "vertex %" PetscInt_FMT \
         " has a non-positive lumped mass %g", vtx, (double)lumped_mass_h_[k]);
   }
   PetscCall(CollectiveFailure(comm_, failed, message));

   // Every label the BCSpec names has to be on some boundary face somewhere
   // (see UnstructuredDG). A boundary face's vertices are all owned by
   // someone, so every one is seen above on some rank
   for (const auto &entry : bcs.faces()) {
      PetscMPIInt seen = 0, any = 0;
      for (const PetscInt value : labels_seen) {
         if (value == entry.first) seen = 1;
      }
      PetscCallMPI(MPI_Allreduce(&seen, &any, 1, MPI_INT, MPI_MAX, comm_));
      PetscCheck(any, comm_, PETSC_ERR_ARG_WRONG, "a boundary condition was given for \"Face Sets\" value %" \
         PetscInt_FMT " but no boundary face of the mesh carries it", entry.first);
   }

   // ~~~~~~~~~~
   // The ordinates, flattened (a * 3 + d)
   // ~~~~~~~~~~
   std::vector<PetscScalar> omega(3 * n_angles, 0.0);
   for (PetscInt a = 0; a < n_angles; a++) {
      omega[3 * a] = mu[a];
      omega[3 * a + 1] = eta[a];
      omega[3 * a + 2] = xi ? xi[a] : 0.0;
   }

   // ~~~~~~~~~~
   // COO coordinates, slot order per row: neighbours (point order), one
   // mirror slot per reflective axis at the vertex, the diagonal last - see
   // the header. The star's (element, element vertex) -> slot map and the
   // boundary faces' mirror slots are per VERTEX (every angle's row of a
   // vertex has the same layout, nulled slots included)
   // ~~~~~~~~~~
   oor_.clear();
   ooc_.clear();
   std::vector<PetscInt> row_slot_offset(local_rows + 1, 0);
   std::vector<PetscInt> diag_slot(local_rows, 0);
   std::vector<PetscInt> is_bc_row(local_rows, 0);
   std::vector<PetscInt> reflect_slot(local_rows, -1);
   std::vector<PetscScalar> dirichlet_value(local_rows, 0.0);
   std::vector<PetscScalar> ghost_inflow(local_rows, 0.0);

   std::vector<PetscInt> star_slot(star_elem.size() * nv, -1);
   std::vector<PetscInt> bface_offset(local_vertices + 1, 0), bface_slot;
   std::vector<PetscScalar> bface_nA, bface_mass;

   for (PetscInt k = 0; k < local_vertices; k++) {

      const PetscInt lv = owned_lv_[k];
      PetscInt own_g = 0;
      PetscBool is_owned = PETSC_FALSE;
      PetscCall(GlobalPointOffset(gsec, v_start_ + lv, &own_g, &is_owned));

      std::vector<PetscInt> nb_g(neighbours[k].size());
      for (size_t n = 0; n < neighbours[k].size(); n++) {
         PetscCall(GlobalPointOffset(gsec, v_start_ + neighbours[k][n], &nb_g[n], &is_owned));
      }
      const PetscInt n_nb = (PetscInt)neighbours[k].size();

      // The reflective axes touching this vertex, ascending
      std::vector<PetscInt> refl_axes;
      for (const auto &bf : bfaces[k]) {
         if (bcs.type(bf.label) == BCType::REFLECT) refl_axes.push_back(bf.axis);
      }
      std::sort(refl_axes.begin(), refl_axes.end());
      refl_axes.erase(std::unique(refl_axes.begin(), refl_axes.end()), refl_axes.end());
      const PetscInt n_refl = (PetscInt)refl_axes.size();
      const PetscInt row_len = n_nb + n_refl + 1;

      // The star's slots inside the row
      for (PetscInt s = star_offset[k]; s < star_offset[k + 1]; s++) {
         const PetscInt e = star_elem[s];
         for (PetscInt j = 0; j < nv; j++) {
            const PetscInt col_lv = elem_vertex_h_[e * nv + j];
            if (col_lv == lv) star_slot[s * nv + j] = row_len - 1;
            else {
               const auto it = std::lower_bound(neighbours[k].begin(), neighbours[k].end(), col_lv);
               star_slot[s * nv + j] = (PetscInt)(it - neighbours[k].begin());
            }
         }
      }

      // The boundary faces' data, per vertex
      const PetscReal inv_m = 1.0 / lumped_mass_h_[k];
      for (const auto &bf : bfaces[k]) {
         for (PetscInt d = 0; d < 3; d++) bface_nA.push_back(bf.nA[d]);
         bface_mass.push_back(inv_m / (PetscReal)bf.n_face_vertices / bf.area);
         PetscInt slot = -1;
         if (bcs.type(bf.label) == BCType::REFLECT) {
            slot = n_nb + (PetscInt)(std::lower_bound(refl_axes.begin(), refl_axes.end(), bf.axis) - refl_axes.begin());
         }
         bface_slot.push_back(slot);
      }
      bface_offset[k + 1] = (PetscInt)bface_slot.size();

      for (PetscInt a = 0; a < n_angles; a++) {

         const PetscInt r = k * n_angles + a;
         const PetscInt row_g = own_g + a;
         row_slot_offset[r] = (PetscInt)oor_.size();

         for (PetscInt n = 0; n < n_nb; n++) {
            oor_.push_back(row_g);
            ooc_.push_back(nb_g[n] + a);
         }
         // A mirror slot is live only for an angle coming in through a
         // reflective face on its axis
         for (PetscInt x = 0; x < n_refl; x++) {
            PetscBool live = PETSC_FALSE;
            for (size_t f = 0; f < bfaces[k].size(); f++) {
               const auto &bf = bfaces[k][f];
               if (bcs.type(bf.label) != BCType::REFLECT || bf.axis != refl_axes[x]) continue;
               if (PetscRealPart(FaceFlux(omega.data(), a, bf.nA, 0)) < 0.0) live = PETSC_TRUE;
            }
            oor_.push_back(live ? row_g : -1);
            ooc_.push_back(live ? own_g + reflect[refl_axes[x]][a] : -1);
         }
         diag_slot[r] = (PetscInt)oor_.size();
         oor_.push_back(row_g);
         ooc_.push_back(row_g);

         // The vacuum inflow, face by face, windowed by the face centroid
         for (size_t f = 0; f < bfaces[k].size(); f++) {
            const auto &bf = bfaces[k][f];
            const BCFace bc = bcs.face(bf.label);
            if (bc.type == BCType::REFLECT) continue;
            const PetscScalar s = FaceFlux(omega.data(), a, bf.nA, 0);
            if (PetscRealPart(s) >= 0.0) continue;
            PetscReal t[2] = {0.0, 0.0};
            PetscInt n_t = 0;
            for (PetscInt d = 0; d < dim; d++) {
               if (d != bf.axis) t[n_t++] = bf.centroid[d];
            }
            if (!InWindow(bc, t, n_t)) continue;
            ghost_inflow[r] += -s * inv_m / (PetscReal)bf.n_face_vertices / bf.area * bc.inflow / sum_weights;
         }

         PetscCheck((PetscInt)oor_.size() - row_slot_offset[r] == row_len, PETSC_COMM_SELF, PETSC_ERR_PLIB, \
            "row %" PetscInt_FMT " has %" PetscInt_FMT " slots, not %" PetscInt_FMT, r, \
            (PetscInt)oor_.size() - row_slot_offset[r], row_len);
      }
   }
   row_slot_offset[local_rows] = (PetscInt)oor_.size();

   PetscCall(set_pattern(row_slot_offset, diag_slot, is_bc_row, reflect_slot, dirichlet_value, ghost_inflow, \
      PETSC_TRUE));

   // ~~~~~~~~~~
   // The device tables - flat rank-1 views only
   // ~~~~~~~~~~
   std::vector<PetscScalar> inv_mass(local_vertices), centroid(3 * n_elem);
   for (PetscInt k = 0; k < local_vertices; k++) inv_mass[k] = 1.0 / lumped_mass_h_[k];
   for (PetscInt i = 0; i < 3 * n_elem; i++) centroid[i] = elem_centroid_h_[i];

   auto upload_scalar = [](const std::vector<PetscScalar> &h, const char *name, PetscScalarKokkosView &d) {
      d = PetscScalarKokkosView(name, h.size());
      if (!h.empty()) {
         PetscScalarKokkosViewHostUnmanaged hv(const_cast<PetscScalar *>(h.data()), h.size());
         Kokkos::deep_copy(d, hv);
      }
   };
   auto upload_int = [](const std::vector<PetscInt> &h, const char *name, PetscIntKokkosView &d) {
      d = PetscIntKokkosView(name, h.size());
      if (!h.empty()) {
         PetscIntKokkosViewHostUnmanaged hv(const_cast<PetscInt *>(h.data()), h.size());
         Kokkos::deep_copy(d, hv);
      }
   };
   upload_scalar(omega, "omega_d", omega_d_);
   upload_scalar(inv_mass, "inv_mass_d", inv_mass_d_);
   upload_int(star_offset, "star_offset_d", star_offset_d_);
   upload_int(star_elem, "star_elem_d", star_elem_d_);
   upload_int(star_li, "star_li_d", star_li_d_);
   upload_int(star_slot, "star_slot_d", star_slot_d_);
   upload_int(elem_vertex_h_, "elem_vertex_d", elem_vertex_d_);
   upload_scalar(mass_h_, "mass_d", mass_d_);
   upload_scalar(grad_h_, "grad_d", grad_d_);
   upload_scalar(stiff_h_, "stiff_d", stiff_d_);
   upload_scalar(centre_grad_h_, "centre_grad_d", centre_grad_d_);
   upload_scalar(centroid, "elem_centroid_d", elem_centroid_d_);
   upload_int(bface_offset, "bface_offset_d", bface_offset_d_);
   upload_scalar(bface_nA, "bface_nA_d", bface_nA_d_);
   upload_scalar(bface_mass, "bface_mass_d", bface_mass_d_);
   upload_int(bface_slot, "bface_slot_d", bface_slot_d_);

   // ~~~~~~~~~~
   // The dof-1 vertex twin, for nodal fields. Its owned vertices sit in point
   // order too, so a nodal global Vec is indexed by owned vertex k
   // ~~~~~~~~~~
   {
      PetscSection vsec = NULL, vgsec = NULL;
      PetscCall(DMClone(dm_, &vertex_dm_));
      PetscCall(PetscSectionCreate(comm_, &vsec));
      PetscCall(PetscSectionSetChart(vsec, p_start, p_end));
      for (PetscInt vtx = v_start_; vtx < v_end_; vtx++) PetscCall(PetscSectionSetDof(vsec, vtx, 1));
      PetscCall(PetscSectionSetUp(vsec));
      PetscCall(DMSetLocalSection(vertex_dm_, vsec));
      PetscCall(PetscSectionDestroy(&vsec));
      PetscCall(DMSetVecType(vertex_dm_, VECKOKKOS));
      PetscCall(DMGetGlobalSection(vertex_dm_, &vgsec));
      PetscCall(CheckCGLayout(vertex_dm_, vgsec, 1, v_start_, v_end_, owned, ps.n_cells));
      // The local Vec is indexed v - v_start: the section's offsets are
      // assigned in chart order, which is point order
      PetscSection lsec = NULL;
      PetscCall(DMGetLocalSection(vertex_dm_, &lsec));
      for (PetscInt vtx = v_start_; vtx < v_end_; vtx++) {
         PetscInt off = 0;
         PetscCall(PetscSectionGetOffset(lsec, vtx, &off));
         PetscCheck(off == vtx - v_start_, PETSC_COMM_SELF, PETSC_ERR_PLIB, "vertex %" PetscInt_FMT \
            " sits at local offset %" PetscInt_FMT ", not %" PetscInt_FMT, vtx, off, vtx - v_start_);
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredCG::create_nodal_vecs(Vec *global, Vec *local) const
{
   PetscFunctionBeginUser;

   PetscCheck(vertex_dm_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "UnstructuredCG::create has to come first");
   PetscCall(DMCreateGlobalVector(vertex_dm_, global));
   PetscCall(DMCreateLocalVector(vertex_dm_, local));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UnstructuredCG::nodal_global_to_local(Vec global, Vec local) const
{
   PetscFunctionBeginUser;

   PetscCall(DMGlobalToLocalBegin(vertex_dm_, global, INSERT_VALUES, local));
   PetscCall(DMGlobalToLocalEnd(vertex_dm_, global, INSERT_VALUES, local));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A free function rather than a local lambda (see docs/dev/kokkos.md). One
// thread per row, gathering over the vertex's star; f_d empty means f = 1
static void WeightedLoadKernel(PetscScalarKokkosView y_d, PetscScalarConstKokkosView f_d, bool have_f, \
   PetscScalarKokkosView sigma_t_e, PetscScalarKokkosView coeff_e, PetscScalar scale, PetscReal zeta, \
   PetscScalarKokkosView omega_d, PetscScalarKokkosView inv_mass_d, PetscIntKokkosView star_offset_d, \
   PetscIntKokkosView star_elem_d, PetscIntKokkosView star_li_d, PetscIntKokkosView elem_vertex_d, \
   PetscScalarKokkosView mass_d, PetscScalarKokkosView grad_d, PetscScalarKokkosView centre_grad_d, \
   PetscInt nv, PetscInt n_angles, PetscInt local_rows)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt r) {

         const PetscInt k = r / n_angles;
         const PetscInt a = r % n_angles;
         PetscScalar acc = 0.0;
         for (PetscInt s = star_offset_d(k); s < star_offset_d(k + 1); s++) {
            const PetscInt e = star_elem_d(s);
            const PetscInt li = star_li_d(s);
            const PetscScalar c = coeff_e(e);
            if (c == 0.0) continue;
            const PetscScalar tau = UboltSUPGTau(centre_grad_d, omega_d, e, a, nv, sigma_t_e(e), zeta);
            for (PetscInt j = 0; j < nv; j++) {
               // (phi_i + tau Omega . grad phi_i, phi_j): M_ij + tau Omega . G_ji
               const PetscInt ji = (e * nv + j) * nv + li;
               const PetscScalar w = mass_d((e * nv + li) * nv + j) + tau * (omega_d(3 * a) * grad_d(ji * 3) + \
                  omega_d(3 * a + 1) * grad_d(ji * 3 + 1) + omega_d(3 * a + 2) * grad_d(ji * 3 + 2));
               const PetscScalar f = have_f ? f_d(elem_vertex_d(e * nv + j)) : 1.0;
               acc += c * w * f;
            }
         }
         y_d(r) += scale * inv_mass_d(k) * acc;
      });
}

PetscErrorCode UnstructuredCG::add_weighted_load(const PetscScalarKokkosView &sigma_t_e, \
   const PetscScalarKokkosView &coeff_e, Vec f_local, PetscScalar scale, Vec y) const
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());
   PetscCheck((PetscInt)sigma_t_e.extent(0) == n_local_elements() && (PetscInt)coeff_e.extent(0) == n_local_elements(), \
      PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "per-element data covers %" PetscInt_FMT " / %" PetscInt_FMT \
      " elements but there are %" PetscInt_FMT " local elements", (PetscInt)sigma_t_e.extent(0), \
      (PetscInt)coeff_e.extent(0), n_local_elements());

   PetscScalarConstKokkosView f_d;
   PetscScalarKokkosView y_d;
   if (f_local) PetscCall(VecGetKokkosView(f_local, &f_d));
   PetscCall(VecGetKokkosView(y, &y_d));

   WeightedLoadKernel(y_d, f_d, f_local != NULL, sigma_t_e, coeff_e, scale, zeta_, omega_d_, inv_mass_d_, \
      star_offset_d_, star_elem_d_, star_li_d_, elem_vertex_d_, mass_d_, grad_d_, centre_grad_d_, nv_, \
      ps_.n_angles, ps_.local_rows());

   PetscCall(VecRestoreKokkosView(y, &y_d));
   if (f_local) PetscCall(VecRestoreKokkosView(f_local, &f_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Painting - every local element, by centroid
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UnstructuredCG::paint_boxes_over(const std::vector<MaterialBox2D> &boxes, PetscIntKokkosView &mat_id_d) const
{
   std::vector<PetscScalar> box_lohi;
   std::vector<PetscInt> box_material;

   PetscFunctionBeginUser;

   PetscCheck(dim_ == 2, comm_, PETSC_ERR_ARG_INCOMP, "2D boxes painted onto a %" PetscInt_FMT "D mesh", dim_);
   PetscCall(ps_.check_decomposed());
   FlattenBoxes(boxes, box_lohi, box_material);
   PetscCall(PaintFlatBoxes(comm_, dim_, n_local_elements(), elem_centroid_d_, (PetscInt)boxes.size(), box_lohi, \
      box_material, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UnstructuredCG::paint_boxes_over(const std::vector<MaterialBox3D> &boxes, PetscIntKokkosView &mat_id_d) const
{
   std::vector<PetscScalar> box_lohi;
   std::vector<PetscInt> box_material;

   PetscFunctionBeginUser;

   PetscCheck(dim_ == 3, comm_, PETSC_ERR_ARG_INCOMP, "3D boxes painted onto a %" PetscInt_FMT "D mesh", dim_);
   PetscCall(ps_.check_decomposed());
   FlattenBoxes(boxes, box_lohi, box_material);
   PetscCall(PaintFlatBoxes(comm_, dim_, n_local_elements(), elem_centroid_d_, (PetscInt)boxes.size(), box_lohi, \
      box_material, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UnstructuredCG::paint_boxes(PetscInt background_material, const std::vector<MaterialBox2D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCall(PetscKokkosInitializeCheck());
   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);
   mat_id_d = PetscIntKokkosView("mat_id_d", n_local_elements());
   Kokkos::deep_copy(mat_id_d, background_material);
   PetscCall(paint_boxes_over(boxes, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UnstructuredCG::paint_boxes(PetscInt background_material, const std::vector<MaterialBox3D> &boxes, \
   PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCall(PetscKokkosInitializeCheck());
   PetscCheck(background_material >= 0, comm_, PETSC_ERR_ARG_OUTOFRANGE, \
      "background material index %" PetscInt_FMT " is negative", background_material);
   mat_id_d = PetscIntKokkosView("mat_id_d", n_local_elements());
   Kokkos::deep_copy(mat_id_d, background_material);
   PetscCall(paint_boxes_over(boxes, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UnstructuredCG::paint_cell_sets(PetscInt background_material, \
   const std::map<PetscInt, PetscInt> &label_to_material, PetscIntKokkosView &mat_id_d) const
{
   PetscFunctionBeginUser;

   PetscCall(ps_.check_decomposed());
   std::vector<PetscInt> cells(n_local_elements());
   for (PetscInt c = c_start_; c < c_end_; c++) cells[c - c_start_] = c;
   PetscCall(PaintCellSets(dm_, comm_, background_material, label_to_material, cells, mat_id_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
