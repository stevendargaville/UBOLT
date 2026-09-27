#ifndef UBOLT_DSA_HPP
#define UBOLT_DSA_HPP

#include "ubolt/types.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"
#include "ubolt/unstructured_dg.hpp"
#include <petscksp.h>
#include <vector>

// Diffusion synthetic acceleration, the second half of the preconditioner in
// Dargaville et al., JCP 518 (2024) 113342, Section 3:
//
//    G^-1_angle = P_angle . D_diff^-1 . R_angle
//
// The composite preconditioner UBOLT already had attacks the hyperbolic part of
// the transport operator (a shell for the removal term, PCAIR for streaming)
// and nothing at all attacks the scattering, so an optically thick problem with
// a scattering ratio near 1 degrades. That is the hole this fills:
//
// - R_angle takes the angular residual to its 0th moment,
//   phi(c) = sum_a w_a r(c, a), with the BC rows masked out of the sum.
// - D_diff is a cell-centred finite difference diffusion operator on the SAME
//   grid, -div(D grad phi) + sigma_a phi with D = 1/(3 sigma_t) and
//   sigma_a = sigma_t - sigma_s(g, g); harmonic-mean face D, a Marshak (Robin)
//   condition on vacuum faces and zero Neumann on reflective ones. SPD, and
//   inverted INEXACTLY - the paper used one BoomerAMG V-cycle, this defaults to
//   KSPPREONLY + PCGAMG (no hypre in the CI images) under the options prefix
//   "dsa_", so -dsa_pc_type hypre and friends select anything at runtime.
//   The Marshak face sits ON the domain boundary, which is where the default
//   ghost-flux vacuum treatment puts the transport boundary too; under the
//   opt-in Dirichlet-cell treatment the transport boundary is the boundary
//   cell's centre instead, and the same face serves both (measured Sep 2026:
//   scaling its coefficient anywhere in 0.25-1.0 moves no count by more than 1
//   on the diffusive recipes under ghost-flux).
// - P_angle broadcasts the scalar correction back isotropically,
//   delta_psi(c, a) = delta_phi(c) / sum_weights, writing zero on the BC rows.
//   That scaling is what makes the two operators consistent: A applied to an
//   isotropic psi = phi / sum_w gives sigma_a phi / sum_w on each angle plus
//   streaming, and R hands back sigma_a phi - exactly D_diff's removal part.
//
// The whole object is a preconditioner, so it is NOT an OperatorTerm; it does
// not contribute to the operator being solved. TransportSolver takes one
// optionally and drives it from a PCShell at composite index 2 - see there.
// Caller-owned: create it, hand the solver a pointer, destroy it afterwards
//
// GEOMETRY, so per-dimension create() overloads, the same split the streaming
// term makes: each one pulls the spacings and the per-axis BC families off the
// concrete backend and its BCSpec and hands them to a dimension-generic
// create_common. Everything after that - the assembly loop included - is
// written for 1/2/3D with the unused axes degenerate
//
// On the DMPlex backend (UnstructuredDG, either order) D_diff is the
// cell-centred finite VOLUME sibling of the same operator: a two-point flux
// across each face, T_f = A_f / (d_c / D_c + d_n / D_n) with d the centroids'
// normal distances to the face (the harmonic face D again - on a uniform box
// d_c = d_n = h / 2 and it IS the structured stencil), a Marshak face
// A_f / (2 + d_c / D_c) on vacuum boundary faces, nothing on reflective ones.
// Volumes vary there, so the matrix is assembled VOLUME-WEIGHTED - V_c times
// the per-unit-volume row the structured backends write - which keeps it SPD,
// and the restricted moment is scaled by V before the inner solve to match
// (on a box that is a constant factor, so the plex twin reproduces the
// structured correction to rounding with an exact inner solve).
//
// At DG1 the diffusion unknown lives in the DG1 space itself: D_diff is the
// modified interior penalty (MIP) form of Wang & Ragusa (NSE 166, 2010) on the
// backend's orthonormal modal basis, one unknown per (cell, basis) node,
//    a(u, v) = sum_c int_c D grad u . grad v + sigma_a u v
//            + sum_{interior f} int_f kappa [[u]][[v]] - {{D du/dn}}[[v]] - [[u]]{{D dv/dn}}
//            + sum_{vacuum f} int_f kappa u v - 1/2 D (du/dn) v - 1/2 u D (dv/dn)
// with kappa = max(C / 2 (D_c / h_c + D_n / h_n), 1/4) on an interior face and
// max(C D_c / h_c, 1/4) on a vacuum one, h = twice the centroid's distance to
// the face (the cell width on a box, a fraction of the height on a simplex,
// which only errs towards more penalty) and C = -dsa_mip_penalty, 4 by
// default. The 1/4 is the upwind jump penalty of an isotropic angular flux, so
// the thick limit is the transport's, and on a vacuum face it is Marshak's
// phi / 2 again. Reflective faces are natural (zero Neumann), nothing. The
// face terms read the backend's exact face matrices (int_f phi_i phi_j, both
// cells' bases) and each side's constant basis gradients. The rows are the
// weak form tested with phi_i - volume-weighted, like the DG0 plex rows, and
// symmetric - so the restriction sums EVERY node's ordinates (row (c, i) of
// the transport is (1 / V) int_c phi_i times the balance) and scales by V, and
// the prolongation corrects every node, the slopes included. The restricted
// residual and the correction are then the same weak moments the transport
// rows are, which is what the cell-average operator this replaced lacked
// (measured Sep 2026: that one took the quad box_diffusive twin 34 -> 34)
//
// VOIDS are masked out, on every backend and order. D = 1/(3 sigma_t) is not
// defined where sigma_t = 0, and a diffusion correction has nothing to say
// about a region with no collisions, so a cell whose group Sigma_t is at or
// below -dsa_void_sigma_t (0 by default: only a true void) is taken out of
// the correction altogether:
// - its diffusion rows are the identity (times the cell volume on the plex,
//   so the plex matrix is still V times the structured one), decoupled from
//   every neighbour in both directions, so the matrix stays SPD;
// - the restriction writes zero there and the prolongation corrects nothing
//   there - the transport rows of a void cell are left entirely to the
//   removal + PCAIR stages, which is where streaming is handled anyway;
// - a face between a real cell and a void is a MARSHAK face for the real
//   cell, exactly the vacuum boundary face (the MIP vacuum form at DG1): the
//   correction treats what crosses into the void as leaked, the
//   zero-incoming-current condition. Not zero Neumann: that would leave a
//   non-absorbing island surrounded by void singular, where Marshak never is.
// The mask is per GROUP (a material can be a void in some groups only), and
// a void's D is stored as ZERO in the staged D vector - no real cell's D can
// be - which is how a neighbour, on this rank or another, is seen to be void.
// With no void cell the arithmetic is the unmasked operator's, bit for bit.
// What this does NOT do is couple the regions a void separates: particles
// streaming across it are the transport stages' business (a void-bridging
// diffusion operator is the known refinement)
//
// Single Mat + single inner KSP, values-refilled per group by set_group().
// Deliberately not folded into TransportSolver::refresh(): the solver has no
// group context, so the driver calls set_group() where it points the terms at
// the group's xsections. Per-group cached Mats/KSPs would be a local change
// here if a sweep ever wanted them
class PETSC_VISIBILITY_PUBLIC DSAPrecon {
public:
   // 1D. quad supplies the weights and their sum, bcs the per-face families
   // (the backend does not keep the BCSpec, so it comes in again here)
   //
   // ps must already carry the decomposition - the discretisation's create()
   // decides it, so build this after the discretisation
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD1D &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);

   // 2D. CAREFUL with the face names: in 2D bottom/top are the Y faces (in 3D
   // they are the Z ones - PETSc's box-mesh convention, see StructuredFD3D)
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD2D &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);

   // 3D. PETSc's box convention, so bottom/top are the Z faces here and the Y
   // ones are front (y-min) and back (y-max) - NOT what 2D calls them
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD3D &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);

   // The DMPlex backend, either DG order. No DMDA twin: the matrix is sized
   // off the backend's owned cells, whose global order is the transport
   // rows' (CheckPlexLayout), and preallocated from its face CSR
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);

   // Point at this group's cross sections and refill the diffusion matrix
   // (values-only after the first call - the sparsity is the grid's). The
   // views are NOT owned, exactly like RemovalTerm::set_sigma_t
   PetscErrorCode set_group(const PetscScalarKokkosView &sigma_t_d, \
      const PetscScalarKokkosView &sigma_s_within_d);

   // y = P D_diff^-1 R x, what the PCShell applies
   PetscErrorCode apply(Vec x, Vec y);

   PetscErrorCode destroy();

   // The inner diffusion solve, for a caller that wants to inspect it
   KSP ksp() const { return ksp_; }

   // How many cells (global) the current group masks out as void - 0 until
   // the first set_group()
   PetscInt n_void_cells() const { return n_void_cells_; }

private:
   // Everything that does not depend on the dimension: the dof-1 twin DMDA,
   // the diffusion matrix and its KSP, the work vectors and the cached
   // quadrature/boundary views. The per-dimension create() fills h_, n_cells_
   // and the BC families first and then calls this
   PetscErrorCode create_common(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
      const AngularQuadrature &quad);

   // The work vectors and the inner KSP, shared by both create_common paths
   PetscErrorCode create_ksp();

   // Refill diff_mat_ from the current group's xsections. Called by set_group()
   PetscErrorCode assemble();
   // Its per-backend halves, after the shared checks: the DMDA star through
   // MatSetValuesStencil, the plex faces through MatSetValuesCOO (the
   // two-point flux at DG0, the interior penalty form at DG1). Host copies of
   // the group's per-cell xsections and of the void mask come in
   PetscErrorCode assemble_structured(const PetscScalar *sigma_t_h, const PetscScalar *sigma_s_h, \
      const PetscInt *void_h);
   PetscErrorCode assemble_plex(const PetscScalar *sigma_t_h, const PetscScalar *sigma_s_h, const PetscInt *void_h);
   PetscErrorCode assemble_plex_dg1(const PetscScalar *sigma_t_h, const PetscScalar *sigma_s_h, \
      const PetscInt *void_h);

   MPI_Comm comm_ = MPI_COMM_NULL;
   PetscInt dim_ = 0;
   PetscInt n_angles_ = 0;
   // Spatial dofs per cell: 1 except DG1, whose diffusion unknown has one per
   // (cell, basis) node too
   PetscInt n_basis_ = 1;
   PetscInt local_cells_ = 0;
   PetscScalar sum_weights_ = 0.0;

   // A dof-1 twin of the backend's DMDA: same grid, same decomposition, one
   // unknown per cell
   DM da_ = NULL;
   Mat diff_mat_ = NULL;
   KSP ksp_ = NULL;
   // Persistent work - never allocate inside an apply
   Vec rhs_ = NULL, sol_ = NULL;
   // The diffusion coefficient, staged globally then ghosted so the harmonic
   // face means can reach a neighbour rank's cell (d_local_ on the DMDA only)
   Vec d_global_ = NULL, d_local_ = NULL;

   // Cell counts and spacings per axis; the unused ones are 1 and 0 so the
   // dimension-generic loops degenerate
   PetscInt n_cells_[3] = {1, 1, 1};
   PetscScalar h_[3] = {0.0, 0.0, 0.0};
   // Is the low/high face of each axis a vacuum (Marshak) face? Reflective
   // faces are zero Neumann, which is no contribution at all
   PetscBool vacuum_lo_[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
   PetscBool vacuum_hi_[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};

   // Is there a vacuum (Marshak) face anywhere? Without one the operator is
   // pure Neumann and only absorption keeps it nonsingular
   PetscBool any_vacuum_ = PETSC_FALSE;

   // ~~~~~~~~~~ DMPlex only (plex_ true, da_ NULL) ~~~~~~~~~~
   PetscBool plex_ = PETSC_FALSE;
   // Per owned cell, the COO entries run [cell_entry_offset_[c],
   // cell_entry_offset_[c + 1]): one per interior face in cone order, then
   // the diagonal. At DG1 each of the cell's n_basis rows in turn, and n_basis
   // entries (the column cell's basis j) where DG0 has one
   std::vector<PetscInt> cell_entry_offset_;
   // Per (cell, face) slot, the backend's CSR: the neighbour's slot in d_nb_
   // (-1 on a boundary face), A_f, the two normal distances, and whether a
   // boundary face is vacuum
   std::vector<PetscInt> cell_face_offset_;
   std::vector<PetscInt> face_nb_;
   std::vector<PetscReal> face_area_;
   std::vector<PetscReal> face_distance_;
   std::vector<PetscBool> face_vacuum_;
   std::vector<PetscReal> volume_;
   std::vector<PetscScalar> coo_v_;
   // Cell volumes as a Vec, one per (cell, basis) node, to weight the
   // restricted moment
   Vec volume_vec_ = NULL;
   // DG1 only, the backend's host geometry (see UnstructuredDG): the face
   // matrices int_f phi_i^c phi_j^{c | n} / (V_c A_f), the unit outward
   // normal per face slot, and the basis gradients of this cell and of the
   // neighbour across each interior face slot. And the penalty constant C
   std::vector<PetscScalar> face_own_;
   std::vector<PetscScalar> face_up_;
   std::vector<PetscReal> face_normal_;
   std::vector<PetscScalar> basis_grad_;
   std::vector<PetscScalar> face_nb_grad_;
   PetscReal mip_penalty_ = 4.0;
   // The neighbours' D: d_global_ scattered onto one seq entry per interior
   // face slot (owned neighbours included - one path for both)
   VecScatter nb_scatter_ = NULL;
   Vec d_nb_ = NULL;
   PetscScalar2DKokkosView w_d_;
   PetscIntKokkosView is_bc_row_d_;
   // Not owned - the group's xsection slices, as handed over by set_group()
   PetscScalarKokkosView sigma_t_d_;
   PetscScalarKokkosView sigma_s_d_;

   // The void mask (see the header): Sigma_t <= void_sigma_t_ is a void.
   // One flag per local cell, refilled per group by assemble(), read by the
   // restriction and prolongation kernels
   PetscReal void_sigma_t_ = 0.0;
   PetscIntKokkosView void_cell_d_;
   PetscInt n_void_cells_ = 0;
};

#endif
