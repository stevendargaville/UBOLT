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
// - D is DISCRETISATION-CONSISTENT by default (-dsa_consistent_d, since Sep
//   2026): the first-order upwind streaming adds a numerical diffusion to the
//   transport, and in thick cells it dwarfs the physical 1/(3 sigma_t). For a
//   linear flux in an infinite pure scatterer the upwind face current is
//   exactly -(D + m h) grad phi, with m = sum_{Omega.n > 0} w (Omega . n) /
//   sum_weights the quadrature's half-range current (1/4 in the continuum)
//   and h the cell width across the face - the upwind jump term m [[phi]]
//   written as a diffusion. So every face coefficient, the Marshak faces'
//   included, blends that in:
//      D_face = (D^p + (m h)^p)^(1/p),   p = -dsa_consistent_d_power, 1.5
//   p = 1 is the exact sum, which is right in both limits but over-diffuses
//   intermediate cells (tau = sigma_t h ~ 0.3-2): a 1D Fourier analysis of
//   step-differenced source iteration + this DSA puts the optimal D below
//   D + m h there, and p = 1.5 tracks that optimum to within a few hundredths
//   of spectral radius at every tau and scattering ratio. On the plex (DG0) m
//   is per face, off the backend's own ordinates and the face normal, and h
//   is the two centroids' distance through the face; DG1 ignores the option,
//   its interior penalty already floors at the same 1/4.
//   -dsa_consistent_d 0 is the physical D the correction had before. The
//   measured effect is in TODO.md (the Phase 4 DSA notes): the diffusive
//   recipes 11 -> 5, the crooked pipe 28 -> 8 and the literature crooked-pipe
//   sets 51-106 -> 17-21, and the eps sweep flat at 10-11 down to eps 1e-4.
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
// structured correction to rounding with an exact inner solve). The
// consistent-D blend above is applied to the face's harmonic D over the
// centroid-to-centroid distance, so on a box it is the structured one too.
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
// VOIDS. D = 1/(3 sigma_t) is not defined where sigma_t = 0. A cell whose
// group Sigma_t is at or below -dsa_void_sigma_t (0 by default: only a true
// void) is a void, and the correction either BRIDGES the voids (the default,
// -dsa_void_bridge, since Sep 2026) or MASKS them (-dsa_void_bridge 0).
//
// Bridging keeps the void cells in the diffusion operator as ordinary cells
// with no absorption and a FREE-FLIGHT diffusion coefficient. D = 1/(3 sigma_t)
// is the random walk's <mu^2> times the flight length 1/sigma_t; in a void the
// flight is ended by the void's walls instead, and the mean flight through a
// region is its mean chord, Cauchy's L = 4 V / S (the classical Behrens void
// correction for streaming cavities in diffusion theory). So a void cell takes
// D = 1 / (3 (sigma_t + 1 / L)) - Wigner's rational combination of the
// collision and the wall, exactly L / 3 in a true void - with V the voids'
// total volume and S the area of their faces that end a flight: faces onto a
// non-void cell and vacuum boundary faces (a reflective face mirrors the
// flight on, so it is not in S). -dsa_void_d fixes D instead. Everything else
// is the unvoided operator's: harmonic face D, the consistent-D blend, the
// restriction and prolongation over the void's nodes. So the correction
// couples the regions a void separates, which the mask cannot: measured Sep
// 2026, the void recipes 6 / 8 / 6 / 8 (slab gap, box channel, cube duct,
// plex channel, masked) -> 6 / 6 / 5 / 6, the channel beating the old tiny-
// Sigma_t workaround's 7, and the no-void count 5 on the duct. L is an
// aggregate over every void in the group (not per connected void), which the
// counts are forgiving of - a factor of ~3 either way of L / 3 costs at most
// one iteration on the scans behind those numbers.
// At DG1 the MIP form above takes, on every face touching a void, the
// weighted interior penalty of Ern, Stephansen & Zunino (IMA J Numer Anal 29,
// 2009): both averages weighted so that {{D grad u}} uses the face's HARMONIC
// D, D_h = 2 D_c D_n / (D_c + D_n), and the penalty kappa computed off D_h.
// With the void D ~100x the material's, the plain arithmetic MIP penalty
// C / 2 (D_c / h_c + D_n / h_n) welds the material's surface to the void
// (and took the channel 10 -> 13); D_h is the DG1 sibling of the harmonic
// face D the DG0 operators already use, and it is still SPD. Measured: with
// an exact inner solve the DG1 channel goes 10 (mask) -> 7, and with the
// default single GAMG V-cycle it ties the mask at 10.
// Bridging needs something to bridge and must stay nonsingular, so a group
// that is void EVERYWHERE, or one with no vacuum face and no absorption
// anywhere (the bridged operator is then pure Neumann), falls back to the
// mask. The void flags cross ranks through the staged D (a negative D marks
// a bridged void on the plex, where DG1 needs to see one).
//
// Masking takes the void cells out of the correction altogether:
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
//   Under the consistent D it is the BLENDED vacuum face, (D_c^p + (m h)^p)^(1/p)
//   on the real cell's side, and the void's D = 0 flag is tested before any
//   blend, so no m h coupling ever reaches a masked cell.
// A masked void's D is stored as ZERO in the staged D vector - no real
// cell's D can be - which is how a neighbour, on this rank or another, is
// seen to be void. What the mask does NOT do is couple the regions a void
// separates: particles streaming across it are left to the transport stages.
//
// Either way it is per GROUP (a material can be a void in some groups only),
// and with no void cell the arithmetic is the plain operator's, bit for bit.
//
// Single Mat + single inner KSP, values-refilled per group by set_group().
// Deliberately not folded into TransportSolver::refresh(): the solver has no
// group context, so the driver calls set_group() where it points the terms at
// the group's xsections. Per-group cached Mats/KSPs would be a local change
// here if a sweep ever wanted them
class PETSC_VISIBILITY_PUBLIC DSAPrecon {
public:
   // 1D. quad supplies the weights and their sum, bcs the per-face families
   // (the backend does not keep the BCSpec, so it comes in again here). On
   // the structured backends quad must BE the dimension's SN set
   // (SNQuadrature here, SNQuadrature2D/3D below) - its cosines give the
   // consistent D's half-range current - or create() errors
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

   // Whether the current group's voids are bridged (PETSC_FALSE when they are
   // masked, or there are none), and the mean chord length L = 4 V / S their
   // free-flight D is L / 3 of (0 unless bridged) - see the header
   PetscBool bridged() const { return bridged_; }
   PetscReal void_chord() const { return void_chord_; }

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
   // The voids' mean chord 4 V / S for bridging (see the header), over every
   // rank; 0 when no face ends a flight
   PetscErrorCode void_chord_length(const PetscInt *is_void, PetscReal *chord);
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

   // -dsa_consistent_d: blend the upwind scheme's numerical diffusion into D
   // (see the header comment), with -dsa_consistent_d_power. half_range_ is
   // the quadrature's half-range current per axis on the structured backends,
   // face_half_range_ per face slot on the plex (DG0 only)
   PetscBool consistent_d_ = PETSC_TRUE;
   PetscReal consistent_power_ = 1.5;
   PetscReal half_range_[3] = {0.0, 0.0, 0.0};
   std::vector<PetscReal> face_half_range_;

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

   // The voids (see the header): Sigma_t <= void_sigma_t_ is a void. The
   // mask is one flag per local cell, refilled per group by assemble(), read
   // by the restriction and prolongation kernels - set only on a MASKED void
   PetscReal void_sigma_t_ = 0.0;
   // -dsa_void_bridge and -dsa_void_d; whether this group's voids ARE
   // bridged (void_bridge_ and a void to bridge and a nonsingular result),
   // their mean chord L, and per local cell the void flag and the D staged
   // for the assembly (0 on a masked void)
   PetscBool void_bridge_ = PETSC_TRUE;
   PetscReal void_d_ = 0.0;
   PetscBool bridged_ = PETSC_FALSE;
   PetscReal void_chord_ = 0.0;
   std::vector<PetscInt> is_void_;
   std::vector<PetscScalar> d_cell_;
   PetscIntKokkosView void_cell_d_;
   PetscInt n_void_cells_ = 0;
};

#endif
