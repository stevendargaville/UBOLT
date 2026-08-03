#ifndef UBOLT_DSA_HPP
#define UBOLT_DSA_HPP

#include "ubolt/types.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include <petscksp.h>

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
// written for 1/2/3D with the unused axes degenerate. 1D and 2D exist today;
// the 3D sibling lands with its own stage
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

private:
   // Everything that does not depend on the dimension: the dof-1 twin DMDA,
   // the diffusion matrix and its KSP, the work vectors and the cached
   // quadrature/boundary views. The per-dimension create() fills h_, n_cells_
   // and the BC families first and then calls this
   PetscErrorCode create_common(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
      const AngularQuadrature &quad);

   // Refill diff_mat_ from the current group's xsections. Called by set_group()
   PetscErrorCode assemble();

   MPI_Comm comm_ = MPI_COMM_NULL;
   PetscInt dim_ = 0;
   PetscInt n_angles_ = 0;
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
   // face means can reach a neighbour rank's cell
   Vec d_global_ = NULL, d_local_ = NULL;

   // Cell counts and spacings per axis; the unused ones are 1 and 0 so the
   // dimension-generic loops degenerate
   PetscInt n_cells_[3] = {1, 1, 1};
   PetscScalar h_[3] = {0.0, 0.0, 0.0};
   // Is the low/high face of each axis a vacuum (Marshak) face? Reflective
   // faces are zero Neumann, which is no contribution at all
   PetscBool vacuum_lo_[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
   PetscBool vacuum_hi_[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};

   PetscScalar2DKokkosView w_d_;
   PetscIntKokkosView is_bc_row_d_;
   // Not owned - the group's xsection slices, as handed over by set_group()
   PetscScalarKokkosView sigma_t_d_;
   PetscScalarKokkosView sigma_s_d_;
};

#endif
