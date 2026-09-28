#ifndef UBOLT_DSA_HPP
#define UBOLT_DSA_HPP

#include "ubolt/types.hpp"
#include <petscksp.h>
#include <vector>

struct PhaseSpace;
class AngularQuadrature;
class BCSpec;
class Discretisation;
class StructuredFD1D;
class StructuredFD2D;
class StructuredFD3D;
class UnstructuredDG;
class UnstructuredCG;
class DSAOperator;

// Diffusion synthetic acceleration, the scattering half of the preconditioner
// (Dargaville et al., JCP 518 (2024) 113342, Section 3):
//
//    G^-1 = P . D_diff^-1 . R
//
// The removal shell and PCAIR attack the hyperbolic part of the operator and
// nothing attacks the scattering, so a thick problem with a scattering ratio
// near 1 degrades without this. It preconditions rather than contributes, so
// it is NOT an OperatorTerm: TransportSolver drives it from a PCShell at
// composite index 2. Caller-owned, built after the discretisation, and
// refilled per group by set_group() (one Mat + one inner KSP; the solver has
// no group context, so the driver calls it where it points the terms at the
// group's xsections).
//
// - R: the 0th angular moment of the residual per (cell, basis) node, BC rows
//   left out. P: the correction broadcast isotropically, / sum_weights (which
//   makes R A P's removal part exactly D_diff's), zero on the BC rows.
// - D_diff, -div(D grad phi) + sigma_a phi with D = 1/(3 sigma_t) and
//   sigma_a = sigma_t - sigma_s(g, g), per backend:
//   structured: the cell-centred star on a dof-1 DMDA twin, harmonic face D,
//     Marshak (Robin) on vacuum faces, zero Neumann on reflective ones;
//   DG0 plex: the two-point-flux sibling, VOLUME-weighted (SPD where volumes
//     vary; R is scaled by V to match; on a box it is V x the structured one);
//   DG1 plex: the MIP interior penalty form on the backend's modal basis, one
//     unknown per node, penalty C = -dsa_mip_penalty (4), floor 1/4;
//   CG-SUPG: continuous P1/Q1 on the transport's vertices, which is exactly
//     m R A P of the SUPG operator on an isotropic flux (R scaled by m_i).
// - Consistent D (structured, DG0): every face D blends in the upwind scheme's
//   numerical diffusion m h (m the quadrature's half-range current) as
//   (D^p + (m h)^p)^(1/p), p = 1.5. Not at DG1 (the penalty floor is the same
//   1/4) or on CG (SUPG's thick limit is already the physical D).
// - Voids (group Sigma_t <= -dsa_void_sigma_t, 0): BRIDGED by default - kept
//   with no absorption and the free-flight D = 1/(3 (sigma_t + 1/L)), L = 4 V / S
//   the voids' aggregate mean chord - falling back to the mask when everything
//   is void or the bridged operator would be singular. MASKED
//   (-dsa_void_bridge 0): identity rows, nothing restricted or corrected there,
//   a Marshak face for the neighbour. CG never masks: an unbridged void keeps
//   the SUPG operator's own tensor. Per group; no void = the plain operator,
//   bit for bit.
//
// Options (all under dsa_): -dsa_consistent_d[_power], -dsa_mip_penalty,
// -dsa_void_sigma_t, -dsa_void_bridge, -dsa_void_d (a fixed void D), and the
// inner solve's -dsa_ksp_* / -dsa_pc_* (default PREONLY + one GAMG V-cycle).
// Derivations: docs/dsa.md. Measurements: TODO.md's DSA notes.
class PETSC_VISIBILITY_PUBLIC DSAPrecon {
public:
   DSAPrecon() = default;
   DSAPrecon(const DSAPrecon &) = delete;
   DSAPrecon &operator=(const DSAPrecon &) = delete;
   DSAPrecon &operator=(DSAPrecon &&) = default;

   // One overload per backend; ps must already carry the decomposition. bcs
   // gives the face families (the backends do not keep it). The structured
   // ones need quad to BE the dimension's SN set (its cosines give the
   // consistent D's half-range current). CAREFUL with the face names: 2D
   // bottom/top are Y, 3D bottom/top are Z (PETSc's box convention)
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD1D &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD2D &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD3D &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);
   // set_group() then takes the per-ELEMENT xsections the CG terms do
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredCG &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);
   // Any backend: dispatches to the overloads above, errors on an unknown one
   PetscErrorCode create(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
      const AngularQuadrature &quad, const BCSpec &bcs);

   // Point at this group's cross sections and refill the diffusion matrix
   // (values only). The views are NOT owned, like RemovalTerm::set_sigma_t
   PetscErrorCode set_group(const PetscScalarKokkosView &sigma_t_d, \
      const PetscScalarKokkosView &sigma_s_within_d);

   // y = P D_diff^-1 R x, what the PCShell applies
   PetscErrorCode apply(Vec x, Vec y);

   // Frees everything and resets the object, so it can be created again
   PetscErrorCode destroy();

   // The inner diffusion solve, for a caller that wants to inspect it
   KSP ksp() const { return ksp_; }

   // The current group's void cells (global; 0 before the first set_group()),
   // whether they are bridged, and their mean chord L = 4 V / S (0 unless)
   PetscInt n_void_cells() const { return n_void_cells_; }
   PetscBool bridged() const { return bridged_; }
   PetscReal void_chord() const { return void_chord_; }

private:
   // What every overload shares before the backend builds its operator, and
   // the inner KSP after
   PetscErrorCode init_common(MPI_Comm comm, const PhaseSpace &ps, const Discretisation &disc, \
      const AngularQuadrature &quad);
   PetscErrorCode create_ksp();
   // The void policy for the current group, then the backend's refill
   PetscErrorCode assemble();

   MPI_Comm comm_ = MPI_COMM_NULL;
   PetscInt n_angles_ = 0;
   PetscInt n_basis_ = 1;
   PetscInt local_cells_ = 0;
   PetscInt n_cells_global_ = 0;
   PetscScalar sum_weights_ = 0.0;
   PetscScalar2DKokkosView w_d_;
   PetscIntKokkosView is_bc_row_d_;

   // The backend's diffusion operator (owned; see src/dsa_operatork.hpp)
   DSAOperator *op_ = nullptr;
   KSP ksp_ = NULL;

   // Not owned - the group's xsection slices, as handed over by set_group()
   PetscScalarKokkosView sigma_t_d_, sigma_s_d_;

   // -dsa_void_sigma_t, -dsa_void_bridge, -dsa_void_d, and this group's state:
   // per xsection unit the void flag and the D staged for the backend, per
   // local cell the mask the restriction/prolongation read (host copy kept
   // so an unchanged mask is not re-uploaded)
   PetscReal void_sigma_t_ = 0.0;
   PetscBool void_bridge_ = PETSC_TRUE;
   PetscReal void_d_ = 0.0;
   PetscBool bridged_ = PETSC_FALSE;
   PetscReal void_chord_ = 0.0;
   PetscInt n_void_cells_ = 0;
   std::vector<PetscInt> is_void_;
   std::vector<PetscScalar> d_unit_;
   std::vector<PetscInt> void_cell_h_;
   PetscIntKokkosView void_cell_d_;
};

#endif
