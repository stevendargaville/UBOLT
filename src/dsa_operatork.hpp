#ifndef UBOLT_DSA_OPERATORK_HPP
#define UBOLT_DSA_OPERATORK_HPP

// Internal to the DSA translation units (never included from include/ubolt/):
// the per-backend diffusion operator behind DSAPrecon, and the helpers the
// backends share. DSAPrecon (src/dsak.kokkos.cxx) owns the policy - the void
// census, the bridge decision, D per xsection unit, the mask - and the
// restriction, prolongation and inner KSP; a DSAOperator owns the matrix and
// the geometry it is built from. See docs/dsa.md for the derivations

#include "ubolt/dsa.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/discretisation.hpp"
#include "petsc_kokkos.hpp"
#include <vector>

// What the policy hands a backend for one group. A "unit" is what the group's
// xsections are per: a local cell, or a local ELEMENT on CG
struct DSAGroup {
   const PetscScalar *sigma_t_h = nullptr, *sigma_s_h = nullptr;
   // The same sigma_t on the device (CG's SUPG-D kernel reads it)
   PetscScalarKokkosView sigma_t_d;
   const PetscInt *is_void = nullptr;
   // D per unit: 1/(3 sigma_t), the free-flight D in a bridged void, 0 in an
   // unbridged one (see StageD for how a neighbour reads it)
   const PetscScalar *d = nullptr;
   // Per local cell, a masked void (can_mask() backends only)
   const PetscInt *masked = nullptr;
   PetscInt n_void = 0;
   PetscBool bridged = PETSC_FALSE;
};

class DSAOperator {
public:
   virtual ~DSAOperator() = default;

   // How many xsection entries a group has, and which of them are this rank's
   // to count as voids (nullptr: all of them). "cells" / "elements" for errors
   virtual PetscInt n_units() const = 0;
   virtual const PetscInt *owned_units() const { return nullptr; }
   virtual const char *unit_name() const { return "cells"; }
   // Whether an unbridged void can be masked. CG cannot: the vertex a void
   // shares with the material is the material's too, so it keeps the SUPG
   // tensor instead - and nothing else to fall back on, so its bridge needs
   // only a face to end a flight and its singular guard runs first
   virtual PetscBool can_mask() const { return PETSC_TRUE; }
   // This rank's share of the voids' volume V and flight-ending area S (faces
   // onto a non-void unit or a vacuum boundary; a reflective face mirrors the
   // flight on). Collective where a neighbour's flag crosses ranks
   virtual PetscErrorCode void_volume_surface(const PetscInt *is_void, PetscReal vs[2]) = 0;
   // Refill mat's values for the group
   virtual PetscErrorCode assemble(const DSAGroup &g) = 0;

   // Frees the backend's own handles, then these
   PetscErrorCode destroy();

   // Created by the backend: the diffusion matrix, its work vectors, and the
   // weight the restricted moment is scaled by (NULL on the structured
   // backends, whose rows are per unit volume)
   Mat mat = NULL;
   Vec rhs = NULL, sol = NULL, weight = NULL;
   // A vacuum (Marshak) face anywhere? Without one only absorption keeps the
   // operator nonsingular
   PetscBool any_vacuum = PETSC_FALSE;

protected:
   virtual PetscErrorCode destroy_own() { return PETSC_SUCCESS; }
};

// One builder per backend, each in its own TU. *op is set before anything
// can fail, so the caller's destroy() cleans up a half-built one
PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD1D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op);
PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD2D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op);
PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const StructuredFD3D &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op);
PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredDG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op);
PetscErrorCode DSAOperatorCreate(MPI_Comm comm, const PhaseSpace &ps, const UnstructuredCG &disc, \
   const AngularQuadrature &quad, const BCSpec &bcs, DSAOperator **op);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Shared helpers
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A host copy of a device view, on PETSc's execution space
template <class View>
static inline auto DSAHostCopy(const View &d)
{
   auto h = Kokkos::create_mirror_view(Kokkos::HostSpace(), d);
   Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), h, d);
   PetscGetKokkosExecutionSpace().fence();
   return h;
}

// The quadrature weights on the host, once per create
static inline void DSAHostWeights(const AngularQuadrature &quad, std::vector<PetscReal> &w)
{
   auto w_h = DSAHostCopy(quad.w_d());
   w.resize(quad.n_angles());
   for (PetscInt a = 0; a < quad.n_angles(); a++) w[a] = PetscRealPart(w_h(a, 0));
}

// The half-range current of an isotropic unit scalar flux through a face:
// sum over the ordinates leaving through it of w_a (Omega_a . n), over
// sum_weights - 1/4 in the continuum. cos_h[a] = Omega_a . n
static inline PetscReal HalfRangeCurrent(const std::vector<PetscReal> &w_h, PetscReal sum_weights, \
   const std::vector<PetscReal> &cos_h)
{
   PetscReal sum = 0.0;
   for (size_t a = 0; a < w_h.size(); a++) {
      if (cos_h[a] > 0.0) sum += w_h[a] * cos_h[a];
   }
   return sum / sum_weights;
}

// A face D with the upwind numerical diffusion m blended in: the p-norm of
// (D, m), plain addition at p = 1
static inline PetscScalar BlendD(PetscScalar d, PetscReal m, PetscReal p)
{
   if (p == 1.0) return d + m;
   return PetscPowReal(PetscPowReal(PetscRealPart(d), p) + PetscPowReal(m, p), 1.0 / p);
}

// -dsa_consistent_d and its power. p < 1 would add MORE than the sum
static inline PetscErrorCode ConsistentDOptions(MPI_Comm comm, PetscBool *on, PetscReal *power)
{
   PetscFunctionBeginUser;

   PetscCall(PetscOptionsGetBool(NULL, "dsa_", "-consistent_d", on, NULL));
   PetscCall(PetscOptionsGetReal(NULL, "dsa_", "-consistent_d_power", power, NULL));
   PetscCheck(*power >= 1.0, comm, PETSC_ERR_ARG_OUTOFRANGE, \
      "-dsa_consistent_d_power must be at least 1, got %g", (double)*power);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// The staged D: one value per cell, the only thing a neighbour (on this rank
// or another, through the ghost exchange) sees of it
//     0  a masked void - no unknown to couple to, a Marshak face for the neighbour
//    <0  a bridged void, |v| its D (DG1 weights the faces touching one)
//    >0  a real cell, its D
// No real cell's D can be 0 or negative, so the encoding is unambiguous
static inline PetscScalar StageD(PetscScalar d, PetscBool bridged_void) { return bridged_void ? -d : d; }
static inline PetscBool StagedMasked(PetscScalar v) { return (PetscBool)(v == 0.0); }
static inline PetscBool StagedBridged(PetscScalar v) { return (PetscBool)(PetscRealPart(v) < 0.0); }
static inline PetscScalar StagedD(PetscScalar v) { return PetscAbsScalar(v); }

// The chord pass stages a void FLAG through the same vectors (a second vector
// would cost a second exchange). Not a D - it is restaged as one before use
static inline PetscScalar StageVoidFlag(PetscInt is_void) { return is_void ? 1.0 : 0.0; }
static inline PetscBool StagedVoidFlag(PetscScalar v) { return (PetscBool)(PetscRealPart(v) != 0.0); }

#endif
