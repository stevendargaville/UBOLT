#ifndef UBOLT_TERMS_CG_HPP
#define UBOLT_TERMS_CG_HPP

#include "ubolt/operator_term.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/material_spec.hpp"
#include "ubolt/multigroup.hpp"
#include "ubolt/unstructured_cg.hpp"
#include <vector>

// The consistent-SUPG pieces of the CG backend (UnstructuredCG): the operator,
// the scatter, and the external source + group transfer. The generic
// RemovalTerm, ScatteringTerm and GroupTransfer do not apply
// on this backend - they are per row, while the SUPG weight couples a vertex to
// its star and depends on the angle - so these are their siblings, and all
// four read the backend's element tables, its tau (UboltSUPGTau) and its one
// shared weighted-load kernel (UnstructuredCG::add_weighted_load), so they
// cannot disagree. Every xsection view here is per LOCAL ELEMENT
// (n_local_elements(), overlap included) - see unstructured_cg.hpp

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Streaming + SUPG streaming + removal + the weak boundary terms, assembled
//
// Everything that sits on the lhs and acts on the group's own unknowns except
// the scatter: tau depends on sigma_t, so the SUPG streaming part is as
// group-dependent as the removal, and splitting them would buy nothing (a
// Galerkin-only streaming matrix is not a usable pmat). One assembled term,
// re-pointed per group like RemovalTerm and refilled by
// TransportOperator::assemble() - the same preallocated matrix every group
//
// has_diagonal/add_diagonal: the diagonal slot's contributions in the same
// order the fill writes them, so the composed diagonal matches the assembled
// one bitwise
class PETSC_VISIBILITY_PUBLIC SUPGTermCG : public OperatorTerm {
public:
   // sigma_t_e is per local element and must outlive the term
   PetscErrorCode create(const PhaseSpace &ps, const UnstructuredCG &disc, const PetscScalarKokkosView &sigma_t_e);
   void set_sigma_t(const PetscScalarKokkosView &sigma_t_e) { sigma_t_e_ = sigma_t_e; }
   // set_sigma_t(xs.sigma_t(g))
   PetscErrorCode set_group(const GroupXSections &xs, PetscInt g) override;
   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(const PetscScalarKokkosView &coo_v_d) const override;
   PetscBool has_diagonal() const override { return PETSC_TRUE; }
   PetscErrorCode add_diagonal(Vec d) const override;
private:
   const UnstructuredCG *disc_ = nullptr;
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalarKokkosView sigma_t_e_;
   CooPattern pattern_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Isotropic scattering with the SUPG weight, matrix-free:
//   y(i, a) -= 1 / m_i sum_e sigma_s,e / W sum_j (M_ij + tau Omega_a . G_ji) phi_j
// phi the nodal scalar flux - owned vertices integrated from x, the overlap
// ones brought in through the backend's vertex twin. Owns two nodal Vecs
// (persistent scratch), hence destroy()
class PETSC_VISIBILITY_PUBLIC ScatteringTermCG : public OperatorTerm {
public:
   // Both views per local element; sigma_t_e is the group's total xsection,
   // which tau is built from
   PetscErrorCode create(const PhaseSpace &ps, const UnstructuredCG &disc, const AngularQuadrature &quad, \
      const PetscScalarKokkosView &sigma_t_e, const PetscScalarKokkosView &sigma_s_e);
   void set_group(const PetscScalarKokkosView &sigma_t_e, const PetscScalarKokkosView &sigma_s_e)
   {
      sigma_t_e_ = sigma_t_e;
      sigma_s_e_ = sigma_s_e;
   }
   // set_group(xs.sigma_t(g), xs.sigma_s(g, g))
   PetscErrorCode set_group(const GroupXSections &xs, PetscInt g) override;
   PetscBool matrix_free() const override { return PETSC_TRUE; }
   PetscErrorCode apply_add(Vec x, Vec y) const override;
   PetscErrorCode destroy();
private:
   const UnstructuredCG *disc_ = nullptr;
   PetscInt n_angles_ = 0;
   PetscScalar sum_weights_ = 0.0;
   PetscScalar2DKokkosView w_d_;
   PetscScalarKokkosView sigma_t_e_, sigma_s_e_;
   PetscScalar2DKokkosView scalar_flux_d_;
   Vec phi_global_ = NULL, phi_local_ = NULL;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// GroupSource with the SUPG weight - GroupTransfer's sibling. Both parts are
// the backend's weighted load, with tau from the TARGET group's sigma_t (the
// source sits in that group's equation, so it takes that group's test
// function):
//   external: b += 1 / m_i sum_e q_e(g) / W sum_j (M_ij + tau_g Omega . G_ji)
//   transfer: b += 1 / m_i sum_e sigma_s,e(g_from -> g_to) / W
//                  sum_j (M_ij + tau_{g_to} Omega . G_ji) phi_j(g_from)
// Each solved group's nodal scalar flux is kept owned + overlap, so the halo
// exchange happens once per group, in set_scalar_flux. Owns Vecs, hence destroy()
class PETSC_VISIBILITY_PUBLIC GroupTransferCG : public GroupSource {
public:
   // xs (sized per local element) and disc must outlive the object; mat_id_d
   // is per local element
   PetscErrorCode create(const PhaseSpace &ps, const UnstructuredCG &disc, const AngularQuadrature &quad, \
      const GroupXSections &xs, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d);
   PetscErrorCode add_external(PetscInt g, Vec b) const override;
   PetscErrorCode set_scalar_flux(PetscInt g, Vec psi_g) override;
   PetscErrorCode add_transfer(PetscInt g_from, PetscInt g_to, Vec b) const override;
   PetscErrorCode destroy();
private:
   const UnstructuredCG *disc_ = nullptr;
   const GroupXSections *xs_ = nullptr;
   PetscInt n_angles_ = 0;
   PetscScalar sum_weights_ = 0.0;
   PetscScalar2DKokkosView w_d_;
   MaterialSourceTable source_;
   // Persistent scratch: the per-element source, the integrated flux and the
   // owned nodal Vec it goes through
   PetscScalarKokkosView q_e_;
   PetscScalar2DKokkosView scalar_flux_d_;
   Vec phi_global_ = NULL;
   // Per group, the nodal scalar flux over the LOCAL vertices
   std::vector<Vec> phi_;
   std::vector<PetscBool> phi_set_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// One-shot GroupTransferCG::add_external: b += the SUPG-weighted external
// source, tau from sigma_t_e (group g's). ADDS, like UboltFillSource: call on
// a zeroed b after UboltFillInflow
PETSC_EXTERN PetscErrorCode UboltFillSourceCG(const UnstructuredCG &disc, const AngularQuadrature &quad, \
   const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, const PetscScalarKokkosView &sigma_t_e, \
   PetscInt g, Vec b);

#endif
