#ifndef UBOLT_TERMS_HPP
#define UBOLT_TERMS_HPP

#include "ubolt/operator_term.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"
#include "ubolt/unstructured_dg.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming: mu dpsi/dx, upwinded onto the discretisation's stencil
//
// Takes the 1D backend by concrete type, not the Discretisation base: it needs
// the geometry (dx) and it owns the upwind slot convention, both of which are
// per-dimension. A 2D mesh wants a 2D sibling, not this term
class PETSC_VISIBILITY_PUBLIC StreamingTerm : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const StructuredFD1D &disc, const SNQuadrature &quad);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

   PetscBool has_diagonal() const override { return PETSC_TRUE; }
   PetscErrorCode add_diagonal(Vec d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalar dx_ = 0.0;
   PetscScalarKokkosView mu_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming in 2D: mu dpsi/dx + eta dpsi/dy, each upwinded onto its own axis
//
// The sibling of StreamingTerm, not a generalisation of it: the slot convention
// is per-dimension, and this is the term that owns it. It writes StructuredFD2D's
// three slots positionally - upwind-x, upwind-y, diagonal - which is the same
// contract the 1D pair have, one axis wider
class PETSC_VISIBILITY_PUBLIC StreamingTerm2D : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const StructuredFD2D &disc, const SNQuadrature2D &quad);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

   PetscBool has_diagonal() const override { return PETSC_TRUE; }
   PetscErrorCode add_diagonal(Vec d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalar dx_ = 0.0;
   PetscScalar dy_ = 0.0;
   PetscScalarKokkosView mu_d_;
   PetscScalarKokkosView eta_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming in 3D: mu dpsi/dx + eta dpsi/dy + xi dpsi/dz, each upwinded onto
// its own axis
//
// The next sibling along: it writes StructuredFD3D's four slots positionally -
// upwind-x, upwind-y, upwind-z, diagonal - the same contract the 1D and 2D
// pairs have, one axis wider again
class PETSC_VISIBILITY_PUBLIC StreamingTerm3D : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const StructuredFD3D &disc, const SNQuadrature3D &quad);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

   PetscBool has_diagonal() const override { return PETSC_TRUE; }
   PetscErrorCode add_diagonal(Vec d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalar dx_ = 0.0;
   PetscScalar dy_ = 0.0;
   PetscScalar dz_ = 0.0;
   PetscScalarKokkosView mu_d_;
   PetscScalarKokkosView eta_d_;
   PetscScalarKokkosView xi_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming on an unstructured mesh: Omega . grad psi as the DG0 upwind face
// flux, sum_f (Omega . nA_f / V_c) psi_upwind(f)
//
// The sibling of the structured streaming terms for UnstructuredDG, and the
// owner of its slot convention: n_faces + 1 slots per row, one per face in cone
// order, then the diagonal. Per face s = Omega_a . nA_f; an OUTFLOW face
// (s > 0) adds s / V_c to the diagonal, an inflow face adds it (negative) to
// that face's slot - which the backend has already pointed at the upwind
// neighbour, or nulled on a boundary face, so the fill never branches on
// anything but the sign of s. Same arithmetic as the structured stencil on a
// quad/hex box, to rounding
//
// The quadrature is NOT a create argument: the ordinates the term needs are
// disc.omega_d(), fixed when the backend classified the rows, so the two can
// never disagree
class PETSC_VISIBILITY_PUBLIC StreamingTermDG0 : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const UnstructuredDG &disc);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

   PetscBool has_diagonal() const override { return PETSC_TRUE; }
   PetscErrorCode add_diagonal(Vec d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalarKokkosView omega_d_;
   PetscIntKokkosView cell_face_offset_d_;
   PetscScalarKokkosView face_nA_d_;
   PetscScalarKokkosView inv_volume_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Streaming on an unstructured mesh at DG1: the upwind DG weak form of
// Omega . grad psi against the backend's orthonormal linear basis, per unit
// volume - for row (c, i, a), with s = Omega_a . nA_f
//   sum_f s * (1 / (V A_f)) int_f phi_i psi^up  -  (Omega_a . grad phi_i) psi_(c, 0)
// psi^up this cell's own trace on an outflow face (s > 0: the face_own_d matrix
// into the row's own block), the upwind trace otherwise (face_up_d into that
// face's block - the backend pointed it at the neighbour, at the mirrored angle
// on a reflective face, or nulled it on a vacuum face, whose inflow is in the
// rhs). The volume term lands on own slot 0. Same fill discipline as DG0: the
// only branch is on the sign of s
//
// The sibling of StreamingTermDG0 for UnstructuredDG at order 1, and the owner
// of its slot convention: n_basis slots per face in cone order, then this
// cell's n_basis, the diagonal being own slot i. Takes no quadrature, for the
// same reason as DG0
class PETSC_VISIBILITY_PUBLIC StreamingTermDG1 : public OperatorTerm {
public:
   PetscErrorCode create(const PhaseSpace &ps, const UnstructuredDG &disc);

   PetscBool assembled() const override { return PETSC_TRUE; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

   PetscBool has_diagonal() const override { return PETSC_TRUE; }
   PetscErrorCode add_diagonal(Vec d) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt n_basis_ = 0;
   PetscInt local_rows_ = 0;
   PetscScalarKokkosView omega_d_;
   PetscIntKokkosView cell_face_offset_d_;
   PetscScalarKokkosView face_nA_d_;
   PetscScalarKokkosView face_own_d_;
   PetscScalarKokkosView face_up_d_;
   PetscScalarKokkosView basis_grad_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Removal: sigma_t psi, a pure diagonal contribution
//
// Nothing here is dimension-specific - it writes the diagonal slot and skips
// the Dirichlet rows - so it takes any Discretisation. At DG1 too: the basis
// is orthonormal on the cell, so the mass matrix is the identity and sigma_t
// (constant on the cell) stays on the diagonal of every basis row
//
// The one term that can go either way. Assembled (the default) it is a
// per-group refill of the shared matrix; matrix-free it leaves the assembled
// matrix carrying STREAMING ONLY, which does not depend on the group, so a
// multigroup sweep assembles once and PCAIR sets up once for the whole sweep.
// Same operator either way, to rounding - see -matfree_removal in
// tests/transportk and its -check_matfree equivalence check
class PETSC_VISIBILITY_PUBLIC RemovalTerm : public OperatorTerm {
public:
   // sigma_t_d is indexed by local cell and must outlive the term
   PetscErrorCode create(const PhaseSpace &ps, const Discretisation &disc, const PetscScalarKokkosView &sigma_t_d);

   // Point the term at a different xsection. The group sweep calls this and
   // then TransportOperator::assemble() to refill the values, which is why
   // nothing here caches anything derived from sigma_t. Matrix-free the apply
   // reads it straight through, so there is nothing to refill at all
   void set_sigma_t(const PetscScalarKokkosView &sigma_t_d) { sigma_t_d_ = sigma_t_d; }

   // Assemble this term (the default) or apply it matrix-free. Set it before
   // the first TransportOperator::assemble() - the operator partitions its
   // terms there, not in add_term, so the order of the two calls does not
   // matter, only that this one comes first
   void set_matrix_free(PetscBool matrix_free) { matrix_free_ = matrix_free; }

   PetscBool assembled() const override { return (PetscBool)!matrix_free_; }
   PetscErrorCode assemble_add(PetscScalarKokkosView &coo_v_d) const override;

   PetscBool matrix_free() const override { return matrix_free_; }
   PetscErrorCode apply_add(Vec x, Vec y) const override;

   // Always: whichever half of the term is live, the removal is on the
   // operator's diagonal and the Jacobi stage of the preconditioner wants it
   PetscBool has_diagonal() const override { return PETSC_TRUE; }
   PetscErrorCode add_diagonal(Vec d) const override;

private:
   PetscInt rows_per_cell_ = 0;
   PetscInt local_rows_ = 0;
   PetscBool matrix_free_ = PETSC_FALSE;
   PetscScalarKokkosView sigma_t_d_;
   CooPattern pattern_;
   BoundaryInfo boundary_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Isotropic scattering, applied matrix-free: integrate the angular flux to get
// the scalar flux, scale by sigma_s, and subtract it from every angle (it is on
// the lhs). Never assembled - that is the whole point of the MatShell
class PETSC_VISIBILITY_PUBLIC ScatteringTerm : public OperatorTerm {
public:
   // sigma_s_d is indexed by local cell and must outlive the term. In a
   // multigroup sweep this is the within-group block sigma_s(g -> g); the
   // off-diagonal blocks go to the rhs through GroupTransfer
   // Any discretisation and any quadrature: the scatter is cell-local, so all
   // it wants from the discretisation is the Dirichlet mask, and all it wants
   // from the quadrature is the weights, never the directions
   PetscErrorCode create(const PhaseSpace &ps, const Discretisation &disc, \
      const AngularQuadrature &quad, const PetscScalarKokkosView &sigma_s_d);

   // Point the term at a different xsection - see RemovalTerm::set_sigma_t
   void set_sigma_s(const PetscScalarKokkosView &sigma_s_d) { sigma_s_d_ = sigma_s_d; }

   PetscBool matrix_free() const override { return PETSC_TRUE; }
   PetscErrorCode apply_add(Vec x, Vec y) const override;

private:
   PetscInt n_angles_ = 0;
   PetscInt n_basis_ = 1;
   PetscScalar sum_weights_ = 0.0;
   PetscScalarKokkosView sigma_s_d_;
   PetscScalar2DKokkosView w_d_;
   BoundaryInfo boundary_;
   // Persistent scratch - never allocate device memory inside an apply
   PetscScalar2DKokkosView scalar_flux_d_;
};

#endif
