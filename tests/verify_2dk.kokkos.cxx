// Phase 4 verification for the 2D discretisation
//
// Two checks, both of which fail the process rather than printing something a
// human has to read:
//
//  1. Pure-streaming closed form. psi = x + y is linear, and the first order
//     upwind difference of a linear function is exact, so the DISCRETE operator
//     has the same solution as the PDE: mu dpsi/dx + eta dpsi/dy = mu + eta,
//     with psi = x + y prescribed on the inflow boundaries. That pins the
//     stencil coefficients, both upwind signs, dx against dy and the Dirichlet
//     rows all at once, to rounding.
//  2. The whole shell operator - including the matrix-free scatter - against a
//     reference matrix built here entry by entry, on a small grid.
//
// There is no hand-layout twin to compare against in 2D (a 2D DMDA's global
// numbering is not the natural ordering), which is why these two exist
//
// Currently run with:
// make build_tests && ./verify_2dk

// ubolt.hpp pulls in petscvec_kokkos.hpp which must come before any other
// PETSc header in a C++ file (see docs/dev/kokkos.md)
#include "ubolt/ubolt.hpp"
#include <petscksp.h>
#include "pflare.h"
#include <vector>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// What this direction's row at this node should be, and for a reflective row
// which angle it couples to
//
// Written out again here rather than read back from the discretisation's BC
// mask, reflect slots or reflection maps on purpose: a check that asked the
// code under test which rows it thought were boundary rows - or which angle
// mirrors which - would agree with it by construction. The partner is found by
// this file's own search over the cosines
enum class RefKind { INTERIOR, DIRICHLET, REFLECT };

static RefKind ClassifyNode(const SNQuadrature2D &quad, PetscInt a, PetscInt i, PetscInt j, \
   PetscInt n_cells_x, PetscInt n_cells_y, BCType left, BCType right, BCType bottom, BCType top, \
   PetscInt *partner)
{
   const PetscScalar *mu = quad.mu_host();
   const PetscScalar *eta = quad.eta_host();

   const bool in_x = (mu[a] > 0.0 && i == 0) || (mu[a] < 0.0 && i == n_cells_x - 1);
   const bool in_y = (eta[a] > 0.0 && j == 0) || (eta[a] < 0.0 && j == n_cells_y - 1);

   *partner = -1;
   if (!in_x && !in_y) return RefKind::INTERIOR;

   // Vacuum wins at mixed corners; only a direction whose every incoming face
   // is reflective reflects, flipping the cosine on each incoming axis
   const BCType x_bc = (mu[a] > 0.0) ? left : right;
   const BCType y_bc = (eta[a] > 0.0) ? bottom : top;
   if ((in_x && x_bc == BCType::VACUUM) || (in_y && y_bc == BCType::VACUUM)) return RefKind::DIRICHLET;

   const PetscScalar mu_want = in_x ? -mu[a] : mu[a];
   const PetscScalar eta_want = in_y ? -eta[a] : eta[a];
   for (PetscInt b = 0; b < quad.n_angles(); b++) {
      if (mu[b] == mu_want && eta[b] == eta_want) { *partner = b; break; }
   }
   return RefKind::REFLECT;
}

// Is this direction entering the box through a face this node sits on - the
// all-vacuum special case the closed-form check uses
static PetscBool IsInflowNode(const SNQuadrature2D &quad, PetscInt a, PetscInt i, PetscInt j, \
   PetscInt n_cells_x, PetscInt n_cells_y)
{
   PetscInt partner = -1;
   return (PetscBool)(ClassifyNode(quad, a, i, j, n_cells_x, n_cells_y, BCType::VACUUM, \
      BCType::VACUUM, BCType::VACUUM, BCType::VACUUM, &partner) == RefKind::DIRICHLET);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 1: pure streaming against the closed form
//
// Both halves of it are worth having. The residual is the sharp one - it is
// exact arithmetic on the assembled coefficients, no solver tolerance in the
// way - and the solve then says the system those coefficients make really does
// have that solution and is nonsingular
static PetscErrorCode CheckStreamingClosedForm(PetscInt n_cells_x, PetscInt n_cells_y, PetscInt n_angles, \
   PetscReal length_x, PetscReal length_y, PetscBool *ok)
{
   PhaseSpace ps;
   SNQuadrature2D quad;
   StructuredFD2D disc;
   StreamingTerm2D streaming;
   TransportOperator op;
   TransportSolver solver;
   Vec psi_exact, b, work;
   PetscReal residual_norm = 0.0, solution_error = 0.0;
   const PetscReal residual_tol = 1e-12;
   const PetscReal solution_tol = 1e-9;

   PetscFunctionBeginUser;

   PetscCall(ps.create(PETSC_COMM_WORLD, n_cells_x * n_cells_y, n_angles));
   PetscCall(quad.create(n_angles));
   PetscCall(disc.create(PETSC_COMM_WORLD, ps, n_cells_x, n_cells_y, length_x, length_y, quad));

   // Streaming alone: no removal, no scatter
   PetscCall(streaming.create(ps, disc, quad));
   PetscCall(op.create(PETSC_COMM_WORLD, ps, disc));
   PetscCall(op.add_term(&streaming));
   PetscCall(op.assemble());

   PetscCall(MatCreateVecs(op.assembled_mat(), &psi_exact, &b));
   PetscCall(VecDuplicate(psi_exact, &work));

   // ~~~~~~~~~~
   // The closed form and the rhs it satisfies
   // ~~~~~~~~~~
   {
      PetscScalar *psi_a, *b_a;
      const PetscScalar *mu = quad.mu_host();
      const PetscScalar *eta = quad.eta_host();
      const PetscScalar dx = disc.dx(), dy = disc.dy();
      const PetscInt local_cells_x = disc.local_cells_x();

      PetscCall(VecGetArray(psi_exact, &psi_a));
      PetscCall(VecGetArray(b, &b_a));

      for (PetscInt c = 0; c < ps.local_cells; c++) {

         // Local cell index runs over this rank's patch lexicographically
         const PetscInt i = disc.cell_start_x() + c % local_cells_x;
         const PetscInt j = disc.cell_start_y() + c / local_cells_x;
         const PetscScalar x = i * dx;
         const PetscScalar y = j * dy;

         for (PetscInt a = 0; a < n_angles; a++) {
            const PetscInt r = c * n_angles + a;
            psi_a[r] = x + y;
            // Inflow rows carry the boundary value, the rest the source that
            // the linear solution streams: mu d(x+y)/dx + eta d(x+y)/dy
            b_a[r] = IsInflowNode(quad, a, i, j, n_cells_x, n_cells_y) ? (x + y) : (mu[a] + eta[a]);
         }
      }

      PetscCall(VecRestoreArray(psi_exact, &psi_a));
      PetscCall(VecRestoreArray(b, &b_a));
   }

   // ~~~~~~~~~~
   // Does the operator send the closed form to that rhs
   // ~~~~~~~~~~
   PetscCall(MatMult(op.mat(), psi_exact, work));
   PetscCall(VecAXPY(work, -1.0, b));
   PetscCall(VecNorm(work, NORM_INFINITY, &residual_norm));

   // ~~~~~~~~~~
   // And does solving it give the closed form back
   // ~~~~~~~~~~
   PetscCall(solver.create(PETSC_COMM_WORLD, op, op.assembled_mat()));
   // After create(), so this wins over anything -ksp_rtol on the command line
   // asked for: this check wants the solver's own error out of the way
   PetscCall(KSPSetTolerances(solver.ksp(), 1e-13, 1e-13, PETSC_CURRENT, 1000));
   PetscCall(VecZeroEntries(work));
   PetscCall(solver.solve(b, work));
   PetscCall(VecAXPY(work, -1.0, psi_exact));
   PetscCall(VecNorm(work, NORM_INFINITY, &solution_error));

   if (solver.converged_reason() <= 0) {
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "  streaming closed form: KSP did not converge: %s\n", \
         KSPConvergedReasons[solver.converged_reason()]));
      *ok = PETSC_FALSE;
   }
   if (residual_norm > residual_tol || solution_error > solution_tol) *ok = PETSC_FALSE;

   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  pure streaming vs psi = x + y on %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT \
      " angles: residual %.3e (tol %.0e), solution error %.3e (tol %.0e)\n", \
      n_cells_x, n_cells_y, n_angles, (double)residual_norm, (double)residual_tol, \
      (double)solution_error, (double)solution_tol));

   PetscCall(solver.destroy());
   PetscCall(op.destroy());
   PetscCall(disc.destroy());
   PetscCall(VecDestroy(&psi_exact));
   PetscCall(VecDestroy(&b));
   PetscCall(VecDestroy(&work));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check 2: the shell operator against a reference matrix built here
//
// MatComputeOperator applies the shell to every unit basis vector, so this sees
// the assembled terms AND the matrix-free scatter, exactly as the KSP does
//
// Serial only. The reference is written in the natural (j * n_cells_x + i)
// ordering, which is what a 2D DMDA's global numbering is only on one rank -
// that difference is precisely what StructuredFD2D's layout assert exists for,
// and it is checked there rather than papered over here
//
// A Dirichlet row is the identity and NOTHING else - the scatter is a term like
// any other and owes the mask the same contract. That is worth checking rather
// than assuming: this check is what caught the matrix-free scatter writing to
// those rows in the first place (fixed Aug 2026, see TODO.md). A reflective
// row is the identity plus the -1 on its mirrored angle and nothing else, so
// the same check pins the reflect slots, the partner angles and the vacuum-wins
// corner rule per entry
static PetscErrorCode CheckOperatorAgainstReference(PetscInt n_cells_x, PetscInt n_cells_y, PetscInt n_angles, \
   PetscReal length_x, PetscReal length_y, PetscScalar sigma_t, PetscScalar sigma_s, \
   BCType left, BCType right, BCType bottom, BCType top, const char *bc_desc, PetscBool *ok)
{
   PhaseSpace ps;
   SNQuadrature2D quad;
   StructuredFD2D disc;
   StreamingTerm2D streaming;
   RemovalTerm removal;
   ScatteringTerm scattering;
   TransportOperator op;
   Mat explicit_mat = NULL;
   PetscMPIInt size;
   PetscReal max_diff = 0.0;
   const PetscReal tol = 1e-12;

   PetscFunctionBeginUser;

   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
   PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, \
      "the reference matrix is written in the natural ordering, which a 2D DMDA only uses on one rank");

   BCSpec bcs;
   bcs.set(StructuredFD2D::FACE_LEFT, left);
   bcs.set(StructuredFD2D::FACE_RIGHT, right);
   bcs.set(StructuredFD2D::FACE_BOTTOM, bottom);
   bcs.set(StructuredFD2D::FACE_TOP, top);

   PetscCall(ps.create(PETSC_COMM_WORLD, n_cells_x * n_cells_y, n_angles));
   PetscCall(quad.create(n_angles));
   PetscCall(disc.create(PETSC_COMM_WORLD, ps, n_cells_x, n_cells_y, length_x, length_y, quad, bcs));

   PetscScalarKokkosView sigma_t_d("sigma_t_d", ps.local_cells);
   PetscScalarKokkosView sigma_s_d("sigma_s_d", ps.local_cells);
   Kokkos::deep_copy(sigma_t_d, sigma_t);
   Kokkos::deep_copy(sigma_s_d, sigma_s);

   PetscCall(streaming.create(ps, disc, quad));
   PetscCall(removal.create(ps, disc, sigma_t_d));
   PetscCall(scattering.create(ps, disc, quad, sigma_s_d));

   PetscCall(op.create(PETSC_COMM_WORLD, ps, disc));
   PetscCall(op.add_term(&streaming));
   PetscCall(op.add_term(&removal));
   PetscCall(op.add_term(&scattering));
   PetscCall(op.assemble());

   PetscCall(MatComputeOperator(op.mat(), MATAIJ, &explicit_mat));

   {
      const PetscScalar *mu = quad.mu_host();
      const PetscScalar *eta = quad.eta_host();
      const PetscScalar dx = disc.dx(), dy = disc.dy();
      // Independently of the quadrature object: a 2D level-symmetric set covers
      // the whole 4 pi sphere and shares it equally between its ordinates
      const PetscScalar sum_weights = 4.0 * PETSC_PI;
      const PetscScalar w = sum_weights / n_angles;
      const PetscScalar scat = sigma_s / sum_weights;
      const PetscInt N = n_cells_x * n_cells_y * n_angles;

      std::vector<PetscInt> cols(N);
      std::vector<PetscScalar> got(N), ref(N);
      for (PetscInt k = 0; k < N; k++) cols[k] = k;

      for (PetscInt j = 0; j < n_cells_y; j++) {
         for (PetscInt i = 0; i < n_cells_x; i++) {
            for (PetscInt a = 0; a < n_angles; a++) {

               const PetscInt row = (j * n_cells_x + i) * n_angles + a;
               ref.assign(N, 0.0);

               PetscInt partner = -1;
               const RefKind kind = ClassifyNode(quad, a, i, j, n_cells_x, n_cells_y, \
                  left, right, bottom, top, &partner);

               if (kind == RefKind::DIRICHLET) {

                  // An inflow row is the identity, and nothing gets to add to
                  // it - not the assembled terms and not the scatter
                  ref[row] = 1.0;

               } else if (kind == RefKind::REFLECT) {

                  // A reflective row is the reflection condition and nothing
                  // else: psi(a) - psi(mirrored a) = 0 in the same cell
                  PetscCheck(partner >= 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
                     "no mirrored ordinate found for angle %" PetscInt_FMT, a);
                  ref[row] = 1.0;
                  ref[(j * n_cells_x + i) * n_angles + partner] = -1.0;

               } else {

                  ref[row] += PetscAbsScalar(mu[a]) / dx + PetscAbsScalar(eta[a]) / dy + sigma_t;
                  if (mu[a] != 0.0) {
                     const PetscInt upwind_i = (mu[a] > 0.0) ? i - 1 : i + 1;
                     ref[(j * n_cells_x + upwind_i) * n_angles + a] += -PetscAbsScalar(mu[a]) / dx;
                  }
                  if (eta[a] != 0.0) {
                     const PetscInt upwind_j = (eta[a] > 0.0) ? j - 1 : j + 1;
                     ref[(upwind_j * n_cells_x + i) * n_angles + a] += -PetscAbsScalar(eta[a]) / dy;
                  }

                  // The scatter is on the lhs, so it comes off this angle for
                  // every angle it scatters from - including its own, which is
                  // why it reaches the diagonal too. Note it integrates over
                  // EVERY angle of the node, Dirichlet ones included: the
                  // prescribed incoming flux is a real part of the flux there
                  for (PetscInt a2 = 0; a2 < n_angles; a2++) {
                     ref[(j * n_cells_x + i) * n_angles + a2] += -scat * w;
                  }
               }

               PetscCall(MatGetValues(explicit_mat, 1, &row, N, cols.data(), got.data()));
               for (PetscInt k = 0; k < N; k++) {
                  const PetscReal d = PetscAbsScalar(got[k] - ref[k]);
                  if (d > max_diff) max_diff = d;
               }
            }
         }
      }
   }

   if (max_diff > tol) *ok = PETSC_FALSE;
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
      "  shell operator vs reference matrix on %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT \
      " angles, %s bcs: max entry difference %.3e (tol %.0e)\n", n_cells_x, n_cells_y, n_angles, \
      bc_desc, (double)max_diff, (double)tol));

   PetscCall(MatDestroy(&explicit_mat));
   PetscCall(op.destroy());
   PetscCall(disc.destroy());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **args) {

   PetscBool ok = PETSC_TRUE;

   PetscFunctionBeginUser;

   PetscCall(PetscInitialize(&argc, &args, (char*)0, NULL));
   // Register the pflare types
   PCRegister_PFLARE();

   // Which checks to run - the reference matrix one is serial only, so the
   // parallel recipe asks for the closed form on its own
   PetscBool skip_reference = PETSC_FALSE;
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-skip_reference", &skip_reference, NULL));

   // No block scope here, unlike the solve drivers: every device view lives
   // and dies inside the Check* functions, well before PetscFinalize
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D verification\n"));

   // S2 and S4, and a non-square grid so an x/y mix-up cannot hide
   PetscCall(CheckStreamingClosedForm(16, 12, 4, 1.0, 2.0, &ok));
   PetscCall(CheckStreamingClosedForm(16, 12, 12, 1.0, 2.0, &ok));

   if (!skip_reference) {
      const BCType V = BCType::VACUUM, R = BCType::REFLECT;
      // All vacuum (the original check), all reflect, and a mixed config
      // with reflect on left + bottom so both flip maps and the vacuum-wins
      // corners are all in the checked matrix
      PetscCall(CheckOperatorAgainstReference(4, 3, 4, 1.0, 2.0, 1.5, 0.7, V, V, V, V, "vacuum", &ok));
      PetscCall(CheckOperatorAgainstReference(4, 3, 12, 1.0, 2.0, 1.5, 0.7, V, V, V, V, "vacuum", &ok));
      PetscCall(CheckOperatorAgainstReference(4, 3, 4, 1.0, 2.0, 1.5, 0.7, R, R, R, R, "reflect", &ok));
      PetscCall(CheckOperatorAgainstReference(4, 3, 12, 1.0, 2.0, 1.5, 0.7, R, R, R, R, "reflect", &ok));
      PetscCall(CheckOperatorAgainstReference(4, 3, 4, 1.0, 2.0, 1.5, 0.7, R, V, R, V, "mixed", &ok));
      PetscCall(CheckOperatorAgainstReference(4, 3, 12, 1.0, 2.0, 1.5, 0.7, R, V, R, V, "mixed", &ok));
   }

   if (!ok) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stderr, "2D verification FAILED\n"));

   PetscCall(PetscFinalize());
   return ok ? 0 : 1;
}
