#include "ubolt/quadrature.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// SN quadrature (Gauss-Legendre for n_angles = 2 or 4)
PetscErrorCode create_sn_quadrature(PetscInt n_angles, PetscScalar *mu, PetscScalar *w, PetscScalar *sum_weights) {

    PetscFunctionBeginUser;

    if (n_angles == 2) {
        mu[0] = -0.5773502692; mu[1] = 0.5773502692;
        w[0] = 1.0;            w[1] = 1.0;
    } else if (n_angles == 4) {
        mu[0] = -0.8611363116; mu[1] = -0.3399810436;
        mu[2] =  0.3399810436; mu[3] =  0.8611363116;
        w[0] = 0.3478548451;   w[1] = 0.6521451549;
        w[2] = 0.6521451549;   w[3] = 0.3478548451;
    } else {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
            "SN quadrature only implemented for 2 or 4 angles, was given %" PetscInt_FMT, n_angles);
    }
    // In 1D we integrate over [-1, 1] so the sum of weights is 2
    *sum_weights = 2.0;

    PetscFunctionReturn(PETSC_SUCCESS);
}
