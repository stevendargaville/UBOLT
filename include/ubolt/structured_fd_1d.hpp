#ifndef UBOLT_STRUCTURED_FD_1D_HPP
#define UBOLT_STRUCTURED_FD_1D_HPP

#include "ubolt/types.hpp"
#include "ubolt/discretisation.hpp"
#include "ubolt/phase_space.hpp"
#include "ubolt/sn_quadrature.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/material_regions.hpp"
#include <vector>

// Uniform-grid upwinded finite difference discretisation of a 1D slab
//
// Two COO entries per row: the upwind neighbour, then the diagonal. Under
// VacuumTreatment::GHOST_FLUX (the default) there are no BC rows: an inflow
// row keeps its diagonal, and its upwind slot is nulled with the
// |mu| / dx-weighted inflow moved to BoundaryInfo::ghost_inflow_d on a vacuum
// face, or pointed at the mirrored angle in the same cell on a reflective one.
// Under DIRICHLET_CELL an inflow row is a BC row: a vacuum one keeps only the
// diagonal and its rhs carries the incoming flux (the face's angle-integrated
// inflow over the ordinates; a point face takes no window), a reflective one
// repurposes the upwind slot for the -1 coupling to the mirrored angle
class PETSC_VISIBILITY_PUBLIC StructuredFD1D : public Discretisation {
public:
   // The boundary label ids this backend hands to the BCSpec - PETSc's
   // DMPlexCreateBoxMesh "Face Sets" ids for a 1D interval. Any other id in a
   // BCSpec is an error
   static constexpr PetscInt FACE_LEFT = 1;
   static constexpr PetscInt FACE_RIGHT = 2;

   // The quadrature decides which neighbour is upwind for each angle, and on a
   // reflective face which angle mirrors which. A default bcs is vacuum
   // (prescribed inflow) on both faces
   //
   // The layout comes from a 1D DMDA, which also decides the parallel
   // decomposition: `ps` is filled in (ps.local_cells) rather than read, so
   // this must be created before anything sized off the phase space
   PetscErrorCode create(MPI_Comm comm, PhaseSpace &ps, PetscReal length, const SNQuadrature &quad, \
      const BCSpec &bcs = BCSpec());

   // Uniform grid so every cell has the same width
   PetscScalar dx() const { return dx_; }

   // This rank's patch of the slab: the 1D DMDA numbers in the natural order,
   // so local cell index c - what the per-cell xsection views are indexed by -
   // is just c = i - cell_start_x
   PetscInt cell_start_x() const { return cell_start_x_; }

   // Paint material indices onto this rank's cells: the background everywhere,
   // then the intervals in order with the later ones winning, membership
   // decided by the cell CENTRE. Allocates mat_id_d sized local_cells - the
   // per-cell material index view MaterialSpec's fill steps consume. The
   // painting is the geometric half of the material split (see MaterialSpec),
   // which is why it lives on the concrete backend
   PetscErrorCode paint_intervals(PetscInt background_material, \
      const std::vector<MaterialInterval1D> &intervals, PetscIntKokkosView &mat_id_d) const;

private:
   PetscScalar dx_ = 0.0;
   PetscInt cell_start_x_ = 0;
};

#endif
