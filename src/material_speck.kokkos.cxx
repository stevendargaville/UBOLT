#include "ubolt/material_spec.hpp"
#include "petsc_kokkos.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MaterialSpec
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode MaterialSpec::create(PetscInt n_materials, PetscInt n_groups)
{
   PetscFunctionBeginUser;

   PetscCheck(n_materials > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "n_materials must be positive, was given %" PetscInt_FMT, n_materials);
   PetscCheck(n_groups > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "n_groups must be positive, was given %" PetscInt_FMT, n_groups);

   n_materials_ = n_materials;
   n_groups_ = n_groups;

   // Zero initialised, so an untouched material is void
   sigma_t_.assign(n_materials * n_groups, 0.0);
   sigma_s_.assign(n_materials * n_groups * n_groups, 0.0);
   source_.assign(n_materials * n_groups, 0.0);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode MaterialSpec::set_sigma_t(PetscInt mat, PetscInt g, PetscScalar value)
{
   PetscFunctionBeginUser;

   PetscCheck(mat >= 0 && mat < n_materials_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "material %" PetscInt_FMT " out of range, n_materials is %" PetscInt_FMT, mat, n_materials_);
   PetscCheck(g >= 0 && g < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, n_groups_);

   sigma_t_[mat * n_groups_ + g] = value;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode MaterialSpec::set_sigma_s(PetscInt mat, PetscInt g_from, PetscInt g_to, PetscScalar value)
{
   PetscFunctionBeginUser;

   PetscCheck(mat >= 0 && mat < n_materials_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "material %" PetscInt_FMT " out of range, n_materials is %" PetscInt_FMT, mat, n_materials_);
   PetscCheck(g_from >= 0 && g_from < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g_from, n_groups_);
   PetscCheck(g_to >= 0 && g_to < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g_to, n_groups_);

   sigma_s_[(mat * n_groups_ + g_from) * n_groups_ + g_to] = value;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode MaterialSpec::set_source(PetscInt mat, PetscInt g, PetscScalar value)
{
   PetscFunctionBeginUser;

   PetscCheck(mat >= 0 && mat < n_materials_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "material %" PetscInt_FMT " out of range, n_materials is %" PetscInt_FMT, mat, n_materials_);
   PetscCheck(g >= 0 && g < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, n_groups_);

   source_[mat * n_groups_ + g] = value;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UboltFillSource
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Free functions rather than local lambdas: an extended device lambda can't be
// defined inside another lambda

// The smallest and largest material index painted onto the cells - a painted
// index outside the spec's table would silently read garbage, so the fill
// checks the range out loud first
static void MaterialIdRange(const PetscIntKokkosView &mat_id_d, PetscInt local_cells, \
   PetscInt *id_min, PetscInt *id_max)
{
   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt c, PetscInt &lmin, PetscInt &lmax) {

         lmin = Kokkos::min(lmin, mat_id_d(c));
         lmax = Kokkos::max(lmax, mat_id_d(c));
      }, Kokkos::Min<PetscInt>(*id_min), Kokkos::Max<PetscInt>(*id_max));
}

static void FillSourceKernel(PetscScalarKokkosView b_d, PetscScalarKokkosView source_tab_d, \
   PetscIntKokkosView mat_id_d, PetscIntKokkosView is_bc_row_d, PetscInt n_groups, PetscInt g, \
   PetscInt n_angles, PetscInt local_cells, PetscScalar sum_weights)
{
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_cells, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // cell
         const PetscInt i = t.league_rank();
         // The isotropic strength shared out over the angular domain
         const PetscScalar q = source_tab_d(mat_id_d(i) * n_groups + g) / sum_weights;

         // The same per-angle share into every angle of the cell
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt j) {

               const PetscInt r = i * n_angles + j;
               // The rhs on a BC row belongs to the boundary condition
               if (is_bc_row_d(r)) return;
               b_d(r) = q;
         });
   });
}

// The source table is tiny, so upload it whole
static PetscScalarKokkosView UploadSourceTable(const MaterialSpec &mats)
{
   PetscScalarKokkosView source_tab_d("source_tab_d", (PetscInt)mats.source_host().size());
   PetscScalarKokkosViewHostUnmanaged source_tab_h( \
      const_cast<PetscScalar *>(mats.source_host().data()), mats.source_host().size());
   Kokkos::deep_copy(source_tab_d, source_tab_h);
   return source_tab_d;
}

// See the declaration in material_spec.hpp for the BC row contract
PetscErrorCode UboltFillSource(const PhaseSpace &ps, const BoundaryInfo &boundary, \
   const AngularQuadrature &quad, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, \
   PetscInt g, Vec b)
{
   PetscInt local_rows_b = 0;
   PetscInt id_min = 0, id_max = 0;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());

   PetscCheck(g >= 0 && g < mats.n_groups(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, mats.n_groups());
   PetscCheck(quad.n_angles() == ps.n_angles, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck((PetscInt)mat_id_d.extent(0) == ps.local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "material ids cover %" PetscInt_FMT " cells but there are %" PetscInt_FMT " local cells", \
      (PetscInt)mat_id_d.extent(0), ps.local_cells);
   PetscCheck((PetscInt)boundary.is_bc_row_d.extent(0) == ps.local_rows(), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "the boundary info covers %" PetscInt_FMT " rows but there are %" \
      PetscInt_FMT " local rows", (PetscInt)boundary.is_bc_row_d.extent(0), ps.local_rows());
   PetscCall(VecGetLocalSize(b, &local_rows_b));
   PetscCheck(local_rows_b == ps.local_rows(), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "b has %" PetscInt_FMT " local rows but the phase space has %" PetscInt_FMT, \
      local_rows_b, ps.local_rows());

   MaterialIdRange(mat_id_d, ps.local_cells, &id_min, &id_max);
   PetscCheck(id_min >= 0 && id_max < mats.n_materials(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "painted material indices span [%" PetscInt_FMT ", %" PetscInt_FMT "] but the spec has %" \
      PetscInt_FMT " materials", id_min, id_max, mats.n_materials());

   PetscScalarKokkosView source_tab_d = UploadSourceTable(mats);

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));
   FillSourceKernel(b_d, source_tab_d, mat_id_d, boundary.is_bc_row_d, mats.n_groups(), g, \
      ps.n_angles, ps.local_cells, quad.sum_weights());
   PetscCall(VecRestoreKokkosView(b, &b_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UboltFillCellSource
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void FillCellSourceKernel(PetscScalarKokkosView cell_source_d, \
   PetscScalarKokkosView source_tab_d, PetscIntKokkosView mat_id_d, PetscInt n_groups, \
   PetscInt g, PetscInt local_cells)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cells), KOKKOS_LAMBDA(PetscInt c) {

         cell_source_d(c) = source_tab_d(mat_id_d(c) * n_groups + g);
      });
}

// See the declaration in material_spec.hpp for what this writes and what it
// deliberately does not do
PetscErrorCode UboltFillCellSource(const PhaseSpace &ps, const MaterialSpec &mats, \
   const PetscIntKokkosView &mat_id_d, PetscInt g, PetscScalarKokkosView cell_source_d)
{
   PetscInt id_min = 0, id_max = 0;

   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(ps.check_decomposed());

   PetscCheck(g >= 0 && g < mats.n_groups(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, mats.n_groups());
   PetscCheck((PetscInt)mat_id_d.extent(0) == ps.local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "material ids cover %" PetscInt_FMT " cells but there are %" PetscInt_FMT " local cells", \
      (PetscInt)mat_id_d.extent(0), ps.local_cells);
   PetscCheck((PetscInt)cell_source_d.extent(0) == ps.local_cells, PETSC_COMM_SELF, \
      PETSC_ERR_ARG_INCOMP, "the destination holds %" PetscInt_FMT " cells but there are %" \
      PetscInt_FMT " local cells", (PetscInt)cell_source_d.extent(0), ps.local_cells);

   MaterialIdRange(mat_id_d, ps.local_cells, &id_min, &id_max);
   PetscCheck(id_min >= 0 && id_max < mats.n_materials(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "painted material indices span [%" PetscInt_FMT ", %" PetscInt_FMT "] but the spec has %" \
      PetscInt_FMT " materials", id_min, id_max, mats.n_materials());

   PetscScalarKokkosView source_tab_d = UploadSourceTable(mats);
   FillCellSourceKernel(cell_source_d, source_tab_d, mat_id_d, mats.n_groups(), g, ps.local_cells);

   PetscFunctionReturn(PETSC_SUCCESS);
}
