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
// UboltCheckMaterialIds
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Free functions rather than local lambdas: an extended device lambda can't be
// defined inside another lambda

static void MaterialIdRange(const PetscIntKokkosView &mat_id_d, PetscInt n, PetscInt *id_min, PetscInt *id_max)
{
   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(PetscInt c, PetscInt &lmin, PetscInt &lmax) {

         lmin = Kokkos::min(lmin, mat_id_d(c));
         lmax = Kokkos::max(lmax, mat_id_d(c));
      }, Kokkos::Min<PetscInt>(*id_min), Kokkos::Max<PetscInt>(*id_max));
}

PetscErrorCode UboltCheckMaterialIds(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d)
{
   PetscInt id_min = 0, id_max = 0;

   PetscFunctionBeginUser;

   MaterialIdRange(mat_id_d, (PetscInt)mat_id_d.extent(0), &id_min, &id_max);
   PetscCheck(id_min >= 0 && id_max < mats.n_materials(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "painted material indices span [%" PetscInt_FMT ", %" PetscInt_FMT "] but the spec has %" \
      PetscInt_FMT " materials", id_min, id_max, mats.n_materials());

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MaterialSourceTable
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode MaterialSourceTable::create(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d)
{
   PetscFunctionBeginUser;

   // We allocate device memory below, and PETSc brings Kokkos up lazily
   PetscCall(PetscKokkosInitializeCheck());

   PetscCall(UboltCheckMaterialIds(mats, mat_id_d));

   n_groups_ = mats.n_groups();
   mat_id_d_ = mat_id_d;
   // The table is tiny, so upload it whole
   source_tab_d_ = PetscScalarKokkosView("source_tab_d", (PetscInt)mats.source_host().size());
   PetscScalarKokkosViewHostUnmanaged source_tab_h( \
      const_cast<PetscScalar *>(mats.source_host().data()), mats.source_host().size());
   Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), source_tab_d_, source_tab_h);
   // The copy is asynchronous and the host table is the caller's
   PetscGetKokkosExecutionSpace().fence();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void FillEntriesKernel(PetscScalarKokkosView out, PetscScalarKokkosView source_tab_d, \
   PetscIntKokkosView mat_id_d, PetscInt n_groups, PetscInt g, PetscInt n)
{
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(PetscInt e) {

         out(e) = source_tab_d(mat_id_d(e) * n_groups + g);
      });
}

PetscErrorCode MaterialSourceTable::fill_entries(PetscInt g, const PetscScalarKokkosView &out) const
{
   PetscFunctionBeginUser;

   PetscCheck(g >= 0 && g < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, n_groups_);
   PetscCheck(out.extent(0) == mat_id_d_.extent(0), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the destination holds %" PetscInt_FMT " entries but the material ids %" PetscInt_FMT, \
      (PetscInt)out.extent(0), (PetscInt)mat_id_d_.extent(0));

   FillEntriesKernel(out, source_tab_d_, mat_id_d_, n_groups_, g, (PetscInt)out.extent(0));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void AddIsotropicKernel(PetscScalarKokkosView b_d, PetscScalarKokkosView source_tab_d, \
   PetscIntKokkosView mat_id_d, PetscIntKokkosView is_bc_row_d, PetscInt n_groups, PetscInt g, \
   PetscInt n_angles, PetscInt n_basis, PetscInt local_cells, PetscScalar sum_weights)
{
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_cells, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // cell
         const PetscInt i = t.league_rank();
         // The isotropic strength shared out over the angular domain
         const PetscScalar q = source_tab_d(mat_id_d(i) * n_groups + g) / sum_weights;

         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, n_angles), [&](const PetscInt j) {

               // Basis 0 of the cell
               const PetscInt r = i * n_basis * n_angles + j;
               if (is_bc_row_d(r)) return;
               // ADD, not assign - see the declaration
               b_d(r) += q;
         });
   });
}

PetscErrorCode MaterialSourceTable::add_isotropic(PetscInt g, PetscInt n_angles, PetscInt n_basis, \
   PetscScalar sum_weights, const PetscIntKokkosView &is_bc_row_d, Vec b) const
{
   const PetscInt local_cells = (PetscInt)mat_id_d_.extent(0);
   PetscInt local_rows_b = 0;

   PetscFunctionBeginUser;

   PetscCheck(g >= 0 && g < n_groups_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "group %" PetscInt_FMT " out of range, n_groups is %" PetscInt_FMT, g, n_groups_);
   PetscCall(VecGetLocalSize(b, &local_rows_b));
   PetscCheck(local_rows_b == local_cells * n_basis * n_angles && \
      (PetscInt)is_bc_row_d.extent(0) == local_rows_b, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "b has %" PetscInt_FMT " local rows and the BC row mask %" PetscInt_FMT ", but the material ids " \
      "cover %" PetscInt_FMT " cells of %" PetscInt_FMT " rows", local_rows_b, \
      (PetscInt)is_bc_row_d.extent(0), local_cells, n_basis * n_angles);

   PetscScalarKokkosView b_d;
   PetscCall(VecGetKokkosView(b, &b_d));
   AddIsotropicKernel(b_d, source_tab_d_, mat_id_d_, is_bc_row_d, n_groups_, g, n_angles, n_basis, \
      local_cells, sum_weights);
   PetscCall(VecRestoreKokkosView(b, &b_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The one-shot fills
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UboltFillSource(const PhaseSpace &ps, const BoundaryInfo &boundary, \
   const AngularQuadrature &quad, const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, \
   PetscInt g, Vec b)
{
   MaterialSourceTable table;

   PetscFunctionBeginUser;

   PetscCall(ps.check_decomposed());
   PetscCheck(quad.n_angles() == ps.n_angles, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "the quadrature has %" PetscInt_FMT " angles but the phase space has %" PetscInt_FMT, \
      quad.n_angles(), ps.n_angles);
   PetscCheck((PetscInt)mat_id_d.extent(0) == ps.local_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, \
      "material ids cover %" PetscInt_FMT " cells but there are %" PetscInt_FMT " local cells", \
      (PetscInt)mat_id_d.extent(0), ps.local_cells);

   PetscCall(table.create(mats, mat_id_d));
   PetscCall(table.add_isotropic(g, ps.n_angles, ps.n_basis, quad.sum_weights(), boundary.is_bc_row_d, b));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode UboltFillCellSource(const MaterialSpec &mats, const PetscIntKokkosView &mat_id_d, PetscInt g, \
   PetscScalarKokkosView cell_source_d)
{
   MaterialSourceTable table;

   PetscFunctionBeginUser;

   PetscCall(table.create(mats, mat_id_d));
   PetscCall(table.fill_entries(g, cell_source_d));

   PetscFunctionReturn(PETSC_SUCCESS);
}
