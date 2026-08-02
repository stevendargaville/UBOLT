#ifndef UBOLT_PROBLEM_SPEC_HPP
#define UBOLT_PROBLEM_SPEC_HPP

#include "ubolt/types.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/material_spec.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include <string>
#include <vector>

// A complete problem definition read from a JSON file: everything physical -
// dimension, mesh, angles, materials, painted regions, boundary conditions,
// inflow and the flux output - so a solve driver needs no problem options of
// its own. What stays OFF the file is everything about how the problem is
// solved: PETSc options (-ksp_*, -pc_*) and driver strategy knobs remain
// command-line, so the same file can be solved many ways
//
// The materials entry is either a path to a standalone materials file
// (resolved relative to the problem file's directory) or the same schema
// inline: n_groups, then per material dense ids 0..n-1, Sigma_t[g] and the
// Sigma_s[from][to] transfer matrix. Materials files authored by other
// transport codes load directly: fission fields (Sigma_f, Nu, Chi) and
// bookkeeping (schema_version, length_unit, outside_id) are accepted and
// ignored, and UBOLT adds one optional field, Source[g], the isotropic
// external strength MaterialSpec carries (absent = 0, void). The materials
// schema is deliberately TOLERANT of unknown keys - external files carry
// fields UBOLT has no use for - while the problem file is STRICT: an unknown
// key there is a typo and an error. "_comment" is ignored everywhere (JSON
// has no comments)
//
// See docs/problem_files.md for the full schema of both files
class PETSC_VISIBILITY_PUBLIC ProblemSpec {
public:
   // Rank 0 reads and resolves the file(s) and broadcasts one JSON string;
   // every rank parses the same bytes, so the spec is identical everywhere.
   // No PETSc handles are owned, so there is no destroy()
   PetscErrorCode create(MPI_Comm comm, const char *problem_path);

   PetscInt dimension = 0;
   // Only the x entries mean anything when dimension == 1
   PetscInt n_cells_x = 0, n_cells_y = 0;
   PetscReal length_x = 0.0, length_y = 0.0;
   PetscInt n_angles = 0;
   // From the materials schema - the phase space takes it from here
   PetscInt n_groups = 0;

   // The materials table, loaded and ready for GroupXSections /
   // UboltFillSource; names[i] is material i's name ("" if none), so regions
   // can refer to materials by name as well as id
   MaterialSpec materials;
   std::vector<std::string> material_names;

   // The geometry half: the background everywhere, then the paint list in
   // order with later entries winning - exactly paint_intervals/paint_boxes
   // semantics. One of the two lists is filled, by dimension
   PetscInt background_material = 0;
   std::vector<MaterialInterval1D> intervals;
   std::vector<MaterialBox2D> boxes;

   // Keyed with the dimension's FACE_* ids, ready for the backend's create.
   // n_reflect_faces lets a driver ask "all faces reflective?" (the
   // infinite-medium check) without re-walking the spec
   BCSpec bcs;
   PetscInt n_reflect_faces = 0;

   // The per-angle incoming flux on the Dirichlet rows of the rhs
   PetscReal inflow = 1.0;

   // Scalar flux output path (.vts/.vtr), empty = no output
   std::string flux_vtk;
};

#endif
