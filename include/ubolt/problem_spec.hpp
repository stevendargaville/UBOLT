#ifndef UBOLT_PROBLEM_SPEC_HPP
#define UBOLT_PROBLEM_SPEC_HPP

#include "ubolt/types.hpp"
#include "ubolt/bc_spec.hpp"
#include "ubolt/material_spec.hpp"
#include "ubolt/structured_fd_1d.hpp"
#include "ubolt/structured_fd_2d.hpp"
#include "ubolt/structured_fd_3d.hpp"
#include <map>
#include <string>
#include <vector>

// A complete problem definition read from a JSON file: everything physical -
// dimension, mesh, SN order, materials, painted regions, boundary conditions
// (with per-face inflow) and the flux output - so a solve driver needs no
// problem options of its own. What stays OFF the file is everything about how
// the problem is solved: PETSc options (-ksp_*, -pc_*) and driver strategy knobs remain
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
   // mesh.type: PETSC_FALSE ("structured", the default) is the DMDA finite
   // difference backends; PETSC_TRUE ("unstructured") is the DG0 backend on a
   // DMPlex, 2D and 3D only. These are plain fields on purpose - this header
   // does not include the unstructured backend, the driver builds its mesh
   // description from them
   PetscBool mesh_unstructured = PETSC_FALSE;
   // Unstructured box only: triangles/tets instead of quads/hexes
   PetscBool mesh_simplex = PETSC_FALSE;
   // Unstructured only: a mesh file PETSc reads (Gmsh .msh, ...), resolved
   // relative to the problem file's directory exactly as a materials path is.
   // Empty = a box built in code, described by n_cells_* / length_* below;
   // non-empty = the file decides the mesh and n_cells_* / length_* stay 0
   std::string mesh_file;
   // Only the first `dimension` axes mean anything
   PetscInt n_cells_x = 0, n_cells_y = 0, n_cells_z = 0;
   PetscReal length_x = 0.0, length_y = 0.0, length_z = 0.0;
   // The SN order, NOT the ordinate count: how many ordinates an order is, is
   // the quadrature's business and differs by dimension (S4 is 4 ordinates in
   // 1D, 12 in 2D, 24 in 3D). So is WHICH orders exist - any even one in 1D,
   // the even 2 to 18 in 2D and 3D - so only positive-and-even is checked here
   PetscInt sn_order = 0;
   // From the materials schema - the phase space takes it from here
   PetscInt n_groups = 0;

   // The materials table, loaded and ready for GroupXSections /
   // UboltFillSource; names[i] is material i's name ("" if none), so regions
   // can refer to materials by name as well as id
   MaterialSpec materials;
   std::vector<std::string> material_names;

   // The geometry half: the background everywhere, then the paint list in
   // order with later entries winning - exactly paint_intervals/paint_boxes
   // semantics. One of the three lists is filled, by dimension. An
   // unstructured mesh adds a layer between the two: background, then
   // cell_sets, then the paint list
   PetscInt background_material = 0;
   // Unstructured only (regions.cell_sets): "Cell Sets" label value ->
   // material index. Cells whose label value is not a key keep the background
   std::map<PetscInt, PetscInt> cell_sets;
   std::vector<MaterialInterval1D> intervals;
   std::vector<MaterialBox2D> boxes;
   std::vector<MaterialBox3D> boxes_3d;

   // Keyed with the dimension's FACE_* ids, ready for the backend's create.
   // On an unstructured mesh the face names map onto the same ids (they are
   // PETSc's box "Face Sets" values) and integer keys are "Face Sets" values
   // stored as given, so an id is an id whichever way the file spelled it.
   // n_reflect_faces lets a driver ask "all faces reflective?" (the
   // infinite-medium check) without re-walking the spec - it counts reflective
   // label ids, so on a file mesh it is only a face count if the ids are
   BCSpec bcs;
   PetscInt n_reflect_faces = 0;

   // Scalar flux output path, empty = no output. .vts/.vtr on a structured
   // mesh, .vtu on an unstructured one - the parser checks which
   std::string flux_vtk;
};

#endif
