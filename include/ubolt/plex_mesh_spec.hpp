#ifndef UBOLT_PLEX_MESH_SPEC_HPP
#define UBOLT_PLEX_MESH_SPEC_HPP

#include <petscsys.h>
#include <string>

// How the plex backends get their mesh: a box built in code, or a file PETSc
// can read (Gmsh .msh, ...). A box's "Face Sets" are PETSc's box ids, the
// structured backends' FACE_* constants
struct PETSC_VISIBILITY_PUBLIC PlexMeshSpec {
   PetscInt dimension = 0;                 // 2 or 3
   PetscInt n_cells[3] = {0, 0, 0};        // box: cells per axis (ignored with a file)
   PetscReal lengths[3] = {0.0, 0.0, 0.0}; // box: extents from the origin
   PetscBool simplex = PETSC_FALSE;        // box: triangles/tets instead of quads/hexes
   std::string file;                       // non-empty: DMPlexCreateFromFile, interpolated
};

#endif
