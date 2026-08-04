#include "ubolt/problem_spec.hpp"
// The vendored parser lives under src/external so it cannot be included from
// the public headers - a quote include resolves against this file's directory
#include "external/nlohmann/json.hpp"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <limits>

using json = nlohmann::json;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// JSON access helpers
//
// Every access is preceded by an explicit contains()/type check and the parse
// runs with exceptions disabled, so no nlohmann exception can ever cross a
// PetscCall frame. `file` in each signature is only for the error messages
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode JsonGetInt(const json &obj, const char *key, const char *file, PetscInt *value)
{
   PetscFunctionBeginUser;

   PetscCheck(obj.contains(key), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: missing required key \"%s\"", file, key);
   const json &v = obj.at(key);
   PetscCheck(v.is_number_integer(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: \"%s\" must be an integer", file, key);
   // Read wide and range-check into PetscInt for the 32-bit-index builds
   const std::int64_t raw = v.get<std::int64_t>();
   PetscCheck(raw >= std::numeric_limits<PetscInt>::min() && raw <= std::numeric_limits<PetscInt>::max(), \
      PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%s: \"%s\" = %" PetscInt64_FMT " does not fit in PetscInt", \
      file, key, (PetscInt64)raw);
   *value = (PetscInt)raw;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode JsonGetReal(const json &obj, const char *key, const char *file, PetscReal *value)
{
   PetscFunctionBeginUser;

   PetscCheck(obj.contains(key), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: missing required key \"%s\"", file, key);
   const json &v = obj.at(key);
   PetscCheck(v.is_number(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: \"%s\" must be a number", file, key);
   *value = (PetscReal)v.get<double>();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A fixed-length array of numbers - mesh entries, intervals, boxes, xsection
// rows. The array itself must exist; expected_len is what the schema says
static PetscErrorCode JsonGetRealArray(const json &arr, const char *key, const char *file, \
   size_t expected_len, std::vector<PetscScalar> &values)
{
   PetscFunctionBeginUser;

   PetscCheck(arr.is_array() && arr.size() == expected_len, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: \"%s\" must be an array of %zu numbers", file, key, expected_len);
   values.resize(expected_len);
   for (size_t i = 0; i < expected_len; i++) {
      PetscCheck(arr.at(i).is_number(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: \"%s\"[%zu] must be a number", file, key, i);
      values[i] = (PetscScalar)arr.at(i).get<double>();
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The problem file is strict: any key not in `allowed` is a typo and an
// error. "_comment" is always allowed and ignored (JSON has no comments)
static PetscErrorCode JsonCheckKeys(const json &obj, const char *what, const char *file, \
   std::initializer_list<const char *> allowed)
{
   PetscFunctionBeginUser;

   for (const auto &item : obj.items()) {
      if (item.key() == "_comment") continue;
      bool known = false;
      for (const char *k : allowed) known = known || (item.key() == k);
      PetscCheck(known, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: unknown key \"%s\" in %s", file, item.key().c_str(), what);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Rank 0's half: read the problem file, splice in a path-valued materials
// file, and hand back one resolved JSON string for the broadcast
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode ReadWholeFile(const char *path, std::string &contents)
{
   PetscFunctionBeginUser;

   std::ifstream f(path, std::ios::binary);
   PetscCheck(f.good(), PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "could not open %s", path);
   std::ostringstream ss;
   ss << f.rdbuf();
   PetscCheck(!f.fail(), PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "could not read %s", path);
   contents = ss.str();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode ReadAndResolve(const char *problem_path, std::string &resolved)
{
   std::string text;

   PetscFunctionBeginUser;

   PetscCall(ReadWholeFile(problem_path, text));
   json root = json::parse(text, nullptr, /*allow_exceptions=*/false);
   PetscCheck(!root.is_discarded(), PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, \
      "%s: not valid JSON", problem_path);

   // A string-valued materials entry is a path, relative to the problem
   // file's own directory - splice the file it names in so the broadcast
   // carries one self-contained tree
   if (root.contains("materials") && root.at("materials").is_string()) {
      const std::string mat_ref = root.at("materials").get<std::string>();
      std::string mat_path = mat_ref;
      if (!mat_ref.empty() && mat_ref[0] != '/') {
         const char *slash = strrchr(problem_path, '/');
         if (slash) mat_path = std::string(problem_path, slash - problem_path + 1) + mat_ref;
      }
      PetscCall(ReadWholeFile(mat_path.c_str(), text));
      json mats = json::parse(text, nullptr, /*allow_exceptions=*/false);
      PetscCheck(!mats.is_discarded(), PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, \
         "%s: not valid JSON (materials file named by %s)", mat_path.c_str(), problem_path);
      root["materials"] = std::move(mats);
   }

   // dump() writes doubles shortest-round-trip, so nothing is lost re-parsing
   resolved = root.dump();

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The materials schema - the multigroup table plus UBOLT's Source.
// Tolerant of unknown keys: materials files authored by other codes carry
// fission fields and bookkeeping UBOLT has no use for yet
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode ParseMaterials(const json &mats_j, const char *file, PetscInt *n_groups, \
   MaterialSpec &materials, std::vector<std::string> &material_names)
{
   std::vector<PetscScalar> row;

   PetscFunctionBeginUser;

   PetscCheck(mats_j.is_object(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: \"materials\" must be a path string or an object", file);

   // The only representation UBOLT reads is the plain multigroup table -
   // absent means multigroup, so files that omit the key keep working
   if (mats_j.contains("representation")) {
      const json &rep = mats_j.at("representation");
      PetscCheck(rep.is_string() && rep.get<std::string>() == "multigroup", PETSC_COMM_SELF, \
         PETSC_ERR_SUP, "%s: only the \"multigroup\" materials representation is supported", file);
   }

   PetscCall(JsonGetInt(mats_j, "n_groups", file, n_groups));
   PetscCheck(*n_groups > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "%s: n_groups must be positive, was given %" PetscInt_FMT, file, *n_groups);
   const size_t ng = (size_t)*n_groups;

   PetscCheck(mats_j.contains("materials") && mats_j.at("materials").is_array() && \
      !mats_j.at("materials").empty(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: \"materials\" must contain a non-empty materials array", file);
   const json &list = mats_j.at("materials");
   const PetscInt n_materials = (PetscInt)list.size();

   PetscCall(materials.create(n_materials, *n_groups));
   material_names.assign((size_t)n_materials, "");

   // ids must be dense 0..n-1 - they ARE the MaterialSpec indices, because
   // the tables reach device kernels (see material_spec.hpp). seen[] gives
   // the duplicate check
   std::vector<bool> seen((size_t)n_materials, false);
   for (const json &m : list) {

      PetscCheck(m.is_object(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: each materials entry must be an object", file);
      PetscInt id = 0;
      PetscCall(JsonGetInt(m, "id", file, &id));
      PetscCheck(id >= 0 && id < n_materials, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
         "%s: material ids must be dense 0..%" PetscInt_FMT ", was given %" PetscInt_FMT, \
         file, n_materials - 1, id);
      PetscCheck(!seen[(size_t)id], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: duplicate material id %" PetscInt_FMT, file, id);
      seen[(size_t)id] = true;

      if (m.contains("name")) {
         PetscCheck(m.at("name").is_string(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
            "%s: material %" PetscInt_FMT ": \"name\" must be a string", file, id);
         material_names[(size_t)id] = m.at("name").get<std::string>();
      }

      PetscCheck(m.contains("Sigma_t"), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: material %" PetscInt_FMT ": missing \"Sigma_t\"", file, id);
      PetscCall(JsonGetRealArray(m.at("Sigma_t"), "Sigma_t", file, ng, row));
      for (size_t g = 0; g < ng; g++) PetscCall(materials.set_sigma_t(id, (PetscInt)g, row[g]));

      // Sigma_s[from][to]: row g is everything scattering OUT of g, which is
      // exactly MaterialSpec's (g_from, g_to) ordering
      PetscCheck(m.contains("Sigma_s") && m.at("Sigma_s").is_array() && m.at("Sigma_s").size() == ng, \
         PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: material %" PetscInt_FMT ": \"Sigma_s\" must be an array of %zu rows", file, id, ng);
      for (size_t g_from = 0; g_from < ng; g_from++) {
         PetscCall(JsonGetRealArray(m.at("Sigma_s").at(g_from), "Sigma_s row", file, ng, row));
         for (size_t g_to = 0; g_to < ng; g_to++) {
            PetscCall(materials.set_sigma_s(id, (PetscInt)g_from, (PetscInt)g_to, row[g_to]));
         }
      }

      // UBOLT's one extension: the isotropic external source. Absent = 0
      if (m.contains("Source")) {
         PetscCall(JsonGetRealArray(m.at("Source"), "Source", file, ng, row));
         for (size_t g = 0; g < ng; g++) PetscCall(materials.set_source(id, (PetscInt)g, row[g]));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A region's material is an id or a name from the materials table
static PetscErrorCode LookupMaterial(const json &ref, const char *file, \
   const std::vector<std::string> &material_names, PetscInt *material)
{
   PetscFunctionBeginUser;

   if (ref.is_string()) {
      const std::string name = ref.get<std::string>();
      for (size_t i = 0; i < material_names.size(); i++) {
         if (material_names[i] == name) {
            *material = (PetscInt)i;
            PetscFunctionReturn(PETSC_SUCCESS);
         }
      }
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%s: no material named \"%s\"", file, name.c_str());
   }

   PetscCheck(ref.is_number_integer(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: a material reference must be an id or a name", file);
   const std::int64_t raw = ref.get<std::int64_t>();
   PetscCheck(raw >= 0 && raw < (std::int64_t)material_names.size(), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_OUTOFRANGE, "%s: material id %" PetscInt64_FMT " out of range, there are %zu materials", \
      file, (PetscInt64)raw, material_names.size());
   *material = (PetscInt)raw;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ProblemSpec
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PetscErrorCode ProblemSpec::create(MPI_Comm comm, const char *problem_path)
{
   PetscMPIInt rank, count;
   PetscInt64 len = 0;
   std::string resolved;
   std::vector<PetscScalar> vals;

   PetscFunctionBeginUser;

   PetscCallMPI(MPI_Comm_rank(comm, &rank));

   // Rank 0 reads and resolves; everyone parses the same broadcast bytes, so
   // the spec cannot disagree across ranks. Under MPIUNI the broadcasts are
   // no-ops
   if (rank == 0) PetscCall(ReadAndResolve(problem_path, resolved));
   len = (PetscInt64)resolved.size();
   PetscCallMPI(MPI_Bcast(&len, 1, MPIU_INT64, 0, comm));
   resolved.resize((size_t)len);
   PetscCall(PetscMPIIntCast(len, &count));
   PetscCallMPI(MPI_Bcast(resolved.data(), count, MPI_CHAR, 0, comm));

   json root = json::parse(resolved, nullptr, /*allow_exceptions=*/false);
   PetscCheck(!root.is_discarded(), PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, \
      "%s: not valid JSON after resolution", problem_path);

   // A file from before inflow went per face would parse into different
   // physics rather than failing, so name it rather than letting the strict
   // key check call it a typo
   PetscCheck(!root.contains("inflow"), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: the global \"inflow\" key was removed - inflow is per face now, and " \
      "angle-integrated like \"Source\" (the old key was per-angle): " \
      "\"boundary_conditions\": {\"left\": {\"type\": \"vacuum\", \"inflow\": 1.0}, ...}", problem_path);

   // The problem file is strict - an unknown key is a typo, not an extension
   PetscCall(JsonCheckKeys(root, "the problem file", problem_path, \
      {"dimension", "mesh", "sn_order", "materials", "regions", "boundary_conditions", \
       "output"}));

   PetscCall(JsonGetInt(root, "dimension", problem_path, &dimension));
   PetscCheck(dimension >= 1 && dimension <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "%s: dimension must be 1, 2 or 3, was given %" PetscInt_FMT, problem_path, dimension);
   const size_t dim = (size_t)dimension;

   // ~~~~~~~~~~~~~
   // Mesh: n_cells and lengths, one entry per dimension
   // ~~~~~~~~~~~~~
   PetscCheck(root.contains("mesh") && root.at("mesh").is_object(), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_WRONG, "%s: missing required object \"mesh\"", problem_path);
   const json &mesh = root.at("mesh");
   PetscCall(JsonCheckKeys(mesh, "\"mesh\"", problem_path, {"n_cells", "lengths"}));

   PetscCheck(mesh.contains("n_cells"), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: mesh is missing \"n_cells\"", problem_path);
   const json &n_cells = mesh.at("n_cells");
   PetscCheck(n_cells.is_array() && n_cells.size() == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: mesh \"n_cells\" must be an array of %zu integers", problem_path, dim);
   for (size_t d = 0; d < dim; d++) {
      PetscInt n = 0;
      PetscCheck(n_cells.at(d).is_number_integer(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: mesh \"n_cells\"[%zu] must be an integer", problem_path, d);
      const std::int64_t raw = n_cells.at(d).get<std::int64_t>();
      PetscCheck(raw > 0 && raw <= std::numeric_limits<PetscInt>::max(), PETSC_COMM_SELF, \
         PETSC_ERR_ARG_OUTOFRANGE, "%s: mesh \"n_cells\"[%zu] must be a positive PetscInt", \
         problem_path, d);
      n = (PetscInt)raw;
      if (d == 0) n_cells_x = n;
      else if (d == 1) n_cells_y = n;
      else n_cells_z = n;
   }

   PetscCheck(mesh.contains("lengths"), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: mesh is missing \"lengths\"", problem_path);
   PetscCall(JsonGetRealArray(mesh.at("lengths"), "lengths", problem_path, dim, vals));
   length_x = (PetscReal)PetscRealPart(vals[0]);
   if (dimension >= 2) length_y = (PetscReal)PetscRealPart(vals[1]);
   if (dimension >= 3) length_z = (PetscReal)PetscRealPart(vals[2]);
   PetscCheck(length_x > 0.0 && (dimension < 2 || length_y > 0.0) && \
      (dimension < 3 || length_z > 0.0), PETSC_COMM_SELF, \
      PETSC_ERR_ARG_OUTOFRANGE, "%s: mesh lengths must be positive", problem_path);

   // The SN order, not the ordinate count - which orders exist is the
   // quadrature's business, so only the obvious nonsense is caught here
   PetscCall(JsonGetInt(root, "sn_order", problem_path, &sn_order));
   PetscCheck(sn_order > 0 && sn_order % 2 == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
      "%s: sn_order must be a positive even integer, was given %" PetscInt_FMT, problem_path, \
      sn_order);

   // ~~~~~~~~~~~~~
   // Materials (already spliced in if it was a path)
   // ~~~~~~~~~~~~~
   PetscCheck(root.contains("materials"), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
      "%s: missing required key \"materials\"", problem_path);
   PetscCall(ParseMaterials(root.at("materials"), problem_path, &n_groups, materials, material_names));

   // ~~~~~~~~~~~~~
   // Regions: the background material plus the paint list
   // ~~~~~~~~~~~~~
   if (root.contains("regions")) {

      PetscCheck(root.at("regions").is_object(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: \"regions\" must be an object", problem_path);
      const json &regions = root.at("regions");
      PetscCall(JsonCheckKeys(regions, "\"regions\"", problem_path, {"background", "paint"}));

      if (regions.contains("background")) {
         PetscCall(LookupMaterial(regions.at("background"), problem_path, material_names, \
            &background_material));
      }

      if (regions.contains("paint")) {

         PetscCheck(regions.at("paint").is_array(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
            "%s: regions \"paint\" must be an array", problem_path);
         const char *shape_key = (dimension == 1) ? "interval" : "box";

         for (const json &p : regions.at("paint")) {

            PetscCheck(p.is_object(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
               "%s: each paint entry must be an object", problem_path);
            PetscCall(JsonCheckKeys(p, "a paint entry", problem_path, {"material", shape_key}));
            PetscCheck(p.contains("material"), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
               "%s: a paint entry is missing \"material\"", problem_path);
            PetscInt material = 0;
            PetscCall(LookupMaterial(p.at("material"), problem_path, material_names, &material));

            PetscCheck(p.contains(shape_key), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
               "%s: a paint entry is missing \"%s\" (dimension is %" PetscInt_FMT ")", \
               problem_path, shape_key, dimension);
            if (dimension == 1) {
               PetscCall(JsonGetRealArray(p.at(shape_key), shape_key, problem_path, 2, vals));
               intervals.push_back({vals[0], vals[1], material});
            }
            else if (dimension == 2) {
               PetscCall(JsonGetRealArray(p.at(shape_key), shape_key, problem_path, 4, vals));
               boxes.push_back({vals[0], vals[1], vals[2], vals[3], material});
            }
            else {
               PetscCall(JsonGetRealArray(p.at(shape_key), shape_key, problem_path, 6, vals));
               boxes_3d.push_back({vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], material});
            }
         }
      }
   }

   // ~~~~~~~~~~~~~
   // Boundary conditions, keyed by face name for the dimension. Unset faces
   // are vacuum, matching BCSpec's default
   // ~~~~~~~~~~~~~
   if (root.contains("boundary_conditions")) {

      PetscCheck(root.at("boundary_conditions").is_object(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: \"boundary_conditions\" must be an object", problem_path);
      const json &bcj = root.at("boundary_conditions");

      // CAREFUL: the face names follow PETSc's box-mesh convention per
      // dimension, and bottom/top move axis - they are the Y faces in 2D but
      // the Z faces in 3D, where the y faces are front/back (see the FACE_*
      // comments on the backends)
      struct Face { const char *name; PetscInt id; };
      const Face faces_1d[] = {{"left", StructuredFD1D::FACE_LEFT}, \
                               {"right", StructuredFD1D::FACE_RIGHT}};
      const Face faces_2d[] = {{"left", StructuredFD2D::FACE_LEFT}, \
                               {"right", StructuredFD2D::FACE_RIGHT}, \
                               {"bottom", StructuredFD2D::FACE_BOTTOM}, \
                               {"top", StructuredFD2D::FACE_TOP}};
      const Face faces_3d[] = {{"left", StructuredFD3D::FACE_LEFT}, \
                               {"right", StructuredFD3D::FACE_RIGHT}, \
                               {"front", StructuredFD3D::FACE_FRONT}, \
                               {"back", StructuredFD3D::FACE_BACK}, \
                               {"bottom", StructuredFD3D::FACE_BOTTOM}, \
                               {"top", StructuredFD3D::FACE_TOP}};
      const Face *faces = (dimension == 1) ? faces_1d : ((dimension == 2) ? faces_2d : faces_3d);
      const size_t n_faces = 2 * (size_t)dimension;

      if (dimension == 1) PetscCall(JsonCheckKeys(bcj, "\"boundary_conditions\"", problem_path, \
         {"left", "right"}));
      else if (dimension == 2) PetscCall(JsonCheckKeys(bcj, "\"boundary_conditions\"", problem_path, \
         {"left", "right", "bottom", "top"}));
      else PetscCall(JsonCheckKeys(bcj, "\"boundary_conditions\"", problem_path, \
         {"left", "right", "front", "back", "bottom", "top"}));

      // A face is either the bare family string - "vacuum" meaning a cold
      // vacuum face, inflow 0 - or an object carrying that family plus what a
      // vacuum face lets in
      for (size_t f = 0; f < n_faces; f++) {
         if (!bcj.contains(faces[f].name)) continue;
         const json &v = bcj.at(faces[f].name);

         std::string kind;
         if (v.is_string()) kind = v.get<std::string>();
         else {
            PetscCheck(v.is_object(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
               "%s: boundary condition \"%s\" must be \"vacuum\", \"reflect\" or an object", \
               problem_path, faces[f].name);
            PetscCall(JsonCheckKeys(v, "a boundary condition", problem_path, \
               {"type", "inflow", "window"}));
            PetscCheck(v.contains("type") && v.at("type").is_string(), PETSC_COMM_SELF, \
               PETSC_ERR_ARG_WRONG, "%s: boundary condition \"%s\" is missing a string \"type\"", \
               problem_path, faces[f].name);
            kind = v.at("type").get<std::string>();
         }

         PetscCheck(kind == "vacuum" || kind == "reflect", PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
            "%s: boundary condition \"%s\" must be \"vacuum\", \"reflect\" or an object, was " \
            "given \"%s\"", problem_path, faces[f].name, kind.c_str());

         if (kind == "reflect") {
            PetscCheck(v.is_string() || (!v.contains("inflow") && !v.contains("window")), \
               PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%s: boundary condition \"%s\" is " \
               "reflective - a \"reflect\" face takes no \"inflow\" or \"window\"", \
               problem_path, faces[f].name);
            bcs.set(faces[f].id, BCType::REFLECT);
            n_reflect_faces++;
            continue;
         }

         // Vacuum. The bare string form and an object with no "inflow" are the
         // same thing: a cold face, the BCSpec default
         if (v.is_string()) continue;

         if (v.contains("inflow")) {
            PetscReal value = 0.0;
            PetscCall(JsonGetReal(v, "inflow", problem_path, &value));
            bcs.set_inflow(faces[f].id, value);
         }

         if (v.contains("window")) {
            PetscCheck(v.contains("inflow"), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
               "%s: boundary condition \"%s\" has a \"window\" without an \"inflow\" - a window " \
               "without an inflow restricts nothing", problem_path, faces[f].name);
            PetscCheck(dimension >= 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
               "%s: boundary condition \"%s\" has a \"window\" - a 1D face is a point and takes " \
               "no \"window\"", problem_path, faces[f].name);
            // One [lo, hi] pair per tangential axis, in ascending global-axis
            // order: 2 numbers on a 2D face, 4 on a 3D one
            std::vector<PetscScalar> lo_hi;
            PetscCall(JsonGetRealArray(v.at("window"), "window", problem_path, \
               2 * (size_t)(dimension - 1), lo_hi));
            std::vector<PetscReal> window(lo_hi.size());
            for (size_t p = 0; p < lo_hi.size(); p += 2) {
               window[p]     = PetscRealPart(lo_hi[p]);
               window[p + 1] = PetscRealPart(lo_hi[p + 1]);
               PetscCheck(window[p] <= window[p + 1], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, \
                  "%s: boundary condition \"%s\" has an inside-out \"window\" - give it as " \
                  "[lo, hi] per tangential axis, ascending axis order, with lo <= hi", \
                  problem_path, faces[f].name);
            }
            bcs.set_window(faces[f].id, dimension - 1, window.data());
         }
      }
   }

   // ~~~~~~~~~~~~~
   // Output
   // ~~~~~~~~~~~~~
   if (root.contains("output")) {
      PetscCheck(root.at("output").is_object(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
         "%s: \"output\" must be an object", problem_path);
      const json &output = root.at("output");
      PetscCall(JsonCheckKeys(output, "\"output\"", problem_path, {"flux_vtk"}));
      if (output.contains("flux_vtk")) {
         PetscCheck(output.at("flux_vtk").is_string(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
            "%s: output \"flux_vtk\" must be a string", problem_path);
         flux_vtk = output.at("flux_vtk").get<std::string>();
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}
