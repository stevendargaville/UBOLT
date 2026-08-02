# ~~~~~~~~~~~~~~~~~
# UBOLT - Steven Dargaville
# Makefile for tests
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Where to find a built PFLARE (exported by the top Makefile, defaulted
# here so make can also be called directly in tests/)
PFLARE_DIR ?= /home/sdargavi/projects/PFLARE
# Where the top Makefile puts libubolt (also exported, defaulted the same way)
LIBDIR ?= $(CURDIR)/../lib

# Get the flags we have on input
# These are appended to the flags set by PETSc
# so that users can add their own flags
# but not override the PETSc ones which we use for our builds
CFLAGS_INPUT := $(CFLAGS)
CPPFLAGS_INPUT := $(CPPFLAGS)
CXXPPFLAGS_INPUT := $(CXXPPFLAGS)
CXXFLAGS_INPUT := $(CXXFLAGS)
CUDAC_FLAGS_INPUT := $(CUDAC_FLAGS)
MPICXX_INCLUDES_INPUT := $(MPICXX_INCLUDES)
HIPC_FLAGS_INPUT := $(HIPC_FLAGS)
SYCLC_FLAGS_INPUT := $(SYCLC_FLAGS)

# ~~~~~~~~~~~
# UBOLT specific changes
# ~~~~~~~~~~~
# Include directories - include top level directory in case compilers output modules there
INCLUDE := -I$(CURDIR)/../ -I../include -I$(PFLARE_DIR)/include

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# We then have to add the flags back in after the petsc rules/variables
# have overwritten
override CFLAGS += $(CFLAGS_INPUT) $(INCLUDE)
override CPPFLAGS += $(CPPFLAGS_INPUT) $(INCLUDE)
override CXXPPFLAGS += $(CXXPPFLAGS_INPUT) $(INCLUDE)
override CXXFLAGS += $(CXXFLAGS_INPUT) $(INCLUDE)
override CUDAC_FLAGS += $(CUDAC_FLAGS_INPUT) $(INCLUDE)
override MPICXX_INCLUDES += $(MPICXX_INCLUDES_INPUT) $(INCLUDE)
override HIPC_FLAGS += $(HIPC_FLAGS_INPUT) $(INCLUDE)
override SYCLC_FLAGS += $(SYCLC_FLAGS_INPUT) $(INCLUDE)

# Detect if PETSc was configured without MPI - exported by the top Makefile,
# derived here the same way so make can also be called directly in tests/
PETSC_HAVE_MPIUNI ?= $(if $(shell awk '/^[[:space:]]*\#define[[:space:]]+PETSC_HAVE_MPIUNI[[:space:]]+1/{print 1; exit}' $(PETSCCONF_H)),1,0)

# Link to ubolt and pflare - order is important
# ubolt and then pflare must come first for static linking
LDFLAGS += -Wl,-rpath,$(LIBDIR) -Wl,-rpath,$(PFLARE_DIR)/lib
LDLIBS  := -L$(LIBDIR) -lubolt -L$(PFLARE_DIR)/lib -lpflare $(LDLIBS)

EXISTING_LD := $(LD_LIBRARY_PATH)
EXISTING_DYLD := $(DYLD_FALLBACK_LIBRARY_PATH)

ifeq ($(shell uname -s 2>/dev/null),Darwin)
  export DYLD_FALLBACK_LIBRARY_PATH=$(LIBDIR):$(PFLARE_DIR)/lib$(if $(EXISTING_DYLD),:$(EXISTING_DYLD))
else
  export LD_LIBRARY_PATH=$(LIBDIR):$(PFLARE_DIR)/lib$(if $(EXISTING_LD),:$(EXISTING_LD))
endif

clean::
	$(RM) -f $(TEST_TARGETS)

# ~~~~~~~~~~~
# Run UBOLT tests
#
# Pass/fail is the process exit code. The drivers exit non-zero unless
# KSPGetConvergedReason > 0, and every recipe pins -ksp_max_it to the
# baseline iteration count, so an iteration-count regression fails the run.
# Baseline counts live in ../docs/dev/testing.md and tests/baselines/
# ~~~~~~~~~~~

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_check:
#
	@echo ""
	@echo "Check 1D slab streaming/removal + matrix-free scatter with AIRG"
	./transportk -problem problems/slab_st2.json -ksp_max_it 6
ifeq ($(PETSC_HAVE_MPIUNI),0)
	@echo "Check 1D slab preconditioned with streaming only in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json -precon_stream -ksp_pc_side right -ksp_max_it 9
endif
	@echo "Check 1D slab multigroup, 4 groups with downscatter"
	./transportk -problem problems/slab_mg4_t05.json -ksp_max_it 6
	@echo "Check 2D box against the closed form and a reference matrix"
	./verify_2dk
	@echo "Check 2D box streaming/removal + matrix-free scatter with AIRG"
	./transportk -problem problems/box_50_st2.json -ksp_max_it 7
	@echo "Check 2D box, reflective left + bottom faces"
	./transportk -problem problems/box_50_reflect_lb.json -ksp_max_it 6

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_short_serial:
#
	@echo ""
	@echo "Test 1D slab, streaming/removal pmat, strong removal"
	./transportk -problem problems/slab_st2.json -ksp_max_it 6
	@echo "Test 1D slab, streaming-only pmat, strong removal"
	./transportk -problem problems/slab_st2.json -precon_stream -ksp_pc_side right -ksp_max_it 9
	@echo "Test 1D slab, streaming/removal pmat, weak removal (advection dominated)"
	./transportk -problem problems/slab_st0.json -ksp_max_it 1
	@echo "Test 1D slab, diagonally scaled"
	./transportk -problem problems/slab_st0.json -diag_scale -ksp_max_it 1
	@echo "Test 1D slab, option prefix reaches PCAIR at composite index 1"
	./transportk -problem problems/slab_st2.json -ksp_max_it 6 -sub_1_pc_air_print_stats_timings
	@echo "Test 1D slab, S2 on a coarser slab"
	./transportk -problem problems/slab_s2_coarse.json -ksp_max_it 9
	@echo "Test 1D slab, one MatSetValuesCOO per term instead of a single summed one"
	./transportk -problem problems/slab_st2.json -ksp_max_it 6 -ubolt_coo_two_call
#
# Reflective boundaries. The infinite-medium runs also check the solution
# against the exact per-group constant; all-reflect needs absorption, so those
# problem files carry within-group Sigma_s below Sigma_t - see
# ../docs/dev/testing.md
	@echo "Test 1D slab, reflective left face"
	./transportk -problem problems/slab_st2_reflect_left.json -ksp_max_it 8
	@echo "Test 1D slab, all faces reflective against the infinite-medium solution"
	./transportk -problem problems/slab_inf_medium.json -check_inf_medium -ksp_rtol 1e-12 -ksp_max_it 10
	@echo "Test 1D slab, reflective face with one MatSetValuesCOO per term"
	./transportk -problem problems/slab_st2_reflect_left.json -ksp_max_it 8 -ubolt_coo_two_call
#
# Multigroup. -ksp_max_it is a single pin for the whole sweep, so it is the MAX
# over the groups - see ../docs/dev/testing.md. The uncoupled t0 file carries
# Source 2.0, so each group's solve is byte-identical to slab_st2.json's
	@echo "Test 1D slab multigroup, uncoupled groups each reproduce the single-group problem"
	./transportk -problem problems/slab_mg4_t0.json -ksp_max_it 6
	@echo "Test 1D slab multigroup, 4 groups with downscatter"
	./transportk -problem problems/slab_mg4_t05.json -ksp_max_it 6
	@echo "Test 1D slab multigroup, strong downscatter"
	./transportk -problem problems/slab_mg4_t2.json -ksp_max_it 6
	@echo "Test 1D slab multigroup, streaming-only pmat reused across the sweep"
	./transportk -problem problems/slab_mg4_t05.json -precon_stream -ksp_pc_side right -ksp_max_it 11
	@echo "Test 1D slab multigroup, weak removal (advection dominated)"
	./transportk -problem problems/slab_mg4_t05_st0.json -ksp_max_it 1
	@echo "Test 1D slab multigroup, S2 on a coarser slab"
	./transportk -problem problems/slab_mg2_coarse.json -ksp_max_it 9
	@echo "Test 1D slab multigroup, per-group refill with one MatSetValuesCOO per term"
	./transportk -problem problems/slab_mg4_t05.json -ksp_max_it 6 -ubolt_coo_two_call
	@echo "Test 1D slab multigroup, reflective left face (right stays vacuum - the last group has ratio 1)"
	./transportk -problem problems/slab_mg4_t05_reflect_left.json -ksp_max_it 8
	@echo "Test 1D slab multigroup, a double-density region painted mid-slab"
	./transportk -problem problems/slab_mg4_t05_dense.json -ksp_max_it 6
#
# 2D. verify_2dk is the discretisation check (closed form + reference matrix);
# the box recipes pin iteration counts the same way the 1D ones do. st2 alone
# is a scattering ratio of exactly 1 (no absorption); st2 with Sigma_s 1.0 is
# ratio 0.5. Both are mesh independent on every grid shape - see
# ../docs/dev/testing.md
	@echo "Test 2D box, closed form and reference matrix"
	./verify_2dk
	@echo "Test 2D box, streaming/removal pmat, pure scattering (ratio 1)"
	./transportk -problem problems/box_50_st2.json -ksp_max_it 7
	@echo "Test 2D box, scattering ratio 0.5"
	./transportk -problem problems/box_50_ratio05.json -ksp_max_it 5
	@echo "Test 2D box, non-square grid (dx is not dy)"
	./transportk -problem problems/box_60x30_st2.json -ksp_max_it 6
	@echo "Test 2D box, non-square box with square cells"
	./transportk -problem problems/box_80x20_st2.json -ksp_max_it 5
	@echo "Test 2D box, weak removal (advection dominated)"
	./transportk -problem problems/box_50_st0.json -ksp_max_it 3
	@echo "Test 2D box, streaming-only pmat"
	./transportk -problem problems/box_50_st2.json -precon_stream -ksp_pc_side right -ksp_max_it 9
	@echo "Test 2D box, S4 (12 angles)"
	./transportk -problem problems/box_30_s4_st2.json -ksp_max_it 6
	@echo "Test 2D box, one MatSetValuesCOO per term instead of a single summed one"
	./transportk -problem problems/box_50_st2.json -ksp_max_it 7 -ubolt_coo_two_call
	@echo "Test 2D box, reflective left + bottom faces"
	./transportk -problem problems/box_50_reflect_lb.json -ksp_max_it 6
	@echo "Test 2D box, S4 with reflective left + bottom faces"
	./transportk -problem problems/box_50_s4_reflect_lb.json -ksp_max_it 6
	@echo "Test 2D box, all faces reflective against the infinite-medium solution"
	./transportk -problem problems/box_50_inf_medium.json -check_inf_medium -ksp_rtol 1e-12 -ksp_max_it 10
#
# Painted regions. The identity file's regions carry the background values, so
# it must land exactly where the uniform 50x50 recipe does - painting and
# expansion change nothing but which table entry a cell reads. The absorber is
# a genuinely heterogeneous pin: a sourceless absorber in a scattering box
	@echo "Test 2D box, two regions equal to the background (painting identity)"
	./transportk -problem problems/box_50_identity.json -ksp_max_it 7
	@echo "Test 2D box, an absorbing sourceless block painted mid-box"
	./transportk -problem problems/box_50_absorber.json -ksp_max_it 5
	@echo "Test 2D box, cold box: zero inflow, a central 0.1 x 0.1 source region and nothing else"
	./transportk -problem problems/box_100_cold.json -ksp_max_it 5

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_short_parallel:
#
	@echo ""
	@echo "Test 1D slab, streaming/removal pmat, strong removal in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json -ksp_max_it 6
	@echo "Test 1D slab, streaming/removal pmat, weak removal in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st0.json -ksp_max_it 1
	@echo "Test 1D slab, streaming-only pmat, strong removal in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json -precon_stream -ksp_pc_side right -ksp_max_it 9
	@echo "Test 1D slab, S2 on a coarser slab in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_s2_coarse.json -ksp_max_it 9
	@echo "Test 1D slab, one MatSetValuesCOO per term in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json -ksp_max_it 6 -ubolt_coo_two_call
	@echo "Test 1D slab, all faces reflective against the infinite-medium solution in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_inf_medium.json -check_inf_medium -ksp_rtol 1e-12 -ksp_max_it 11
	@echo "Test 1D slab multigroup, 4 groups with downscatter in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t05.json -ksp_max_it 6
	@echo "Test 1D slab multigroup, streaming-only pmat in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t05.json -precon_stream -ksp_pc_side right -ksp_max_it 11
	@echo "Test 1D slab multigroup, uncoupled groups in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t0.json -ksp_max_it 6
#
# 2D. -n 4 as well as -n 2 for the closed form: a 2D DMDA splits in one
# direction only at -n 2, so -n 4 is the first decomposition with a genuinely
# 2D processor grid, and the patch-lexicographic global numbering the layout
# assert pins is at its least like the natural one there.
# The reference matrix check is serial (it is written in natural ordering)
	@echo "Test 2D box, closed form in parallel"
	$(MPIEXEC) -n 2 ./verify_2dk -skip_reference
	@echo "Test 2D box, closed form on a 2D processor grid"
	$(MPIEXEC) -n 4 ./verify_2dk -skip_reference
	@echo "Test 2D box, pure scattering in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_st2.json -ksp_max_it 6
	@echo "Test 2D box, scattering ratio 0.5 in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_ratio05.json -ksp_max_it 5
	@echo "Test 2D box, non-square grid in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_60x30_st2.json -ksp_max_it 6
	@echo "Test 2D box, weak removal in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_st0.json -ksp_max_it 3
	@echo "Test 2D box, streaming-only pmat in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_st2.json -precon_stream -ksp_pc_side right -ksp_max_it 10
	@echo "Test 2D box, S4 in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_30_s4_st2.json -ksp_max_it 6
	@echo "Test 2D box, one MatSetValuesCOO per term in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_st2.json -ksp_max_it 6 -ubolt_coo_two_call
	@echo "Test 2D box, reflective left + bottom faces in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_reflect_lb.json -ksp_max_it 6
	@echo "Test 2D box, all faces reflective against the infinite-medium solution in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_inf_medium.json -check_inf_medium -ksp_rtol 1e-12 -ksp_max_it 10
#
# Painted regions in parallel. The identity runs at -n 4: painting maps cell
# centres through the patch-lexicographic local ordering, and the 2D processor
# grid is where that ordering is least like the natural one
	@echo "Test 2D box, painting identity on a 2D processor grid"
	$(MPIEXEC) -n 4 ./transportk -problem problems/box_50_identity.json -ksp_max_it 7
	@echo "Test 2D box, an absorbing sourceless block in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_50_absorber.json -ksp_max_it 5
	@echo "Test 2D box, cold box with a central source region in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_100_cold.json -ksp_max_it 5
	@echo "Test 1D slab multigroup, a double-density region in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t05_dense.json -ksp_max_it 6

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_serial:
#
	@echo ""
	@echo "Test 1D slab, streaming-only pmat, weak removal"
	./transportk -problem problems/slab_st0.json -precon_stream -ksp_pc_side right -ksp_max_it 1
	@echo "Test 1D slab, diagonally scaled, streaming-only pmat"
	./transportk -problem problems/slab_st0.json -diag_scale -precon_stream -ksp_pc_side right -ksp_max_it 3
	@echo "Test 2D box, non-square grid, four times as fine (mesh independence)"
	./transportk -problem problems/box_120x60_st2.json -ksp_max_it 7
	@echo "Test 2D box, S4 on a larger grid"
	./transportk -problem problems/box_100_s4_st2.json -ksp_max_it 7
	@echo "Test 2D box, streaming-only pmat on a larger grid"
	./transportk -problem problems/box_120x60_st2.json -precon_stream -ksp_pc_side right -ksp_max_it 10
	@echo "Test 2D box, pure scattering on a larger grid"
	./transportk -problem problems/box_100_st2.json -ksp_max_it 7

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_parallel:
#
	@echo ""
	@echo "Test 1D slab, weak removal in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st0.json -ksp_max_it 1
	@echo "Test 1D slab, diagonally scaled in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st0.json -diag_scale -ksp_max_it 1
	@echo "Test 1D slab multigroup, weak removal in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t05_st0.json -ksp_max_it 1
	@echo "Test 1D slab multigroup, strong downscatter in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t2.json -ksp_max_it 6
	@echo "Test 2D box, non-square grid, four times as fine, in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_120x60_st2.json -ksp_max_it 7
	@echo "Test 2D box, S4 on a larger grid in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_100_s4_st2.json -ksp_max_it 6
	@echo "Test 2D box, weak removal on a larger grid in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_100_st0.json -ksp_max_it 3
	@echo "Test 2D box, pure scattering on a larger grid in parallel"
	$(MPIEXEC) -n 2 ./transportk -problem problems/box_100_st2.json -ksp_max_it 7

# ~~~~~~~~~~~
# Re-capture reference baselines into baselines/
# Only do this deliberately - see ../docs/dev/testing.md
# The full matrix: {default pc, precon_stream} x {sigma_t 0, 2}
#                  x {np 1, 2} x {diag_scale off, on}
# Capped at 200 iterations: ALL FOUR st=2 diag_scale configs are pathological
# (they hit the cap or break down) and a 200-iteration residual history is a
# strong enough fingerprint for refactor comparison
#
# None of those four converges, so the driver exits 1 on them by design - they
# get a || true so the capture runs to completion
# ~~~~~~~~~~~
BASE := -ksp_monitor -ksp_converged_reason -ksp_max_it 200
capture_baselines:
	./transportk -problem problems/slab_st0.json $(BASE) > baselines/slab1d_default_st0_np1.log
	./transportk -problem problems/slab_st2.json $(BASE) > baselines/slab1d_default_st2_np1.log
	./transportk -problem problems/slab_st0.json -diag_scale $(BASE) > baselines/slab1d_default_st0_np1_ds.log
	./transportk -problem problems/slab_st2.json -diag_scale $(BASE) > baselines/slab1d_default_st2_np1_ds.log || true
	./transportk -problem problems/slab_st0.json -precon_stream -ksp_pc_side right $(BASE) > baselines/slab1d_stream_st0_np1.log
	./transportk -problem problems/slab_st2.json -precon_stream -ksp_pc_side right $(BASE) > baselines/slab1d_stream_st2_np1.log
	./transportk -problem problems/slab_st0.json -precon_stream -ksp_pc_side right -diag_scale $(BASE) > baselines/slab1d_stream_st0_np1_ds.log
	./transportk -problem problems/slab_st2.json -precon_stream -ksp_pc_side right -diag_scale $(BASE) > baselines/slab1d_stream_st2_np1_ds.log || true
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st0.json $(BASE) > baselines/slab1d_default_st0_np2.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json $(BASE) > baselines/slab1d_default_st2_np2.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st0.json -diag_scale $(BASE) > baselines/slab1d_default_st0_np2_ds.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json -diag_scale $(BASE) > baselines/slab1d_default_st2_np2_ds.log || true
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st0.json -precon_stream -ksp_pc_side right $(BASE) > baselines/slab1d_stream_st0_np2.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json -precon_stream -ksp_pc_side right $(BASE) > baselines/slab1d_stream_st2_np2.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st0.json -precon_stream -ksp_pc_side right -diag_scale $(BASE) > baselines/slab1d_stream_st0_np2_ds.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_st2.json -precon_stream -ksp_pc_side right -diag_scale $(BASE) > baselines/slab1d_stream_st2_np2_ds.log || true
#
# Multigroup: 4 groups, {default pc, precon_stream} x {transfer 0, 0.5} x {np 1, 2}.
# Each log is the whole group sweep, one KSP history after another. The
# transfer 0 logs are 4 copies of the single-group history, which is the
# uncoupled-groups verification; the transfer 0.5 logs are the fingerprint of
# the downscatter source itself. slab_mg4_t0.json carries Source 2.0: the
# isotropic strength that puts 1.0 on each ordinate (sum_weights is 2 in 1D),
# i.e. exactly the rhs the single-group files carry, which is what keeps those
# logs byte-for-byte copies of the single-group ones
	./transportk -problem problems/slab_mg4_t0.json $(BASE) > baselines/slab1dmg_default_st2_g4_t0_np1.log
	./transportk -problem problems/slab_mg4_t05.json $(BASE) > baselines/slab1dmg_default_st2_g4_t05_np1.log
	./transportk -problem problems/slab_mg4_t05_st0.json $(BASE) > baselines/slab1dmg_default_st0_g4_t05_np1.log
	./transportk -problem problems/slab_mg4_t05.json -precon_stream -ksp_pc_side right $(BASE) > baselines/slab1dmg_stream_st2_g4_t05_np1.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t0.json $(BASE) > baselines/slab1dmg_default_st2_g4_t0_np2.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t05.json $(BASE) > baselines/slab1dmg_default_st2_g4_t05_np2.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t05_st0.json $(BASE) > baselines/slab1dmg_default_st0_g4_t05_np2.log
	$(MPIEXEC) -n 2 ./transportk -problem problems/slab_mg4_t05.json -precon_stream -ksp_pc_side right $(BASE) > baselines/slab1dmg_stream_st2_g4_t05_np2.log
	@echo "Baselines captured in baselines/"
