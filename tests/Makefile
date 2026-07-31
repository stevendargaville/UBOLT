# ~~~~~~~~~~~~~~~~~
# UBOLT - Steven Dargaville
# Makefile for tests
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Where to find a built PFLARE (exported by the top Makefile, defaulted
# here so make can also be called directly in tests/)
PFLARE_DIR ?= /home/sdargavi/projects/PFLARE

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

# Link to pflare - order is important
# pflare must be first for static linking
# (Phase 1 adds -L$(LIBDIR) -lubolt in front)
LDFLAGS += -Wl,-rpath,$(PFLARE_DIR)/lib
LDLIBS  := -L$(PFLARE_DIR)/lib -lpflare $(LDLIBS)

EXISTING_LD := $(LD_LIBRARY_PATH)
EXISTING_DYLD := $(DYLD_FALLBACK_LIBRARY_PATH)

ifeq ($(shell uname -s 2>/dev/null),Darwin)
  export DYLD_FALLBACK_LIBRARY_PATH=$(PFLARE_DIR)/lib$(if $(EXISTING_DYLD),:$(EXISTING_DYLD))
else
  export LD_LIBRARY_PATH=$(PFLARE_DIR)/lib$(if $(EXISTING_LD),:$(EXISTING_LD))
endif

clean::
	$(RM) -f $(TEST_TARGETS)

# ~~~~~~~~~~~
# Run UBOLT tests
#
# Pass/fail is the process exit code. Until the drivers check
# KSPGetConvergedReason themselves (Phase 1), every recipe pins
# -ksp_max_it to the baseline iteration count and adds
# -ksp_error_if_not_converged -on_error_abort so an iteration-count
# regression aborts the run with a non-zero exit code.
# Baseline counts live in ../docs/dev/testing.md and tests/baselines/
# ~~~~~~~~~~~

FAIL_NOT_CONVERGED := -ksp_error_if_not_converged -on_error_abort

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_check:
#
	@echo ""
	@echo "Check 1D slab streaming/removal + matrix-free scatter with AIRG"
	./slab_1dk -max_exponent 2 -ksp_max_it 10 $(FAIL_NOT_CONVERGED)
	@echo "Check 1D slab preconditioned with streaming only in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -ksp_max_it 16 $(FAIL_NOT_CONVERGED)

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_short_serial:
#
	@echo ""
	@echo "Test 1D slab, streaming/removal pmat, strong removal"
	./slab_1dk -max_exponent 2 -ksp_max_it 10 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, streaming-only pmat, strong removal"
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -ksp_max_it 16 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, streaming/removal pmat, weak removal (advection dominated)"
	./slab_1dk -max_exponent 0 -ksp_max_it 1 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, diagonally scaled"
	./slab_1dk -diag_scale -max_exponent 0 -ksp_max_it 1 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, option prefix reaches PCAIR at composite index 1"
	./slab_1dk -max_exponent 2 -ksp_max_it 10 -sub_1_pc_air_print_stats_timings $(FAIL_NOT_CONVERGED)

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_short_parallel:
#
	@echo ""
	@echo "Test 1D slab, streaming/removal pmat, strong removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 2 -ksp_max_it 10 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, streaming/removal pmat, weak removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 -ksp_max_it 1 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, streaming-only pmat, strong removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -ksp_max_it 16 $(FAIL_NOT_CONVERGED)

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_serial:
#
	@echo ""
	@echo "Test 1D slab, streaming-only pmat, weak removal"
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 -ksp_max_it 1 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, diagonally scaled, streaming-only pmat"
	./slab_1dk -diag_scale -precon_stream -ksp_pc_side right -max_exponent 0 -ksp_max_it 3 $(FAIL_NOT_CONVERGED)

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_parallel:
#
	@echo ""
	@echo "Test 1D slab, weak removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 -ksp_max_it 1 $(FAIL_NOT_CONVERGED)
	@echo "Test 1D slab, diagonally scaled in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -diag_scale -max_exponent 0 -ksp_max_it 1 $(FAIL_NOT_CONVERGED)

# ~~~~~~~~~~~
# Re-capture reference baselines into baselines/
# Only do this deliberately - see ../docs/dev/testing.md
# The full matrix: {default pc, precon_stream} x {max_exponent 0, 2}
#                  x {np 1, 2} x {diag_scale off, on}
# Capped at 200 iterations: two diag_scale configs are pathological under the
# current code (6305 its / divergence) and a 200-iteration residual history is
# a strong enough fingerprint for refactor comparison
# ~~~~~~~~~~~
BASE := -ksp_monitor -ksp_converged_reason -ksp_max_it 200
capture_baselines:
	./slab_1dk -max_exponent 0 $(BASE) > baselines/slab1d_default_me0_np1.log
	./slab_1dk -max_exponent 2 $(BASE) > baselines/slab1d_default_me2_np1.log
	./slab_1dk -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_default_me0_np1_ds.log
	./slab_1dk -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_default_me2_np1_ds.log
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 $(BASE) > baselines/slab1d_stream_me0_np1.log
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 $(BASE) > baselines/slab1d_stream_me2_np1.log
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_stream_me0_np1_ds.log
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_stream_me2_np1_ds.log
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 $(BASE) > baselines/slab1d_default_me0_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 2 $(BASE) > baselines/slab1d_default_me2_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_default_me0_np2_ds.log
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_default_me2_np2_ds.log
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 $(BASE) > baselines/slab1d_stream_me0_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 $(BASE) > baselines/slab1d_stream_me2_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_stream_me0_np2_ds.log
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_stream_me2_np2_ds.log
	@echo "Baselines captured in baselines/"
