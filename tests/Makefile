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
	./slab_1dk -max_exponent 2 -ksp_max_it 10
	@echo "Check 1D slab preconditioned with streaming only in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -ksp_max_it 16

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_short_serial:
#
	@echo ""
	@echo "Test 1D slab, streaming/removal pmat, strong removal"
	./slab_1dk -max_exponent 2 -ksp_max_it 10
	@echo "Test 1D slab, streaming-only pmat, strong removal"
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -ksp_max_it 16
	@echo "Test 1D slab, streaming/removal pmat, weak removal (advection dominated)"
	./slab_1dk -max_exponent 0 -ksp_max_it 1
	@echo "Test 1D slab, diagonally scaled"
	./slab_1dk -diag_scale -max_exponent 0 -ksp_max_it 1
	@echo "Test 1D slab, option prefix reaches PCAIR at composite index 1"
	./slab_1dk -max_exponent 2 -ksp_max_it 10 -sub_1_pc_air_print_stats_timings
	@echo "Test 1D slab, runtime phase space sizes (default n_cells/n_angles/length)"
	./slab_1dk -n_cells 1000 -n_angles 4 -length 1.0 -max_exponent 2 -ksp_max_it 10
	@echo "Test 1D slab, S2 on a coarser slab"
	./slab_1dk -n_cells 500 -n_angles 2 -length 2.0 -max_exponent 2 -ksp_max_it 14
	@echo "Test 1D slab, one MatSetValuesCOO per term instead of a single summed one"
	./slab_1dk -max_exponent 2 -ksp_max_it 10 -ubolt_coo_two_call

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_short_parallel:
#
	@echo ""
	@echo "Test 1D slab, streaming/removal pmat, strong removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 2 -ksp_max_it 10
	@echo "Test 1D slab, streaming/removal pmat, weak removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 -ksp_max_it 1
	@echo "Test 1D slab, streaming-only pmat, strong removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -ksp_max_it 16
	@echo "Test 1D slab, S2 on a coarser slab in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -n_cells 500 -n_angles 2 -length 2.0 -max_exponent 2 -ksp_max_it 14
	@echo "Test 1D slab, one MatSetValuesCOO per term in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 2 -ksp_max_it 10 -ubolt_coo_two_call

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_serial:
#
	@echo ""
	@echo "Test 1D slab, streaming-only pmat, weak removal"
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 -ksp_max_it 1
	@echo "Test 1D slab, diagonally scaled, streaming-only pmat"
	./slab_1dk -diag_scale -precon_stream -ksp_pc_side right -max_exponent 0 -ksp_max_it 3

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_parallel:
#
	@echo ""
	@echo "Test 1D slab, weak removal in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 -ksp_max_it 1
	@echo "Test 1D slab, diagonally scaled in parallel"
	$(MPIEXEC) -n 2 ./slab_1dk -diag_scale -max_exponent 0 -ksp_max_it 1

# ~~~~~~~~~~~
# Re-capture reference baselines into baselines/
# Only do this deliberately - see ../docs/dev/testing.md
# The full matrix: {default pc, precon_stream} x {max_exponent 0, 2}
#                  x {np 1, 2} x {diag_scale off, on}
# Capped at 200 iterations: two diag_scale configs are pathological under the
# current code (6305 its / divergence) and a 200-iteration residual history is
# a strong enough fingerprint for refactor comparison
#
# The four me=2 diag_scale captures do not converge, so the driver exits 1 on
# them by design - they get a || true so the capture runs to completion
# ~~~~~~~~~~~
BASE := -ksp_monitor -ksp_converged_reason -ksp_max_it 200
capture_baselines:
	./slab_1dk -max_exponent 0 $(BASE) > baselines/slab1d_default_me0_np1.log
	./slab_1dk -max_exponent 2 $(BASE) > baselines/slab1d_default_me2_np1.log
	./slab_1dk -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_default_me0_np1_ds.log
	./slab_1dk -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_default_me2_np1_ds.log || true
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 $(BASE) > baselines/slab1d_stream_me0_np1.log
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 $(BASE) > baselines/slab1d_stream_me2_np1.log
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_stream_me0_np1_ds.log
	./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_stream_me2_np1_ds.log || true
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 $(BASE) > baselines/slab1d_default_me0_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 2 $(BASE) > baselines/slab1d_default_me2_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_default_me0_np2_ds.log
	$(MPIEXEC) -n 2 ./slab_1dk -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_default_me2_np2_ds.log || true
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 $(BASE) > baselines/slab1d_stream_me0_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 $(BASE) > baselines/slab1d_stream_me2_np2.log
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 0 -diag_scale $(BASE) > baselines/slab1d_stream_me0_np2_ds.log
	$(MPIEXEC) -n 2 ./slab_1dk -precon_stream -ksp_pc_side right -max_exponent 2 -diag_scale $(BASE) > baselines/slab1d_stream_me2_np2_ds.log || true
	@echo "Baselines captured in baselines/"
