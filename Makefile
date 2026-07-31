# ~~~~~~~~~~~~~~~~~
# UBOLT - Steven Dargaville
# Makefile for UBOLT
#
# Must have defined PETSC_DIR and PETSC_ARCH before calling
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
#
# PETSc must be configured with Kokkos - UBOLT is Kokkos-mandatory
# Requires a built PFLARE - override with PFLARE_DIR=/path/to/PFLARE
# ~~~~~~~~~~~~~~~~~

# Check PETSc version is at least 3.25.0
PETSC_VERSION_MIN := $(shell ${PETSC_DIR}/lib/petsc/bin/petscversion ge 3.25)
ifeq ($(PETSC_VERSION_MIN),0)
$(error PETSc version is too old. UBOLT requires at least version 3.25.0)
endif

# Where to find a built PFLARE
export PFLARE_DIR ?= /home/sdargavi/projects/PFLARE

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

# Directories we want
INCLUDEDIR  := include
SRCDIR      := src
export LIBDIR := $(CURDIR)/lib

# Include directories
INCLUDE := -I$(CURDIR) -I$(INCLUDEDIR) -I$(PFLARE_DIR)/include

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

# ~~~~~~~~~~~~~~~~~~~~~~~~
# Check if petsc has been configured with various options
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Read petscconf.h via awk (portable on macOS)
define _have_conf
$(shell awk '/^[[:space:]]*#define[[:space:]]+$(1)[[:space:]]+1/{print 1; exit}' $(PETSCCONF_H))
endef

export PETSC_USE_SHARED_LIBRARIES := $(if $(call _have_conf,PETSC_USE_SHARED_LIBRARIES),1,0)
export PETSC_HAVE_KOKKOS := $(if $(call _have_conf,PETSC_HAVE_KOKKOS),1,0)
# Detect if PETSc was configured without MPI
export PETSC_HAVE_MPIUNI := $(if $(call _have_conf,PETSC_HAVE_MPIUNI),1,0)

# UBOLT is Kokkos-mandatory: assembly and matrix-free terms are Kokkos kernels
# and pflare's GPU path dispatches on MATAIJKOKKOS
ifeq ($(PETSC_HAVE_KOKKOS),0)
$(error PETSc has not been configured with Kokkos. UBOLT requires Kokkos)
endif

# To prevent overlinking with conda builds, only explicitly link
# to the libraries we use in ubolt
ifeq ($(CONDA_BUILD),1)
	PETSC_LINK_LIBS = -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc ${BLASLAPACK_LIB} ${KOKKOS_LIB} ${KOKKOS_KERNELS_LIB}
# Otherwise just use everything petsc uses to be safe
else
	PETSC_LINK_LIBS = $(LDLIBS)
endif

# On macOS, strip any -Wl,-rpath,* when linking the shared library to avoid duplicate LC_RPATH
ifeq ($(shell uname -s 2>/dev/null),Darwin)
PETSC_LINK_LIBS_NORPATH := $(strip $(foreach w,$(PETSC_LINK_LIBS),$(if $(findstring -Wl,-rpath,$(w)),,$(w))))
else
PETSC_LINK_LIBS_NORPATH := $(PETSC_LINK_LIBS)
endif

# ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~

# All the files required by libubolt
# Every translation unit is a Kokkos one: the Xk.o objects come from
# Xk.kokkos.cxx via PETSc's Kokkos build rules
OBJS := $(SRCDIR)/sn_quadraturek.o \
		  $(SRCDIR)/structured_fd_1dk.o \
		  $(SRCDIR)/termsk.o \
		  $(SRCDIR)/transport_operatork.o \
		  $(SRCDIR)/transport_solverk.o

# Define a variable containing all the tests
export TEST_TARGETS = slab_1dk
# Define a variable containing all the tests that the make check runs
export CHECK_TARGETS = slab_1dk

# Output the library - either static or dynamic
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
OUT = $(LIBDIR)/libubolt.a
else
# mac osx name is different
ifeq ($(shell uname -s 2>/dev/null),Darwin)
OUT = $(LIBDIR)/libubolt.dylib
else
OUT = $(LIBDIR)/libubolt.so
endif
endif
export OUT

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.DEFAULT_GOAL := all
.PHONY: all
all: $(OUT)

# Create our directory structure and build the library
# (either static or dynamic depending on what petsc was configured with)
# Linked with the Kokkos linker as libubolt contains Kokkos kernels
$(OUT): $(OBJS)
	@mkdir -p $(LIBDIR)
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
	$(AR) $(AR_FLAGS) $(OUT) $(OBJS)
	$(RANLIB) $(OUT)
else
ifeq ($(shell uname -s 2>/dev/null),Darwin)
# macOS: Use -dynamiclib and set a relocatable @rpath install_name. Do not embed rpaths.
	$(LINK.kokkos.cxx) -dynamiclib -o $(OUT) $(OBJS) $(PETSC_LINK_LIBS_NORPATH) -install_name @rpath/$(notdir $(OUT))
else
# Linux: Use -shared and set the soname.
	$(LINK.kokkos.cxx) -shared -o $(OUT) $(OBJS) $(PETSC_LINK_LIBS) -Wl,-soname,$(notdir $(OUT))
endif
endif

# Build the tests (in parallel)
.PHONY: build_tests
build_tests: $(OUT)
	+$(MAKE) -C tests $(TEST_TARGETS)

# Build the tests used in the check
.PHONY: build_tests_check
build_tests_check: $(OUT)
	+$(MAKE) -C tests $(CHECK_TARGETS)

.PHONY: tests_short_serial
tests_short_serial: build_tests
	$(MAKE) -C tests run_tests_short_serial

.PHONY: tests_short_parallel
tests_short_parallel: build_tests
ifeq ($(PETSC_HAVE_MPIUNI),0)
	$(MAKE) -C tests run_tests_short_parallel
endif

# Very quick tests
.PHONY: tests_short
tests_short: build_tests
	$(MAKE) tests_short_serial
	$(MAKE) tests_short_parallel

# Build and run all the tests
.PHONY: tests
tests: build_tests
	($(MAKE) tests_short || (echo "Short tests failed" && exit 1)) && \
	($(MAKE) -C tests run_tests_serial || (echo "Serial tests failed" && exit 1)) && \
	(if [ "$(PETSC_HAVE_MPIUNI)" = "0" ]; then $(MAKE) -C tests run_tests_parallel; else true; fi || (echo "Parallel tests failed" && exit 1)) && \
	echo "All tests passed: OK"

# A quick sanity check with simple tests
.PHONY: check
check: build_tests_check
	@$(MAKE) --no-print-directory -C tests run_check

# Re-capture the reference baselines in tests/baselines
# (see docs/dev/testing.md before doing this)
.PHONY: baselines
baselines: build_tests
	$(MAKE) -C tests capture_baselines

# Cleanup
clean::
	$(RM) -r $(LIBDIR)
	$(RM) $(SRCDIR)/*.o
	$(MAKE) -C tests clean
