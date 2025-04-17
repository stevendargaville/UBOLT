# ~~~~~~~~~~~~~~~~~
# UBOLT - Steven Dargaville
# Makefile for UBOLT
#
# Must have defined PETSC_DIR and PETSC_ARCH before calling
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Check PETSc version is at least 3.23.0
PETSC_VERSION_MIN := $(shell ${PETSC_DIR}/lib/petsc/bin/petscversion ge 3.23)
ifeq ($(PETSC_VERSION_MIN),0)
$(error PETSc version is too old. UBOLT requires at least version 3.23.0)
endif

PFLARE_DIR := /home/sdargavi/projects/PFLARE
export CXXFLAGS:=${CXXFLAGS} -I${PFLARE_DIR}/include
export LDLIBS:=${LDLIBS} -Wl,-rpath,${PFLARE_DIR}/lib -L${PFLARE_DIR}/lib -lpflare

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~

# Library to output
OUT := UBOLTk

# All the files required by UBOLT
OBJS := UBOLTk.o

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.DEFAULT_GOAL := all		  	
all: $(OUT)

# Cleanup
clean::
	$(RM) *.mod
	$(RM) *.o
	$(RM) UBOLTk