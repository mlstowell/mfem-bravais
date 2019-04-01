# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443271. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM-Bravais library. For more information and
# source code availability see http://https://github.com/mlstowell/mfem-bravais.
#
# MFEM-Bravais is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

define BRAVAIS_HELP_MSG

MFEM-Bravais makefile targets:

   make
   make status/info
   make install
   make clean
   make distclean
   make style

Examples:

make -j 4
   Build MFEM-Bravais using the current configuration options from MFEM.
   (MFEM-Bravais requires the MFEM finite element library, and uses its
    compiler and linker options in its build process.)
make status
   Display information about the current configuration.
make install PREFIX=<dir>
   Install the executables in <dir>.
make clean
   Clean the executables, library and object files.
make distclean
   In addition to "make clean", remove the local installation directory and some
   run-time generated files.
make style
   Format the MFEM-Bravais C++ source files using the Artistic Style (astyle)
   settings from MFEM.

endef

# Save the MAKEOVERRIDES for cases where we explicitly want to pass the command
# line overrides to sub-make:
override MAKEOVERRIDES_SAVE := $(MAKEOVERRIDES)
# Do not pass down variables from the command-line to sub-make:
MAKEOVERRIDES =

# Path to the mfem source directory, defaults to this makefile's directory:
THIS_MK := $(lastword $(MAKEFILE_LIST))
$(if $(wildcard $(THIS_MK)),,$(error Makefile not found "$(THIS_MK)"))
BRAVAIS_DIR ?= $(patsubst %/,%,$(dir $(THIS_MK)))
BRAVAIS_REAL_DIR := $(realpath $(BRAVAIS_DIR))
$(if $(BRAVAIS_REAL_DIR),,$(error Source directory "$(BRAVAIS_DIR)" is not valid))
SRC := $(if $(BRAVAIS_REAL_DIR:$(CURDIR)=),$(BRAVAIS_DIR)/,)
$(if $(word 2,$(SRC)),$(error Spaces in SRC = "$(SRC)" are not supported))

BRAVAIS_GIT_STRING = $(shell [ -d $(BRAVAIS_DIR)/.git ] && \
   git -C $(BRAVAIS_DIR) \
   describe --all --long --abbrev=40 --dirty --always 2> /dev/null)

# Custom configuration flags
BRAVAIS_CONFIG_MK ?=
-include $(BRAVAIS_CONFIG_MK)

# Default installation location
PREFIX ?= ./bin
INSTALL ?= /usr/bin/install

# Archiver: AR and ARFLAGS are defined by default, RANLIB is not.
# The default value of AR is 'ar' and we do not want to change that.
# The default value of ARFLAGS is 'rv', however, we want to set a different
# default, so we modify ARFLAGS, unless it was already changed on the command
# line or in the configuration file $(BRAVAIS_CONFIG_MK).
ifeq ($(origin ARFLAGS),default)
   ARFLAGS = cruv
endif
RANLIB ?= ranlib

# Use the MFEM build directory
#MFEM_DIR ?= ../mfem
MFEM_DIR ?= ../dev/pyramid-dev
CONFIG_MK ?= $(MFEM_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_DIR = ../mfem/mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk

# Use two relative paths to MFEM: first one for compilation in '.' and second
# one for compilation in 'lib'.
MFEM_DIR1 := $(MFEM_DIR)
MFEM_DIR2 := $(realpath $(MFEM_DIR))

# Use the compiler used by MFEM. Get the compiler and the options for compiling
# and linking from MFEM's config.mk. (Skip this if the target does not require
# building.)
ifeq (,$(filter help clean distclean style,$(MAKECMDGOALS)))
   -include $(CONFIG_MK)
endif

CXX = $(MFEM_CXX)
CPPFLAGS = $(MFEM_CPPFLAGS)
CXXFLAGS = $(MFEM_CXXFLAGS)

# MFEM config does not define C compiler
CC     ?= gcc
CFLAGS ?= -O3

# Optional compile/link flags
BRAVAIS_OPTS ?=
BRAVAIS_LDFLAGS ?=

OPTIM_OPTS = -O3
DEBUG_OPTS = -g -Wall
BRAVAIS_DEBUG ?= $(MFEM_DEBUG)
ifneq ($(BRAVAIS_DEBUG),$(MFEM_DEBUG))
   ifeq ($(BRAVAIS_DEBUG),YES)
      CXXFLAGS = $(DEBUG_OPTS)
   else
      CXXFLAGS = $(OPTIM_OPTS)
   endif
endif

BRAVAIS_FLAGS = $(CPPFLAGS) $(CXXFLAGS) $(MFEM_INCFLAGS) $(BRAVAIS_OPTS)
BRAVAIS_LIBS = $(MFEM_LIBS) -lmfem-extras

ifeq ($(BRAVAIS_DEBUG),YES)
   BRAVAIS_FLAGS += -DBRAVAIS_DEBUG
endif

NOTMAC := $(subst Darwin,,$(shell uname -s))
SO_EXT = $(if $(NOTMAC),so,dylib)

# Default multisampling mode and multisampling line-width
BRAVAIS_MULTISAMPLE  ?= 4
BRAVAIS_MS_LINEWIDTH ?= $(if $(NOTMAC),1.4,0.01)
BRAVAIS_FLAGS += -DBRAVAIS_MULTISAMPLE=$(BRAVAIS_MULTISAMPLE)\
 -DBRAVAIS_MS_LINEWIDTH=$(BRAVAIS_MS_LINEWIDTH)

# Macro that searches for a file in a list of directories returning the first
# directory that contains the file.
# $(1) - the file to search for
# $(2) - list of directories to search
define find_dir
$(patsubst %/$(1),%,$(firstword $(wildcard $(foreach d,$(2),$(d)/$(1)))))
endef

PTHREAD_LIB = -lpthread
BRAVAIS_LIBS += $(PTHREAD_LIB)

LIBS = $(strip $(BRAVAIS_LIBS) $(BRAVAIS_LDFLAGS))
CCC  = $(strip $(CXX) $(BRAVAIS_FLAGS))
Ccc  = $(strip $(CC) $(CFLAGS))

EXAMPLE_SUBDIRS = 
EXAMPLE_DIRS := examples $(addprefix examples/,$(EXAMPLE_SUBDIRS))
EXAMPLE_TEST_DIRS := examples

MINIAPP_SUBDIRS = common electromagnetics meshing performance tools nurbs
MINIAPP_DIRS := $(addprefix miniapps/,$(MINIAPP_SUBDIRS))
MINIAPP_TEST_DIRS := $(filter-out %/common,$(MINIAPP_DIRS))
MINIAPP_USE_COMMON := $(addprefix miniapps/,electromagnetics tools)

EM_DIRS = $(EXAMPLE_DIRS) $(MINIAPP_DIRS)

# Use BUILD_DIR on the command line; set BRAVAIS_BUILD_DIR before including this
# makefile or config/config.mk from a separate $(BUILD_DIR).
BRAVAIS_BUILD_DIR ?= .
BUILD_DIR := $(BRAVAIS_BUILD_DIR)
BUILD_REAL_DIR := $(abspath $(BUILD_DIR))
ifneq ($(BUILD_REAL_DIR),$(BRAVAIS_REAL_DIR))
   BUILD_SUBDIRS = $(DIRS) config $(EM_DIRS) doc $(TEST_DIRS)
   BUILD_DIR_DEF = -DMFEM_BUILD_DIR="$(BUILD_REAL_DIR)"
   BLD := $(if $(BUILD_REAL_DIR:$(CURDIR)=),$(BUILD_DIR)/,)
   $(if $(word 2,$(BLD)),$(error Spaces in BLD = "$(BLD)" are not supported))
else
   BUILD_DIR = $(BRAVAIS_DIR)
   BLD := $(SRC)
endif
BRAVAIS_BUILD_DIR := $(BUILD_DIR)

# generated with 'echo lib/*.c*'
SOURCE_FILES = lib/bravais.cpp
OBJECT_FILES1 = $(SOURCE_FILES:.cpp=.o)
OBJECT_FILES = $(OBJECT_FILES1:.c=.o)
# generated with 'echo lib/*.h*'
HEADER_FILES = lib/bravais.hpp

# Targets

.PHONY: lib clean distclean install status info opt debug style

.SUFFIXES: .c .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)
.c.o:
	cd $(<D); $(Ccc) -c $(<F)

# Default rule.
lib: lib/libbravais.a
#lib: $(if $(static),$(BLD)libbravais.a) \
#        $(if $(shared),$(BLD)libbravais.$(SO_EXT))

# Rules for compiling all source files.
$(OBJECT_FILES): $(BLD)%.o: $(SRC)%.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(BRAVAIS_FLAGS) -c $(<) -o $(@)

$(BLD)libbravais.a: $(OBJECT_FILES)
	$(AR) $(ARFLAGS) $(@) $(OBJECT_FILES)
	$(RANLIB) $(@)

$(BLD)libbravais.$(SO_EXT): $(BLD)libbravais.$(SO_VER)
	cd $(@D) && ln -sf $(<F) $(@F)

periodic-bravais-mesh: meshing/periodic-bravais-mesh.cpp lib/libbravais.a $(MFEM_LIB_FILE)
	$(CCC) -o periodic-bravais-mesh meshing/periodic-bravais-mesh.cpp -Llib -lbravais $(LIBS)

display-bravais: meshing/display-bravais.cpp lib/libbravais.a $(MFEM_LIB_FILE)
	$(CCC) -o display-bravais meshing/display-bravais.cpp -Llib -lbravais $(LIBS)

glvis: override MFEM_DIR = $(MFEM_DIR1)
glvis:	glvis.cpp lib/libbravais.a $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(CCC) -o glvis glvis.cpp -Llib -lglvis $(LIBS)

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
ifeq (,$(and $(findstring B,$(MAKEFLAGS)),$(wildcard $(CONFIG_MK))))
	$(error The MFEM library is not built)
endif

opt:
	$(MAKE) "BRAVAIS_DEBUG=NO"

debug:
	$(MAKE) "BRAVAIS_DEBUG=YES"

$(OBJECT_FILES): override MFEM_DIR = $(MFEM_DIR2)
$(OBJECT_FILES): $(HEADER_FILES) $(CONFIG_MK)

lib/libbravais.a: $(OBJECT_FILES)
	cd lib;	$(AR) $(ARFLAGS) libbravais.a *.o; $(RANLIB) libbravais.a

clean:
	rm -rf lib/*.o lib/*~ *~ glvis lib/libbravais.a *.dSYM

distclean: clean
	rm -rf bin/

install: glvis
	mkdir -p $(PREFIX)
	$(INSTALL) -m 750 glvis $(PREFIX)

help:
	$(info $(value BRAVAIS_HELP_MSG))
	@true

status info:
	$(info MFEM_DIR    = $(MFEM_DIR))
	$(info BRAVAIS_FLAGS = $(BRAVAIS_FLAGS))
	$(info BRAVAIS_LIBS  = $(value BRAVAIS_LIBS))
	$(info BRAVAIS_LIBS  = $(BRAVAIS_LIBS))
	$(info PREFIX      = $(PREFIX))
	@true

ASTYLE = astyle --options=$(MFEM_DIR1)/config/mfem.astylerc
ALL_FILES = ./glvis.cpp $(SOURCE_FILES) $(HEADER_FILES)
EXT_FILES = 
FORMAT_FILES := $(filter-out $(EXT_FILES), $(ALL_FILES))

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
