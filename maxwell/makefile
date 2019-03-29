# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Use the MFEM build directory
MFEM_DIR = ../..
CONFIG_MK = $(MFEM_DIR)/config/config.mk
TEST_MK = $(MFEM_DIR)/config/test.mk
# Use the MFEM install directory
# MFEM_DIR = ../../mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS =
PAR_MINIAPPS = \
	maxwell_dispersion maxwell_dispersion_amr maxwell_dispersion_amr_eh \
	maxwell_dispersion_uniform_amr maxwell_dispersion_shift \
	maxwell_dispersion_krylov maxwell_homogenization \
	thermal_resistivity meta_material
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

COMMON_HPP=../common/pfem_extras.hpp ../common/bravais.hpp
COMMON_O=$(COMMON_HPP:.hpp=.o)

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Replace the default implicit rule for *.cpp files
%: %.cpp %_solver.o $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $@_solver.o $(COMMON_O) $(MFEM_LIBS)
#%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) ../common/pfem_extras.o bravais.o
#	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ ../common/pfem_extras.o bravais.o $(MFEM_LIBS)

# Replace the default implicit rule for *.cpp files
%.o: %.cpp %.hpp $(COMMON_HPP) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) \
	-I/Users/stowell1/Projects/MFEM/EVSLv1.1.0/EVSL_1.1.0/INC \
	-I/Users/stowell1/Projects/MFEM/EVSLv1.1.0/EVSL_1.1.0/EXTERNAL \
	-c $(<) -o $(@)

# maxwell_bloch.o: maxwell_bloch.cpp maxwell_bloch.hpp ../common/pfem_extras.hpp \
# 	$(CONFIG_MK) $(MFEM_LIB_FILE)
# 	$(MFEM_CXX) -c $(MFEM_FLAGS) maxwell_bloch.cpp -o $@ $(MFEM_LIBS)

# maxwell_bloch_new.o: maxwell_bloch_new.cpp maxwell_bloch_new.hpp ../common/pfem_extras.hpp \
# 	$(CONFIG_MK) $(MFEM_LIB_FILE)
# 	$(MFEM_CXX) -c $(MFEM_FLAGS) maxwell_bloch_new.cpp -o $@ $(MFEM_LIBS)

# bravais.o: bravais.cpp bravais.hpp \
# 	$(CONFIG_MK) $(MFEM_LIB_FILE)
# 	$(MFEM_CXX) -c $(MFEM_FLAGS) bravais.cpp -o $@ $(MFEM_LIBS)

vector3d: vector3d.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) $(COMMON_O) 
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ $(COMMON_O) $(MFEM_LIBS)

write_matrices: write_matrices.cpp $(CONFIG_MK) $(COMMON_O) $(MFEM_LIB_FILE) 
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ $(COMMON_O) $(MFEM_LIBS)

maxwell_dispersion: maxwell_dispersion.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) maxwell_bloch.o $(COMMON_O)
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch.o $(COMMON_O) $(MFEM_LIBS)
#	/Users/stowell1/Projects/MFEM/EVSLv1.1.0/EVSL_1.1.0/EXTERNAL/evsl_suitesparse.o \
#	-L/Users/stowell1/Projects/MFEM/EVSLv1.1.0/EVSL_1.1.0 -levsl \
#	-L/Users/stowell1/Projects/MFEM/SuiteSparse/UMFPACK/Lib -lumfpack -L/Users/stowell1/Projects/MFEM/SuiteSparse/SuiteSparse_config -lsuitesparseconfig -L/Users/stowell1/Projects/MFEM/SuiteSparse/CHOLMOD/Lib -lcholmod -L/Users/stowell1/Projects/MFEM/SuiteSparse/AMD/Lib/ -lamd -L/Users/stowell1/Projects/MFEM/SuiteSparse/COLAMD/Lib -lcolamd -L/Users/stowell1/Projects/MFEM/SuiteSparse/CCOLAMD/Lib -lccolamd -L/Users/stowell1/Projects/MFEM/SuiteSparse/CAMD/Lib -lcamd

maxwell_dispersion_amr: maxwell_dispersion_amr.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) $(COMMON_O) maxwell_bloch_amr.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch_amr.o $(COMMON_O) $(MFEM_LIBS)

maxwell_dispersion_uniform_amr: maxwell_dispersion_uniform_amr.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) $(COMMON_O) maxwell_bloch_uniform_amr.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch_uniform_amr.o $(COMMON_O) $(MFEM_LIBS)

maxwell_dispersion_amr_eh: maxwell_dispersion_amr_eh.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) $(COMMON_O) maxwell_bloch_amr_eh.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch_amr_eh.o $(COMMON_O) $(MFEM_LIBS)

maxwell_dispersion_shift: maxwell_dispersion_shift.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) $(COMMON_O) maxwell_bloch_shift.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch_shift.o $(COMMON_O) $(MFEM_LIBS)

maxwell_dispersion_krylov: maxwell_dispersion_krylov.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) $(COMMON_O) maxwell_bloch_krylov.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch_krylov.o $(COMMON_O) $(MFEM_LIBS)

maxwell_dispersion_new: maxwell_dispersion_new.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) maxwell_bloch_new.o bravais.o ../common/pfem_extras.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch_new.o $(COMMON_O) $(MFEM_LIBS)

maxwell_homogenization: maxwell_homogenization.cpp $(CONFIG_MK) $(COMMON_O) maxwell_bloch.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ maxwell_bloch.o $(COMMON_O) $(MFEM_LIBS)

test_bravais: test_bravais.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) $(COMMON_O)
	$(MFEM_CXX) $(MFEM_FLAGS) -I../common  $(@).cpp -o $@ $(COMMON_O) $(MFEM_LIBS)

#meta_material: meta_material.cpp $(CONFIG_MK) $(COMMON_O) $(MFEM_LIB_FILE)
#	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ $(COMMON_O) $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ scalar3d vector3d
	rm -f maxwell_dispersion maxwell_homogenization
	rm -r thermal_resistivity meta_material
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	rm -f refined.mesh displaced.mesh mesh.*
	rm -f sphere_refined.* sol.* sol_u.* sol_p.*
	rm -f deformed.* velocity.* elastic_energy.*
	rm -rf *.dSYM
