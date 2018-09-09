# The MIT License (MIT)
#
# Copyright (c) 2014-2018 David Medina and Tim Warburton
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

rmSlash = $(patsubst %/,%,$1)

OCCA_DIR := $(call rmSlash,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJ_DIR := $(OCCA_DIR)

# Clear the list of suffixes
.SUFFIXES:
.SUFFIXES: .hpp .h .tpp .cpp .o .so
# Cancel some implicit rules
%:   %.o
%.o: %.cpp
%:   %.cpp

include $(OCCA_DIR)/scripts/Makefile

#---[ Working Paths ]-----------------------------
ifeq ($(usingWinux),0)
  compilerFlags  += $(picFlag)
  fCompilerFlags += $(picFlag)
else
  sharedFlag += $(picFlag)
endif

# [-L$OCCA_DIR/lib -locca] are kept for applications
#   using $OCCA_DIR/scripts/Makefile
paths += -I$(srcPath)
paths := $(filter-out -L$(OCCA_DIR)/lib,$(paths))
linkerFlags := $(filter-out -locca,$(linkerFlags))
#=================================================


#---[ Flags ]-------------------------------------
# Expand and overwrite flag variables to avoid
# multiple expansions which are slow.
compilerFlags := $(compilerFlags)
flags := $(flags)
sharedFlag := $(sharedFlag)
pthreadFlag := $(pthreadFlag)
#=================================================


#---[ variables ]---------------------------------
srcToObject  = $(subst $(PROJ_DIR)/src,$(PROJ_DIR)/obj,$(1:.cpp=.o))

dontCompile = $(OCCA_DIR)/src/core/kernelOperators.cpp $(OCCA_DIR)/src/tools/runFunction.cpp

sources     = $(realpath $(shell find $(PROJ_DIR)/src -type f -name '*.cpp'))
sources    := $(filter-out $(dontCompile),$(sources))
headers     = $(realpath $(shell find $(PROJ_DIR)/include -type f -name '*.hpp'))
testSources = $(realpath $(shell find $(PROJ_DIR)/tests/src -type f -name '*.cpp'))
tests       = $(subst $(testPath)/src,$(testPath)/bin,$(testSources:.cpp=))

objects = $(call srcToObject,$(sources))

outputs = $(libPath)/libocca.so $(binPath)/occa

ifndef VALGRIND_ENABLED
  testFlags = $(compilerFlags) -fsanitize=address -fno-omit-frame-pointer
else
  testFlags = $(compilerFlags)
endif
#=================================================


#---[ Compile Library ]---------------------------
# Setup compiled defines and force rebuild if defines changed
COMPILED_DEFINES     := $(OCCA_DIR)/include/occa/defines/compiledDefines.hpp
NEW_COMPILED_DEFINES := $(OCCA_DIR)/.compiledDefines

MAKE_COMPILED_DEFINES := $(shell cat "$(OCCA_DIR)/scripts/compiledDefinesTemplate.hpp" | \
                                 sed "s,@@OCCA_OS@@,$(OCCA_OS),g;\
                                      s,@@OCCA_USING_VS@@,$(OCCA_USING_VS),g;\
                                      s,@@OCCA_UNSAFE@@,$(OCCA_UNSAFE),g;\
                                      s,@@OCCA_MPI_ENABLED@@,$(OCCA_MPI_ENABLED),g; \
                                      s,@@OCCA_OPENMP_ENABLED@@,$(OCCA_OPENMP_ENABLED),g;\
                                      s,@@OCCA_OPENCL_ENABLED@@,$(OCCA_OPENCL_ENABLED),g;\
                                      s,@@OCCA_CUDA_ENABLED@@,$(OCCA_CUDA_ENABLED),g;\
                                      s,@@OCCA_HIP_ENABLED@@,$(OCCA_HIP_ENABLED),g;" > "$(NEW_COMPILED_DEFINES)")

MAKE_COMPILED_DEFINES := $(shell \
 [ ! -f "$(COMPILED_DEFINES)" -o -n "$(shell diff -q $(COMPILED_DEFINES) $(NEW_COMPILED_DEFINES))" ] \
 && cp "$(NEW_COMPILED_DEFINES)" "$(COMPILED_DEFINES)" \
)
MAKE_COMPILED_DEFINES := $(shell rm $(NEW_COMPILED_DEFINES))

all: $(objects) $(outputs)
	@(. $(OCCA_DIR)/include/occa/scripts/shellTools.sh && installOcca)
	@echo "> occa info"
	@$(OCCA_DIR)/bin/occa info

$(COMPILED_DEFINES_CHANGED):
#=================================================


#---[ Builds ]------------------------------------
#  ---[ libocca ]-------------
$(libPath)/libocca.so:$(objects) $(headers) $(COMPILED_DEFINES)
	mkdir -p $(libPath)
	$(compiler) $(compilerFlags) $(sharedFlag) $(pthreadFlag) -o $(libPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(linkerFlags))

$(binPath)/occa:$(OCCA_DIR)/bin/occa.cpp $(libPath)/libocca.so $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(binPath)
	$(compiler) $(compilerFlags) -o $(binPath)/occa -Wl,-rpath,$(libPath) $(flags) $(OCCA_DIR)/bin/occa.cpp $(paths) $(linkerFlags) -L$(OCCA_DIR)/lib -locca
#  ===========================

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp $(OCCA_DIR)/include/occa/%.tpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.h $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<
#=================================================


#---[ Test ]--------------------------------------
tests: $(tests)

test: unit-tests e2e-tests

unit-tests: $(tests)
	@$(testPath)/run_tests

e2e-tests: unit-tests
	@$(testPath)/run_examples

$(testPath)/bin/%:$(testPath)/src/%.cpp $(outputs)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(testFlags) $(pthreadFlag) -o $@ $(flags) $< $(paths) $(linkerFlags) -L$(OCCA_DIR)/lib -locca
#=================================================


#---[ Clean ]-------------------------------------
clean:
	rm -rf $(objPath)/*
	rm -rf $(binPath)/occa
	rm -rf $(testPath)/bin
	rm -rf $(testPath)/src/io/locks
	rm  -f $(libPath)/libocca.so
#=================================================


#---[ Info ]--------------------------------------
info:
	$(info --------------------------------)
	$(info CXX            = $(or $(CXX),(empty)))
	$(info CXXFLAGS       = $(or $(CXXFLAGS),(empty)))
	$(info LDFLAGS        = $(or $(LDFLAGS),(empty)))
	$(info --------------------------------)
	$(info compiler       = $(value compiler))
	$(info compilerFlags  = $(compilerFlags))
	$(info flags          = $(flags))
	$(info paths          = $(paths))
	$(info sharedFlag     = $(sharedFlag))
	$(info pthreadFlag    = $(pthreadFlag))
	$(info linkerFlags    = $(linkerFlags))
	$(info --------------------------------)
#	$(info OCCA_DEVELOPER = $(OCCA_DEVELOPER))
	$(info debugEnabled   = $(debugEnabled))
	$(info checkEnabled   = $(checkEnabled))
	$(info debugFlags     = $(debugFlags))
	$(info releaseFlags   = $(releaseFlags))
	$(info picFlag        = $(picFlag))
	$(info --------------------------------)
	$(info mpiEnabled     = $(mpiEnabled))
	$(info openmpEnabled  = $(openmpEnabled))
	$(info openclEnabled  = $(openclEnabled))
	$(info cudaEnabled    = $(cudaEnabled))
	$(info hipEnabled     = $(hipEnabled))
	$(info --------------------------------)
	@true
#=================================================


#---[ Print ]-------------------------------------
# Print the contents of a makefile variable, e.g.: 'make print-compiler'.
print-%:
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true
#=================================================
