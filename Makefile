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
links := $(filter-out -locca,$(links))
#=================================================

#---[ variables ]---------------------------------
srcToObject  = $(subst $(PROJ_DIR)/src,$(PROJ_DIR)/obj,$(1:.cpp=.o))

sources     = $(realpath $(shell find $(PROJ_DIR)/src     -type f -name '*.cpp'))
headers     = $(realpath $(shell find $(PROJ_DIR)/include -type f -name '*.hpp'))
sources    := $(filter-out $(OCCA_DIR)/src/operators/%,$(sources))
testSources = $(realpath $(shell find $(PROJ_DIR)/tests   -type f -name '*.cpp'))
tests       = $(subst $(testPath)/,$(testPath)/bin/,$(testSources:.cpp=))

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
                                      s,@@OCCA_CUDA_ENABLED@@,$(OCCA_CUDA_ENABLED),g;" > "$(NEW_COMPILED_DEFINES)")
MAKE_COMPILED_DEFINES := $(shell \
 [ ! -f "$(COMPILED_DEFINES)" -o -n "$(shell diff -q $(COMPILED_DEFINES) $(NEW_COMPILED_DEFINES))" ] \
 && cp "$(NEW_COMPILED_DEFINES)" "$(COMPILED_DEFINES)" \
)
MAKE_COMPILED_DEFINES := $(shell rm $(NEW_COMPILED_DEFINES))

all: $(objects) $(outputs)
	@(. $(OCCA_DIR)/scripts/shellTools.sh && installOcca)
	@echo "> occa info"
	@$(OCCA_DIR)/bin/occa info

$(COMPILED_DEFINES_CHANGED):
#=================================================


#---[ Builds ]------------------------------------
#  ---[ libocca ]-------------
$(libPath)/libocca.so:$(objects) $(headers) $(COMPILED_DEFINES)
	mkdir -p $(libPath)
	$(compiler) $(compilerFlags) $(sharedFlag) $(pthreadFlag) -o $(libPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(links))

$(binPath)/occa:$(OCCA_DIR)/scripts/occa.cpp $(libPath)/libocca.so $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(binPath)
	$(compiler) $(compilerFlags) -o $(binPath)/occa -Wl,-rpath,$(libPath) $(flags) $(OCCA_DIR)/scripts/occa.cpp $(paths) $(links) -L$(OCCA_DIR)/lib -locca
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
examples =                 \
	1_add_vectors/cpp        \
	1_add_vectors/c          \
	2_background_device      \
  3_reduction              \
  4_building_kernels       \
  5_unified_memory         \
  6_arrays                 \
  7_streams

tests: $(tests)

test: unit-tests e2e-tests

unit-tests: $(tests)
	@for test in $(tests); do                             \
	   	testname=$$(basename "$$test");                   \
      chars=$$(echo "$${testname}" | wc -c);            \
      linechars=$$((60 - $${chars}));                   \
	    line=$$(printf '%*s' $${linechars} | tr ' ' '-'); \
	    echo -e "\n---[ $${testname} ]$${line}";          \
	    ASAN_OPTIONS=protect_shadow_gap=0 $$test 2>&1 | head -n 100; \
	done

e2e-tests: unit-tests
	@for dir in $(examples); do                           \
	  echo "Compiling example [$$dir]";                   \
	  cd $(OCCA_DIR)/examples/$$dir &&                    \
	  rm -f main                    &&                    \
	  CXXFLAGS='-g' make            &&                    \
	  OCCA_VERBOSE=1 ./main;                              \
	done

$(testPath)/bin/%:$(testPath)/%.cpp $(libPath)/libocca.so
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(testFlags) $(pthreadFlag) -o $@ $(flags) $< $(paths) $(links) -L$(OCCA_DIR)/lib -locca
#=================================================


#---[ Clean ]-------------------------------------
clean:
	rm -rf $(objPath)/*
	rm -rf $(binPath)/*
	rm -rf $(testPath)/bin/*;
	rm  -f $(libPath)/libocca.so
	rm  -f $(OCCA_DIR)/scripts/main
#=================================================
