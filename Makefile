# The MIT License (MIT)
#
# Copyright (c) 2014-2017 David Medina and Tim Warburton
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

ifndef PREFIX
  OCCA_COMPILED_DIR = $(OCCA_DIR)
else
  OCCA_COMPILED_DIR = $(PREFIX)
endif

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

sources  = $(realpath $(shell find $(PROJ_DIR)/src     -type f -name '*.cpp'))
headers  = $(realpath $(shell find $(PROJ_DIR)/include -type f -name '*.hpp'))
sources := $(filter-out $(OCCA_DIR)/src/operators/%,$(sources))

objects = $(call srcToObject,$(sources))

outputs = $(libPath)/libocca.so $(binPath)/occa
#=================================================


#---[ Compile Library ]---------------------------
# Setup compiled defines and force rebuild if defines changed
NEW_COMPILED_DEFINES     := $(OCCA_DIR)/include/occa/defines/compiledDefines.hpp
OLD_COMPILED_DEFINES     := $(OCCA_DIR)/.old_compiledDefines
COMPILED_DEFINES_CHANGED := $(OCCA_DIR)/.compiledDefinesChanged

MAKE_COMPILED_DEFINES := $(shell touch "$(NEW_COMPILED_DEFINES)")
MAKE_COMPILED_DEFINES := $(shell cp "$(NEW_COMPILED_DEFINES)" "$(OLD_COMPILED_DEFINES)")
MAKE_COMPILED_DEFINES := $(shell cat "$(OCCA_DIR)/scripts/compiledDefinesTemplate.hpp" | \
                                 sed "s,@@OCCA_OS@@,$(OCCA_OS),g;\
                                      s,@@OCCA_USING_VS@@,$(OCCA_USING_VS),g;\
                                      s,@@OCCA_COMPILED_DIR@@,\"$(OCCA_COMPILED_DIR)\",g;\
                                      s,@@OCCA_DEBUG_ENABLED@@,$(OCCA_DEBUG_ENABLED),g;\
                                      s,@@OCCA_CHECK_ENABLED@@,$(OCCA_CHECK_ENABLED),g;\
                                      s,@@OCCA_OPENMP_ENABLED@@,$(OCCA_OPENMP_ENABLED),g;\
                                      s,@@OCCA_OPENCL_ENABLED@@,$(OCCA_OPENCL_ENABLED),g;\
                                      s,@@OCCA_CUDA_ENABLED@@,$(OCCA_CUDA_ENABLED),g;\
                                      s,@@OCCA_MPI_ENABLED@@,$(OCCA_MPI_ENABLED),g;" > "$(NEW_COMPILED_DEFINES)")
MAKE_COMPILED_DEFINES := $(shell [ -n "$(shell diff -q $(OLD_COMPILED_DEFINES) $(NEW_COMPILED_DEFINES))" ] && touch "$(COMPILED_DEFINES_CHANGED)")
MAKE_COMPILED_DEFINES := $(shell rm $(OLD_COMPILED_DEFINES))

all: $(objects) $(outputs)
	@(. $(OCCA_DIR)/scripts/shellTools.sh && installOcca)
	@echo -e ""
	@echo -e "---[ Compiled With ]--------------------------------------------------------"
	@echo -e "    OCCA_OS             : $(OCCA_OS)"
	@echo -e "    OCCA_USING_VS       : $(OCCA_USING_VS)"
	@echo -e "    OCCA_COMPILED_DIR   : \"$(OCCA_COMPILED_DIR)\"\n"
	@echo -e "    OCCA_DEBUG_ENABLED  : $(OCCA_DEBUG_ENABLED)"
	@echo -e "    OCCA_CHECK_ENABLED  : $(OCCA_CHECK_ENABLED)\n"
	@echo -e "    OCCA_OPENMP_ENABLED : $(OCCA_OPENMP_ENABLED)"
	@echo -e "    OCCA_OPENCL_ENABLED : $(OCCA_OPENCL_ENABLED)"
	@echo -e "    OCCA_CUDA_ENABLED   : $(OCCA_CUDA_ENABLED)"
	@echo -e "    OCCA_MPI_ENABLED    : $(OCCA_MPI_ENABLED)"
	@echo -e "============================================================================"

$(COMPILED_DEFINES_CHANGED):
#=================================================


#---[ Builds ]------------------------------------
#  ---[ libocca ]-------------
$(libPath)/libocca.so:$(objects) $(headers) $(COMPILED_DEFINES_CHANGED)
	mkdir -p $(libPath)
	$(compiler) $(compilerFlags) $(sharedFlag) -o $(libPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(links))

$(binPath)/occa:$(OCCA_DIR)/scripts/occa.cpp $(libPath)/libocca.so $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(binPath)
	$(compiler) $(compilerFlags) -o $(binPath)/occa $(flags) $(OCCA_DIR)/scripts/occa.cpp $(paths) $(links) -L$(OCCA_DIR)/lib -locca
#  ===========================

#  ---[ C++ ]-----------------
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp $(OCCA_DIR)/include/occa/%.tpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<
#  ===========================
#=================================================


#---[ Test ]--------------------------------------
examples =                 \
	addVectors/cpp           \
	addVectors/c             \
	backgroundDevices        \
	customReduction          \
	reduction                \
	unifiedMemoryAddVectors  \
	usingArrays              \
	usingStreams

test:
	@for dir in $(examples); do         \
	  echo "Compiling example [$$dir]"; \
	  cd $(OCCA_DIR)/examples/$$dir &&  \
	  rm -f main                    &&  \
	  CXXFLAGS='-g' make            &&  \
	  ./main;                           \
	done
#=================================================


#---[ Clean ]-------------------------------------
clean:
	rm -rf $(objPath)/*
	rm -rf $(binPath)/*
	rm  -f $(libPath)/libocca.so
	rm  -f $(OCCA_DIR)/scripts/main
#=================================================