# The MIT License (MIT)
#
# Copyright (c) 2014-2016 David Medina and Tim Warburton
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

#---[ WORKING PATHS ]-----------------------------
ifeq ($(usingWinux),0)
  compilerFlags  += $(picFlag) -DOCCA_COMPILED_DIR="$(OCCA_DIR)"
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

#---[ VARIABLES ]---------------------------------
srcToObject  = $(subst $(PROJ_DIR)/src,$(PROJ_DIR)/obj,$(patsubst %.f90,%.o,$(1:.cpp=.o)))

sources  = $(realpath $(shell find $(PROJ_DIR)/src     -type f -name '*.cpp'))
headers  = $(realpath $(shell find $(PROJ_DIR)/include -type f -name '*.hpp'))
fsources = $(realpath $(shell find $(PROJ_DIR)/src     -type f -name '*.f90'))

sources := $(filter-out $(OCCA_DIR)/src/operators/%,$(sources))

#  ---[ Languages ]-----------
ifndef OCCA_COMPILE_PYTHON
	sources := $(filter-out $(OCCA_DIR)/src/lang/python=/%,$(sources))
endif

ifndef OCCA_COMPILE_JAVA
	sources := $(filter-out $(OCCA_DIR)/src/lang/java/%,$(sources))
endif

ifndef OCCA_COMPILE_OBJC
	sources := $(filter-out $(OCCA_DIR)/src/lang/objc/%,$(sources))
endif

ifndef OCCA_COMPILE_FORTRAN
	sources := $(filter-out $(OCCA_DIR)/src/lang/fortran/%,$(sources))
endif
#  ===========================

objects = $(call srcToObject,$(sources))

outputs = $(libPath)/libocca.so $(binPath)/occa
#=================================================


#---[ COMPILE LIBRARY ]---------------------------
all: $(objects) $(outputs)
#=================================================


#---[ PYTHON ]------------------------------------
python: $(libPath)/_C_occa.so
	python $(OCCA_DIR)/scripts/make.py compile
#=================================================


#---[ BUILDS ]------------------------------------
#  ---[ libocca ]-------------
$(libPath)/libocca.so:$(objects) $(headers)
	$(compiler) $(compilerFlags) $(sharedFlag) -o $(libPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(links))

$(binPath)/occa:$(OCCA_DIR)/scripts/occa.cpp $(libPath)/libocca.so
	@mkdir -p $(binPath)
	$(compiler) $(compilerFlags) -o $(binPath)/occa $(flags) $(OCCA_DIR)/scripts/occa.cpp $(paths) $(links) -L$(OCCA_DIR)/lib -locca
#  ===========================

#  ---[ C++ ]-----------------
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp $(OCCA_DIR)/include/occa/%.tpp
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<
#  ===========================

#  ---[ Fortran ]-------------
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.f90
	@mkdir -p $(abspath $(dir $@))
	$(fCompiler) $(fCompilerFlags) $(fModDirFlag) $(libPath) -o $@ $(fFlags) -c $<
#  ===========================

#  ---[ Python ]-------------
pyflags = -I${OCCA_PYTHON_DIR}/ -I${OCCA_NUMPY_DIR} -L${OCCA_LIBPYTHON_DIR} -l${OCCA_LIBPYTHON}

$(libPath)/_C_occa.so: $(libPath)/libocca.so $(incPath)/occa/lang/python/_C_occa.h $(incPath)/occa/lang/python/_C_occa.h
	gcc $(compilerFlags) $(sharedFlag) $(srcPath)/python/_C_occa.c -o $@ -I$(incPath) -I$(incPath)/occa/python -L$(libPath) $(pyFlags) -locca
#  ===========================
#=================================================


#---[ TEST ]--------------------------------------
test:
	echo '---[ Testing ]--------------------------'
	cd $(OCCA_DIR); \
	make -j 4 CXXFLAGS='-g' FCFLAGS='-g'

	cd $(OCCA_DIR)/examples/addVectors/cpp; \
	make -j 4 CXXFLAGS='-g' FCFLAGS='-g'; \
	./main

	cd $(OCCA_DIR)/examples/addVectors/c; \
	make -j 4 CXXFLAGS='-g' FCFLAGS='-g'; \
	./main

	cd $(OCCA_DIR)/examples/addVectors/f90; \
	make -j 4 CXXFLAGS='-g' FCFLAGS='-g'; \
	./main

	cd $(OCCA_DIR)/examples/reduction/; \
	make -j 4 CXXFLAGS='-g' FCFLAGS='-g'; \
	./main

	cd $(OCCA_DIR)/examples/usingArrays/; \
	make -j 4 CXXFLAGS='-g' FCFLAGS='-g'; \
	./main
#=================================================


#---[ CLEAN ]-------------------------------------
clean:
	rm -rf $(objPath)/*
	rm -rf $(binPath)/*
	rm  -f $(libPath)/libocca.so
	rm  -f $(libPath)/*.mod
	rm  -f $(libPath)/_C_occa.so
	rm  -f $(OCCA_DIR)/scripts/main
#=================================================

touch:
	mkdir -p /Users/dsm5/gitRepos/night/obj/lang/c/
	touch /Users/dsm5/gitRepos/night/obj/lang/c/c_wrapper.o