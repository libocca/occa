#---[ OCCA_DIR ]----------------------------------
OCCA_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJ_DIR:=$(OCCA_DIR)

include $(OCCA_DIR)/scripts/makefile

#---[ WORKING PATHS ]-----------------------------
ifeq ($(usingWinux),0)
  compilerFlags  += $(picFlag) -DOCCA_COMPILED_DIR="$(OCCA_DIR)"
  fCompilerFlags += $(picFlag)
else
  sharedFlag += $(picFlag)
endif

# [-L$OCCA_DIR/lib -locca] are kept for applications
#   using $OCCA_DIR/scripts/makefile
paths := $(filter-out -L$(OCCA_DIR)/lib,$(paths))
links := $(filter-out -locca,$(links))

iPath := $(iPath)/occa
#=================================================

#---[ COMPILATION ]-------------------------------
headers  = $(wildcard $(iPath)/*.hpp)        $(wildcard $(iPath)/*.tpp)
headers += $(wildcard $(iPath)/parser/*.hpp) $(wildcard $(iPath)/parser/*.tpp)
headers += $(wildcard $(iPath)/array/*.hpp)  $(wildcard $(iPath)/array/*.tpp)

sources  = $(wildcard $(sPath)/*.cpp)
sources += $(wildcard $(sPath)/parser/*.cpp)

fsources = $(wildcard $(sPath)/*.f90)

objects = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))
outputs = $(lPath)/libocca.so $(bPath)/occa $(bPath)/occainfo

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
  objects += $(subst $(sPath)/,$(oPath)/,$(fsources:.f90=.o))
endif
endif

ifdef OCCA_LIBPYTHON_DIR
  ifdef OCCA_LIBPYTHON
    ifdef OCCA_PYTHON_DIR
      ifdef OCCA_NUMPY_DIR
        outputs += $(lPath)/_C_occa.so

        pyFlags = -I${OCCA_PYTHON_DIR}/ -I${OCCA_NUMPY_DIR} -L${OCCA_LIBPYTHON_DIR}

        ifeq ($(usingLinux),1)
          pyFlags += -l${OCCA_LIBPYTHON}
        else ifeq ($(usingOSX),1)
          pyFlags += -framework Python
        endif
      endif
    endif
  endif
endif

all: objdirs $(outputs)

objdirs: $(oPath) $(oPath)/parser $(oPath)/python
$(oPath):
	mkdir -p $(oPath)
$(oPath)/parser:
	mkdir -p $(oPath)/parser
$(oPath)/python:
	mkdir -p $(oPath)/python

$(lPath)/libocca.so:$(objects) $(headers)
	$(compiler) $(compilerFlags) $(sharedFlag) -o $(lPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(links))

$(oPath)/%.o:$(sPath)/%.cpp $(iPath)/%.hpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(oPath)/parser/%.o:$(sPath)/parser/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/parser/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/parser/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(oPath)/array/%.o:$(sPath)/array/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/array/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/array/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(oPath)/fTypes.mod:$(sPath)/fTypes.f90 $(oPath)/fTypes.o
	@true

$(oPath)/fTypes.o:$(sPath)/fTypes.f90
	$(fCompiler) $(fCompilerFlags) $(fModDirFlag) $(lPath) -o $@ $(fFlags) -c $<

$(oPath)/fBase.o:$(sPath)/fBase.f90 $(sPath)/fTypes.f90 $(oPath)/fTypes.o
	$(fCompiler) $(fCompilerFlags) $(fModDirFlag) $(lPath) -o $@ $(fFlags) -c $<

$(lPath)/_C_occa.so: $(lPath)/libocca.so $(iPath)/python/_C_occa.h $(sPath)/occa/python/_C_occa.c
	clang -shared -fPIC $(sPath)/occa/python/_C_occa.c -o $(lPath)/_C_occa.so \
	-I$(OCCA_DIR)/include/ -I$(OCCA_DIR)/include/occa/python -L$(OCCA_DIR)/lib $(pyFlags) -locca

$(bPath)/occa:$(OCCA_DIR)/scripts/occa.cpp $(lPath)/libocca.so
	$(compiler) $(compilerFlags) -o $(bPath)/occa $(flags) $(OCCA_DIR)/scripts/occa.cpp $(paths) $(links) -L$(OCCA_DIR)/lib -locca

$(bPath)/occainfo:$(OCCA_DIR)/scripts/occaInfo.cpp $(lPath)/libocca.so
	$(compiler) $(compilerFlags) -o $(bPath)/occainfo $(flags) $(OCCA_DIR)/scripts/occaInfo.cpp $(paths) $(links) -L$(OCCA_DIR)/lib -locca

$(oPath)/occaKernelDefines.o:        \
	$(iPath)/defines/OpenMP.hpp        \
	$(iPath)/defines/OpenCL.hpp        \
	$(iPath)/defines/CUDA.hpp          \
	$(iPath)/defines/Pthreads.hpp      \
	$(iPath)/defines/COI.hpp           \
	$(iPath)/defines/occaCOIMain.hpp   \
	$(iPath)/kernelDefines.hpp	       \
	$(OCCA_DIR)/scripts/occaKernelDefines.py
	$(compiler) $(compilerFlags) -o $(oPath)/occa/kernelDefines.o $(flags) -c $(paths) $(sPath)/occa/kernelDefines.cpp

ifdef OCCA_DEVELOPER
ifeq ($(OCCA_DEVELOPER), 1)
$(iPath)/occaKernelDefines.hpp:$(OCCA_DIR)/scripts/occaKernelDefines.py
	python $(OCCA_DIR)/scripts/occaKernelDefines.py
endif
endif
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
	rm -rf $(oPath)/*
	rm -rf $(bPath)/*
	rm  -f $(lPath)/libocca.so
	rm  -f $(lPath)/*.mod
	rm  -f $(lPath)/_C_occa.so
	rm  -f $(OCCA_DIR)/scripts/main
#=================================================