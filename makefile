ifndef OCCA_DIR
.FORCE:

$(occaLPath)/libocca.so: .FORCE
	@echo "Error: You need to set the environment variable [OCCA_DIR]"
	@echo "For example:"
	@echo "  export OCCA_DIR='$(shell pwd)'"

clean: .FORCE
	@echo "Error: You need to set the environment variable [OCCA_DIR]"
	@echo "For example:"
	@echo "  export OCCA_DIR='$(shell pwd)'"
else

include ${OCCA_DIR}/scripts/makefile

$(shell . ${OCCA_DIR}/scripts/shellTools.sh; setupObjDir)

#---[ WORKING PATHS ]-----------------------------
ifeq ($(usingWinux),0)
  compilerFlags  += $(picFlag)
  fCompilerFlags += $(picFlag)
else
  sharedFlag     += $(picFlag)
endif

# [-L$OCCA_DIR/lib -locca] are kept for applications
#   using $OCCA_DIR/scripts/makefile
paths := $(filter-out -L${OCCA_DIR}/lib,$(paths))
links := $(filter-out -locca,$(links))

occaBPath = ${OCCA_DIR}/$(bPath)
occaIPath = ${OCCA_DIR}/$(iPath)/occa
occaOPath = ${OCCA_DIR}/$(oPath)
occaSPath = ${OCCA_DIR}/$(sPath)
occaLPath = ${OCCA_DIR}/$(lPath)
#=================================================

#---[ COMPILATION ]-------------------------------
headers   = $(wildcard $(occaIPath)/*.hpp)        $(wildcard $(occaIPath)/*.tpp)
headers  += $(wildcard $(occaIPath)/parser/*.hpp) $(wildcard $(occaIPath)/parser/*.tpp)
headers  += $(wildcard $(occaIPath)/array/*.hpp)  $(wildcard $(occaIPath)/array/*.tpp)

sources   = $(wildcard $(occaSPath)/*.cpp)
sources  += $(wildcard $(occaSPath)/parser/*.cpp)

fsources  = $(wildcard $(occaSPath)/*.f90)

objects = $(subst $(occaSPath)/,$(occaOPath)/,$(sources:.cpp=.o))
outputs = $(occaLPath)/libocca.so $(occaBPath)/occa $(occaBPath)/occainfo

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
  objects += $(subst $(occaSPath)/,$(occaOPath)/,$(fsources:.f90=.o))
endif
endif

ifdef OCCA_LIBPYTHON_DIR
  ifdef OCCA_LIBPYTHON
    ifdef OCCA_PYTHON_DIR
      ifdef OCCA_NUMPY_DIR
        outputs += $(occaLPath)/_C_occa.so

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

.SUFFIXES:

all: $(outputs)

$(occaLPath)/libocca.so:$(objects) $(headers)
	$(compiler) $(compilerFlags) $(sharedFlag) -o $(occaLPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(links))

$(occaOPath)/%.o:$(occaSPath)/%.cpp $(occaIPath)/%.hpp $(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(occaOPath)/parser/%.o:$(occaSPath)/parser/%.cpp $(wildcard $(subst $(occaSPath)/,$(occaIPath)/parser/,$(<:.cpp=.hpp))) $(wildcard $(subst $(occaSPath)/,$(occaIPath)/parser/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(occaOPath)/array/%.o:$(occaSPath)/array/%.cpp $(wildcard $(subst $(occaSPath)/,$(occaIPath)/array/,$(<:.cpp=.hpp))) $(wildcard $(subst $(occaSPath)/,$(occaIPath)/array/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(occaOPath)/fTypes.mod:$(occaSPath)/fTypes.f90 $(occaOPath)/fTypes.o
	@true

$(occaOPath)/fTypes.o:$(occaSPath)/fTypes.f90
	$(fCompiler) $(fCompilerFlags) $(fModDirFlag) $(occaLPath) -o $@ $(fFlags) -c $<

$(occaOPath)/fBase.o:$(occaSPath)/fBase.f90 $(occaSPath)/fTypes.f90 $(occaOPath)/fTypes.o
	$(fCompiler) $(fCompilerFlags) $(fModDirFlag) $(occaLPath) -o $@ $(fFlags) -c $<

$(occaLPath)/_C_occa.so: $(occaLPath)/libocca.so $(occaIPath)/python/_C_occa.h $(occaSPath)/occa/python/_C_occa.c
	clang -shared -fPIC $(occaSPath)/occa/python/_C_occa.c -o $(occaLPath)/_C_occa.so \
	-I${OCCA_DIR}/include/ -I${OCCA_DIR}/include/occa/python -L${OCCA_DIR}/lib $(pyFlags) -locca

$(occaBPath)/occa:$(OCCA_DIR)/scripts/occa.cpp $(occaLPath)/libocca.so
	$(compiler) $(compilerFlags) -o $(occaBPath)/occa $(flags) $(OCCA_DIR)/scripts/occa.cpp $(paths) $(links) -L${OCCA_DIR}/lib -locca

$(occaBPath)/occainfo:$(OCCA_DIR)/scripts/occaInfo.cpp $(occaLPath)/libocca.so
	$(compiler) $(compilerFlags) -o $(occaBPath)/occainfo $(flags) $(OCCA_DIR)/scripts/occaInfo.cpp $(paths) $(links) -L${OCCA_DIR}/lib -locca

$(occaOPath)/occaKernelDefines.o:        \
	$(occaIPath)/defines/OpenMP.hpp        \
	$(occaIPath)/defines/OpenCL.hpp        \
	$(occaIPath)/defines/CUDA.hpp          \
	$(occaIPath)/defines/Pthreads.hpp      \
	$(occaIPath)/defines/COI.hpp           \
	$(occaIPath)/defines/occaCOIMain.hpp   \
	$(occaIPath)/kernelDefines.hpp	       \
	$(OCCA_DIR)/scripts/occaKernelDefines.py
	$(compiler) $(compilerFlags) -o $(occaOPath)/occa/kernelDefines.o $(flags) -c $(paths) $(occaSPath)/occa/kernelDefines.cpp

ifdef OCCA_DEVELOPER
ifeq ($(OCCA_DEVELOPER), 1)
$(occaIPath)/occaKernelDefines.hpp:$(OCCA_DIR)/scripts/occaKernelDefines.py
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
	rm -rf $(occaOPath)/*
	rm -rf $(occaBPath)/*
	rm  -f ${OCCA_DIR}/scripts/main
	rm  -f $(occaLPath)/libocca.so
	rm  -f $(occaLPath)/*.mod
	rm  -f $(occaLPath)/_C_occa.so
endif
#=================================================
