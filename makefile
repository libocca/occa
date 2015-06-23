ifndef OCCA_DIR
  OCCA_DIR = $(shell pwd)
  occaDirWasInitialized = 1
endif

include ${OCCA_DIR}/scripts/makefile

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
occaIPath = ${OCCA_DIR}/$(iPath)
occaOPath = ${OCCA_DIR}/$(oPath)
occaSPath = ${OCCA_DIR}/$(sPath)
occaLPath = ${OCCA_DIR}/$(lPath)
#=================================================

#---[ COMPILATION ]-------------------------------
headers  = $(wildcard $(occaIPath)/*.hpp) $(wildcard $(occaIPath)/*.tpp)
sources  = $(wildcard $(occaSPath)/*.cpp)
fsources = $(wildcard $(occaSPath)/*.f90)

objects = $(subst $(occaSPath)/,$(occaOPath)/,$(sources:.cpp=.o))

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
  objects += $(subst $(occaSPath)/,$(occaOPath)/,$(fsources:.f90=.o))
endif
endif

ifdef occaDirWasInitialized
.FORCE:

$(occaLPath)/libocca.so: .FORCE
	@echo "Error: You need to set the environment variable [OCCA_DIR]"
	@echo "For example:"
	@echo "  export OCCA_DIR='$(shell pwd)'"
else

.SUFFIXES:

all: $(occaLPath)/libocca.so $(occaBPath)/occa $(occaBPath)/occainfo

$(occaLPath)/libocca.so:$(objects) $(headers)
	$(compiler) $(compilerFlags) $(sharedFlag) -o $(occaLPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(links))
endif

$(occaOPath)/%.o:$(occaSPath)/%.cpp $(occaIPath)/%.hpp $(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(occaOPath)/occaFTypes.mod:$(occaSPath)/occaFTypes.f90 $(occaOPath)/occaFTypes.o
	@true

$(occaOPath)/occaFTypes.o:$(occaSPath)/occaFTypes.f90
	$(fCompiler) $(fCompilerFlags) $(fModDirFlag) $(occaLPath) -o $@ $(fFlags) -c $<

$(occaOPath)/occaF.o:$(occaSPath)/occaF.f90 $(occaSPath)/occaFTypes.f90 $(occaOPath)/occaFTypes.o
	$(fCompiler) $(fCompilerFlags) $(fModDirFlag) $(occaLPath) -o $@ $(fFlags) -c $<

# Ingore [-Wl,--enable-new-dtags] warnings if COI isn't being compiled
ifeq (coiEnabled, 1)
$(occaOPath)/occaCOI.o:$(occaSPath)/occaCOI.cpp $(occaIPath)/occaCOI.hpp
	$(compiler) $(compilerFlags) -o $@ $(flags) -Wl,--enable-new-dtags -c $(paths) $<
endif

$(occaBPath)/occa:$(OCCA_DIR)/scripts/occa.cpp $(occaLPath)/libocca.so
	$(compiler) $(compilerFlags) -o $(occaBPath)/occa $(flags) $(OCCA_DIR)/scripts/occa.cpp $(paths) $(links) -L${OCCA_DIR}/lib -locca

$(occaBPath)/occainfo:$(OCCA_DIR)/scripts/occaInfo.cpp $(occaLPath)/libocca.so
	$(compiler) $(compilerFlags) -o $(occaBPath)/occainfo $(flags) $(OCCA_DIR)/scripts/occaInfo.cpp $(paths) $(links) -L${OCCA_DIR}/lib -locca

$(occaOPath)/occaKernelDefines.o:              \
	$(occaIPath)/defines/occaOpenMPDefines.hpp   \
	$(occaIPath)/defines/occaOpenCLDefines.hpp   \
	$(occaIPath)/defines/occaCUDADefines.hpp     \
	$(occaIPath)/defines/occaPthreadsDefines.hpp \
	$(occaIPath)/defines/occaCOIDefines.hpp      \
	$(occaIPath)/defines/occaCOIMain.hpp         \
	$(occaIPath)/occaKernelDefines.hpp	         \
	$(OCCA_DIR)/scripts/occaKernelDefines.py
	$(compiler) $(compilerFlags) -o $(occaOPath)/occaKernelDefines.o $(flags) -c $(paths) $(occaSPath)/occaKernelDefines.cpp

ifdef OCCA_DEVELOPER
ifeq ($(OCCA_DEVELOPER), 1)
$(occaIPath)/occaKernelDefines.hpp:$(OCCA_DIR)/scripts/occaKernelDefines.py
	python $(OCCA_DIR)/scripts/occaKernelDefines.py
endif
endif

ifdef occaDirWasInitialized
clean: .FORCE
	@echo "Error: You need to set the environment variable [OCCA_DIR]"
	@echo "For example:"
	@echo "  export OCCA_DIR='$(shell pwd)'"
else
clean:
	rm -f $(occaOPath)/*;
	rm -f $(occaBPath)/*;
	rm -f ${OCCA_DIR}/scripts/main;
	rm -f $(occaLPath)/libocca.so;
	rm -f $(occaLPath)/*.mod;
endif
#=================================================
