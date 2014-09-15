ifndef OCCA_DIR
	OCCA_DIR = $(shell pwd)
endif

include ${OCCA_DIR}/scripts/makefile

#---[ WORKING PATHS ]-----------------------------
compilerFlags += -fPIC
FcompilerFlags += -fPIC
lPath = lib

occaIPath = ${OCCA_DIR}/$(iPath)
occaOPath = ${OCCA_DIR}/$(oPath)
occaSPath = ${OCCA_DIR}/$(sPath)
occaLPath = ${OCCA_DIR}/$(lPath)
#=================================================

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(occaIPath)/*.hpp) $(wildcard $(occaIPath)/*.tpp)
sources = $(wildcard $(occaSPath)/*.cpp)
fsources = $(wildcard $(occaSPath)/*.f90)

objects = $(subst $(occaSPath)/,$(occaOPath)/,$(sources:.cpp=.o))

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
	objects += $(subst $(occaSPath)/,$(occaOPath)/,$(fsources:.f90=.o))
endif
endif

ifdef OCCA_DEVELOPER
ifeq ($(OCCA_DEVELOPER), 1)
	developerDependencies =	$(occaOPath)/occaKernelDefines.o
else
	developerDependencies =
endif
else
	developerDependencies =
endif

$(occaLPath)/libocca.so:$(objects) $(headers)
	$(compiler) $(compilerFlags) -shared -o $(occaLPath)/libocca.so $(flags) $(objects) $(paths) $(filter-out -locca, $(links))

$(occaOPath)/%.o:$(occaSPath)/%.cpp $(occaIPath)/%.hpp$(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.tpp))) $(developerDependencies)
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(occaOPath)/occaFTypes.o:$(occaSPath)/occaFTypes.f90
	$(Fcompiler) $(FcompilerFlags) $(FcompilerModFlag) $(occaLPath) -o $@ -c $<

$(occaOPath)/occaFTypes.mod:$(occaSPath)/occaFTypes.f90 $(occaOPath)/occaFTypes.o
	@true

$(occaOPath)/occaF.o:$(occaSPath)/occaF.f90 $(occaOPath)/occaFTypes.mod
	$(Fcompiler) $(FcompilerFlags) $(FcompilerModFlag) $(occaLPath) -o $@ -c $<

$(occaOPath)/occaCOI.o:$(occaSPath)/occaCOI.cpp $(occaIPath)/occaCOI.hpp
	$(compiler) $(compilerFlags) -o $@ $(flags) -Wl,--enable-new-dtags -c $(paths) $<

ifdef OCCA_DEVELOPER
ifeq ($(OCCA_DEVELOPER), 1)
$(occaOPath)/occaKernelDefines.o:            \
	$(occaIPath)/defines/occaOpenMPDefines.hpp   \
	$(occaIPath)/defines/occaOpenCLDefines.hpp   \
	$(occaIPath)/defines/occaCUDADefines.hpp     \
	$(occaIPath)/defines/occaPthreadsDefines.hpp \
	$(occaIPath)/defines/occaCOIDefines.hpp      \
	$(occaIPath)/defines/occaCOIMain.hpp         \
	$(occaIPath)/occaKernelDefines.hpp
	$(compiler) $(compilerFlags) -o $(occaOPath)/occaKernelDefines.o $(flags) -c $(paths) $(occaSPath)/occaKernelDefines.cpp

$(OCCA_DIR)/scripts/occaKernelDefinesGenerator:\
	$(occaIPath)/defines/occaOpenMPDefines.hpp     \
	$(occaIPath)/defines/occaOpenCLDefines.hpp     \
	$(occaIPath)/defines/occaCUDADefines.hpp       \
	$(occaIPath)/defines/occaPthreadsDefines.hpp   \
	$(occaIPath)/defines/occaCOIDefines.hpp        \
	$(occaIPath)/defines/occaCOIMain.hpp
	$(compiler) -o $(OCCA_DIR)/scripts/occaKernelDefinesGenerator $(OCCA_DIR)/scripts/occaKernelDefinesGenerator.cpp

$(occaIPath)/occaKernelDefines.hpp:$(OCCA_DIR)/scripts/occaKernelDefinesGenerator
	$(OCCA_DIR)/scripts/occaKernelDefinesGenerator $(OCCA_DIR)
endif
endif

clean:
	rm -f $(occaOPath)/*;
	rm -f ${OCCA_DIR}/scripts/main;
	rm -f $(occaLPath)/libocca.a;
	rm -f $(occaLPath)/*.mod;
	rm -f $(OCCA_DIR)/scripts/occaKernelDefinesGenerator
#=================================================
