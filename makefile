include ${OCCA_DIR}/scripts/makefile

#---[ WORKING PATHS ]-----------------------------
lPath = lib

occaIPath = ${OCCA_DIR}/$(iPath)
occaOPath = ${OCCA_DIR}/$(oPath)
occaSPath = ${OCCA_DIR}/$(sPath)
occaLPath = ${OCCA_DIR}/$(lPath)
#=================================================

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(occaIPath)/*.hpp) $(wildcard $(occaIPath)/*.tpp)
sources = $(wildcard $(occaSPath)/*.cpp)

objects = $(subst $(occaSPath)/,$(occaOPath)/,$(sources:.cpp=.o))

$(occaLPath)/libocca.a:$(objects) $(headers)
	ar rcs $(occaLPath)/libocca.a $(objects)

$(occaOPath)/%.o:$(occaSPath)/%.cpp $(occaIPath)/%.hpp$(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(occaSPath)/,$(occaIPath)/,$(<:.cpp=.tpp)))\
	$(occaOPath)/occaKernelDefines.o
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(occaOPath)/occaKernelDefines.o:$(occaIPath)/occaOpenMPDefines.hpp $(occaIPath)/occaOpenCLDefines.hpp $(occaIPath)/occaCUDADefines.hpp $(occaIPath)/occaKernelDefines.hpp
	$(compiler) $(compilerFlags) -o $(occaOPath)/occaKernelDefines.o $(flags) -c $(paths) $(occaSPath)/occaKernelDefines.cpp

$(OCCA_DIR)/scripts/occaKernelDefinesGenerator:$(occaIPath)/occaOpenMPDefines.hpp $(occaIPath)/occaOpenCLDefines.hpp $(occaIPath)/occaCUDADefines.hpp
	$(compiler) -o $(OCCA_DIR)/scripts/occaKernelDefinesGenerator $(OCCA_DIR)/scripts/occaKernelDefinesGenerator.cpp

$(occaIPath)/occaKernelDefines.hpp:$(OCCA_DIR)/scripts/occaKernelDefinesGenerator
	$(OCCA_DIR)/scripts/occaKernelDefinesGenerator

clean:
	rm -f $(occaOPath)/*;
	rm -f ${OCCA_DIR}/scripts/main;
	rm -f $(occaLPath)/libocca.a;
#=================================================
