PROJ_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../../scripts/makefile
else
  include ${OCCA_DIR}/scripts/makefile
endif

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

executables = ${PROJ_DIR}/main ${PROJ_DIR}/main_c

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
	executables += ${PROJ_DIR}/main_f90
endif
endif

all: $(executables)

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(links)

${PROJ_DIR}/main_c: ${PROJ_DIR}/main.c
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main_c $(flags) $(objects) ${PROJ_DIR}/main.c $(paths) $(links)

${PROJ_DIR}/main_f90: ${PROJ_DIR}/main.f90
	$(fCompiler) $(fCompilerFlags) -o ${PROJ_DIR}/main_f90 $(flags) $(objects) ${PROJ_DIR}/main.f90 $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f ${PROJ_DIR}/main
	rm -f ${PROJ_DIR}/main_c
	rm -f ${PROJ_DIR}/main_f90
#=================================================
