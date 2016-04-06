PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../../scripts/Makefile
else
  include ${OCCA_DIR}/scripts/Makefile
endif

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects = $(subst $(sPath)/,$(objPath)/,$(sources:.cpp=.o))

${PROJ_DIR}/main_f90: ${PROJ_DIR}/main.f90
	$(fCompiler) $(fCompilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.f90 $(paths) $(links)

clean:
	rm -f $(objPath)/*;
	rm -f ${PROJ_DIR}/main
#=================================================
