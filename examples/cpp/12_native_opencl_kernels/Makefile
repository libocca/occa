PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../../scripts/build/Makefile
else
  include ${OCCA_DIR}/scripts/build/Makefile
endif

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources = $(wildcard $(srcPath)/*.cpp)

objects = $(subst $(srcPath)/,$(objPath)/,$(sources:.cpp=.o))

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(linkerFlags)
	@if which install_name_tool > /dev/null 2>&1; then \
		install_name_tool -add_rpath "${OCCA_DIR}/lib" ${PROJ_DIR}/main; \
	fi

$(objPath)/%.o:$(srcPath)/%.cpp $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(objPath)/*;
	rm -f $(PROJ_DIR)/main
#=================================================
