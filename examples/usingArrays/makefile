PROJ_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../scripts/makefile
else
  include ${OCCA_DIR}/scripts/makefile
endif

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

executables = ${PROJ_DIR}/main

all: $(executables)

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f ${PROJ_DIR}/main
#=================================================
