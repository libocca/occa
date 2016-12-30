PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../scripts/Makefile
else
  include ${OCCA_DIR}/scripts/Makefile
endif

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources = $(wildcard $(srcPath)/*.cpp)
tests   = $(wildcard $(testPath)/test*.cpp)

objects      = $(subst $(srcPath)/,$(objPath)/,$(sources:.cpp=.o))
testBinaries = $(subst $(testPath)/,$(testPath)/bin/,$(tests:.cpp=))

flags += -I$(incPath)

executables = ${PROJ_DIR}/main

all: $(executables)

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp $(testBinaries)
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(links)

$(objPath)/%.o:$(srcPath)/%.cpp $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.tpp)))
	@mkdir -p $(objPath)
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(testPath)/bin/%:$(testPath)/%.cpp
	@mkdir -p $(testPath)/bin
	$(compiler) $(compilerFlags) -o $@ $(flags) $(objects) $< $(paths) $(links)

clean:
	rm -f $(objPath)/*;
	rm -f $(testPath)/bin/*;
	rm -f ${PROJ_DIR}/main
#=================================================
