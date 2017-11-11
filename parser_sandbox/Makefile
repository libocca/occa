PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../scripts/Makefile
else
  include ${OCCA_DIR}/scripts/Makefile
endif

#---[ COMPILATION ]-------------------------------
headers     = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources     = $(wildcard $(srcPath)/*.cpp)
testSources = $(wildcard $(testPath)/test*.cpp)

objects = $(subst $(srcPath)/,$(objPath)/,$(sources:.cpp=.o))
tests   = $(subst $(testPath)/,$(testPath)/bin/,$(testSources:.cpp=))

flags += -I$(incPath)

executables = ${PROJ_DIR}/main

all: $(executables)

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp $(tests)
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) -L ${OCCA_DIR}/lib $(objects) ${PROJ_DIR}/main.cpp $(paths) $(links)

$(objPath)/%.o:$(srcPath)/%.cpp $(incPath)/%.hpp
	@mkdir -p $(objPath)
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(objPath)/%.o:$(srcPath)/%.cpp $(incPath)/%.hpp $(incPath)/%.tpp
	@mkdir -p $(objPath)
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(testPath)/bin/%:$(testPath)/%.cpp $(objects) $(headers)
	@mkdir -p $(testPath)/bin
	$(compiler) $(compilerFlags) -o $@ $(flags) -L ${OCCA_DIR}/lib $(objects) $< $(paths) $(links)

test: $(tests)
	@for test in $(tests); do   \
	    echo "Running: $$test"; \
	    $$test;                 \
	done

clean:
	rm -f $(objPath)/*;
	rm -f $(testPath)/bin/*;
	rm -f ${PROJ_DIR}/main
#=================================================
