PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../scripts/Makefile
else
  include ${OCCA_DIR}/scripts/Makefile
endif

#---[ COMPILATION ]-------------------------------
srcToObject  = $(subst $(PROJ_DIR)/src,$(PROJ_DIR)/obj,$(1:.cpp=.o))

headers     = $(realpath $(shell find $(PROJ_DIR)/include -type f -name '*.hpp'))
sources     = $(realpath $(shell find $(PROJ_DIR)/src     -type f -name '*.cpp'))
testSources = $(wildcard $(testPath)/test*.cpp)

objects = $(call srcToObject,$(sources))
tests   = $(subst $(testPath)/,$(testPath)/bin/,$(testSources:.cpp=))

flags += -I$(incPath)

executables = ${PROJ_DIR}/main

all: $(executables)

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp $(tests)
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) -L ${OCCA_DIR}/lib $(objects) ${PROJ_DIR}/main.cpp $(paths) $(links)

$(objPath)/%.o:$(srcPath)/%.cpp $(incPath)/%.hpp
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(objPath)/%.o:$(srcPath)/%.cpp $(incPath)/%.hpp $(incPath)/%.tpp
	@mkdir -p $(abspath $(dir $@))
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
	rm -rf $(objPath)/*;
	rm -f  $(testPath)/bin/*;
	rm -f  ${PROJ_DIR}/main
#=================================================
