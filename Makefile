rmSlash = $(patsubst %/,%,$1)

OCCA_DIR := $(call rmSlash,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJ_DIR := $(OCCA_DIR)

# Clear the list of suffixes
.SUFFIXES:
.SUFFIXES: .hpp .h .tpp .cpp .mm .o .so .dylib

# Cancel some implicit rules
%:   %.o
%.o: %.cpp
%:   %.cpp
%:   %.mm

include $(OCCA_DIR)/scripts/Makefile

#---[ Working Paths ]-----------------------------
ifeq ($(usingWinux),0)
  compilerFlags  += $(picFlag)
  fCompilerFlags += $(picFlag)
else
  sharedFlag += $(picFlag)
endif

# [-L$OCCA_DIR/lib -locca] are kept for applications
#   using $OCCA_DIR/scripts/Makefile
paths += -I$(srcPath)
paths := $(filter-out -L$(OCCA_DIR)/lib,$(paths))
linkerFlags := $(filter-out -locca,$(linkerFlags))

ifeq (${PREFIX},)
  OCCA_BUILD_DIR := $(OCCA_DIR)
else
  OCCA_BUILD_DIR := ${PREFIX}
endif
#=================================================


#---[ Flags ]-------------------------------------
# Expand and overwrite flag variables to avoid
# multiple expansions which are slow.
compilerFlags := $(compilerFlags)
flags := $(flags)
sharedFlag := $(sharedFlag)
pthreadFlag := $(pthreadFlag)
#=================================================


#---[ Compilation Variables ]---------------------
srcToObject     = $(subst $(PROJ_DIR)/src,$(PROJ_DIR)/obj,$(1:.cpp=.o))

dontCompile = $(OCCA_DIR)/src/core/kernelOperators.cpp $(OCCA_DIR)/src/tools/runFunction.cpp

sources      = $(realpath $(shell find $(PROJ_DIR)/src -type f -name '*.cpp'))
sources     := $(filter-out $(dontCompile),$(sources))
headers      = $(realpath $(shell find $(PROJ_DIR)/include -type f -name '*.hpp' -o -name "*.h"))
testSources  = $(realpath $(shell find $(PROJ_DIR)/tests/src -type f -name '*.cpp'))
testSources := $(filter-out $(testPath)/src/fortran/%.cpp,$(testSources))
tests        = $(subst $(testPath)/src,$(testPath)/bin,$(testSources:.cpp=))

objects = $(call srcToObject,$(sources))

# Only compile Objective-C++ if Metal is enabled
ifeq ($(metalEnabled),1)
  objcSrcToObject = $(subst $(PROJ_DIR)/src,$(PROJ_DIR)/obj,$(1:.mm=.o))
  objcSources = $(realpath $(shell find $(PROJ_DIR)/src -type f -name '*.mm'))
  objects += $(call objcSrcToObject,$(objcSources))
endif

outputs = $(libPath)/libocca.$(soExt) $(binPath)/occa

# Add Fortran lib and tests
ifeq ($(fortranEnabled),1)
  fPaths += $(fModuleDirFlag)$(modPath)
  fPaths := $(filter-out -I$(modPath),$(fPaths))
  outputs += $(libPath)/libocca_fortran.$(soExt)
  tests += $(fTests)
endif

ifdef SANITIZER_ENABLED
  testFlags = $(compilerFlags) -fsanitize=address -fno-omit-frame-pointer
else
  testFlags = $(compilerFlags)
endif
#=================================================


#---[ Compile Library ]---------------------------
# Setup compiled defines and force rebuild if defines changed
COMPILED_DEFINES     := $(OCCA_DIR)/include/occa/defines/compiledDefines.hpp
NEW_COMPILED_DEFINES := $(OCCA_DIR)/.compiledDefines

MAKE_COMPILED_DEFINES := $(shell cat "$(OCCA_DIR)/scripts/compiledDefinesTemplate.hpp" | \
                                 sed "s,@@OCCA_OS@@,$(OCCA_OS),g;\
                                      s,@@OCCA_USING_VS@@,$(OCCA_USING_VS),g;\
                                      s,@@OCCA_UNSAFE@@,$(OCCA_UNSAFE),g;\
                                      s,@@OCCA_MPI_ENABLED@@,$(OCCA_MPI_ENABLED),g; \
                                      s,@@OCCA_OPENMP_ENABLED@@,$(OCCA_OPENMP_ENABLED),g;\
                                      s,@@OCCA_CUDA_ENABLED@@,$(OCCA_CUDA_ENABLED),g;\
                                      s,@@OCCA_HIP_ENABLED@@,$(OCCA_HIP_ENABLED),g;\
                                      s,@@OCCA_OPENCL_ENABLED@@,$(OCCA_OPENCL_ENABLED),g;\
                                      s,@@OCCA_METAL_ENABLED@@,$(OCCA_METAL_ENABLED),g;\
                                      s,@@OCCA_BUILD_DIR@@,$(OCCA_BUILD_DIR),g;"\
                                      > "$(NEW_COMPILED_DEFINES)")

MAKE_COMPILED_DEFINES := $(shell \
 [ ! -f "$(COMPILED_DEFINES)" -o -n "$(shell diff -q $(COMPILED_DEFINES) $(NEW_COMPILED_DEFINES))" ] \
 && cp "$(NEW_COMPILED_DEFINES)" "$(COMPILED_DEFINES)" \
)
MAKE_COMPILED_DEFINES := $(shell rm $(NEW_COMPILED_DEFINES))

all: $(objects) $(outputs)
	@(. $(OCCA_DIR)/include/occa/scripts/shellTools.sh && installOcca)
	@echo "> occa info"
	@$(OCCA_DIR)/bin/occa info

$(COMPILED_DEFINES_CHANGED):
#=================================================


#---[ Builds ]------------------------------------
#  ---[ libocca ]-------------
$(libPath)/libocca.$(soExt):$(objects) $(headers) $(COMPILED_DEFINES)
	mkdir -p $(libPath)
	$(compiler) $(compilerFlags) $(sharedFlag) $(pthreadFlag) $(soNameFlag) -o $(libPath)/libocca.$(soExt) $(flags) $(objects) $(paths) $(linkerFlags)

$(libPath)/libocca_fortran.$(soExt): $(fObjects) $(libPath)/libocca.$(soExt)
	@mkdir -p $(libPath)
	$(fCompiler) $(fCompilerFlags) $(sharedFlag) $(pthreadFlag) $(fSoNameFlag) -o $@ $^ $(flags) $(fPaths) $(filter-out -locca_fortran, $(fLinkerFlags))

$(binPath)/occa:$(OCCA_DIR)/bin/occa.cpp $(libPath)/libocca.$(soExt) $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(binPath)
	$(compiler) $(compilerFlags) -o $(binPath)/occa -Wl,-rpath,$(libPath) $(flags) $(OCCA_DIR)/bin/occa.cpp $(paths) $(linkerFlags) -L$(OCCA_DIR)/lib -locca
#  ===========================

# Sources with C++ headers and template headers
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp $(OCCA_DIR)/include/occa/%.tpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

# Sources with C++ headers
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.hpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

# Sources with C headers
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(OCCA_DIR)/include/occa/%.h $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

# Objective-C++ sources
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.mm $(OCCA_DIR)/include/occa/%.hpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	clang++ -x objective-c++ -o $@ $(flags) -c $(paths) $<

# Header-less sources
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.cpp $(COMPILED_DEFINES_CHANGED)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

# Fortran sources
include $(OCCA_DIR)/scripts/Make.fortran_rules
$(OCCA_DIR)/obj/%.o:$(OCCA_DIR)/src/%.f90
	@mkdir -p $(modPath)
	@mkdir -p $(abspath $(dir $@))
	$(fCompiler) $(fCompilerFlags) -o $@ $(flags) -c $(fPaths) $<
#=================================================


#---[ Test ]--------------------------------------
tests: $(tests)

test: unit-tests e2e-tests bin-tests

unit-tests: $(tests)
	@$(testPath)/run_tests
	@if [ $$? -ne 0 ]; then \
	  @exit 1;              \
	fi

e2e-tests: unit-tests
	@FORTRAN_EXAMPLES=$(fortranEnabled) $(testPath)/run_examples
	@if [ $$? -ne 0 ]; then \
	  @exit 1;              \
	fi

bin-tests: e2e-tests
	@$(testPath)/run_bin_tests
	@if [ $$? -ne 0 ]; then \
	  @exit 1;              \
	fi

$(testPath)/bin/%:$(testPath)/src/%.cpp $(outputs)
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(testFlags) $(pthreadFlag) -o $@ -Wl,-rpath,$(libPath) $(flags) $< $(paths) $(linkerFlags) -L$(OCCA_DIR)/lib -locca

# Fortran tests
fTests: $(fTests)

$(objPath)/%.o: $(testPath)/src/fortran/typedefs_helper.cpp
	@mkdir -p $(abspath $(dir $@))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(testPath)/bin/fortran/typedefs: $(objPath)/tests/fortran/typedefs_helper.o

$(testPath)/bin/%: $(testPath)/src/%.f90 $(libPath)/libocca_fortran.$(soExt)
	@mkdir -p $(abspath $(dir $@))
	$(fCompiler) $(fCompilerFlags) -o $@ $^ -Wl,-rpath,$(libPath) $(flags) $(fPaths) $(fLinkerFlags)
#=================================================


#---[ Clean ]-------------------------------------
clean:
	rm -rf $(objPath)/*
	rm -rf $(modPath)/*
	rm -rf $(binPath)/occa
	rm -rf $(testPath)/bin
	rm -rf $(testPath)/src/io/locks
	rm  -f $(libPath)/libocca.$(soExt)
	rm  -f $(libPath)/libocca_fortran.$(soExt)
#=================================================


#---[ Info ]--------------------------------------
info:
	$(info --------------------------------)
	$(info CXX            = $(or $(CXX),(empty)))
	$(info CXXFLAGS       = $(or $(CXXFLAGS),(empty)))
	$(info CC             = $(or $(CC),(empty)))
	$(info CFLAGS         = $(or $(CFLAGS),(empty)))
	$(info FC             = $(or $(FC),(empty)))
	$(info FFLAGS         = $(or $(FFLAGS),(empty)))
	$(info LDFLAGS        = $(or $(LDFLAGS),(empty)))
	$(info --------------------------------)
	$(info compiler       = $(value compiler))
	$(info vendor         = $(vendor))
	$(info compilerFlags  = $(compilerFlags))
	$(info cCompiler      = $(value cCompiler))
	$(info cCompilerFlags = $(cCompilerFlags))
	$(info fCompiler      = $(value fCompiler))
	$(info fCompilerFlags = $(fCompilerFlags))
	$(info flags          = $(flags))
	$(info paths          = $(paths))
	$(info fPaths         = $(fPaths))
	$(info sharedFlag     = $(sharedFlag))
	$(info pthreadFlag    = $(pthreadFlag))
	$(info linkerFlags    = $(linkerFlags))
	$(info --------------------------------)
	$(info debugEnabled   = $(debugEnabled))
	$(info checkEnabled   = $(checkEnabled))
	$(info debugFlags     = $(debugFlags))
	$(info releaseFlags   = $(releaseFlags))
	$(info picFlag        = $(picFlag))
	$(info --------------------------------)
	$(info fortranEnabled = $(fortranEnabled))
	$(info mpiEnabled     = $(mpiEnabled))
	$(info openmpEnabled  = $(openmpEnabled))
	$(info openclEnabled  = $(openclEnabled))
	$(info metalEnabled   = $(metalEnabled))
	$(info cudaEnabled    = $(cudaEnabled))
	$(info hipEnabled     = $(hipEnabled))
	$(info --------------------------------)
	@true
#=================================================


#---[ Print ]-------------------------------------
# Print the contents of a makefile variable, e.g.: 'make print-compiler'.
print-%:
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true
#=================================================
