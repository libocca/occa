PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ifndef OCCA_DIR
  OCCA_DIR=$(PROJ_DIR)/../../..
endif
include $(OCCA_DIR)/scripts/build/Makefile

# The OCCA source directory
libSrcPath = $(OCCA_DIR)/src

# Here we want to statically compile the example, thus remove libocca_fortran
# from the linker flags, the global include path from fPaths and set the module
# path to the local object directory
fLinkerFlags := $(filter-out -locca_fortran, $(fLinkerFlags))
fPaths := $(filter-out -I$(subst $(PROJ_DIR),$(OCCA_DIR),$(modPath)), $(fPaths))
fPaths += $(fModuleDirFlag)$(objPath)/fortran


#---[ COMPILATION ]-------------------------------
all: main

main: $(fObjects)

include $(OCCA_DIR)/scripts/build/Make.fortran_rules

%: %.f90
	@mkdir -p $(abspath $(dir $@))
	$(fCompiler) $(fCompilerFlags) -o $@ $^ $(flags) $(fPaths) $(fLinkerFlags)

$(objPath)/%.o: $(libSrcPath)/%.f90
	@mkdir -p $(abspath $(dir $@))
	$(fCompiler) $(fCompilerFlags) -o $@ $(flags) -c $(fPaths) $<

clean:
	rm -rf $(objPath)/*
	rm -rf $(PROJ_DIR)/main
#=================================================


#---[ RUN ]---------------------------------------
run: main
	$(PROJ_DIR)/main --verbose
#=================================================
