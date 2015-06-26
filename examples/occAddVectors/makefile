ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

include ${OCCA_DIR}/scripts/makefile

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

executables = main main_c

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
	executables += main_f90
endif
endif

ifdef OCCA_FLOOPY_ENABLED
ifeq ($(OCCA_FLOOPY_ENABLED), 1)
	executables += mainFloopy

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
	executables += mainFloopy_f90
endif
endif

endif
endif

all: $(executables)

main: $(objects) $(headers) main.cpp
	$(compiler) $(compilerFlags) -o main $(flags) $(objects) main.cpp $(paths) $(links)

main_c: main.c
	$(compiler) $(compilerFlags) -o main_c $(flags) $(objects) main.c $(paths) $(links)

main_f90: main.f90
	$(fCompiler) $(fCompilerFlags) -o main_f90 $(flags) $(objects) main.f90 $(paths) $(links)

mainFloopy: $(objects) $(headers) mainFloopy.cpp
	$(compiler) $(compilerFlags) -o mainFloopy $(flags) $(objects) mainFloopy.cpp $(paths) $(links)

mainFloopy_f90: $(objects) $(headers) mainFloopy.f90
	$(fCompiler) $(fCompilerFlags) -o mainFloopy_f90 $(flags) $(objects) mainFloopy.f90 $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f main main_c main_f90 mainFloopy mainFloopy_f90
#=================================================
