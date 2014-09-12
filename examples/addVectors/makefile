ifndef OCCA_DIR
	OCCA_DIR = ../../
endif

include ${OCCA_DIR}/scripts/makefile

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

examples = main main_c

ifdef OCCA_FORTRAN_ENABLED
ifeq ($(OCCA_FORTRAN_ENABLED), 1)
	examples += main_f90
endif
endif

executables: $(examples)

main_f90: main.f90
	$(Fcompiler) $(FcompilerFlags) -o main_f90 $(flags) $(objects) main.f90 $(paths) $(links)

main_c: main.c
	$(compiler) $(compilerFlags) -o main_c $(flags) $(objects) main.c $(paths) $(links)

main: $(objects) $(headers) main.cpp
	$(compiler) $(compilerFlags) -o main $(flags) $(objects) main.cpp $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f main_f90
	rm -f main_c
	rm -f main;
#=================================================
