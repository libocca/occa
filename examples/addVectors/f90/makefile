ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

include ${OCCA_DIR}/scripts/makefile

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

main_f90: main.f90
	$(fCompiler) $(fCompilerFlags) -o main $(flags) $(objects) main.f90 $(paths) $(links)

clean:
	rm -f $(oPath)/*;
	rm -f main
#=================================================
