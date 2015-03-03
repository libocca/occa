ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

include ${OCCA_DIR}/scripts/makefile

#---[ COMPILATION ]-------------------------------
main: main.cpp
	$(compiler) $(compilerFlags) -o main $(flags) $(objects) main.cpp $(paths) $(links)

clean:
	rm -f $(oPath)/*;
	rm -f main
#=================================================
