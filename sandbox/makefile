PROJ_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../../scripts/makefile
else
  include ${OCCA_DIR}/scripts/makefile
endif

#---[ COMPILATION ]-------------------------------
${PROJ_DIR}/main: ${PROJ_DIR}/main.cpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(links)

clean:
	rm -f $(oPath)/*;
	rm -f ${PROJ_DIR}/main
#=================================================
