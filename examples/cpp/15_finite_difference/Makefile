PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../../scripts/build/Makefile
else
  include ${OCCA_DIR}/scripts/build/Makefile
endif

ifeq ($(usingLinux),1)
  flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
  linkerFlags += -L/usr/lib/fglrx/ -lGL -lX11 -lGLU -lglut -lGL
  paths += -I/opt/X11/include -L/opt/X11/lib
else ifeq ($(usingMacOS),1)
  flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
  linkerFlags += -framework OpenGL -lX11 -lGL -framework OpenGL -framework GLUT
  paths += -I/opt/X11/include -L/opt/X11/lib
endif

flags += -D tFloat=float

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources = $(wildcard $(srcPath)/*.cpp)

objects = $(subst $(srcPath)/,$(objPath)/,$(sources:.cpp=.o))

flags += -I$(PROJ_DIR)/include

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(linkerFlags)
	@if which install_name_tool > /dev/null 2>&1; then \
		install_name_tool -add_rpath "${OCCA_DIR}/lib" ${PROJ_DIR}/main; \
	fi

$(objPath)/%.o:$(srcPath)/%.cpp $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(objPath)/*;
	rm -f ${PROJ_DIR}/main
#=================================================
