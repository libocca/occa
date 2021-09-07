PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../../scripts/build/Makefile
else
  include ${OCCA_DIR}/scripts/build/Makefile
endif

DEBUG=0

ifeq ($(GL), 0)
    flags += -D OCCA_GL_ENABLED=0
else
  ifeq ($(usingLinux),1)
    flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
    linkerFlags += -L/usr/lib/fglrx/ -lGL -lX11 -lGLU -lglut -lGL
    paths += -I/opt/X11/include -L/opt/X11/lib
  else ifeq ($(usingMacOS),1)
    flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
    linkerFlags += -framework OpenGL -lX11 -lGL -framework OpenGL -framework GLUT
    paths += -I/opt/X11/include -L/opt/X11/lib
  endif
endif

flags += -I$(PROJ_DIR)/include

ifeq ($(DOUBLE), 1)
  flags += -D tFloat=double -D tFloat3=occa::double3
else
  flags += -D tFloat=float -D tFloat3=occa::float3
endif

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources = $(wildcard $(srcPath)/*.cpp)

objects = $(subst $(srcPath)/,$(objPath)/,$(sources:.cpp=.o))

${PROJ_DIR}/main: $(objPath) $(objects) $(headers) ${PROJ_DIR}/main.cpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(linkerFlags)
	@if which install_name_tool > /dev/null 2>&1; then \
		install_name_tool -add_rpath "${OCCA_DIR}/lib" ${PROJ_DIR}/main; \
	fi

$(objPath)/%.o:$(srcPath)/%.cpp $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(objPath):
	mkdir -p $(objPath)

clean:
	rm -f $(objPath)/*;
	rm -f ${PROJ_DIR}/main
#=================================================
