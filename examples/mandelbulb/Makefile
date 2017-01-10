PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../scripts/Makefile
else
  include ${OCCA_DIR}/scripts/Makefile
endif

DEBUG=0

ifeq ($(GL), 0)
    flags += -D OCCA_GL_ENABLED=0
else
  ifeq ($(usingLinux),1)
    flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
    links += -L/usr/lib/fglrx/ -lGL -lX11 -lGLU -lglut -lGL
    paths += -I/opt/X11/include -L/opt/X11/lib
  else ifeq ($(usingOSX),1)
    flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
    links += -framework OpenGL -lX11 -lGL -framework OpenGL -framework GLUT
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
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) $(links)

$(objPath)/%.o:$(srcPath)/%.cpp $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

$(objPath):
	mkdir -p $(objPath)

clean:
	rm -f $(objPath)/*;
	rm -f ${PROJ_DIR}/main
#=================================================
