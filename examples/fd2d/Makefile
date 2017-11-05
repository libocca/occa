PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifndef OCCA_DIR
  include $(PROJ_DIR)/../../scripts/Makefile
else
  include ${OCCA_DIR}/scripts/Makefile
endif

ifeq ($(usingLinux),1)
  flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
  links += -L/usr/lib/fglrx/ -lGL -lX11 -lGLU -lglut -lGL
  paths += -I/opt/X11/include -L/opt/X11/lib
else ifeq ($(usingOSX),1)
  flags += -D OCCA_GL_ENABLED=1 -D GL_GLEXT_PROTOTYPES
  links += -framework OpenGL -lX11 -lGL -framework OpenGL -framework GLUT
  paths += -I/opt/X11/include -L/opt/X11/lib
endif

flags += -D tFloat=float

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(incPath)/*.hpp) $(wildcard $(incPath)/*.tpp)
sources = $(wildcard $(srcPath)/*.cpp)

objects = $(subst $(srcPath)/,$(objPath)/,$(sources:.cpp=.o))

flags += -I$(PROJ_DIR)/include

${PROJ_DIR}/main: $(objects) $(headers) ${PROJ_DIR}/main.cpp
	$(compiler) $(compilerFlags) -o ${PROJ_DIR}/main $(flags) $(objects) ${PROJ_DIR}/main.cpp $(paths) -L${OCCA_DIR}/lib $(links)

$(objPath)/%.o:$(srcPath)/%.cpp $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(srcPath)/,$(incPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(objPath)/*;
	rm -f ${PROJ_DIR}/main
#=================================================
