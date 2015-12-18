ifndef OCCA_DIR
ERROR:
  @echo "Error, environment variable [OCCA_DIR] is not set"
endif

DEBUG=0

include ${OCCA_DIR}/scripts/makefile

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

ifeq ($(DOUBLE), 1)
  flags += -D tFloat=double -D tFloat3=occa::double3
else
  flags += -D tFloat=float -D tFloat3=occa::float3
endif

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

main: $(objects) $(headers) main.cpp
	$(compiler) $(compilerFlags) -o main $(flags) $(objects) main.cpp $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f main;
#=================================================
