ifndef OCCA_DIR
ERROR:
  @echo "Error, environment variable [OCCA_DIR] is not set"
endif

include ${OCCA_DIR}/scripts/makefile

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
