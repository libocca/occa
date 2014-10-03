#---[ Paths ]-------------------------------------
oPath = obj
sPath = src
iPath = inc
#=================================================

paths = -I./$(sPath) -I./$(iPath)
flags = -Qunused-arguments -D__extern_always_inline=inline -g -Wno-unused-value

headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

main: main.cpp $(objects) $(headers)
	clang++ -o main $(flags) $(objects) $(paths) -lm main.cpp

$(oPath)/%.o:$(sPath)/%.cpp $(iPath)/%.hpp$(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	clang++ -o $@ $(flags) -c $(paths) $<

clean:
	rm -f main
	rm -f $(oPath)/*;
