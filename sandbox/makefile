#---[ Paths ]-------------------------------------
oPath = obj
sPath = src
iPath = inc
#=================================================

paths = -I${OCCA_DIR}/include -L${OCCA_DIR}/lib -locca
flags = -Qunused-arguments -D__extern_always_inline=inline -g -Wno-unused-value

main: main.cpp
	clang++ -o main $(flags) $(paths) -lm main.cpp

$(oPath)/%.o:$(sPath)/%.cpp $(iPath)/%.hpp$(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	clang++ -o $@ $(flags) -c $(paths) $<

clean:
	rm -f main
	rm -f $(oPath)/*;
