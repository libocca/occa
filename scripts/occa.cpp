#include "occa.hpp"

int main(int argc, char **argv){
  for(int i = 0; i < argc; ++i)
    std::cout << "argv[" << i << "] = " << argv[i] << '\n';

  return 0;
}