#include "occa.hpp"

int main(int argc, char **argv){
  occa::kernelInfo kInfo;

  kInfo.addDefine("A", (int)    0);
  kInfo.addDefine("B", (double) 0.0);
  kInfo.addDefine("C", (char)   'C');
  kInfo.addDefine("D", "This is a string");

  return 0;
}
