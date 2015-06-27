#include "occa_c.h"

int main(int argc, char **argv){
  occaKernelInfo kInfo = occaCreateKernelInfo();

  occaKernelInfoAddDefine(kInfo, "A", occaInt(0));
  occaKernelInfoAddDefine(kInfo, "B", occaDouble(0.0));
  occaKernelInfoAddDefine(kInfo, "C", occaChar('C'));
  occaKernelInfoAddDefine(kInfo, "D", occaString("D"));

  occaKernelInfoFree(kInfo);

  return 0;
}
