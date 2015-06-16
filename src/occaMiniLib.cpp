#include "occaMiniLib.hpp"

#define OCCA_MEMSET_KERNEL_SOURCE(TYPE, CAP_TYPE)       \
  "kernel void memset" #CAP_TYPE "K(" #TYPE " *ptr,\n"  \
  "                        const " #TYPE " value,\n"    \
  "                        const int count){\n"         \
  "  for(int b = 0; b < count; b += 128; outer){\n"     \
  "    for(int i = b; i < (b + 128); ++i; inner){\n"    \
  "      if(i < count)"                                 \
  "        ptr[i] = value;"                             \
  "    }\n"                                             \
  "  }\n"                                               \
  "}"

namespace occa {
  void memsetBool(void *ptr, const bool &value, uintptr_t count){
    static occa::kernel *memsetBoolK = NULL;

    if(memsetBoolK != NULL){
      (*memsetBoolK)(ptr, value, (int) count);
    }
    else {
      memsetBoolK  = new occa::kernel;
      *memsetBoolK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(bool, Bool), "memsetBoolK");
      (*memsetBoolK)(ptr, value, count);
    }
  }

  void memsetChar(void *ptr, const char &value, uintptr_t count){
    static occa::kernel *memsetCharK = NULL;

    if(memsetCharK != NULL){
      (*memsetCharK)(ptr, value, (int) count);
    }
    else {
      memsetCharK  = new occa::kernel;
      *memsetCharK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(char, Char), "memsetCharK");
      (*memsetCharK)(ptr, value, count);
    }
  }

  void memsetShort(void *ptr, const short &value, uintptr_t count){
    static occa::kernel *memsetShortK = NULL;

    if(memsetShortK != NULL){
      (*memsetShortK)(ptr, value, (int) count);
    }
    else {
      memsetShortK  = new occa::kernel;
      *memsetShortK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(short, Short), "memsetShortK");
      (*memsetShortK)(ptr, value, count);
    }
  }

  void memsetInt(void *ptr, const int &value, uintptr_t count){
    static occa::kernel *memsetIntK = NULL;

    if(memsetIntK != NULL){
      (*memsetIntK)(ptr, value, (int) count);
    }
    else {
      memsetIntK  = new occa::kernel;
      *memsetIntK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(int, Int), "memsetIntK");
      (*memsetIntK)(ptr, value, count);
    }
  }

  void memsetLong(void *ptr, const long &value, uintptr_t count){
    static occa::kernel *memsetLongK = NULL;

    if(memsetLongK != NULL){
      (*memsetLongK)(ptr, value, (int) count);
    }
    else {
      memsetLongK  = new occa::kernel;
      *memsetLongK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(long, Long), "memsetLongK");
      (*memsetLongK)(ptr, value, count);
    }
  }

  void memsetFloat(void *ptr, const float &value, uintptr_t count){
    static occa::kernel *memsetFloatK = NULL;

    if(memsetFloatK != NULL){
      (*memsetFloatK)(ptr, value, (int) count);
    }
    else {
      memsetFloatK  = new occa::kernel;
      *memsetFloatK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(float, Float), "memsetFloatK");
      (*memsetFloatK)(ptr, value, count);
    }
  }

  void memsetDouble(void *ptr, const double &value, uintptr_t count){
    static occa::kernel *memsetDoubleK = NULL;

    if(memsetDoubleK != NULL){
      (*memsetDoubleK)(ptr, value, (int) count);
    }
    else {
      memsetDoubleK  = new occa::kernel;
      *memsetDoubleK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(double, Double), "memsetDoubleK");
      (*memsetDoubleK)(ptr, value, count);
    }
  }
};
