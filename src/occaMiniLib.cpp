#include "occaMiniLib.hpp"

#define OCCA_MEMSET_KERNEL_SOURCE(TYPE, CAP_TYPE)       \
  "kernel void memset" #CAP_TYPE "K(" #TYPE " *ptr,\n"  \
  "                        const " #TYPE " value,\n"    \
  "                        const int count){\n"         \
  "  for(int b = 0; b < count; b += 128; outer){\n"     \
  "    for(int i = b; i < (b + 128); ++i; inner){\n"    \
  "      if(i < count)\n"                               \
  "        ptr[i] = value;\n"                           \
  "    }\n"                                             \
  "  }\n"                                               \
  "}"

namespace occa {
  template <>
  void memset<bool>(void *ptr, const bool &value, uintptr_t count){
    static occa::kernel *memsetBoolK = NULL;

    if(memsetBoolK != NULL){
      (*memsetBoolK)(ptr, value, (int) count);
    }
    else {
      const bool vc = verboseCompilation_f;

      if(vc)
        verboseCompilation_f = false;

      memsetBoolK  = new occa::kernel;
      *memsetBoolK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(bool, Bool),
                                       "memsetBoolK");

      (*memsetBoolK)(ptr, value, (int) count);

      verboseCompilation_f = vc;
    }
  }

  template <>
  void memset<char>(void *ptr, const char &value, uintptr_t count){
    static occa::kernel *memsetCharK = NULL;

    if(memsetCharK != NULL){
      (*memsetCharK)(ptr, value, (int) count);
    }
    else {
      const bool vc = verboseCompilation_f;

      if(vc)
        verboseCompilation_f = false;

      memsetCharK  = new occa::kernel;
      *memsetCharK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(char, Char),
                                       "memsetCharK");

      (*memsetCharK)(ptr, value, (int) count);

      verboseCompilation_f = vc;
    }
  }

  template <>
  void memset<short>(void *ptr, const short &value, uintptr_t count){
    static occa::kernel *memsetShortK = NULL;

    if(memsetShortK != NULL){
      (*memsetShortK)(ptr, value, (int) count);
    }
    else {
      const bool vc = verboseCompilation_f;

      if(vc)
        verboseCompilation_f = false;

      memsetShortK  = new occa::kernel;
      *memsetShortK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(short, Short),
                                        "memsetShortK");

      (*memsetShortK)(ptr, value, (int) count);

      verboseCompilation_f = vc;
    }
  }

  template <>
  void memset<int>(void *ptr, const int &value, uintptr_t count){
    static occa::kernel *memsetIntK = NULL;

    if(memsetIntK != NULL){
      (*memsetIntK)(ptr, value, (int) count);
    }
    else {
      const bool vc = verboseCompilation_f;

      if(vc)
        verboseCompilation_f = false;

      memsetIntK  = new occa::kernel;
      *memsetIntK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(int, Int),
                                      "memsetIntK");

      (*memsetIntK)(ptr, value, (int) count);

      verboseCompilation_f = vc;
    }
  }

  template <>
  void memset<long>(void *ptr, const long &value, uintptr_t count){
    static occa::kernel *memsetLongK = NULL;

    if(memsetLongK != NULL){
      (*memsetLongK)(ptr, value, (int) count);
    }
    else {
      const bool vc = verboseCompilation_f;

      if(vc)
        verboseCompilation_f = false;

      memsetLongK  = new occa::kernel;
      *memsetLongK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(long, Long),
                                       "memsetLongK");

      (*memsetLongK)(ptr, value, (int) count);

      verboseCompilation_f = vc;
    }
  }

  template <>
  void memset<float>(void *ptr, const float &value, uintptr_t count){
    static occa::kernel *memsetFloatK = NULL;

    if(memsetFloatK != NULL){
      (*memsetFloatK)(ptr, value, (int) count);
    }
    else {
      const bool vc = verboseCompilation_f;

      if(vc)
        verboseCompilation_f = false;

      memsetFloatK  = new occa::kernel;
      *memsetFloatK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(float, Float),
                                        "memsetFloatK");

      (*memsetFloatK)(ptr, value, (int) count);

      verboseCompilation_f = vc;
    }
  }

  template <>
  void memset<double>(void *ptr, const double &value, uintptr_t count){
    static occa::kernel *memsetDoubleK = NULL;

    if(memsetDoubleK != NULL){
      (*memsetDoubleK)(ptr, value, (int) count);
    }
    else {
      const bool vc = verboseCompilation_f;

      if(vc)
        verboseCompilation_f = false;

      memsetDoubleK  = new occa::kernel;
      *memsetDoubleK = occa::buildKernel(OCCA_MEMSET_KERNEL_SOURCE(double, Double),
                                         "memsetDoubleK");

      (*memsetDoubleK)(ptr, value, (int) count);

      verboseCompilation_f = vc;
    }
  }
};
