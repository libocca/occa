#ifndef OCCA_C_EXPERIMENTAL_HEADER
#define OCCA_C_EXPERIMENTAL_HEADER

#include <occa/c/experimental/kernelBuilder.h>

#ifdef OCCA_JIT
#  undef OCCA_JIT
#endif

#define OCCA_JIT(OCCA_SCOPE, OKL_SOURCE)                \
  do {                                                  \
    static occaKernelBuilder _occaJitKernelBuilder;     \
    static int _occaJitKernelIsDefined = 0;             \
    if (!_occaJitKernelIsDefined) {                     \
      _occaJitKernelBuilder = occaCreateKernelBuilder(  \
        #OKL_SOURCE,                                    \
        "_occa_jit_kernel"                              \
      );                                                \
      _occaJitKernelIsDefined = 1;                      \
    }                                                   \
    occaKernelBuilderRun(_occaJitKernelBuilder,         \
                         OCCA_SCOPE);                   \
  } while (0)

#endif
