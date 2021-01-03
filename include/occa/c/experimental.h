#ifndef OCCA_C_EXPERIMENTAL_HEADER
#define OCCA_C_EXPERIMENTAL_HEADER

#include <occa/c/experimental/kernelBuilder.h>
#include <occa/c/experimental/scope.h>

#ifdef OCCA_JIT
#  undef OCCA_JIT
#endif

#define OCCA_JIT(OCCA_SCOPE, OKL_SOURCE)            \
  do {                                              \
    static occaKernelBuilder _inlinedKernelBuilder; \
    static int _inlinedKernelIsDefined = 0;         \
    if (!_inlinedKernelIsDefined) {                 \
      _inlinedKernelBuilder = (                     \
        occaKernelBuilderFromInlinedOkl(            \
          OCCA_SCOPE,                               \
          #OKL_SOURCE,                              \
          OCCA_INLINED_KERNEL_NAME                  \
        )                                           \
      );                                            \
      _inlinedKernelIsDefined = 1;                  \
    }                                               \
    occaKernelBuilderRun(_inlinedKernelBuilder,     \
                         OCCA_SCOPE);               \
  } while (0)

#endif
