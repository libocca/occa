#ifndef OCCA_C_HEADER
#define OCCA_C_HEADER

#include <occa/c/base.h>
#include <occa/c/cli.h>
#include <occa/c/device.h>
#include <occa/c/dtype.h>
#include <occa/c/io.h>
#include <occa/c/json.h>
#include <occa/c/kernel.h>
#include <occa/c/kernelBuilder.h>
#include <occa/c/memory.h>
#include <occa/c/properties.h>
#include <occa/c/scope.h>
#include <occa/c/uva.h>

// Just in case someone wants to run with an older format than C99
#ifndef OCCA_DISABLE_VARIADIC_MACROS

#define OCCA_C_RUN_KERNEL3(kernel, N, ...)      \
  occaKernelRunN(kernel, N, __VA_ARGS__)

#define OCCA_C_RUN_KERNEL2(kernel, N, ...)      \
  OCCA_C_RUN_KERNEL3(kernel, N, __VA_ARGS__)

#define OCCA_C_RUN_KERNEL1(kernel, N, ...)      \
  OCCA_C_RUN_KERNEL2(kernel, N, __VA_ARGS__)

#define occaKernelRun(kernel, ...)                                      \
  OCCA_C_RUN_KERNEL1(kernel, OCCA_ARG_COUNT(__VA_ARGS__), __VA_ARGS__)

#endif // OCCA_DISABLE_VARIADIC_MACROS

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
