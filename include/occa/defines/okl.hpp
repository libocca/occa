// Just in case someone wants to run with an older format than C99
#ifndef OCCA_DISABLE_VARIADIC_MACROS
#  ifndef OCCA_DEFINES_OKL_HEADER
#  define OCCA_DEFINES_OKL_HEADER


#define OCCA_INLINED_KERNEL_NAME "_occa_inlinedKernel"

#define OCCA_JIT(OKL_SCOPE, OKL_SOURCE)                               \
  do {                                                                \
    static occa::kernelBuilder _inlinedKernelBuilder = (              \
      occa::kernelBuilder::fromString(                                \
        occa::formatInlinedKernelFromScope(OKL_SCOPE,                 \
                                           #OKL_SOURCE,               \
                                           OCCA_INLINED_KERNEL_NAME), \
        OCCA_INLINED_KERNEL_NAME                                      \
      )                                                               \
    );                                                                \
    _inlinedKernelBuilder.run(OKL_SCOPE);                             \
  } while (false)


#  endif
#endif
