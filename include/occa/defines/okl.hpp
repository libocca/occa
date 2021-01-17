#ifndef OCCA_DEFINES_OKL_HEADER
#define OCCA_DEFINES_OKL_HEADER

#ifdef OCCA_JIT
#  undef OCCA_JIT
#endif

#define OCCA_JIT(OKL_SCOPE, OKL_SOURCE)                 \
  do {                                                  \
    static ::occa::kernelBuilder _occaJitKernelBuilder( \
      #OKL_SOURCE,                                      \
      "_occa_jit_kernel"                                \
    );                                                  \
    _occaJitKernelBuilder.run(OKL_SCOPE);               \
  } while (false)

#endif
