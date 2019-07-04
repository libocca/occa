#ifndef OCCA_DISABLE_VARIADIC_MACROS
#  ifndef OCCA_DEFINES_OKL_HEADER
#  define OCCA_DEFINES_OKL_HEADER


#define OCCA_INLINED_KERNEL_NAME "_occa_inlinedKernel"


#define OCCA_INLINED_OKL(OKL_SCOPE, OKL_SOURCE)               \
  do {                                                        \
    static occa::kernelBuilder _inlinedKernelBuilder = (      \
      occa::kernelBuilder::fromString(                        \
        occa::formatInlinedKernel(OKL_SCOPE,                  \
                                  #OKL_SOURCE,                \
                                  OCCA_INLINED_KERNEL_NAME),  \
        OCCA_INLINED_KERNEL_NAME                              \
      )                                                       \
    );                                                        \
    _inlinedKernelBuilder.run(scope.getDevice(), OKL_SCOPE);  \
  } while (false)


#define OCCA_C_INLINED_OKL(OKL_SCOPE, OKL_SOURCE)           \
  do {                                                      \
    static occaKernelBuilder _inlinedKernelBuilder = (      \
      occaKernelBuilderFromString(                          \
        occaFormatInlinedKernel(OKL_SCOPE,                  \
                                #OKL_SOURCE,                \
                                OCCA_INLINED_KERNEL_NAME),  \
        OCCA_INLINED_KERNEL_NAME                            \
      )                                                     \
    );                                                      \
    occaKernelBuilderRun(_inlinedKernelBuilder,             \
                         occaScopeGetDevice(OKL_SCOPE),     \
                         OKL_SCOPE);                        \
  } while (false)


#  endif
#endif
