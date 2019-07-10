#ifndef OCCA_DISABLE_VARIADIC_MACROS
#  ifndef OCCA_DEFINES_OKL_HEADER
#  define OCCA_DEFINES_OKL_HEADER


#define OCCA_INLINED_KERNEL_NAME "_occa_inlinedKernel"

#define OCCA_JIT(OCCA_PROPS, OKL_ARGS, OKL_SOURCE)        \
  do {                                                    \
    static occa::kernelBuilder _inlinedKernelBuilder = (  \
      occa::kernelBuilder::fromString(                    \
        occa::formatInlinedKernelFromArgs(                \
          occa::getInlinedKernelUnnamedScope OKL_ARGS,    \
          #OKL_ARGS,                                      \
          #OKL_SOURCE,                                    \
          OCCA_INLINED_KERNEL_NAME                        \
        ),                                                \
        OCCA_INLINED_KERNEL_NAME                          \
      )                                                   \
    );                                                    \
    occa::kernel _inlinedKernel = (                       \
      _inlinedKernelBuilder.build(occa::getDevice(),      \
                                  OCCA_PROPS)             \
    );                                                    \
    _inlinedKernel OKL_ARGS;                              \
  } while (false)


#define OCCA_JIT_WITH_SCOPE(OKL_SCOPE, OKL_SOURCE)                    \
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
