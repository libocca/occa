// Just in case someone wants to run with an older format than C99
#ifndef OCCA_DISABLE_VARIADIC_MACROS
#  ifndef OCCA_DEFINES_OKL_HEADER
#  define OCCA_DEFINES_OKL_HEADER


#define OCCA_INLINED_KERNEL_NAME "_occa_inlinedKernel"

#define OCCA_JIT_2(OKL_ARGS, OKL_SOURCE)        \
  OCCA_JIT_3(occa::properties(), OKL_ARGS, OKL_SOURCE)

#define OCCA_JIT_3(OCCA_PROPS, OKL_ARGS, OKL_SOURCE)      \
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

#define OCCA_JIT(...)                                 \
  EXPAND_OCCA_JIT(OCCA_ARG_COUNT(__VA_ARGS__), __VA_ARGS__)

#define EXPAND_OCCA_JIT(ARG_COUNT_FUNC, ...)      \
  EXPAND_OCCA_JIT_2(ARG_COUNT_FUNC, __VA_ARGS__)

#define EXPAND_OCCA_JIT_2(ARG_COUNT, ...)                 \
  EXPAND_OCCA_JIT_3(OCCA_JIT_ ## ARG_COUNT, __VA_ARGS__)

#define EXPAND_OCCA_JIT_3(OCCA_JIT_MACRO, ...)  \
  OCCA_JIT_MACRO (__VA_ARGS__)


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
