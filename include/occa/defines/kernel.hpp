#ifndef OCCA_DISABLE_VARIADIC_MACROS
#  ifndef OCCA_DEFINES_KERNEL_HEADER
#  define OCCA_DEFINES_KERNEL_HEADER

#define OCCA_INLINED_KERNEL_NAME "_occaInlinedKernel"

#define OCCA_INLINED_KERNEL(KERNEL_ARGS, KERNEL_PROPS, KERNEL_SOURCE)   \
  do {                                                                  \
    static occa::kernelBuilder _inlinedKernelBuilder = (                \
      occa::kernelBuilder::fromString(                                  \
        occa::formatInlinedKernel(occa::getInlinedKernelArgTypes KERNEL_ARGS, \
                                  #KERNEL_ARGS,                         \
                                  #KERNEL_SOURCE,                       \
                                  OCCA_INLINED_KERNEL_NAME),            \
        OCCA_INLINED_KERNEL_NAME                                        \
      )                                                                 \
    );                                                                  \
    occa::kernel _inlinedKernel = (                                     \
      _inlinedKernelBuilder.build(occa::getDevice(),                    \
                                  KERNEL_PROPS)                         \
    );                                                                  \
    _inlinedKernel KERNEL_ARGS;                                         \
  } while (false)

#  endif
#endif
