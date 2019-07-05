#ifndef OCCA_C_KERNELBUILDER_HEADER
#define OCCA_C_KERNELBUILDER_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC occaKernelBuilder OCCA_RFUNC occaKernelBuilderFromInlinedOkl(
  occaScope scope,
  const char *kernelSource,
  const char *kernelName
);

OCCA_LFUNC void OCCA_RFUNC occaKernelBuilderRun(
  occaKernelBuilder kernelBuilder,
  occaScope scope
);

OCCA_END_EXTERN_C

#endif
