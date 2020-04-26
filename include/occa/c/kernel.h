#ifndef OCCA_C_KERNEL_HEADER
#define OCCA_C_KERNEL_HEADER

#include <stdarg.h>

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC bool OCCA_RFUNC occaKernelIsInitialized(occaKernel kernel);

OCCA_LFUNC occaProperties OCCA_RFUNC occaKernelGetProperties(occaKernel kernel);

OCCA_LFUNC occaDevice OCCA_RFUNC occaKernelGetDevice(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelName(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelSourceFilename(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelBinaryFilename(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelHash(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelFullHash(occaKernel kernel);

OCCA_LFUNC int OCCA_RFUNC occaKernelMaxDims(occaKernel kernel);

OCCA_LFUNC occaDim OCCA_RFUNC occaKernelMaxOuterDims(occaKernel kernel);

OCCA_LFUNC occaDim OCCA_RFUNC occaKernelMaxInnerDims(occaKernel kernel);

OCCA_LFUNC void OCCA_RFUNC occaKernelSetRunDims(occaKernel kernel,
                                                occaDim outerDims,
                                                occaDim innerDims);

OCCA_LFUNC void OCCA_RFUNC occaKernelPushArg(occaKernel kernel,
                                             occaType arg);

OCCA_LFUNC void OCCA_RFUNC occaKernelClearArgs(occaKernel kernel);

OCCA_LFUNC void OCCA_RFUNC occaKernelRunFromArgs(occaKernel kernel);

// `occaKernelRun` is reserved for a variadic macro
//    which is more user-friendly
OCCA_LFUNC void OCCA_RFUNC occaKernelRunN(occaKernel kernel,
                                         const int argc,
                                         ...);

OCCA_LFUNC void OCCA_RFUNC occaKernelVaRun(occaKernel kernel,
                                           const int argc,
                                           va_list args);

OCCA_END_EXTERN_C

#endif
