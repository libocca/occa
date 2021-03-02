#ifndef OCCA_C_KERNEL_HEADER
#define OCCA_C_KERNEL_HEADER

#include <stdarg.h>

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

bool occaKernelIsInitialized(occaKernel kernel);

occaJson occaKernelGetProperties(occaKernel kernel);

occaDevice occaKernelGetDevice(occaKernel kernel);

const char* occaKernelName(occaKernel kernel);

const char* occaKernelSourceFilename(occaKernel kernel);

const char* occaKernelBinaryFilename(occaKernel kernel);

const char* occaKernelHash(occaKernel kernel);

const char* occaKernelFullHash(occaKernel kernel);

int occaKernelMaxDims(occaKernel kernel);

occaDim occaKernelMaxOuterDims(occaKernel kernel);

occaDim occaKernelMaxInnerDims(occaKernel kernel);

void occaKernelSetRunDims(occaKernel kernel,
                          occaDim outerDims,
                          occaDim innerDims);

void occaKernelPushArg(occaKernel kernel,
                       occaType arg);

void occaKernelClearArgs(occaKernel kernel);

void occaKernelRunFromArgs(occaKernel kernel);

// `occaKernelRun` is reserved for a variadic macro
//    which is more user-friendly
void occaKernelRunN(occaKernel kernel,
                    const int argc,
                    ...);

void occaKernelVaRun(occaKernel kernel,
                     const int argc,
                     va_list args);

void occaKernelRunWithArgs(occaKernel kernel,
                           const int argc,
                           occaType *args);

OCCA_END_EXTERN_C

#endif
