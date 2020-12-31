#ifndef OCCA_C_BASE_HEADER
#define OCCA_C_BASE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

//---[ Globals & Flags ]----------------
occaProperties occaSettings();

void occaPrintModeInfo();
//======================================

//---[ Device ]-------------------------
occaDevice occaHost();

occaDevice occaGetDevice();

void occaSetDevice(occaDevice device);

void occaSetDeviceFromString(const char *info);

occaProperties occaDeviceProperties();

void occaLoadKernels(const char *library);

void occaFinish();

occaStream occaCreateStream(occaProperties props);

occaStream occaGetStream();

void occaSetStream(occaStream stream);

occaStreamTag occaTagStream();

void occaWaitForTag(occaStreamTag tag);

double occaTimeBetweenTags(occaStreamTag startTag,
                           occaStreamTag endTag);
//======================================

//---[ Kernel ]-------------------------
occaKernel occaBuildKernel(const char *filename,
                           const char *kernelName,
                           const occaProperties props);

occaKernel occaBuildKernelFromString(const char *source,
                                     const char *kernelName,
                                     const occaProperties props);

occaKernel occaBuildKernelFromBinary(const char *filename,
                                     const char *kernelName,
                                     const occaProperties props);
//======================================

//---[ Memory ]-------------------------
occaMemory occaMalloc(const occaUDim_t bytes,
                      const void *src,
                      occaProperties props);

occaMemory occaTypedMalloc(const occaUDim_t entries,
                           const occaDtype type,
                           const void *src,
                           occaProperties props);

void* occaUMalloc(const occaUDim_t bytes,
                  const void *src,
                  occaProperties props);

void* occaTypedUMalloc(const occaUDim_t entries,
                       const occaDtype type,
                       const void *src,
                       occaProperties props);
//======================================

OCCA_END_EXTERN_C

#endif
