#ifndef OCCA_C_BASE_HEADER
#define OCCA_C_BASE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

//---[ Globals & Flags ]----------------
occaJson occaSettings();

void occaPrintModeInfo();
//======================================

//---[ Device ]-------------------------
occaDevice occaHost();

occaDevice occaGetDevice();

void occaSetDevice(occaDevice device);

void occaSetDeviceFromString(const char *info);

occaJson occaDeviceProperties();

void occaFinish();

occaStream occaCreateStream(occaJson props);

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
                           const occaJson props);

occaKernel occaBuildKernelFromString(const char *source,
                                     const char *kernelName,
                                     const occaJson props);

occaKernel occaBuildKernelFromBinary(const char *filename,
                                     const char *kernelName,
                                     const occaJson props);
//======================================

//---[ Memory ]-------------------------
occaMemory occaMalloc(const occaUDim_t bytes,
                      const void *src,
                      occaJson props);

occaMemory occaTypedMalloc(const occaUDim_t entries,
                           const occaDtype type,
                           const void *src,
                           occaJson props);

occaMemory occaWrapMemory(const void *ptr,
                          const occaUDim_t bytes,
                          occaJson props);

occaMemory occaTypedWrapMemory(const void *ptr,
                               const occaUDim_t entries,
                               const occaDtype dtype,
                               occaJson props);
//======================================

//---[ MemoryPool ]---------------------
occaMemoryPool occaCreateMemoryPool(occaJson props);
//======================================

OCCA_END_EXTERN_C

#endif
