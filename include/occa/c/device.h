#ifndef OCCA_C_DEVICE_HEADER
#define OCCA_C_DEVICE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

occaDevice occaCreateDevice(occaType info);
occaDevice occaCreateDeviceFromString(const char *info);

bool occaDeviceIsInitialized(occaDevice device);

const char* occaDeviceMode(occaDevice device);

occaProperties occaDeviceGetProperties(occaDevice device);

occaProperties occaDeviceGetKernelProperties(occaDevice device);

occaProperties occaDeviceGetMemoryProperties(occaDevice device);

occaProperties occaDeviceGetStreamProperties(occaDevice device);

occaUDim_t occaDeviceMemorySize(occaDevice device);

occaUDim_t occaDeviceMemoryAllocated(occaDevice device);

void occaDeviceFinish(occaDevice device);

bool occaDeviceHasSeparateMemorySpace(occaDevice device);

//---[ Stream ]-------------------------
occaStream occaDeviceCreateStream(occaDevice device,
                                  occaProperties props);

occaStream occaDeviceGetStream(occaDevice device);

void occaDeviceSetStream(occaDevice device,
                         occaStream stream);

occaStreamTag occaDeviceTagStream(occaDevice device);

void occaDeviceWaitForTag(occaDevice device,
                          occaStreamTag tag);

double occaDeviceTimeBetweenTags(occaDevice device,
                                 occaStreamTag startTag,
                                 occaStreamTag endTag);
//======================================

//---[ Kernel ]-------------------------
occaKernel occaDeviceBuildKernel(occaDevice device,
                                 const char *filename,
                                 const char *kernelName,
                                 const occaProperties props);

occaKernel occaDeviceBuildKernelFromString(occaDevice device,
                                           const char *str,
                                           const char *kernelName,
                                           const occaProperties props);

occaKernel occaDeviceBuildKernelFromBinary(occaDevice device,
                                           const char *filename,
                                           const char *kernelName,
                                           const occaProperties props);
//======================================

//---[ Memory ]-------------------------
occaMemory occaDeviceMalloc(occaDevice device,
                            const occaUDim_t bytes,
                            const void *src,
                            occaProperties props);

occaMemory occaDeviceTypedMalloc(occaDevice device,
                                 const occaUDim_t entries,
                                 const occaDtype dtype,
                                 const void *src,
                                 occaProperties props);

void* occaDeviceUMalloc(occaDevice device,
                        const occaUDim_t bytes,
                        const void *src,
                        occaProperties props);

void* occaDeviceTypedUMalloc(occaDevice device,
                             const occaUDim_t entries,
                             const occaDtype type,
                             const void *src,
                             occaProperties props);
//======================================

OCCA_END_EXTERN_C

#endif
