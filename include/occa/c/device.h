#ifndef OCCA_C_DEVICE_HEADER
#define OCCA_C_DEVICE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

occaDevice occaCreateDevice(occaType info);
occaDevice occaCreateDeviceFromString(const char *info);

bool occaDeviceIsInitialized(occaDevice device);

const char* occaDeviceMode(occaDevice device);

occaJson occaDeviceGetProperties(occaDevice device);

occaJson occaDeviceGetKernelProperties(occaDevice device);

occaJson occaDeviceGetMemoryProperties(occaDevice device);

occaJson occaDeviceGetStreamProperties(occaDevice device);

occaUDim_t occaDeviceMemorySize(occaDevice device);

occaUDim_t occaDeviceMemoryAllocated(occaDevice device);

void occaDeviceFinish(occaDevice device);

bool occaDeviceHasSeparateMemorySpace(occaDevice device);

//---[ Stream ]-------------------------
occaStream occaDeviceCreateStream(occaDevice device,
                                  occaJson props);

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
                                 const occaJson props);

occaKernel occaDeviceBuildKernelFromString(occaDevice device,
                                           const char *str,
                                           const char *kernelName,
                                           const occaJson props);

occaKernel occaDeviceBuildKernelFromBinary(occaDevice device,
                                           const char *filename,
                                           const char *kernelName,
                                           const occaJson props);
//======================================

//---[ Memory ]-------------------------
occaMemory occaDeviceMalloc(occaDevice device,
                            const occaUDim_t bytes,
                            const void *src,
                            occaJson props);

occaMemory occaDeviceTypedMalloc(occaDevice device,
                                 const occaUDim_t entries,
                                 const occaDtype dtype,
                                 const void *src,
                                 occaJson props);

void* occaDeviceUMalloc(occaDevice device,
                        const occaUDim_t bytes,
                        const void *src,
                        occaJson props);

void* occaDeviceTypedUMalloc(occaDevice device,
                             const occaUDim_t entries,
                             const occaDtype dtype,
                             const void *src,
                             occaJson props);

occaMemory occaDeviceWrapMemory(occaDevice device,
                                const void *ptr,
                                const occaUDim_t bytes,
                                occaJson props);

occaMemory occaDeviceTypedWrapMemory(occaDevice device,
                                     const void *ptr,
                                     const occaUDim_t entries,
                                     const occaDtype dtype,
                                     occaJson props);
//======================================

OCCA_END_EXTERN_C

#endif
