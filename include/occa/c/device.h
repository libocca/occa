#ifndef OCCA_C_DEVICE_HEADER
#define OCCA_C_DEVICE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC occaDevice OCCA_RFUNC occaCreateDevice(occaType info);
OCCA_LFUNC occaDevice OCCA_RFUNC occaCreateDeviceFromString(const char *info);

OCCA_LFUNC int OCCA_RFUNC occaDeviceIsInitialized(occaDevice device);

OCCA_LFUNC const char* OCCA_RFUNC occaDeviceMode(occaDevice device);

OCCA_LFUNC occaProperties OCCA_RFUNC occaDeviceGetProperties(occaDevice device);

OCCA_LFUNC occaProperties OCCA_RFUNC occaDeviceGetKernelProperties(occaDevice device);

OCCA_LFUNC occaProperties OCCA_RFUNC occaDeviceGetMemoryProperties(occaDevice device);

OCCA_LFUNC occaUDim_t OCCA_RFUNC occaDeviceMemorySize(occaDevice device);

OCCA_LFUNC occaUDim_t OCCA_RFUNC occaDeviceMemoryAllocated(occaDevice device);

OCCA_LFUNC void OCCA_RFUNC occaDeviceFinish(occaDevice device);

OCCA_LFUNC int OCCA_RFUNC occaDeviceHasSeparateMemorySpace(occaDevice device);

//---[ Stream ]-------------------------
OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device,
                                                        occaProperties props);

OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceGetStream(occaDevice device);

OCCA_LFUNC void OCCA_RFUNC occaDeviceSetStream(occaDevice device,
                                               occaStream stream);

OCCA_LFUNC occaStreamTag OCCA_RFUNC occaDeviceTagStream(occaDevice device);

OCCA_LFUNC void OCCA_RFUNC occaDeviceWaitForTag(occaDevice device,
                                                occaStreamTag tag);

OCCA_LFUNC double OCCA_RFUNC occaDeviceTimeBetweenTags(occaDevice device,
                                                       occaStreamTag startTag,
                                                       occaStreamTag endTag);
//======================================

//---[ Kernel ]-------------------------
OCCA_LFUNC occaKernel OCCA_RFUNC occaDeviceBuildKernel(occaDevice device,
                                                       const char *filename,
                                                       const char *kernelName,
                                                       const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaDeviceBuildKernelFromString(occaDevice device,
                                                                 const char *str,
                                                                 const char *kernelName,
                                                                 const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaDeviceBuildKernelFromBinary(occaDevice device,
                                                                 const char *filename,
                                                                 const char *kernelName,
                                                                 const occaProperties props);
//======================================

//---[ Memory ]-------------------------
OCCA_LFUNC occaMemory OCCA_RFUNC occaDeviceMalloc(occaDevice device,
                                                  const occaUDim_t bytes,
                                                  const void *src,
                                                  occaProperties props);

OCCA_LFUNC occaMemory OCCA_RFUNC occaDeviceTypedMalloc(occaDevice device,
                                                       const occaUDim_t entries,
                                                       const occaDtype dtype,
                                                       const void *src,
                                                       occaProperties props);

OCCA_LFUNC void* OCCA_RFUNC occaDeviceUMalloc(occaDevice device,
                                              const occaUDim_t bytes,
                                              const void *src,
                                              occaProperties props);

OCCA_LFUNC void* OCCA_RFUNC occaDeviceTypedUMalloc(occaDevice device,
                                                   const occaUDim_t entries,
                                                   const occaDtype type,
                                                   const void *src,
                                                   occaProperties props);
//======================================

OCCA_END_EXTERN_C

#endif
