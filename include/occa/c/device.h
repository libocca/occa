/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_C_DEVICE_HEADER
#define OCCA_C_DEVICE_HEADER

#include "occa/c/defines.h"
#include "occa/c/types.h"

OCCA_START_EXTERN_C

OCCA_LFUNC occaDevice OCCA_RFUNC occaCreateDevice(occaType info);

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
OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device);

OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceGetStream(occaDevice device);

OCCA_LFUNC void       OCCA_RFUNC occaDeviceSetStream(occaDevice device,
                                                     occaStream stream);

OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceWrapStream(occaDevice device,
                                                      void *handle_,
                                                      const occaProperties props);

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

OCCA_LFUNC void* OCCA_RFUNC occaDeviceUmalloc(occaDevice device,
                                              const occaUDim_t bytes,
                                              const void *src,
                                              occaProperties props);
//======================================

OCCA_END_EXTERN_C

#endif
