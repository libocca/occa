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

#ifndef OCCA_C_BASE_HEADER
#define OCCA_C_BASE_HEADER

#include "occa/c/defines.h"
#include "occa/c/types.h"

OCCA_START_EXTERN_C
//---[ Globals & Flags ]----------------
OCCA_LFUNC occaProperties OCCA_RFUNC occaSettings();

OCCA_LFUNC void OCCA_RFUNC occaPrintModeInfo();
//======================================

//---[ Device ]-------------------------
OCCA_LFUNC occaDevice OCCA_RFUNC occaHost();

OCCA_LFUNC occaDevice OCCA_RFUNC occaGetDevice();

OCCA_LFUNC void OCCA_RFUNC occaSetDevice(occaDevice device);

OCCA_LFUNC void OCCA_RFUNC occaSetDeviceFromInfo(const char *infos);

OCCA_LFUNC occaProperties OCCA_RFUNC occaDeviceProperties();

OCCA_LFUNC void OCCA_RFUNC occaLoadKernels(const char *library);

OCCA_LFUNC void OCCA_RFUNC occaFinish();

OCCA_LFUNC void OCCA_RFUNC occaWaitFor(occaStreamTag tag);

OCCA_LFUNC occaStream OCCA_RFUNC occaCreateStream();

OCCA_LFUNC occaStream OCCA_RFUNC occaGetStream();

OCCA_LFUNC void OCCA_RFUNC occaSetStream(occaStream stream);

OCCA_LFUNC occaStream OCCA_RFUNC occaWrapStream(void *handle_,
                                                const occaProperties props);

OCCA_LFUNC occaStreamTag OCCA_RFUNC occaTagStream();
//======================================

//---[ Kernel ]-------------------------
OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernel(const char *filename,
                                                 const char *kernelName,
                                                 const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromString(const char *str,
                                                           const char *kernelName,
                                                           const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromBinary(const char *filename,
                                                           const char *kernelName,
                                                           const occaProperties props);
//======================================

//---[ Memory ]-------------------------
OCCA_LFUNC occaMemory OCCA_RFUNC occaMalloc(const occaUDim_t bytes,
                                            const void *src,
                                            occaProperties props);

OCCA_LFUNC void* OCCA_RFUNC occaUMalloc(const occaUDim_t bytes,
                                        const void *src,
                                        occaProperties props);
//======================================

OCCA_END_EXTERN_C

#endif
