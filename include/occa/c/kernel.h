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

#ifndef OCCA_C_KERNEL_HEADER
#define OCCA_C_KERNEL_HEADER

#include "occa/c/defines.h"
#include "occa/c/types.h"

OCCA_START_EXTERN_C

OCCA_LFUNC int OCCA_RFUNC occaKernelIsInitialized(occaKernel kernel);

OCCA_LFUNC occaProperties OCCA_RFUNC occaKernelGetProperties(occaKernel kernel);

OCCA_LFUNC occaDevice OCCA_RFUNC occaKernelGetDevice(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelName(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelSourceFilename(occaKernel kernel);

OCCA_LFUNC const char* OCCA_RFUNC occaKernelBinaryFilename(occaKernel kernel);

OCCA_LFUNC int OCCA_RFUNC occaKernelMaxDims(occaKernel kernel);

OCCA_LFUNC occaDim OCCA_RFUNC occaKernelMaxOuterDims(occaKernel kernel);

OCCA_LFUNC occaDim OCCA_RFUNC occaKernelMaxInnerDims(occaKernel kernel);

OCCA_LFUNC void OCCA_RFUNC occaKernelSetRunDims(occaKernel kernel,
                                                occaDim groups,
                                                occaDim items);

OCCA_LFUNC void OCCA_RFUNC occaKernelRunN(occaKernel kernel,
                                         const int argc,
                                         ...);

OCCA_END_EXTERN_C

#endif
