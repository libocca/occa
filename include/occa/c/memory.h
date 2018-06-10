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

#ifndef OCCA_C_MEMORY_HEADER
#define OCCA_C_MEMORY_HEADER

#include "occa/c/defines.h"
#include "occa/c/types.h"

OCCA_START_EXTERN_C

OCCA_LFUNC int OCCA_RFUNC occaMemoryIsInitialized(occaMemory memory);

OCCA_LFUNC void* OCCA_RFUNC occaMemoryPtr(occaMemory memory);

OCCA_LFUNC occaDevice OCCA_RFUNC occaMemoryGetDevice(occaMemory memory);

OCCA_LFUNC occaProperties OCCA_RFUNC occaMemoryGetProperties(occaMemory memory);

OCCA_LFUNC occaUDim_t OCCA_RFUNC occaMemorySize(occaMemory memory);

OCCA_LFUNC occaMemory OCCA_RFUNC occaMemorySlice(occaMemory memory,
                                                 const occaDim_t offset,
                                                 const occaDim_t bytes);

//---[ UVA ]----------------------------
OCCA_LFUNC int OCCA_RFUNC occaMemoryIsManaged(occaMemory memory);

OCCA_LFUNC int OCCA_RFUNC occaMemoryInDevice(occaMemory memory);

OCCA_LFUNC int OCCA_RFUNC occaMemoryIsStale(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemoryStartManaging(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemoryStopManaging(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemorySyncToDevice(occaMemory memory,
                                                  const occaDim_t bytes,
                                                  const occaDim_t offset);

OCCA_LFUNC void OCCA_RFUNC occaMemorySyncToHost(occaMemory memory,
                                                const occaDim_t bytes,
                                                const occaDim_t offset);
//======================================

OCCA_LFUNC void OCCA_RFUNC occaMemcpy(void *dest,
                                      const void *src,
                                      const occaUDim_t bytes,
                                      occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
                                            const occaUDim_t bytes,
                                            const occaUDim_t destOffset,
                                            const occaUDim_t srcOffset,
                                            occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest,
                                            const void *src,
                                            const occaUDim_t bytes,
                                            const occaUDim_t offset,
                                            occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyMemToPtr(void *dest,
                                            occaMemory src,
                                            const occaUDim_t bytes,
                                            const occaUDim_t offset,
                                            occaProperties props);

OCCA_LFUNC occaMemory OCCA_RFUNC occaMemoryClone(occaMemory memory);

OCCA_LFUNC void OCCA_RFUNC occaMemoryDetach(occaMemory memory);

OCCA_LFUNC occaMemory OCCA_RFUNC occaWrapCpuMemory(void *ptr,
                                                   occaUDim_t bytes);

OCCA_END_EXTERN_C

#endif
