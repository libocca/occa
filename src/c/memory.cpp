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

#include "occa/c/types.hpp"
#include "occa/c/memory.h"

OCCA_START_EXTERN_C

//---[ Background ]---------------------
occaMemory OCCA_RFUNC occaMalloc(const occaUDim_t bytes,
                                 const void *src,
                                 occaProperties props) {
  occa::memory memory;

  if (occa::c::isDefault(props)) {
    memory = occa::malloc(bytes, src);
  } else {
    memory = occa::malloc(bytes,
                          src,
                          occa::c::properties(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}

void* OCCA_RFUNC occaUmalloc(const occaUDim_t bytes,
                             const void *src,
                             occaProperties props) {

  if (occa::c::isDefault(props)) {
    return occa::umalloc(bytes, src);
  }
  return occa::umalloc(bytes,
                       src,
                       occa::c::properties(props));
}
//======================================

const char* OCCA_RFUNC occaMemoryMode(occaMemory memory) {
  return occa::c::memory(memory).mode().c_str();
}

void OCCA_RFUNC occaMemcpy(void *dest, const void *src,
                           const occaUDim_t bytes,
                           occaProperties props) {
  if (occa::c::isDefault(props)) {
    occa::memcpy(dest, src, bytes);
  } else {
    occa::memcpy(dest, src, bytes,
                 occa::c::properties(props));
  }
}

void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
                                 const occaUDim_t bytes,
                                 const occaUDim_t destOffset,
                                 const occaUDim_t srcOffset,
                                 occaProperties props) {

  occa::memory src_ = occa::c::memory(src);
  occa::memory dest_ = occa::c::memory(dest);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest_, src_,
                 bytes,
                 destOffset, srcOffset);
  } else {
    occa::memcpy(dest_, src_,
                 bytes,
                 destOffset, srcOffset,
                 occa::c::properties(props));
  }
}

void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest, const void *src,
                                 const occaUDim_t bytes,
                                 const occaUDim_t offset,
                                 occaProperties props) {

  occa::memory dest_ = occa::c::memory(dest);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest_, src,
                 bytes,
                 offset);
  } else {
    occa::memcpy(dest_, src,
                 bytes,
                 offset,
                 occa::c::properties(props));
  }
}

void OCCA_RFUNC occaCopyMemToPtr(void *dest, occaMemory src,
                                 const occaUDim_t bytes,
                                 const occaUDim_t offset,
                                 occaProperties props) {

  occa::memory src_ = occa::c::memory(src);

  if (occa::c::isDefault(props)) {
    occa::memcpy(dest, src_,
                 bytes,
                 offset);
  } else {
    occa::memcpy(dest, src_,
                 bytes,
                 offset,
                 occa::c::properties(props));
  }
}

OCCA_END_EXTERN_C
