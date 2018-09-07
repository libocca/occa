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

#include <occa/c/types.hpp>
#include <occa/c/uva.h>
#include <occa/tools/uva.hpp>

OCCA_START_EXTERN_C

OCCA_LFUNC int OCCA_RFUNC occaIsManaged(void *ptr) {
  return occa::isManaged(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaStartManaging(void *ptr) {
  occa::startManaging(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaStopManaging(void *ptr) {
  occa::stopManaging(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaSyncToDevice(void *ptr,
                                            const occaUDim_t bytes) {
  occa::syncToDevice(ptr, bytes);
}

OCCA_LFUNC void OCCA_RFUNC occaSyncToHost(void *ptr,
                                          const occaUDim_t bytes) {
  occa::syncToHost(ptr, bytes);
}

OCCA_LFUNC int OCCA_RFUNC occaNeedsSync(void *ptr) {
  return occa::needsSync(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaSync(void *ptr) {
  occa::sync(ptr);
}

OCCA_LFUNC void OCCA_RFUNC occaDontSync(void *ptr) {
  occa::dontSync(ptr);
}

void OCCA_RFUNC occaFreeUvaPtr(void *ptr) {
  occa::freeUvaPtr(ptr);
}

OCCA_END_EXTERN_C
