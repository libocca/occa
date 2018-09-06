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
#include <occa/c/base.h>
#include <occa/tools/env.hpp>

OCCA_START_EXTERN_C

//---[ Globals & Flags ]----------------
occaProperties OCCA_RFUNC occaSettings() {
  return occa::c::newOccaType(occa::settings(),
                              false);
}

void OCCA_RFUNC occaPrintModeInfo() {
  occa::printModeInfo();
}
//======================================

//---[ Device ]-------------------------
occaDevice OCCA_RFUNC occaHost() {
  return occa::c::newOccaType(occa::host());
}

occaDevice OCCA_RFUNC occaGetDevice() {
  return occa::c::newOccaType(occa::getDevice());
}

void OCCA_RFUNC occaSetDevice(occaDevice device) {
  occa::setDevice(occa::c::device(device));
}

void OCCA_RFUNC occaSetDeviceFromString(const char *info) {
  occa::setDevice(info);
}

occaProperties OCCA_RFUNC occaDeviceProperties() {
  return occa::c::newOccaType(occa::deviceProperties(),
                              false);
}

void OCCA_RFUNC occaLoadKernels(const char *library) {
  occa::loadKernels(library);
}

void OCCA_RFUNC occaFinish() {
  occa::finish();
}

occaStream OCCA_RFUNC occaCreateStream(occaProperties props) {
  occa::stream stream;
  if (occa::c::isDefault(props)) {
    stream = occa::createStream();
  } else {
    stream = occa::createStream(occa::c::properties(props));
  }
  stream.dontUseRefs();

  return occa::c::newOccaType(stream);
}

occaStream OCCA_RFUNC occaGetStream() {
  return occa::c::newOccaType(occa::getStream());
}

void OCCA_RFUNC occaSetStream(occaStream stream) {
  occa::setStream(occa::c::stream(stream));
}

occaStreamTag OCCA_RFUNC occaTagStream() {
  occa::streamTag tag = occa::tagStream();
  tag.dontUseRefs();

  return occa::c::newOccaType(tag);
}

void OCCA_RFUNC occaWaitForTag(occaStreamTag tag) {
  occa::waitFor(occa::c::streamTag(tag));
}

double OCCA_RFUNC occaTimeBetweenTags(occaStreamTag startTag,
                                      occaStreamTag endTag) {
  return occa::timeBetween(occa::c::streamTag(startTag),
                           occa::c::streamTag(endTag));
}
//======================================

//---[ Kernel ]-------------------------
occaKernel OCCA_RFUNC occaBuildKernel(const char *filename,
                                      const char *kernelName,
                                      const occaProperties props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernel(filename,
                               kernelName);
  } else {
    kernel = occa::buildKernel(filename,
                               kernelName,
                               occa::c::properties(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel OCCA_RFUNC occaBuildKernelFromString(const char *source,
                                                const char *kernelName,
                                                const occaProperties props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernelFromString(source,
                                         kernelName);
  } else {
    kernel = occa::buildKernelFromString(source,
                                         kernelName,
                                         occa::c::properties(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel OCCA_RFUNC occaBuildKernelFromBinary(const char *filename,
                                                const char *kernelName,
                                                const occaProperties props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernelFromBinary(filename,
                                         kernelName);
  } else {
    kernel = occa::buildKernelFromBinary(filename,
                                         kernelName,
                                         occa::c::properties(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}
//======================================

//---[ Memory ]-------------------------
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

void* OCCA_RFUNC occaUMalloc(const occaUDim_t bytes,
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

OCCA_END_EXTERN_C
