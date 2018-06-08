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
#include <occa/c/device.h>

OCCA_START_EXTERN_C

occaDevice OCCA_RFUNC occaCreateDevice(occaType info) {
  occa::device device;
  if (info.type == occa::c::typeType::properties) {
    device = occa::device(occa::c::properties(info));
  }
  else if (info.type == occa::c::typeType::string) {
    device = occa::device(std::string(info.value.ptr));
  }
  else {
    OCCA_FORCE_ERROR("occaCreateDevice expects an occaProperties or occaString");
  }
  device.dontUseRefs();

  return occa::c::newOccaType(device);
}

int OCCA_RFUNC occaDeviceIsInitialized(occaDevice device) {
  return (int) occa::c::device(device).isInitialized();
}

const char* OCCA_RFUNC occaDeviceMode(occaDevice device) {
  return occa::c::device(device).mode().c_str();
}

occaProperties OCCA_RFUNC occaDeviceGetProperties(occaDevice device) {
  occa::properties &props = occa::c::device(device).properties();
  return occa::c::newOccaType(props);
}

occaProperties OCCA_RFUNC occaDeviceGetKernelProperties(occaDevice device) {
  occa::properties &props = occa::c::device(device).kernelProperties();
  return occa::c::newOccaType(props);
}

occaProperties OCCA_RFUNC occaDeviceGetMemoryProperties(occaDevice device) {
  occa::properties &props = occa::c::device(device).memoryProperties();
  return occa::c::newOccaType(props);
}

occaUDim_t OCCA_RFUNC occaDeviceMemorySize(occaDevice device) {
  return occa::c::device(device).memorySize();
}

occaUDim_t OCCA_RFUNC occaDeviceMemoryAllocated(occaDevice device) {
  return occa::c::device(device).memoryAllocated();
}

void OCCA_RFUNC occaDeviceFinish(occaDevice device) {
  occa::c::device(device).finish();
}

OCCA_LFUNC int OCCA_RFUNC occaDeviceHasSeparateMemorySpace(occaDevice device) {
  return (int) occa::c::device(device).hasSeparateMemorySpace();
}

//---[ Stream ]-------------------------
occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device) {
  occa::device device_ = occa::c::device(device);
  return occa::c::newOccaType(device_.createStream());
}

occaStream OCCA_RFUNC occaDeviceGetStream(occaDevice device) {
  occa::device device_ = occa::c::device(device);
  return occa::c::newOccaType(device_.getStream());
}

void OCCA_RFUNC occaDeviceSetStream(occaDevice device,
                                    occaStream stream) {
  occa::device device_ = occa::c::device(device);
  device_.setStream(occa::c::stream(stream));
}

occaStream OCCA_RFUNC occaDeviceWrapStream(occaDevice device,
                                           void *handle_,
                                           const occaProperties props) {
  occa::device device_ = occa::c::device(device);
  occa::stream stream;
  if (occa::c::isDefault(props)) {
    stream = device_.wrapStream(handle_);
  } else {
    stream = device_.wrapStream(handle_,
                                occa::c::properties(props));
  }
  return occa::c::newOccaType(stream);
}

occaStreamTag OCCA_RFUNC occaDeviceTagStream(occaDevice device) {
  occa::device device_ = occa::c::device(device);
  return occa::c::newOccaType(device_.tagStream());
}

void OCCA_RFUNC occaDeviceWaitFor(occaDevice device,
                                  occaStreamTag tag) {
  occa::device device_ = occa::c::device(device);
  device_.waitFor(occa::c::streamTag(tag));
}

double OCCA_RFUNC occaDeviceTimeBetweenTags(occaDevice device,
                                            occaStreamTag startTag,
                                            occaStreamTag endTag) {
  occa::device device_ = occa::c::device(device);
  return device_.timeBetween(occa::c::streamTag(startTag),
                             occa::c::streamTag(endTag));
}
//======================================

//---[ Kernel ]-------------------------
occaKernel OCCA_RFUNC occaDeviceBuildKernel(occaDevice device,
                                            const char *filename,
                                            const char *kernelName,
                                            const occaProperties props) {
  occa::device device_ = occa::c::device(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernel(filename,
                                 kernelName);
  } else {
    kernel = device_.buildKernel(filename,
                                 kernelName,
                                 occa::c::properties(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel OCCA_RFUNC occaDeviceBuildKernelFromString(occaDevice device,
                                                      const char *str,
                                                      const char *kernelName,
                                                      const occaProperties props) {
  occa::device device_ = occa::c::device(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernelFromString(str,
                                           kernelName);
  } else {
    kernel = device_.buildKernelFromString(str,
                                           kernelName,
                                           occa::c::properties(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel OCCA_RFUNC occaDeviceBuildKernelFromBinary(occaDevice device,
                                                      const char *filename,
                                                      const char *kernelName,
                                                      const occaProperties props) {
  occa::device device_ = occa::c::device(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernelFromBinary(filename,
                                           kernelName);
  } else {
    kernel = device_.buildKernelFromBinary(filename,
                                           kernelName,
                                           occa::c::properties(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}
//======================================

//---[ Memory ]-------------------------
occaMemory OCCA_RFUNC occaDeviceMalloc(occaDevice device,
                                       const occaUDim_t bytes,
                                       const void *src,
                                       occaProperties props) {
  occa::device device_ = occa::c::device(device);
  occa::memory memory;
  if (occa::c::isDefault(props)) {
    memory = device_.malloc(bytes, src);
  } else {
    memory = device_.malloc(bytes,
                            src,
                            occa::c::properties(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}

void* OCCA_RFUNC occaDeviceUmalloc(occaDevice device,
                                   const occaUDim_t bytes,
                                   const void *src,
                                   occaProperties props) {
  occa::device device_ = occa::c::device(device);

  if (occa::c::isDefault(props)) {
    return device_.umalloc(bytes, src);
  }
  return device_.umalloc(bytes,
                         src,
                         occa::c::properties(props));
}
//======================================

OCCA_END_EXTERN_C
