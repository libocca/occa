#include <occa/internal/c/types.hpp>
#include <occa/c/base.h>
#include <occa/internal/utils/env.hpp>

OCCA_START_EXTERN_C

//---[ Globals & Flags ]----------------
occaProperties occaSettings() {
  return occa::c::newOccaType(occa::settings(),
                              false);
}

void occaPrintModeInfo() {
  occa::printModeInfo();
}
//======================================

//---[ Device ]-------------------------
occaDevice occaHost() {
  return occa::c::newOccaType(occa::host());
}

occaDevice occaGetDevice() {
  return occa::c::newOccaType(occa::getDevice());
}

void occaSetDevice(occaDevice device) {
  occa::setDevice(occa::c::device(device));
}

void occaSetDeviceFromString(const char *info) {
  occa::setDevice(info);
}

occaProperties occaDeviceProperties() {
  return occa::c::newOccaType(occa::deviceProperties(),
                              false);
}

void occaLoadKernels(const char *library) {
  occa::loadKernels(library);
}

void occaFinish() {
  occa::finish();
}

occaStream occaCreateStream(occaProperties props) {
  occa::stream stream;
  if (occa::c::isDefault(props)) {
    stream = occa::createStream();
  } else {
    stream = occa::createStream(occa::c::properties(props));
  }
  stream.dontUseRefs();

  return occa::c::newOccaType(stream);
}

occaStream occaGetStream() {
  return occa::c::newOccaType(occa::getStream());
}

void occaSetStream(occaStream stream) {
  occa::setStream(occa::c::stream(stream));
}

occaStreamTag occaTagStream() {
  occa::streamTag tag = occa::tagStream();
  tag.dontUseRefs();

  return occa::c::newOccaType(tag);
}

void occaWaitForTag(occaStreamTag tag) {
  occa::waitFor(occa::c::streamTag(tag));
}

double occaTimeBetweenTags(occaStreamTag startTag,
                           occaStreamTag endTag) {
  return occa::timeBetween(occa::c::streamTag(startTag),
                           occa::c::streamTag(endTag));
}
//======================================

//---[ Kernel ]-------------------------
occaKernel occaBuildKernel(const char *filename,
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

occaKernel occaBuildKernelFromString(const char *source,
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

occaKernel occaBuildKernelFromBinary(const char *filename,
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
occaMemory occaMalloc(const occaUDim_t bytes,
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

occaMemory occaTypedMalloc(const occaUDim_t entries,
                           const occaDtype dtype,
                           const void *src,
                           occaProperties props) {
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  occa::memory memory;
  if (occa::c::isDefault(props)) {
    memory = occa::malloc(entries, dtype_, src);
  } else {
    memory = occa::malloc(entries,
                          dtype_,
                          src,
                          occa::c::properties(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}

void* occaUMalloc(const occaUDim_t bytes,
                  const void *src,
                  occaProperties props) {

  if (occa::c::isDefault(props)) {
    return occa::umalloc(bytes,
                         occa::dtype::byte,
                         src);
  }
  return occa::umalloc(bytes,
                       occa::dtype::byte,
                       src,
                       occa::c::properties(props));
}

void* occaTypedUMalloc(const occaUDim_t entries,
                       const occaDtype dtype,
                       const void *src,
                       occaProperties props) {
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  if (occa::c::isDefault(props)) {
    return occa::umalloc(entries, dtype_, src);
  }
  return occa::umalloc(entries,
                       dtype_,
                       src,
                       occa::c::properties(props));
}
//======================================

OCCA_END_EXTERN_C
