#include <occa/internal/c/types.hpp>
#include <occa/c/base.h>
#include <occa/c/dtype.h>
#include <occa/internal/utils/env.hpp>

OCCA_START_EXTERN_C

//---[ Globals & Flags ]----------------
occaJson occaSettings() {
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
  occa::setDevice(
    occa::json::parse(info)
  );
}

occaJson occaDeviceProperties() {
  return occa::c::newOccaType(occa::deviceProperties(),
                              false);
}

void occaFinish() {
  occa::finish();
}

occaStream occaCreateStream(occaJson props) {
  occa::stream stream;
  if (occa::c::isDefault(props)) {
    stream = occa::createStream();
  } else {
    stream = occa::createStream(occa::c::json(props));
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
                           const occaJson props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernel(filename,
                               kernelName);
  } else {
    kernel = occa::buildKernel(filename,
                               kernelName,
                               occa::c::json(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel occaBuildKernelFromString(const char *source,
                                     const char *kernelName,
                                     const occaJson props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernelFromString(source,
                                         kernelName);
  } else {
    kernel = occa::buildKernelFromString(source,
                                         kernelName,
                                         occa::c::json(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel occaBuildKernelFromBinary(const char *filename,
                                     const char *kernelName,
                                     const occaJson props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernelFromBinary(filename,
                                         kernelName);
  } else {
    kernel = occa::buildKernelFromBinary(filename,
                                         kernelName,
                                         occa::c::json(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}
//======================================

//---[ Memory ]-------------------------
occaMemory occaMalloc(const occaUDim_t bytes,
                      const void *src,
                      occaJson props) {
  return occaTypedMalloc(bytes,
                         occaDtypeByte,
                         src,
                         props);
}

occaMemory occaTypedMalloc(const occaUDim_t entries,
                           const occaDtype dtype,
                           const void *src,
                           occaJson props) {
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  occa::memory memory;
  if (occa::c::isDefault(props)) {
    memory = occa::malloc(entries, dtype_, src);
  } else {
    memory = occa::malloc(entries,
                          dtype_,
                          src,
                          occa::c::json(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}

void* occaUMalloc(const occaUDim_t bytes,
                  const void *src,
                  occaJson props) {
  return occaTypedUMalloc(bytes,
                          occaDtypeByte,
                          src,
                          props);
}

void* occaTypedUMalloc(const occaUDim_t entries,
                       const occaDtype dtype,
                       const void *src,
                       occaJson props) {
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  if (occa::c::isDefault(props)) {
    return occa::umalloc(entries, dtype_, src);
  }
  return occa::umalloc(entries,
                       dtype_,
                       src,
                       occa::c::json(props));
}

occaMemory occaWrapMemory(const void *ptr,
                          const occaUDim_t bytes,
                          occaJson props) {
  return occaTypedWrapMemory(ptr,
                             bytes,
                             occaDtypeByte,
                             props);
}

occaMemory occaTypedWrapMemory(const void *ptr,
                               const occaUDim_t entries,
                               const occaDtype dtype,
                               occaJson props) {
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  occa::memory memory;
  if (occa::c::isDefault(props)) {
    memory = occa::wrapMemory(ptr, entries, dtype_);
  } else {
    memory = occa::wrapMemory(ptr, entries, dtype_, occa::c::json(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}
//======================================

OCCA_END_EXTERN_C
