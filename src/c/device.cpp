#include <occa/internal/c/types.hpp>
#include <occa/c/device.h>
#include <occa/c/dtype.h>

OCCA_START_EXTERN_C

occaDevice occaCreateDevice(occaType info) {
  occa::device device;
  if (info.type == occa::c::typeType::json) {
    device = occa::device(occa::c::json(info));
  }
  else if (info.type == occa::c::typeType::string) {
    device = occa::device(
      occa::json::parse(info.value.ptr)
    );
  }
  else {
    OCCA_FORCE_ERROR("occaCreateDevice expects: occaJson or occaString");
  }
  device.dontUseRefs();

  return occa::c::newOccaType(device);
}

occaDevice occaCreateDeviceFromString(const char *info) {
  occa::device device(
    occa::json::parse(info)
  );
  device.dontUseRefs();
  return occa::c::newOccaType(device);
}

bool occaDeviceIsInitialized(occaDevice device) {
  return (int) occa::c::device(device).isInitialized();
}

const char* occaDeviceMode(occaDevice device) {
  return occa::c::device(device).mode().c_str();
}

occaJson occaDeviceGetProperties(occaDevice device) {
  const occa::json &props = occa::c::device(device).properties();
  return occa::c::newOccaType(props, false);
}

occaJson occaDeviceGetKernelProperties(occaDevice device) {
  const occa::json &props = occa::c::device(device).kernelProperties();
  return occa::c::newOccaType(props, false);
}

occaJson occaDeviceGetMemoryProperties(occaDevice device) {
  const occa::json &props = occa::c::device(device).memoryProperties();
  return occa::c::newOccaType(props, false);
}

occaJson occaDeviceGetStreamProperties(occaDevice device) {
  const occa::json &props = occa::c::device(device).streamProperties();
  return occa::c::newOccaType(props, false);
}

occaUDim_t occaDeviceMemorySize(occaDevice device) {
  return occa::c::device(device).memorySize();
}

occaUDim_t occaDeviceMemoryAllocated(occaDevice device) {
  return occa::c::device(device).memoryAllocated();
}

void occaDeviceFinish(occaDevice device) {
  occa::c::device(device).finish();
}

bool occaDeviceHasSeparateMemorySpace(occaDevice device) {
  return (int) occa::c::device(device).hasSeparateMemorySpace();
}

//---[ Stream ]-------------------------
occaStream occaDeviceCreateStream(occaDevice device,
                                  occaJson props) {
  occa::device device_ = occa::c::device(device);
  occa::stream stream;
  if (occa::c::isDefault(props)) {
    stream = device_.createStream();
  } else {
    stream = device_.createStream(occa::c::json(props));
  }
  stream.dontUseRefs();

  return occa::c::newOccaType(stream);
}

occaStream occaDeviceGetStream(occaDevice device) {
  occa::device device_ = occa::c::device(device);
  return occa::c::newOccaType(device_.getStream());
}

void occaDeviceSetStream(occaDevice device,
                         occaStream stream) {
  occa::device device_ = occa::c::device(device);
  device_.setStream(occa::c::stream(stream));
}

occaStreamTag occaDeviceTagStream(occaDevice device) {
  occa::device device_ = occa::c::device(device);
  occa::streamTag tag = device_.tagStream();
  tag.dontUseRefs();

  return occa::c::newOccaType(tag);
}

void occaDeviceWaitForTag(occaDevice device,
                          occaStreamTag tag) {
  occa::device device_ = occa::c::device(device);
  device_.waitFor(occa::c::streamTag(tag));
}

double occaDeviceTimeBetweenTags(occaDevice device,
                                 occaStreamTag startTag,
                                 occaStreamTag endTag) {
  occa::device device_ = occa::c::device(device);
  return device_.timeBetween(occa::c::streamTag(startTag),
                             occa::c::streamTag(endTag));
}
//======================================

//---[ Kernel ]-------------------------
occaKernel occaDeviceBuildKernel(occaDevice device,
                                 const char *filename,
                                 const char *kernelName,
                                 const occaJson props) {
  occa::device device_ = occa::c::device(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernel(filename,
                                 kernelName);
  } else {
    kernel = device_.buildKernel(filename,
                                 kernelName,
                                 occa::c::json(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel occaDeviceBuildKernelFromString(occaDevice device,
                                           const char *str,
                                           const char *kernelName,
                                           const occaJson props) {
  occa::device device_ = occa::c::device(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernelFromString(str,
                                           kernelName);
  } else {
    kernel = device_.buildKernelFromString(str,
                                           kernelName,
                                           occa::c::json(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}

occaKernel occaDeviceBuildKernelFromBinary(occaDevice device,
                                           const char *filename,
                                           const char *kernelName,
                                           const occaJson props) {
  occa::device device_ = occa::c::device(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernelFromBinary(filename,
                                           kernelName);
  } else {
    kernel = device_.buildKernelFromBinary(filename,
                                           kernelName,
                                           occa::c::json(props));
  }
  kernel.dontUseRefs();

  return occa::c::newOccaType(kernel);
}
//======================================

//---[ Memory ]-------------------------
occaMemory occaDeviceMalloc(occaDevice device,
                            const occaUDim_t bytes,
                            const void *src,
                            occaJson props) {
  return occaDeviceTypedMalloc(device,
                               bytes,
                               occaDtypeByte,
                               src,
                               props);
}

occaMemory occaDeviceTypedMalloc(occaDevice device,
                                 const occaUDim_t entries,
                                 const occaDtype dtype,
                                 const void *src,
                                 occaJson props) {
  occa::device device_ = occa::c::device(device);
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  occa::memory memory;
  if (occa::c::isDefault(props)) {
    memory = device_.malloc(entries, dtype_, src);
  } else {
    memory = device_.malloc(entries,
                            dtype_,
                            src,
                            occa::c::json(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}

void* occaDeviceUMalloc(occaDevice device,
                        const occaUDim_t bytes,
                        const void *src,
                        occaJson props) {
  return occaDeviceTypedUMalloc(device,
                                bytes,
                                occaDtypeByte,
                                src,
                                props);
}

void* occaDeviceTypedUMalloc(occaDevice device,
                             const occaUDim_t entries,
                             const occaDtype dtype,
                             const void *src,
                             occaJson props) {
  occa::device device_ = occa::c::device(device);
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  if (occa::c::isDefault(props)) {
    return device_.umalloc(entries, dtype_, src);
  }
  return device_.umalloc(entries,
                         dtype_,
                         src,
                         occa::c::json(props));
}

occaMemory occaDeviceWrapMemory(occaDevice device,
                                const void *ptr,
                                const occaUDim_t bytes,
                                occaJson props) {
  return occaDeviceTypedWrapMemory(device,
                                   ptr,
                                   bytes,
                                   occaDtypeByte,
                                   props);
}

occaMemory occaDeviceTypedWrapMemory(occaDevice device,
                                     const void *ptr,
                                     const occaUDim_t entries,
                                     const occaDtype dtype,
                                     occaJson props) {
  occa::device device_ = occa::c::device(device);
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  occa::memory memory;
  if (occa::c::isDefault(props)) {
    memory = device_.wrapMemory(ptr, entries, dtype_);
  } else {
    memory = device_.wrapMemory(ptr, entries, dtype_, occa::c::json(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}
//======================================

OCCA_END_EXTERN_C
