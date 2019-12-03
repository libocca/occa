#include <occa/c/types.hpp>
#include <occa/c/device.h>

OCCA_START_EXTERN_C

occaDevice OCCA_RFUNC occaCreateDevice(occaType info) {
  occa::device device;
  if (info.type == occa::c::typeType::properties) {
    device = occa::device(occa::c::properties(info));
  }
  else if (info.type == occa::c::typeType::json) {
    device = occa::device(occa::c::json(info));
  }
  else if (info.type == occa::c::typeType::string) {
    device = occa::device(std::string(info.value.ptr));
  }
  else {
    OCCA_FORCE_ERROR("occaCreateDevice expects: occaProperties, occaJson, or occaString");
  }
  device.dontUseRefs();

  return occa::c::newOccaType(device);
}

occaDevice OCCA_RFUNC occaCreateDeviceFromString(const char *info) {
  occa::device device(info);
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
  const occa::properties &props = occa::c::device(device).properties();
  return occa::c::newOccaType(props, false);
}

occaProperties OCCA_RFUNC occaDeviceGetKernelProperties(occaDevice device) {
  const occa::properties &props = occa::c::device(device).kernelProperties();
  return occa::c::newOccaType(props, false);
}

occaProperties OCCA_RFUNC occaDeviceGetMemoryProperties(occaDevice device) {
  const occa::properties &props = occa::c::device(device).memoryProperties();
  return occa::c::newOccaType(props, false);
}

occaProperties OCCA_RFUNC occaDeviceGetStreamProperties(occaDevice device) {
  const occa::properties &props = occa::c::device(device).streamProperties();
  return occa::c::newOccaType(props, false);
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
occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device,
                                             occaProperties props) {
  occa::device device_ = occa::c::device(device);
  occa::stream stream;
  if (occa::c::isDefault(props)) {
    stream = device_.createStream();
  } else {
    stream = device_.createStream(occa::c::properties(props));
  }
  stream.dontUseRefs();

  return occa::c::newOccaType(stream);
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

occaStreamTag OCCA_RFUNC occaDeviceTagStream(occaDevice device) {
  occa::device device_ = occa::c::device(device);
  occa::streamTag tag = device_.tagStream();
  tag.dontUseRefs();

  return occa::c::newOccaType(tag);
}

void OCCA_RFUNC occaDeviceWaitForTag(occaDevice device,
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

occaMemory OCCA_RFUNC occaDeviceTypedMalloc(occaDevice device,
                                            const occaUDim_t entries,
                                            const occaDtype dtype,
                                            const void *src,
                                            occaProperties props) {
  occa::device device_ = occa::c::device(device);
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  occa::memory memory;
  if (occa::c::isDefault(props)) {
    memory = device_.malloc(entries, dtype_, src);
  } else {
    memory = device_.malloc(entries,
                            dtype_,
                            src,
                            occa::c::properties(props));
  }
  memory.dontUseRefs();

  return occa::c::newOccaType(memory);
}

void* OCCA_RFUNC occaDeviceUMalloc(occaDevice device,
                                   const occaUDim_t bytes,
                                   const void *src,
                                   occaProperties props) {
  occa::device device_ = occa::c::device(device);

  if (occa::c::isDefault(props)) {
    return device_.umalloc(bytes,
                           occa::dtype::byte,
                           src);
  }
  return device_.umalloc(bytes,
                         occa::dtype::byte,
                         src,
                         occa::c::properties(props));
}

void* OCCA_RFUNC occaDeviceTypedUMalloc(occaDevice device,
                                        const occaUDim_t entries,
                                        const occaDtype dtype,
                                        const void *src,
                                        occaProperties props) {
  occa::device device_ = occa::c::device(device);
  const occa::dtype_t &dtype_ = occa::c::dtype(dtype);

  if (occa::c::isDefault(props)) {
    return device_.umalloc(entries, dtype_, src);
  }
  return device_.umalloc(entries,
                         dtype_,
                         src,
                         occa::c::properties(props));
}
//======================================

OCCA_END_EXTERN_C
