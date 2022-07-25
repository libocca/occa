#include <occa/internal/core/device.hpp>
#include <occa/internal/core/kernel.hpp>
#include <occa/internal/core/buffer.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/core/stream.hpp>
#include <occa/internal/core/streamTag.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>

namespace occa {
  modeDevice_t::modeDevice_t(const occa::json &properties_) :
    mode((std::string) properties_["mode"]),
    properties(properties_),
    needsLauncherKernel(false),
    bytesAllocated(0),
    maxBytesAllocated(0) {}

  modeDevice_t::~modeDevice_t() {
    // Null all wrappers
    while (deviceRing.head) {
      device *mem = (device*) deviceRing.head;
      deviceRing.removeRef(mem);
      mem->modeDevice = NULL;
    }
  }

  // Must be called before ~modeDevice_t()!
  void modeDevice_t::freeResources() {
    freeRing<modeKernel_t>(kernelRing);
    freeRing<modeBuffer_t>(memoryRing);
    freeRing<modeStream_t>(streamRing);
    freeRing<modeStreamTag_t>(streamTagRing);
  }

  void modeDevice_t::dontUseRefs() {
    deviceRing.dontUseRefs();
  }

  void modeDevice_t::addDeviceRef(device *dev) {
    deviceRing.addRef(dev);
  }

  void modeDevice_t::removeDeviceRef(device *dev) {
    deviceRing.removeRef(dev);
  }

  bool modeDevice_t::needsFree() const {
    return deviceRing.needsFree();
  }

  void modeDevice_t::addKernelRef(modeKernel_t *kernel) {
    kernelRing.addRef(kernel);
  }

  void modeDevice_t::removeKernelRef(modeKernel_t *kernel) {
    kernelRing.removeRef(kernel);
  }

  void modeDevice_t::addMemoryRef(modeBuffer_t *buffer) {
    memoryRing.addRef(buffer);
  }

  void modeDevice_t::removeMemoryRef(modeBuffer_t *buffer) {
    memoryRing.removeRef(buffer);
  }

  void modeDevice_t::addStreamRef(modeStream_t *stream) {
    streamRing.addRef(stream);
  }

  void modeDevice_t::removeStreamRef(modeStream_t *stream) {
    streamRing.removeRef(stream);
  }

  void modeDevice_t::addStreamTagRef(modeStreamTag_t *streamTag) {
    streamTagRing.addRef(streamTag);
  }

  void modeDevice_t::removeStreamTagRef(modeStreamTag_t *streamTag) {
    streamTagRing.removeRef(streamTag);
  }

  void modeDevice_t::finish() const {
    currentStream.getModeStream()->finish();
  }

  void modeDevice_t::finishAll() const {
    for(auto* stream : streams) {
      if(stream) stream->finish();
    }
  }

  hash_t modeDevice_t::versionedHash() const {
    return (occa::hash(settings()["version"])
            ^ hash());
  }

  void modeDevice_t::writeKernelBuildFile(const std::string &filename,
                                          const hash_t &kernelHash,
                                          const occa::json &kernelProps,
                                          const lang::sourceMetadata_t &sourceMetadata) const {
    occa::json infoProps;

    infoProps["device"]       = properties;
    infoProps["device/hash"]  = versionedHash().getFullString();
    infoProps["kernel/props"] = kernelProps;
    infoProps["kernel/hash"]  = kernelHash.getFullString();
    infoProps["kernel/metadata"] = sourceMetadata.getKernelMetadataJson();
    infoProps["kernel/dependencies"] = sourceMetadata.getDependencyJson();

    io::writeBuildFile(filename, infoProps);
  }

  std::string modeDevice_t::getKernelHash(const std::string &fullHash,
                                          const std::string &kernelName) {
    return (fullHash + "-" + kernelName);
  }

  std::string modeDevice_t::getKernelHash(const hash_t &kernelHash,
                                          const std::string &kernelName) {
    return getKernelHash(kernelHash.getFullString(),
                         kernelName);
  }

  std::string modeDevice_t::getKernelHash(modeKernel_t *kernel) {
    return getKernelHash(kernel->properties["hash"],
                         kernel->name);
  }

  kernel& modeDevice_t::getCachedKernel(const hash_t &kernelHash,
                                        const std::string &kernelName) {

    return cachedKernels[getKernelHash(kernelHash, kernelName)];
  }

  void modeDevice_t::removeCachedKernel(modeKernel_t *kernel) {
    if (kernel == NULL) {
      return;
    }
    cachedKernelMapIterator it = cachedKernels.find(getKernelHash(kernel));
    if (it != cachedKernels.end()) {
      cachedKernels.erase(it);
    }
  }
}
