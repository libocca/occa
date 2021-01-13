#include <occa/core/base.hpp>
#include <occa/internal/core/device.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/utils/tls.hpp>
#include <occa/internal/utils/uva.hpp>

namespace occa {
  //---[ Device Functions ]-------------
  device host() {
    static tls<device> tdev;
    device &dev = tdev.value();
    if (!dev.isInitialized()) {
      dev = occa::device({
        {"mode", "Serial"}
      });
      dev.dontUseRefs();
    }
    return dev;
  }

  device& getDevice() {
    static tls<device> tdev;
    device &dev = tdev.value();
    if (!dev.isInitialized()) {
      dev = host();
    }
    return dev;
  }

  void setDevice(device d) {
    getDevice() = d;
  }

  void setDevice(const std::string &props) {
    getDevice() = device(props);
  }

  void setDevice(const occa::json &props) {
    getDevice() = device(props);
  }

  void setDevice(jsonInitializerList initializer) {
    setDevice(json(initializer));
  }

  const occa::json& deviceProperties() {
    return getDevice().properties();
  }

  void finish() {
    getDevice().finish();
  }

  void waitFor(streamTag tag) {
    getDevice().waitFor(tag);
  }

  double timeBetween(const streamTag &startTag,
                     const streamTag &endTag) {
    return getDevice().timeBetween(startTag, endTag);
  }

  stream createStream(const occa::json &props) {
    return getDevice().createStream(props);
  }

  stream getStream() {
    return getDevice().getStream();
  }

  void setStream(stream s) {
    getDevice().setStream(s);
  }

  streamTag tagStream() {
    return getDevice().tagStream();
  }

  //---[ Kernel Functions ]-------------
  kernel buildKernel(const std::string &filename,
                     const std::string &kernelName,
                     const occa::json &props) {

    return getDevice().buildKernel(filename,
                                   kernelName,
                                   props);
  }

  kernel buildKernelFromString(const std::string &content,
                               const std::string &kernelName,
                               const occa::json &props) {

    return getDevice().buildKernelFromString(content, kernelName, props);
  }

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &kernelName,
                               const occa::json &props) {

    return getDevice().buildKernelFromBinary(filename, kernelName, props);
  }

  //---[ Memory Functions ]-------------
  occa::memory malloc(const dim_t entries,
                      const dtype_t &dtype,
                      const void *src,
                      const occa::json &props) {
    return getDevice().malloc(entries, dtype, src, props);
  }

  template <>
  occa::memory malloc<void>(const dim_t entries,
                            const void *src,
                            const occa::json &props) {
    return getDevice().malloc(entries, dtype::byte, src, props);
  }

  void* umalloc(const dim_t entries,
                const dtype_t &dtype,
                const void *src,
                const occa::json &props) {
    return getDevice().umalloc(entries, dtype, src, props);
  }

  template <>
  void* umalloc<void>(const dim_t entries,
                      const void *src,
                      const occa::json &props) {
    return getDevice().umalloc(entries, dtype::byte, src, props);
  }

  occa::memory wrapMemory(const void *ptr,
                          const dim_t entries,
                          const dtype_t &dtype,
                          const occa::json &props) {
    return getDevice().wrapMemory(ptr, entries, dtype, props);
  }

  template <>
  occa::memory wrapMemory<void>(const void *ptr,
                                const dim_t entries,
                                const occa::json &props) {
    return getDevice().wrapMemory(ptr, entries, dtype::byte, props);
  }

  void memcpy(void *dest, const void *src,
              const dim_t bytes,
              const occa::json &props) {

    ptrRangeMap::iterator srcIt  = uvaMap.find(const_cast<void*>(src));
    ptrRangeMap::iterator destIt = uvaMap.find(dest);

    occa::modeMemory_t *srcMem  = ((srcIt  != uvaMap.end()) ? (srcIt->second)  : NULL);
    occa::modeMemory_t *destMem = ((destIt != uvaMap.end()) ? (destIt->second) : NULL);

    const udim_t srcOff  = (srcMem
                            ? (((char*) src)  - srcMem->uvaPtr)
                            : 0);
    const udim_t destOff = (destMem
                            ? (((char*) dest) - destMem->uvaPtr)
                            : 0);

    const bool usingSrcPtr  = (!srcMem ||
                               ((srcMem->isManaged() && !srcMem->inDevice())));
    const bool usingDestPtr = (!destMem ||
                               ((destMem->isManaged() && !destMem->inDevice())));

    if (usingSrcPtr && usingDestPtr) {
      udim_t bytes_ = bytes;
      if (bytes == -1) {
        OCCA_ERROR("Unable to determine bytes to copy",
                   srcMem || destMem);
        bytes_ = (srcMem
                  ? srcMem->size
                  : destMem->size);
      }

      ::memcpy(dest, src, bytes_);
      return;
    }

    if (usingSrcPtr) {
      destMem->copyFrom(src, bytes, destOff, props);
    } else if (usingDestPtr) {
      srcMem->copyTo(dest, bytes, srcOff, props);
    } else {
      // Auto-detects peer-to-peer stuff
      occa::memory srcMemory(srcMem);
      occa::memory destMemory(destMem);
      destMemory.copyFrom(srcMemory, bytes, destOff, srcOff, props);
    }
  }

  void memcpy(memory dest, const void *src,
              const dim_t bytes,
              const dim_t offset,
              const occa::json &props) {

    dest.copyFrom(src, bytes, offset, props);
  }

  void memcpy(void *dest, memory src,
              const dim_t bytes,
              const dim_t offset,
              const occa::json &props) {

    src.copyTo(dest, bytes, offset, props);
  }

  void memcpy(memory dest, memory src,
              const dim_t bytes,
              const dim_t destOffset,
              const dim_t srcOffset,
              const occa::json &props) {

    dest.copyFrom(src, bytes, destOffset, srcOffset, props);
  }

  void memcpy(void *dest, const void *src,
              const occa::json &props) {
    memcpy(dest, src, -1, props);
  }

  void memcpy(memory dest, const void *src,
              const occa::json &props) {
    memcpy(dest, src, -1, 0, props);
  }

  void memcpy(void *dest, memory src,
              const occa::json &props) {
    memcpy(dest, src, -1, 0, props);
  }

  void memcpy(memory dest, memory src,
              const occa::json &props) {
    memcpy(dest, src, -1, 0, 0, props);
  }
  //====================================

  //---[ Free Functions ]---------------
  void free(device d) {
    d.free();
  }

  void free(stream s) {
    s.free();
  }

  void free(kernel k) {
    k.free();
  }

  void free(memory m) {
    m.free();
  }
  //====================================

  //---[ Helper Methods ]---------------
  bool modeIsEnabled(const std::string &mode) {
    return getMode(mode);
  }

  int getDeviceCount(const std::string &props) {
    return getDeviceCount(json::parse(props));
  }

  int getDeviceCount(const occa::json &props) {
    std::string modeName = props["mode"];
    mode_t *mode = getMode(modeName);

    if (mode) {
      return mode->getDeviceCount(props);
    } else {
      return 0;
    }
  }

  void printModeInfo() {
    strToModeMap &modeMap = getModeMap();
    strToModeMap::iterator it = modeMap.begin();

    styling::table table;
    int serialIndex = 0;
    int index = 0;

    for (; it != modeMap.end(); ++it) {
      if (it->second->name() == "Serial") {
        serialIndex = index;
      }
      table.add(it->second->getDescription());
      ++index;
    }

    // Set so Serial mode is first to show up
    if (serialIndex != 0) {
      styling::section serialSection = table.sections[serialIndex];
      table.sections[serialIndex] = table.sections[0];
      table.sections[0] = serialSection;
    }

    io::stdout << table;
  }
  //====================================
}
