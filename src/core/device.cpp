#include <occa/core/device.hpp>
#include <occa/core/base.hpp>
#include <occa/internal/core/device.hpp>
#include <occa/internal/core/kernel.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/core/memoryPool.hpp>
#include <occa/internal/modes.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>

namespace occa {
  //---[ Utils ]------------------------
  occa::json getModeSpecificProps(const std::string &mode,
                                  const occa::json &props) {
    occa::json allProps = (
      props
      + props["modes/" + mode]
    );

    allProps.remove("modes");

    return allProps;
  }

  occa::json getObjectSpecificProps(const std::string &mode,
                                    const std::string &object,
                                    const occa::json &props) {
    occa::json allProps = (
      props[object]
      + props[object + "/modes/" + mode]
      + props["modes/" + mode + "/" + object]
    );

    allProps.remove(object + "/modes");
    allProps.remove("modes");

    return allProps;
  }

  occa::json initialObjectProps(const std::string &mode,
                                const std::string &object,
                                const occa::json &props) {
    occa::json objectProps = (
      getObjectSpecificProps(mode, object, settings())
      + getObjectSpecificProps(mode, object, props)
    );
    objectProps["mode"] = mode;
    return objectProps;
  }
  //====================================

  device::device() :
    modeDevice(NULL) {}

  device::device(modeDevice_t *modeDevice_) :
    modeDevice(NULL) {
    setModeDevice(modeDevice_);
  }

  device::device(const std::string &props) :
    device(json::parse(props)) {}

  device::device(const occa::json &props) :
    modeDevice(NULL) {
    setup(props);
  }

  device::device(jsonInitializerList initializer) :
    device(json(initializer)) {}

  device::device(const device &other) :
    modeDevice(NULL) {
    setModeDevice(other.modeDevice);
  }

  device& device::operator = (const device &other) {
    setModeDevice(other.modeDevice);
    return *this;
  }

  device::~device() {
    removeDeviceRef();
  }

  void device::assertInitialized() const {
    OCCA_ERROR("Device not initialized or has been freed",
               modeDevice != NULL);
  }

  void device::setModeDevice(modeDevice_t *modeDevice_) {
    if (modeDevice != modeDevice_) {
      removeDeviceRef();
      modeDevice = modeDevice_;
      if (modeDevice) {
        modeDevice->addDeviceRef(this);
      }
    }
  }

  void device::removeDeviceRef() {
    if (!modeDevice) {
      return;
    }
    modeDevice->removeDeviceRef(this);
    if (modeDevice->modeDevice_t::needsFree()) {
      free();
    }
  }

  void device::dontUseRefs() {
    if (modeDevice) {
      modeDevice->modeDevice_t::dontUseRefs();
    }
  }

  bool device::operator == (const occa::device &other) const {
    return (modeDevice == other.modeDevice);
  }

  bool device::operator != (const occa::device &other) const {
    return (modeDevice != other.modeDevice);
  }

  bool device::isInitialized() const {
    return (modeDevice != NULL);
  }

  modeDevice_t* device::getModeDevice() const {
    return modeDevice;
  }

  void device::setup(const std::string &props) {
    setup(json::parse(props));
  }

  void device::setup(const occa::json &props) {
    free();

    const std::string mode_ = props["mode"];

    occa::json deviceProps = (
      getObjectSpecificProps(mode_, "device", settings())
      + getModeSpecificProps(mode_, props)
    );

    deviceProps["kernel"] = initialObjectProps(mode_, "kernel", props);
    deviceProps["memory"] = initialObjectProps(mode_, "memory", props);
    deviceProps["stream"] = initialObjectProps(mode_, "stream", props);

    setModeDevice(occa::newModeDevice(deviceProps));

    // Create an initial stream
    setStream(createStream());
  }

  void device::free() {
    if (modeDevice) {
      modeDevice->freeResources();

      // ~modeDevice_t NULLs all wrappers
      delete modeDevice;
      modeDevice = NULL;
    }
  }

  const std::string& device::mode() const {
    static const std::string noMode = "No Mode";
    return (modeDevice
            ? modeDevice->mode
            : noMode);
  }

  const occa::json& device::properties() const {
    assertInitialized();
    return modeDevice->properties;
  }

  const occa::json& device::kernelProperties() const {
    assertInitialized();
    return (const occa::json&) modeDevice->properties["kernel"];
  }

  occa::json device::kernelProperties(const occa::json &additionalProps) const {
    return (
      kernelProperties()
      + getModeSpecificProps(mode(), additionalProps)
    );
  }

  const occa::json& device::memoryProperties() const {
    assertInitialized();
    return (const occa::json&) modeDevice->properties["memory"];
  }

  occa::json device::memoryProperties(const occa::json &additionalProps) const {
    return (
      memoryProperties()
      + getModeSpecificProps(mode(), additionalProps)
    );
  }

  const occa::json& device::streamProperties() const {
    assertInitialized();
    return (const occa::json&) modeDevice->properties["stream"];
  }

  occa::json device::streamProperties(const occa::json &additionalProps) const {
    return (
      streamProperties()
      + getModeSpecificProps(mode(), additionalProps)
    );
  }

  hash_t device::hash() const {
    if (modeDevice) {
      return modeDevice->versionedHash();
    }
    return hash_t();
  }

  udim_t device::memorySize() const {
    if (modeDevice) {
      return modeDevice->memorySize();
    }
    return 0;
  }

  udim_t device::memoryAllocated() const {
    if (modeDevice) {
      return modeDevice->bytesAllocated;
    }
    return 0;
  }

  udim_t device::maxMemoryAllocated() const {
    if (modeDevice) {
      return modeDevice->maxBytesAllocated;
    }
    return 0;
  }

  void device::finish() {
    if (modeDevice) {
      modeDevice->finish();
    }
  }

  void device::finishAll() {
    if (modeDevice) {
      modeDevice->finishAll();
    }
  }

  bool device::hasSeparateMemorySpace() {
    return (modeDevice &&
            modeDevice->hasSeparateMemorySpace());
  }

  //  |---[ Stream ]--------------------
  stream device::createStream(const occa::json &props) {
    assertInitialized();
    return modeDevice->createStream(streamProperties(props));
  }

  stream device::wrapStream(void* ptr, const occa::json &props) {
    assertInitialized();

    occa::json streamProps = streamProperties(props);

    return modeDevice->wrapStream(ptr, streamProps);
  }

  stream device::getStream() {
    assertInitialized();
    return modeDevice->currentStream;
  }

  void device::setStream(stream s) {
    assertInitialized();
    modeDevice->currentStream = s;
  }

  streamTag device::tagStream() {
    assertInitialized();
    return modeDevice->tagStream();
  }

  void device::waitFor(streamTag tag) {
    assertInitialized();
    modeDevice->waitFor(tag);
  }

  double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
    assertInitialized();
    return modeDevice->timeBetween(startTag, endTag);
  }
  //  |=================================

  //  |---[ Kernel ]--------------------
  void device::setupKernelInfo(const occa::json &props,
                               const hash_t &sourceHash,
                               occa::json &kernelProps,
                               hash_t &kernelHash) const {
    assertInitialized();

    kernelProps = kernelProperties(props);

    kernelHash = (
      hash()
      ^ modeDevice->kernelHash(kernelProps)
      ^ kernelHeaderHash(kernelProps)
      ^ sourceHash
    );

    kernelHash = applyDependencyHash(kernelHash);
  }

  hash_t device::applyDependencyHash(const hash_t &kernelHash) const {
    // Check if the build.json exists to compare dependencies
    const std::string buildFile = io::hashDir(kernelHash) + kc::buildFile;
    if (!io::exists(buildFile)) {
      return kernelHash;
    }

    json buildJson = json::read(buildFile);
    json dependenciesJson = buildJson["kernel/dependencies"];
    if (!dependenciesJson.isInitialized()) {
      return kernelHash;
    }

    hash_t newKernelHash = kernelHash;
    bool foundDependencyChanges = false;

    jsonObject dependencyHashes = dependenciesJson.object();
    jsonObject::iterator it = dependencyHashes.begin();
    while (it != dependencyHashes.end()) {
      const std::string &dependency = it->first;
      const hash_t dependencyHash = hash_t::fromString(it->second);

      if (io::exists(dependency)) {
        // Check whether the dependency changed
        hash_t newDependencyHash = hashFile(dependency);
        newKernelHash ^= newDependencyHash;

        if (dependencyHash != newDependencyHash) {
          foundDependencyChanges = true;
        }
      } else {
        // Dependency is missing so something changed
        foundDependencyChanges = true;
      }

      ++it;
    }

    if (foundDependencyChanges) {
      // Recursively check if new kernels had their dependencies changed
      return applyDependencyHash(newKernelHash);
    }
    return kernelHash;
  }

  kernel device::buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::json &props) const {
    occa::json allProps;
    hash_t kernelHash;
    const std::string realFilename = io::findInPaths(filename, env::OCCA_KERNEL_PATH);
    setupKernelInfo(props, hashFile(realFilename),
                    allProps, kernelHash);

    // TODO: [#185] Fix kernel cache frees
    // // Check cache first
    // kernel &cachedKernel = modeDevice->getCachedKernel(kernelHash,
    //                                                    kernelName);
    // if (cachedKernel.isInitialized()) {
    //   return cachedKernel;
    // }

    const std::string hashDir = io::hashDir(realFilename, kernelHash);
    allProps["hash"] = kernelHash.getFullString();

    kernel cachedKernel = modeDevice->buildKernel(realFilename,
                                                  kernelName,
                                                  kernelHash,
                                                  allProps);

    if (cachedKernel.isInitialized()) {
      cachedKernel.modeKernel->hash = kernelHash;
    } else {
      sys::rmrf(hashDir);
    }

    return cachedKernel;
  }

  kernel device::buildKernelFromString(const std::string &content,
                                       const std::string &kernelName,
                                       const occa::json &props) const {
    occa::json allProps;
    hash_t kernelHash;
    setupKernelInfo(props, occa::hash(content),
                    allProps, kernelHash);

    std::string stringSourceFile = (
      io::hashDir(kernelHash)
      + "string_source.cpp"
    );

    io::stageFile(
      stringSourceFile,
      true,
      [&](const std::string &tempFilename) -> bool {
        io::write(tempFilename, content);
        return true;
      }
    );

    return buildKernel(stringSourceFile,
                       kernelName,
                       props);
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &kernelName,
                                       const occa::json &props) const {
    assertInitialized();

    return kernel(modeDevice->buildKernelFromBinary(filename,
                                                    kernelName,
                                                    props));
  }
  //  |=================================

  //  |---[ Memory ]--------------------
  occa::memory device::malloc(const dim_t entries,
                              const dtype_t &dtype,
                              const void *src,
                              const occa::json &props) {
    assertInitialized();

    if (entries == 0) {
      return memory();
    }

    const dim_t bytes = entries * dtype.bytes();
    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= 0);

    occa::json memProps = memoryProperties(props);

    memory mem(modeDevice->malloc(bytes, src, memProps));
    mem.setDtype(dtype);

    modeDevice->bytesAllocated += bytes;
    modeDevice->maxBytesAllocated = std::max(
      modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
    );

    return mem;
  }

  occa::memory device::malloc(const dim_t entries,
                              const dtype_t &dtype,
                              const occa::memory src,
                              const occa::json &props) {
    memory mem = malloc(entries, dtype, NULL, props);
    if (entries && src.size()) {
      mem.copyFrom(src);
    }
    return mem;
  }

  occa::memory device::malloc(const dim_t entries,
                              const dtype_t &dtype,
                              const occa::json &props) {
    return malloc(entries, dtype, NULL, props);
  }

  template <>
  memory device::malloc<void>(const dim_t entries,
                              const void *src,
                              const occa::json &props) {
    return malloc(entries, dtype::byte, src, props);
  }

  template <>
  memory device::malloc<void>(const dim_t entries,
                              const occa::memory src,
                              const occa::json &props) {
    return malloc(entries, dtype::byte, src, props);
  }

  template <>
  memory device::malloc<void>(const dim_t entries,
                              const occa::json &props) {
    return malloc(entries, dtype::byte, NULL, props);
  }

  template <>
  occa::memory device::wrapMemory<void>(const void *ptr,
                                        const dim_t entries,
                                        const occa::json &props) {
    return wrapMemory(ptr, entries, dtype::byte, props);
  }

  occa::memory device::wrapMemory(const void *ptr,
                                  const dim_t entries,
                                  const dtype_t &dtype,
                                  const occa::json &props) {
    assertInitialized();

    const dim_t bytes = entries * dtype.bytes();
    OCCA_ERROR("Trying to wrap a pointer with negative bytes (" << bytes << ")",
               bytes >= 0);

    occa::json memProps = memoryProperties(props);

    memory mem(modeDevice->wrapMemory(ptr, bytes, memProps));

    mem.setDtype(dtype);

    return mem;
  }

  memoryPool device::createMemoryPool(const occa::json &props) {
    assertInitialized();

    occa::json memProps = memoryProperties(props);

    memoryPool memPool(modeDevice->createMemoryPool(memProps));

    return memPool;
  }
  //  |=================================

  void* device::unwrap() {
    assertInitialized();
    return modeDevice->unwrap();
  }

  //  |=================================

  template <>
  hash_t hash(const occa::device &device) {
    return device.hash();
  }

  std::ostream& operator << (std::ostream &out,
                             const occa::device &device) {
    out << device.properties();
    return out;
  }
  //====================================
}
