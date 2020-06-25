#include <occa/core/device.hpp>
#include <occa/core/base.hpp>
#include <occa/modes.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/io.hpp>

namespace occa {
  //---[ modeDevice_t ]-----------------
  modeDevice_t::modeDevice_t(const occa::properties &properties_) :
    mode((std::string) properties_["mode"]),
    properties(properties_),
    needsLauncherKernel(false),
    bytesAllocated(0) {}

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
    freeRing<modeMemory_t>(memoryRing);
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

  void modeDevice_t::addMemoryRef(modeMemory_t *memory) {
    memoryRing.addRef(memory);
  }

  void modeDevice_t::removeMemoryRef(modeMemory_t *memory) {
    memoryRing.removeRef(memory);
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

  hash_t modeDevice_t::versionedHash() const {
    return (occa::hash(settings()["version"])
            ^ hash());
  }

  void modeDevice_t::writeKernelBuildFile(const std::string &filename,
                                          const hash_t &kernelHash,
                                          const occa::properties &kernelProps,
                                          const lang::sourceMetadata_t &sourceMetadata) const {
    occa::properties infoProps;

    infoProps["device"]       = properties;
    infoProps["device/hash"]  = versionedHash().getFullString();
    infoProps["kernel/props"] = kernelProps;
    infoProps["kernel/hash"]  = kernelHash.getFullString();
    infoProps["kernel/metadata"] = sourceMetadata.getKernelMetadataJson();
    infoProps["kernel/dependencies"] = sourceMetadata.getDependencyJson();

    io::writeBuildFile(filename, kernelHash, infoProps);
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
  //====================================

  //---[ device ]-----------------------
  device::device() :
    modeDevice(NULL) {}

  device::device(modeDevice_t *modeDevice_) :
    modeDevice(NULL) {
    setModeDevice(modeDevice_);
  }

  device::device(const occa::properties &props) :
    modeDevice(NULL) {
    setup(props);
  }

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

  bool device::isInitialized() {
    return (modeDevice != NULL);
  }

  modeDevice_t* device::getModeDevice() const {
    return modeDevice;
  }

  void device::setup(const occa::properties &props) {
    free();

    const std::string mode_ = props["mode"];

    occa::properties deviceProps = (
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

  const occa::properties& device::properties() const {
    assertInitialized();
    return modeDevice->properties;
  }

  const occa::properties& device::kernelProperties() const {
    assertInitialized();
    return (const occa::properties&) modeDevice->properties["kernel"];
  }

  occa::properties device::kernelProperties(const occa::properties &additionalProps) const {
    return (
      kernelProperties()
      + getModeSpecificProps(mode(), additionalProps)
    );
  }

  const occa::properties& device::memoryProperties() const {
    assertInitialized();
    return (const occa::properties&) modeDevice->properties["memory"];
  }

  occa::properties device::memoryProperties(const occa::properties &additionalProps) const {
    return (
      memoryProperties()
      + getModeSpecificProps(mode(), additionalProps)
    );
  }

  const occa::properties& device::streamProperties() const {
    assertInitialized();
    return (const occa::properties&) modeDevice->properties["stream"];
  }

  occa::properties device::streamProperties(const occa::properties &additionalProps) const {
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

  void device::finish() {
    if (!modeDevice) {
      return;
    }
    if (modeDevice->hasSeparateMemorySpace()) {
      const size_t staleEntries = uvaStaleMemory.size();
      for (size_t i = 0; i < staleEntries; ++i) {
        occa::modeMemory_t *mem = uvaStaleMemory[i];

        mem->copyTo(mem->uvaPtr, mem->size, 0, "async: true");

        mem->memInfo &= ~uvaFlag::inDevice;
        mem->memInfo &= ~uvaFlag::isStale;
      }
      if (staleEntries) {
        uvaStaleMemory.clear();
      }
    }

    modeDevice->finish();
  }

  bool device::hasSeparateMemorySpace() {
    return (modeDevice &&
            modeDevice->hasSeparateMemorySpace());
  }

  //  |---[ Stream ]--------------------
  stream device::createStream(const occa::properties &props) {
    assertInitialized();
    return modeDevice->createStream(streamProperties(props));
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
  void device::setupKernelInfo(const occa::properties &props,
                               const hash_t &sourceHash,
                               occa::properties &kernelProps,
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
                             const occa::properties &props) const {
    occa::properties allProps;
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
                                       const occa::properties &props) const {
    occa::properties allProps;
    hash_t kernelHash;
    setupKernelInfo(props, occa::hash(content),
                    allProps, kernelHash);

    io::lock_t lock(kernelHash, "occa-device");
    std::string stringSourceFile = io::hashDir(kernelHash);
    stringSourceFile += "string_source.cpp";

    if (lock.isMine()) {
      if (!io::isFile(stringSourceFile)) {
        io::write(stringSourceFile, content);
      }
      lock.release();
    }

    return buildKernel(stringSourceFile,
                       kernelName,
                       props);
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &kernelName,
                                       const occa::properties &props) const {
    assertInitialized();

    return kernel(modeDevice->buildKernelFromBinary(filename,
                                                    kernelName,
                                                    props));
  }

  void device::loadKernels(const std::string &library) {
    // TODO 1.1: Load kernels
#if 0
    assertInitialized();

    std::string devHash = hash().getFullString();
    strVector dirs = io::directories("occa://" + library);
    const int dirCount = (int) dirs.size();
    int kernelsLoaded = 0;

    for (int d = 0; d < dirCount; ++d) {
      const std::string buildFile = dirs[d] + kc::buildFile;

      if (!io::isFile(buildFile)) {
        continue;
      }

      json info = json::read(buildFile)["info"];
      if ((std::string) info["device/hash"] != devHash) {
        continue;
      }
      ++kernelsLoaded;

      const std::string sourceFilename = dirs[d] + kc::parsedSourceFile;

      json &kInfo = info["kernel"];
      hash_t hash = hash_t::fromString(kInfo["hash"]);
      jsonArray metadataArray = kInfo["metadata"].array();
      occa::properties kernelProps = kInfo["props"];

      // Ignore how the kernel was setup, turn off verbose
      kernelProps["verbose"] = false;

      const int kernels = metadataArray.size();
      for (int k = 0; k < kernels; ++k) {
        buildKernel(sourceFilename,
                    hash,
                    kernelProps,
                    lang::kernelMetadata_t::fromJson(metadataArray[k]));
      }
    }

    // Print loaded info
    if (properties().get("verbose", false) && kernelsLoaded) {
      io::stdout << "Loaded " << kernelsLoaded;
      if (library.size()) {
        io::stdout << " ["<< library << "]";
      } else {
        io::stdout << " cached";
      }
      io::stdout << ((kernelsLoaded == 1)
                     ? " kernel\n"
                     : " kernels\n");
    }
#endif
  }
  //  |=================================

  //  |---[ Memory ]--------------------
  occa::memory device::malloc(const dim_t entries,
                              const dtype_t &dtype,
                              const void *src,
                              const occa::properties &props) {
    assertInitialized();

    if (entries == 0) {
      return memory();
    }

    const dim_t bytes = entries * dtype.bytes();
    OCCA_ERROR("Trying to allocate "
               << "negative bytes (" << bytes << ")",
               bytes >= 0);

    occa::properties memProps = memoryProperties(props);

    memory mem(modeDevice->malloc(bytes, src, memProps));
    mem.setDtype(dtype);

    modeDevice->bytesAllocated += bytes;

    return mem;
  }

  occa::memory device::malloc(const dim_t entries,
                              const dtype_t &dtype,
                              const occa::memory src,
                              const occa::properties &props) {
    memory mem = malloc(entries, dtype, NULL, props);
    if (entries && src.size()) {
      mem.copyFrom(src);
    }
    return mem;
  }

  occa::memory device::malloc(const dim_t entries,
                              const dtype_t &dtype,
                              const occa::properties &props) {
    return malloc(entries, dtype, NULL, props);
  }

  template <>
  memory device::malloc<void>(const dim_t entries,
                              const void *src,
                              const occa::properties &props) {
    return malloc(entries, dtype::byte, src, props);
  }

  template <>
  memory device::malloc<void>(const dim_t entries,
                              const occa::memory src,
                              const occa::properties &props) {
    return malloc(entries, dtype::byte, src, props);
  }

  template <>
  memory device::malloc<void>(const dim_t entries,
                              const occa::properties &props) {
    return malloc(entries, dtype::byte, NULL, props);
  }

  void* device::umalloc(const dim_t entries,
                        const dtype_t &dtype,
                        const void *src,
                        const occa::properties &props) {
    void *ptr = umalloc(entries, dtype, occa::memory(), props);

    if (src && entries) {
      const dim_t bytes = entries * dtype.bytes();
      ::memcpy(ptr, src, bytes);
    }

    return ptr;
  }

  void* device::umalloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::memory src,
                        const occa::properties &props) {
    assertInitialized();

    if (entries == 0) {
      return NULL;
    }

    occa::properties memProps = memoryProperties(props);

    memory mem = malloc(entries, dtype, src, memProps);
    mem.setDtype(dtype);
    mem.dontUseRefs();
    mem.setupUva();

    if (memProps.get("managed", true)) {
      mem.startManaging();
    }
    void *ptr = mem.modeMemory->uvaPtr;
    if (src.size()) {
      mem.copyTo(ptr);
    }
    return ptr;
  }

  void* device::umalloc(const dim_t entries,
                        const dtype_t &dtype,
                        const occa::properties &props) {
    return umalloc(entries, dtype, NULL, props);
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

  //---[ Utils ]------------------------
  occa::properties getModeSpecificProps(const std::string &mode,
                                        const occa::properties &props) {
    occa::properties allProps = (
      props
      + props["modes/" + mode]
    );

    allProps.remove("modes");

    return allProps;
  }

  occa::properties getObjectSpecificProps(const std::string &mode,
                                          const std::string &object,
                                          const occa::properties &props) {
    occa::properties allProps = (
      props[object]
      + props[object + "/modes/" + mode]
      + props["modes/" + mode + "/" + object]
    );

    allProps.remove(object + "/modes");
    allProps.remove("modes");

    return allProps;
  }

  occa::properties initialObjectProps(const std::string &mode,
                                      const std::string &object,
                                      const occa::properties &props) {
    occa::properties objectProps = (
      getObjectSpecificProps(mode, object, settings())
      + getObjectSpecificProps(mode, object, props)
    );
    objectProps["mode"] = mode;
    return objectProps;
  }
  //====================================
}
