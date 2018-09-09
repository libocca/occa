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

#include <occa/core/device.hpp>
#include <occa/core/base.hpp>
#include <occa/mode.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/io.hpp>

namespace occa {
  //---[ modeDevice_t ]-----------------
  modeDevice_t::modeDevice_t(const occa::properties &properties_) :
    mode((std::string) properties_["mode"]),
    properties(properties_),
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
                                          const lang::kernelMetadataMap &metadataMap) const {
    occa::properties infoProps;

    infoProps["device"]       = properties;
    infoProps["device/hash"]  = versionedHash().toFullString();
    infoProps["kernel/props"] = kernelProps;
    infoProps["kernel/hash"]  = kernelHash.toFullString();

    json &metadataJson = infoProps["kernel/metadata"].asArray();
    lang::kernelMetadataMap::const_iterator kIt = metadataMap.begin();
    while (kIt != metadataMap.end()) {
      metadataJson += (kIt->second).toJson();
      ++kIt;
    }

    io::writeBuildFile(filename, kernelHash, infoProps);
  }

  std::string modeDevice_t::getKernelHash(const std::string &fullHash,
                                          const std::string &kernelName) {
    return (fullHash + "-" + kernelName);
  }

  std::string modeDevice_t::getKernelHash(const hash_t &kernelHash,
                                          const std::string &kernelName) {
    return getKernelHash(kernelHash.toFullString(),
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
    occa::properties settings_ = settings();
    occa::properties defaults;

    std::string paths[2] = {"", ""};
    paths[1] = "mode/";
    paths[1] += (std::string) props["mode"];
    paths[1] += '/';

    for (int i = 0; i < 2; ++i) {
      const std::string &path = paths[i];

      if (settings_.has(path + "device")) {
        defaults += settings_[path + "device"];
      }
      if (settings_.has(path + "kernel")) {
        defaults["kernel"] += settings_[path + "kernel"];
      }
      if (settings_.has(path + "memory")) {
        defaults["memory"] += settings_[path + "memory"];
      }
      if (settings_.has(path + "stream")) {
        defaults["stream"] += settings_[path + "stream"];
      }
    }

    setModeDevice(occa::newModeDevice(defaults + props));

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

  occa::properties& device::properties() {
    assertInitialized();
    return modeDevice->properties;
  }

  const occa::properties& device::properties() const {
    assertInitialized();
    return modeDevice->properties;
  }

  occa::properties& device::kernelProperties() {
    assertInitialized();
    return (occa::properties&) modeDevice->properties["kernel"];
  }

  const occa::properties& device::kernelProperties() const {
    assertInitialized();
    return (const occa::properties&) modeDevice->properties["kernel"];
  }

  occa::properties& device::memoryProperties() {
    assertInitialized();
    return (occa::properties&) modeDevice->properties["memory"];
  }

  const occa::properties& device::memoryProperties() const {
    assertInitialized();
    return (const occa::properties&) modeDevice->properties["memory"];
  }

  occa::properties& device::streamProperties() {
    assertInitialized();
    return (occa::properties&) modeDevice->properties["stream"];
  }

  const occa::properties& device::streamProperties() const {
    assertInitialized();
    return (const occa::properties&) modeDevice->properties["stream"];
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
    return modeDevice->createStream(props + streamProperties());
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

    kernelProps = props + kernelProperties();
    kernelProps["mode"] = mode();

    kernelHash = (hash()
                  ^ modeDevice->kernelHash(kernelProps)
                  ^ kernelHeaderHash(kernelProps)
                  ^ sourceHash);
  }

  kernel device::buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props) const {
    occa::properties allProps;
    hash_t kernelHash;
    setupKernelInfo(props, hashFile(filename),
                    allProps, kernelHash);

    // TODO: [#185] Fix kernel cache frees
    // // Check cache first
    // kernel &cachedKernel = modeDevice->getCachedKernel(kernelHash,
    //                                                    kernelName);
    // if (cachedKernel.isInitialized()) {
    //   return cachedKernel;
    // }

    const std::string realFilename = io::filename(filename);
    const std::string hashDir = io::hashDir(realFilename, kernelHash);
    allProps["hash"] = kernelHash.toFullString();

    kernel cachedKernel = modeDevice->buildKernel(realFilename,
                                                  kernelName,
                                                  kernelHash,
                                                  allProps);

    if (!cachedKernel.isInitialized()) {
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

    std::string devHash = hash().toFullString();
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
                    lang::kernelMetadata::fromJson(metadataArray[k]));
      }
    }

    // Print loaded info
    if (properties().get("verbose", false) && kernelsLoaded) {
      std::cout << "Loaded " << kernelsLoaded;
      if (library.size()) {
        std::cout << " ["<< library << "]";
      } else {
        std::cout << " cached";
      }
      std::cout << ((kernelsLoaded == 1)
                    ? " kernel\n"
                    : " kernels\n");
    }
#endif
  }
  //  |=================================

  //  |---[ Memory ]--------------------
  memory device::malloc(const dim_t bytes,
                        const void *src,
                        const occa::properties &props) {
    assertInitialized();

    if (bytes == 0) {
      return memory();
    }

    OCCA_ERROR("Trying to allocate "
               << "negative bytes (" << bytes << ")",
               bytes >= 0);

    occa::properties memProps = props + memoryProperties();
    memory mem(modeDevice->malloc(bytes, src, memProps));

    modeDevice->bytesAllocated += bytes;

    return mem;
  }

  memory device::malloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::properties &props) {
    assertInitialized();

    memory mem = malloc(bytes, NULL, props);
    if (bytes && src.size()) {
      mem.copyFrom(src);
    }
    return mem;
  }

  memory device::malloc(const dim_t bytes,
                        const occa::properties &props) {

    return malloc(bytes, NULL, props);
  }

  void* device::umalloc(const dim_t bytes,
                        const void *src,
                        const occa::properties &props) {
    assertInitialized();

    void *ptr = umalloc(bytes, occa::memory(), props);
    if (src) {
      ::memcpy(ptr, src, bytes);
    }
    return ptr;
  }

  void* device::umalloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::properties &props) {
    assertInitialized();

    if (bytes == 0) {
      return NULL;
    }

    OCCA_ERROR("Trying to allocate "
               << "negative bytes (" << bytes << ")",
               bytes >= 0);

    occa::properties memProps = props + memoryProperties();

    memory mem = malloc(bytes, src, memProps);
    mem.dontUseRefs();
    mem.setupUva();

    if (memProps.get("managed", true)) {
      mem.startManaging();
    }
    void *ptr = mem.modeMemory->uvaPtr;
    if (src.size()) {
      src.copyTo(ptr, bytes);
    }
    return ptr;
  }

  void* device::umalloc(const dim_t bytes,
                        const occa::properties &props) {

    return umalloc(bytes, NULL, props);
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
