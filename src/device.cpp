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

#include <occa/device.hpp>
#include <occa/base.hpp>
#include <occa/mode.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/io.hpp>

namespace occa {
  //---[ device_v ]---------------------
  device_v::device_v(const occa::properties &properties_) {
    mode = (std::string) properties_["mode"];
    properties = properties_;

    currentStream = NULL;
    bytesAllocated = 0;
  }

  device_v::~device_v() {}

  hash_t device_v::versionedHash() const {
    return (occa::hash(settings()["version"])
            ^ hash());
  }

  void device_v::writeKernelBuildFile(const std::string &filename,
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

  std::string device_v::getKernelHash(const std::string &fullHash,
                                      const std::string &kernelName) {
    return (fullHash + "-" + kernelName);
  }

  std::string device_v::getKernelHash(const hash_t &kernelHash,
                                      const std::string &kernelName) {
    return getKernelHash(kernelHash.toFullString(),
                         kernelName);
  }

  std::string device_v::getKernelHash(kernel_v *kernel) {
    return getKernelHash(kernel->properties["hash"],
                         kernel->name);
  }

  kernel& device_v::getCachedKernel(const hash_t &kernelHash,
                                    const std::string &kernelName) {

    return cachedKernels[getKernelHash(kernelHash, kernelName)];
  }

  void device_v::removeCachedKernel(kernel_v *kernel) {
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

  device::device(device_v *modeDevice_) :
    modeDevice(NULL) {
    setModeDevice(modeDevice_);
  }

  device::device(const occa::properties &props) :
    modeDevice(NULL) {
    setup(props);
  }

  device::device(const device &d) :
    modeDevice(NULL) {
    setModeDevice(d.modeDevice);
  }

  device& device::operator = (const device &d) {
    setModeDevice(d.modeDevice);
    return *this;
  }

  device::~device() {
    removeRef();
  }

  void device::setModeDevice(device_v *modeDevice_) {
    if (modeDevice != modeDevice_) {
      removeRef();
      modeDevice = modeDevice_;
      modeDevice->addRef();
    }
  }

  void device::removeRef() {
    if (modeDevice && !modeDevice->removeRef()) {
      free();
    }
  }

  void device::dontUseRefs() {
    if (modeDevice) {
      modeDevice->dontUseRefs();
    }
  }

  bool device::operator == (const occa::device &d) const {
    return (modeDevice == d.modeDevice);
  }

  bool device::isInitialized() {
    return (modeDevice != NULL);
  }

  device_v* device::getModeDevice() const {
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
    }

    setModeDevice(occa::newModeDevice(defaults + props));

    stream newStream = createStream();
    modeDevice->currentStream = newStream.modeStream;
  }

  void device::free() {
    if (modeDevice == NULL) {
      return;
    }
    const int streamCount = modeDevice->streams.size();

    for (int i = 0; i < streamCount; ++i) {
      modeDevice->freeStream(modeDevice->streams[i]);
    }
    modeDevice->streams.clear();
    modeDevice->free();

    delete modeDevice;
    modeDevice = NULL;
  }

  const std::string& device::mode() const {
    static const std::string noMode = "No Mode";
    return (modeDevice
            ? modeDevice->mode
            : noMode);
  }

  occa::properties& device::properties() {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return modeDevice->properties;
  }

  const occa::properties& device::properties() const {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return modeDevice->properties;
  }

  occa::properties& device::kernelProperties() {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return (occa::properties&) modeDevice->properties["kernel"];
  }

  const occa::properties& device::kernelProperties() const {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return (const occa::properties&) modeDevice->properties["kernel"];
  }

  occa::properties& device::memoryProperties() {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return (occa::properties&) modeDevice->properties["memory"];
  }

  const occa::properties& device::memoryProperties() const {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return (const occa::properties&) modeDevice->properties["memory"];
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
        occa::memory_v *mem = uvaStaleMemory[i];

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
  stream device::createStream() {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

    stream newStream(modeDevice, modeDevice->createStream());
    modeDevice->streams.push_back(newStream.modeStream);

    return newStream;
  }

  void device::freeStream(stream s) {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

    const int streamCount = modeDevice->streams.size();

    for (int i = 0; i < streamCount; ++i) {
      if (modeDevice->streams[i] == s.modeStream) {
        if (modeDevice->currentStream == s.modeStream)
          modeDevice->currentStream = NULL;

        modeDevice->freeStream(modeDevice->streams[i]);
        modeDevice->streams.erase(modeDevice->streams.begin() + i);

        break;
      }
    }
  }

  stream device::getStream() {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return stream(modeDevice, modeDevice->currentStream);
  }

  void device::setStream(stream s) {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    modeDevice->currentStream = s.modeStream;
  }

  streamTag device::tagStream() {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return modeDevice->tagStream();
  }

  void device::waitFor(streamTag tag) {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    modeDevice->waitFor(tag);
  }

  double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);
    return modeDevice->timeBetween(startTag, endTag);
  }
  //  |=================================

  //  |---[ Kernel ]--------------------
  kernel device::buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props) const {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    // TODO: [#184] Properly hash through device
    hash_t kernelHash = (hash()
                         ^ occa::hash(allProps)
                         ^ hashFile(filename));

    // Check cache first
    kernel &cachedKernel = modeDevice->getCachedKernel(kernelHash,
                                                    kernelName);
    if (cachedKernel.isInitialized()) {
      return cachedKernel;
    }

    const std::string realFilename = io::filename(filename);
    const std::string hashDir = io::hashDir(realFilename, kernelHash);
    allProps["hash"] = kernelHash.toFullString();

    cachedKernel = modeDevice->buildKernel(realFilename,
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
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    // Store in the same directory as cached outputs
    hash_t kernelHash = (hash()
                         ^ occa::hash(allProps)
                         ^ occa::hash(content));

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
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

    return kernel(modeDevice->buildKernelFromBinary(filename,
                                                 kernelName,
                                                 props));
  }

  void device::loadKernels(const std::string &library) {
    // TODO 1.1: Load kernels
#if 0
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

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
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

    if (bytes == 0) {
      return memory();
    }

    OCCA_ERROR("Trying to allocate "
               << "negative bytes (" << bytes << ")",
               bytes >= 0);

    occa::properties memProps = props + memoryProperties();

    memory mem(modeDevice->malloc(bytes, src, memProps));
    mem.setModeDevice(modeDevice);

    modeDevice->bytesAllocated += bytes;

    return mem;
  }

  memory device::malloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::properties &props) {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

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
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

    void *ptr = umalloc(bytes, occa::memory(), props);
    if (src) {
      ::memcpy(ptr, src, bytes);
    }
    return ptr;
  }

  void* device::umalloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::properties &props) {
    OCCA_ERROR("Device not initialized",
               modeDevice != NULL);

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

  //---[ stream ]-----------------------
  stream::stream() :
    modeDevice(NULL),
    modeStream(NULL) {}

  stream::stream(device_v *modeDevice_,
                 stream_t modeStream_) :
    modeDevice(modeDevice_),
    modeStream(modeStream_) {}

  stream::stream(const stream &other) :
    modeDevice(other.modeDevice),
    modeStream(other.modeStream) {}

  stream& stream::operator = (const stream &other) {
    modeDevice = other.modeDevice;
    modeStream  = other.modeStream;
    return *this;
  }

  bool stream::operator == (const stream &other) const {
    return ((modeDevice == other.modeDevice) &&
            (modeStream == other.modeStream));
  }

  stream_t stream::getModeStream() {
    return modeStream;
  }

  void stream::free() {
    if (modeDevice != NULL) {
      device(modeDevice).freeStream(*this);
    }
  }

  streamTag::streamTag() :
    tagTime(0),
    modeTag(NULL) {}

  streamTag::streamTag(const double tagTime_,
                       void *modeTag_) :
    tagTime(tagTime_),
    modeTag(modeTag_) {}
  //====================================
}
