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

#include "occa/device.hpp"
#include "occa/base.hpp"
#include "occa/mode.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/io.hpp"

namespace occa {
  //---[ device_v ]---------------------
  device_v::device_v(const occa::properties &properties_) {
    mode = properties_["mode"].string();
    properties = properties_;

    currentStream = NULL;
    bytesAllocated = 0;
  }

  device_v::~device_v() {}

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
    return getKernelHash(kernel->properties["hash"].string(),
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
    dHandle(NULL) {}

  device::device(device_v *dHandle_) :
    dHandle(NULL) {
    setDHandle(dHandle_);
  }

  device::device(const occa::properties &props) :
    dHandle(NULL) {
    setup(props);
  }

  device::device(const device &d) :
    dHandle(NULL) {
    setDHandle(d.dHandle);
  }

  device& device::operator = (const device &d) {
    setDHandle(d.dHandle);
    return *this;
  }

  device::~device() {
    removeDHandleRef();
  }

  void device::setDHandle(device_v *dHandle_) {
    if (dHandle != dHandle_) {
      removeDHandleRef();
      dHandle = dHandle_;
      dHandle->addRef();
    }
  }

  void device::removeDHandleRef() {
    removeDHandleRefFrom(dHandle);
  }

  void device::removeDHandleRefFrom(device_v *&dHandle_) {
    if (dHandle_ && !dHandle_->removeRef()) {
      free(dHandle_);
      delete dHandle_;
      dHandle_ = NULL;
    }
  }

  void device::dontUseRefs() {
    if (dHandle) {
      dHandle->dontUseRefs();
    }
  }

  bool device::operator == (const occa::device &d) const {
    return (dHandle == d.dHandle);
  }

  bool device::isInitialized() {
    return (dHandle != NULL);
  }

  device_v* device::getDHandle() const {
    return dHandle;
  }

  void device::setup(const occa::properties &props) {
    occa::properties settings_ = settings();
    occa::properties defaults;

    std::string modePath = "mode/" + props["mode"].string() + "/";
    std::string paths[2] = {"", modePath};

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

    setDHandle(occa::newModeDevice(defaults + props));

    stream newStream = createStream();
    dHandle->currentStream = newStream.handle;
  }

  void device::free() {
    free(dHandle);
  }

  void device::free(device_v *&dHandle_) {
    if (dHandle_ == NULL) {
      return;
    }
    const int streamCount = dHandle_->streams.size();

    for (int i = 0; i < streamCount; ++i) {
      dHandle_->freeStream(dHandle_->streams[i]);
    }
    dHandle_->streams.clear();
    dHandle_->free();
  }

  const std::string& device::mode() const {
    return dHandle->mode;
  }

  occa::properties& device::properties() {
    return dHandle->properties;
  }

  const occa::properties& device::properties() const {
    return dHandle->properties;
  }

  occa::properties& device::kernelProperties() {
    occa::properties &ret = (occa::properties&) dHandle->properties["kernel"];
    return ret;
  }

  const occa::properties& device::kernelProperties() const {
    const occa::properties &ret = (const occa::properties&) dHandle->properties["kernel"];
    return ret;
  }

  occa::properties& device::memoryProperties() {
    occa::properties &ret = (occa::properties&) dHandle->properties["memory"];
    return ret;
  }

  const occa::properties& device::memoryProperties() const {
    const occa::properties &ret = (const occa::properties&) dHandle->properties["memory"];
    return ret;
  }

  hash_t device::hash() const {
    return (occa::hash(settings()["version"])
            ^ dHandle->hash());
  }

  udim_t device::memorySize() const {
    return dHandle->memorySize();
  }

  udim_t device::memoryAllocated() const {
    return dHandle->bytesAllocated;
  }

  void device::finish() {
    if (dHandle->hasSeparateMemorySpace()) {
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

    dHandle->finish();
  }

  bool device::hasSeparateMemorySpace() {
    return dHandle->hasSeparateMemorySpace();
  }

  //  |---[ Stream ]--------------------
  stream device::createStream() {
    stream newStream(dHandle, dHandle->createStream());
    dHandle->streams.push_back(newStream.handle);

    return newStream;
  }

  void device::freeStream(stream s) {
    const int streamCount = dHandle->streams.size();

    for (int i = 0; i < streamCount; ++i) {
      if (dHandle->streams[i] == s.handle) {
        if (dHandle->currentStream == s.handle)
          dHandle->currentStream = NULL;

        dHandle->freeStream(dHandle->streams[i]);
        dHandle->streams.erase(dHandle->streams.begin() + i);

        break;
      }
    }
  }

  stream device::getStream() {
    return stream(dHandle, dHandle->currentStream);
  }

  void device::setStream(stream s) {
    dHandle->currentStream = s.handle;
  }

  stream device::wrapStream(void *handle_, const occa::properties &props) {
    return stream(dHandle, dHandle->wrapStream(handle_, props));
  }

  streamTag device::tagStream() {
    return dHandle->tagStream();
  }

  void device::waitFor(streamTag tag) {
    dHandle->waitFor(tag);
  }

  double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
    return dHandle->timeBetween(startTag, endTag);
  }
  //  |=================================

  //  |---[ Kernel ]--------------------
  void device::storeCacheInfo(const std::string &filename,
                              const hash_t &kernelHash,
                              const occa::properties &kernelProps,
                              const lang::kernelMetadataMap &metadataMap) const {
    occa::properties infoProps;
    infoProps["device"]       = dHandle->properties;
    infoProps["device/hash"]  = hash().toFullString();
    infoProps["kernel/props"] = kernelProps;
    infoProps["kernel/hash"]  = kernelHash.toFullString();

    json &metadataJson = infoProps["kernel/metadata"].asArray();
    lang::kernelMetadataMap::const_iterator kIt = metadataMap.begin();
    while (kIt != metadataMap.end()) {
      metadataJson += (kIt->second).toJson();
      ++kIt;
    }

    io::storeCacheInfo(filename, kernelHash, infoProps);
  }

  void device::loadKernels(const std::string &library) {
    std::string devHash = hash().toFullString();
    strVector dirs = io::directories("occa://" + library);
    const int dirCount = (int) dirs.size();
    int kernelsLoaded = 0;

    for (int d = 0; d < dirCount; ++d) {
      const std::string infoFile = dirs[d] + kc::infoFile;

      if (!sys::fileExists(infoFile)) {
        continue;
      }

      json info = json::read(infoFile)["info"];
      if (info["device/hash"].string() != devHash) {
        continue;
      }
      ++kernelsLoaded;

      const std::string sourceFilename = dirs[d] + kc::parsedSourceFile;

      json &kInfo = info["kernel"];
      hash_t hash = hash_t::fromString(kInfo["hash"].string());
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
  }

  kernel device::buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props) const {

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    hash_t kernelHash = (hash()
                         ^ occa::hash(allProps)
                         ^ hashFile(filename));

    const std::string realFilename = io::filename(filename);
    const std::string hashDir = io::hashDir(realFilename, kernelHash);
    std::string sourceFilename = realFilename;

    lang::kernelMetadata metadata;
    if (allProps.get("okl", true)) {
      sourceFilename = hashDir + kc::parsedSourceFile;

      lang::kernelMetadataMap metadataMap = io::parseFile(realFilename,
                                                          sourceFilename,
                                                          allProps);

      lang::kernelMetadataMap::iterator kIt = metadataMap.find(kernelName);
      OCCA_ERROR("Could not find kernel ["
                 << kernelName << "] in file ["
                 << io::shortname(filename) << "]",
                 kIt != metadataMap.end());

      metadata = kIt->second;

      storeCacheInfo(filename, kernelHash, allProps, metadataMap);
    } else {
      metadata.name = kernelName;
    }

    return buildKernel(sourceFilename,
                       kernelHash,
                       allProps,
                       metadata);
  }

  kernel device::buildKernelFromString(const std::string &content,
                                       const std::string &kernelName,
                                       const occa::properties &props) const {

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    hash_t kernelHash = (hash()
                         ^ occa::hash(allProps)
                         ^ occa::hash(content));

    const std::string hashDir = io::hashDir(kernelHash);
    const std::string hashTag = "occa-device";

    std::string stringSourceFile = hashDir;
    stringSourceFile += "stringSource.okl";

    if (!io::haveHash(kernelHash, hashTag)) {
      io::waitForHash(kernelHash, hashTag);
    } else {
      if (!sys::fileExists(stringSourceFile)) {
        io::write(stringSourceFile, content);
      }
      io::releaseHash(kernelHash, hashTag);
    }

    return buildKernel(stringSourceFile,
                       kernelName,
                       props);
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &kernelName,
                                       const occa::properties &props) const {

    return kernel(dHandle->buildKernelFromBinary(filename,
                                                 kernelName,
                                                 props));
  }

  occa::kernel device::buildKernel(const std::string &filename,
                                   const hash_t &hash,
                                   const occa::properties &kernelProps,
                                   const lang::kernelMetadata &metadata) const {

    // Native kernels don't need a host() to launch them
    device_v *launcherHandle = ((metadata.nestedKernels > 0)
                                ? host().getDHandle()
                                : dHandle);
    // Check cache first
    kernel &ker = launcherHandle->getCachedKernel(hash, metadata.name);
    if (ker.isInitialized()) {
      return ker;
    }

    // Store hash to clean-up cachedKernels during free()
    occa::properties allProps = kernelProps;
    allProps["hash"] = hash.toFullString();

    if (metadata.nestedKernels == 0) {
      ker = launcherHandle->buildKernel(filename,
                                        metadata.name,
                                        hash,
                                        allProps);
      return ker;
    }

    // Create launch kernel
    occa::properties launchProps = host().kernelProperties();
    launchProps["defines/OCCA_LAUNCH_KERNEL"] = 1;
    launchProps["hash"] = hash.toFullString();

    ker = launcherHandle->buildKernel(filename,
                                      metadata.name,
                                      hash,
                                      launchProps);

    // Load nested kernels
    if (metadata.nestedKernels) {
      for (int ki = 0; ki < metadata.nestedKernels; ++ki) {
        lang::kernelMetadata sMetadata = metadata.getNestedKernelMetadata(ki);
        const std::string &sKerName    = sMetadata.name;

        kernel &sKer = dHandle->getCachedKernel(hash,
                                                sKerName);
        sKer = dHandle->buildKernel(filename,
                                    sKerName,
                                    hash,
                                    allProps);
        sKer.kHandle->metadata = sMetadata;

        ker.kHandle->nestedKernels.push_back(sKer);
      }
    }

    return ker;
  }
  //  |=================================

  //  |---[ Memory ]--------------------
  memory device::malloc(const dim_t bytes,
                        const void *src,
                        const occa::properties &props) {
    if (bytes == 0) {
      return memory();
    }

    OCCA_ERROR("Trying to allocate "
               << "negative bytes (" << bytes << ")",
               bytes >= 0);

    occa::properties memProps = props + memoryProperties();

    memory mem(dHandle->malloc(bytes, src, memProps));
    mem.setDHandle(dHandle);

    dHandle->bytesAllocated += bytes;

    return mem;
  }

  memory device::malloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::properties &props) {

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

    void *ptr = umalloc(bytes, occa::memory(), props);
    if (src) {
      ::memcpy(ptr, src, bytes);
    }
    return ptr;
  }

  void* device::umalloc(const dim_t bytes,
                        const occa::memory src,
                        const occa::properties &props) {
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
    void *ptr = mem.mHandle->uvaPtr;
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
  //====================================

  //---[ stream ]-----------------------
  stream::stream() :
    dHandle(NULL),
    handle(NULL) {}

  stream::stream(device_v *dHandle_,
                 stream_t handle_) :
    dHandle(dHandle_),
    handle(handle_) {}

  stream::stream(const stream &s) :
    dHandle(s.dHandle),
    handle(s.handle) {}

  stream& stream::operator = (const stream &s) {
    dHandle = s.dHandle;
    handle  = s.handle;

    return *this;
  }

  void* stream::getHandle(const occa::properties &props) {
    return handle;
  }

  void stream::free() {
    if (dHandle != NULL) {
      device(dHandle).freeStream(*this);
    }
  }

  streamTag::streamTag() :
    tagTime(0),
    handle(NULL) {}
  streamTag::streamTag(const double tagTime_,
                       void *handle_) :
    tagTime(tagTime_),
    handle(handle_) {}
  //====================================
}
