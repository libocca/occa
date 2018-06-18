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
    static std::string noMode = "No Mode";
    if (dHandle) {
      return dHandle->mode;
    }
    return noMode;
  }

  occa::properties& device::properties() {
    return dHandle->properties;
  }

  const occa::properties& device::properties() const {
    return dHandle->properties;
  }

  occa::properties& device::kernelProperties() {
    return (occa::properties&) dHandle->properties["kernel"];
  }

  const occa::properties& device::kernelProperties() const {
    return (const occa::properties&) dHandle->properties["kernel"];
  }

  occa::properties& device::memoryProperties() {
    return (occa::properties&) dHandle->properties["memory"];
  }

  const occa::properties& device::memoryProperties() const {
    return (const occa::properties&) dHandle->properties["memory"];
  }

  hash_t device::hash() const {
    if (dHandle) {
      return dHandle->versionedHash();
    }
    return hash_t();
  }

  udim_t device::memorySize() const {
    if (dHandle) {
      return dHandle->memorySize();
    }
    return 0;
  }

  udim_t device::memoryAllocated() const {
    if (dHandle) {
      return dHandle->bytesAllocated;
    }
    return 0;
  }

  void device::finish() {
    if (!dHandle) {
      return;
    }
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
    if (dHandle) {
      return dHandle->hasSeparateMemorySpace();
    }
    return false;
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
  kernel device::buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props) const {

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    hash_t kernelHash = (hash()
                         ^ occa::hash(allProps)
                         ^ hashFile(filename));

    // Check cache first
    kernel &cachedKernel = dHandle->getCachedKernel(kernelHash,
                                                    kernelName);
    if (cachedKernel.isInitialized()) {
      return cachedKernel;
    }

    const std::string realFilename = io::filename(filename);
    const std::string hashDir = io::hashDir(realFilename, kernelHash);

    cachedKernel = dHandle->buildKernel(realFilename,
                                        kernelName,
                                        kernelHash,
                                        allProps);
    return cachedKernel;
  }

  kernel device::buildKernelFromString(const std::string &content,
                                       const std::string &kernelName,
                                       const occa::properties &props) const {

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    hash_t kernelHash = (hash()
                         ^ occa::hash(allProps)
                         ^ occa::hash(content));

    io::lock_t lock(kernelHash, "occa-device");
    std::string stringSourceFile = io::hashDir(kernelHash);
    stringSourceFile += "stringSource.okl";

    if (lock.isMine()) {
      if (!sys::fileExists(stringSourceFile)) {
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

    return kernel(dHandle->buildKernelFromBinary(filename,
                                                 kernelName,
                                                 props));
  }

  void device::loadKernels(const std::string &library) {
    // TODO 1.1: Load kernels
#if 0
    std::string devHash = hash().toFullString();
    strVector dirs = io::directories("occa://" + library);
    const int dirCount = (int) dirs.size();
    int kernelsLoaded = 0;

    for (int d = 0; d < dirCount; ++d) {
      const std::string buildFile = dirs[d] + kc::buildFile;

      if (!sys::fileExists(buildFile)) {
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
