/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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
#include "occa/parser/parser.hpp"

namespace occa {
  //---[ device_v ]---------------------
  device_v::device_v(const occa::properties &properties_) {
    mode = properties_["mode"].string();
    properties = properties_;

    currentStream = NULL;
    bytesAllocated = 0;
  }

  device_v::~device_v() {}
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

  bool device::isInitialized() {
    return (dHandle != NULL);
  }

  device_v* device::getDHandle() const {
    return dHandle;
  }

  void device::setup(const occa::properties &props) {
    setDHandle(occa::newModeDevice(props));

    stream newStream = createStream();
    dHandle->currentStream = newStream.handle;
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

  const occa::json& device::kernelProperties() const {
    occa::json &ret = dHandle->properties["kernel"];
    ret["mode"] = mode();
    return ret;
  }

  const occa::json& device::memoryProperties() const {
    occa::json &ret = dHandle->properties["memory"];
    ret["mode"] = mode();
    return ret;
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
                              const hash_t &hash,
                              const occa::properties &kernelProps,
                              const kernelMetadataMap_t &metadataMap) const {
    occa::properties infoProps;
    infoProps["device"]          = dHandle->properties;
    infoProps["device/hash"]     = dHandle->hash().toFullString();
    infoProps["kernel/props"]    = kernelProps;
    infoProps["kernel/hash"]     = hash.toFullString();

    json &metadataJson = infoProps["kernel/metadata"].asArray();
    cKernelMetadataMapIterator kIt = metadataMap.begin();
    while (kIt != metadataMap.end()) {
      metadataJson += (kIt->second).toJson();
      ++kIt;
    }

    io::storeCacheInfo(filename, hash, infoProps);
  }


  std::string device::cacheHash(const hash_t &hash,
                                const std::string &kernelName) const {
    std::string str = hash.toFullString();
    str += '-';
    return str + kernelName;
  }

  void device::loadKernels(const std::string &library) {
    std::string devHash = dHandle->hash().toFullString();
    strVector_t dirs = io::directories("occa://" + library);
    const int dirCount = (int) dirs.size();

    const bool isVerbose = settings().get("verboseCompilation", true);
    settings()["verboseCompilation"] = false;
    int kernelsLoaded = 0;

    for (int d = 0; d < dirCount; ++d) {
      const std::string infoFile = dirs[d] + kc::infoFile;

      if (!sys::fileExists(infoFile)) {
        continue;
      }

      json info = json::loads(infoFile)["info"];
      if (info["device/hash"].string() != devHash) {
        continue;
      }

      ++kernelsLoaded;

      json &kInfo = info["kernel"];
      hash_t hash = hash_t::fromString(kInfo["hash"].string());
      jsonArray_t metadataArray = kInfo["metadata"].array();
      occa::properties kernelProps = kInfo["props"];
      const std::string sourceFilename = dirs[d] + kc::parsedSourceFile;

      const int kernels = metadataArray.size();
      for (int k = 0; k < kernels; ++k) {
        kernelMetadata metadata = kernelMetadata::fromJson(metadataArray[k]);
        kernel &ker = dHandle->cachedKernels[cacheHash(hash, metadata.name)];
        if (!ker.isInitialized()) {
          ker = buildKernel(sourceFilename,
                            hash,
                            kernelProps,
                            metadata);
        }
      }
    }

    settings()["verboseCompilation"] = isVerbose;
    if (isVerbose && kernelsLoaded) {
      std::cout << "Loaded " << kernelsLoaded;
      if (library.size()) {
        std::cout << " ["<< library << "]";
      } else {
        std::cout << " cached";
      }
      std::cout << ((kernelsLoaded == 1) ? " kernel\n" : " kernels\n");
    }
  }

  kernel device::buildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props) const {

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    hash_t hash = (dHandle->hash()
                   ^ occa::hash(allProps)
                   ^ hashFile(filename));

    kernel &ker = dHandle->cachedKernels[cacheHash(hash, kernelName)];

    if (!ker.isInitialized()) {
      const std::string realFilename = io::filename(filename);
      const std::string hashDir = io::hashDir(realFilename, hash);
      std::string sourceFilename = realFilename;

      kernelMetadata metadata;
      if (allProps.get("OKL", true)) {
        sourceFilename = hashDir + kc::parsedSourceFile;

        kernelMetadataMap_t metadataMap = io::parseFile(realFilename,
                                                        sourceFilename,
                                                        allProps);

        kernelMetadataMapIterator kIt = metadataMap.find(kernelName);
        OCCA_ERROR("Could not find kernel ["
                   << kernelName << "] in file ["
                   << io::shortname(filename) << "]",
                   kIt != metadataMap.end());

        metadata = kIt->second;

        storeCacheInfo(filename, hash, allProps, metadataMap);
      } else {
        metadata.name = kernelName;
      }

      ker = buildKernel(sourceFilename,
                        hash,
                        allProps,
                        metadata);
    }

    return ker;
  }

  occa::kernel device::buildKernel(const std::string &filename,
                                   const hash_t &hash,
                                   const occa::properties &kernelProps,
                                   const kernelMetadata &metadata) const {

    if (metadata.nestedKernels == 0) {
      return kernel(dHandle->buildKernel(filename,
                                         metadata.name,
                                         hash,
                                         kernelProps));
    }

    // Create launch kernel
    occa::properties launchProps = host().kernelProperties();
    launchProps["defines/OCCA_LAUNCH_KERNEL"] = 1;

    kernel_v *launchKHandle = host().getDHandle()->buildKernel(filename,
                                                               metadata.name,
                                                               hash,
                                                               launchProps);

    // Load nested kernels
    if (metadata.nestedKernels) {
      const bool vc_f = settings()["verboseCompilation"];

      for (int ki = 0; ki < metadata.nestedKernels; ++ki) {
        kernelMetadata sMetadata    = metadata.getNestedKernelMetadata(ki);
        const std::string &sKerName = sMetadata.name;

        kernel sKer(dHandle->buildKernel(filename, sKerName, hash, kernelProps));
        sKer.kHandle->metadata = sMetadata;

        launchKHandle->nestedKernels.push_back(sKer);

        // Only show compilation the first time
        if (ki == 0) {
          settings()["verboseCompilation"] = false;
        }
      }
      settings()["verboseCompilation"] = vc_f;
    }
    return occa::kernel(launchKHandle);
  }

  kernel device::buildKernelFromString(const std::string &content,
                                       const std::string &kernelName,
                                       const occa::properties &props) const {

    occa::properties allProps = props + kernelProperties();
    allProps["mode"] = mode();

    hash_t hash = (dHandle->hash()
                   ^ occa::hash(allProps)
                   ^ occa::hash(content));

    const std::string hashDir = io::hashDir(hash);
    const std::string hashTag = "occa-device";

    std::string stringSourceFile = hashDir;
    stringSourceFile += "stringSource.okl";

    if (!io::haveHash(hash, hashTag)) {
      io::waitForHash(hash, hashTag);
    } else {
      if (!sys::fileExists(stringSourceFile)) {
        io::write(stringSourceFile, content);
      }
      io::releaseHash(hash, hashTag);
    }

    return buildKernel(stringSourceFile,
                       kernelName,
                       props);
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &kernelName,
                                       const occa::properties &props) const {

    return kernel(dHandle->buildKernelFromBinary(filename, kernelName, props));
  }
  //  |=================================

  //  |---[ Memory ]--------------------
  memory device::malloc(const dim_t bytes,
                        const void *src,
                        const occa::properties &props) {

    OCCA_ERROR("Trying to allocate "
               << (bytes ? "negative" : "zero") << " bytes (" << bytes << ")",
               bytes > 0);

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
    if (src.size()) {
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

    OCCA_ERROR("Trying to allocate "
               << (bytes ? "negative" : "zero") << " bytes (" << bytes << ")",
               bytes > 0);

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
    return device.getDHandle()->hash();
  }
  //====================================

  //---[ stream ]-----------------------
  stream::stream() :
    dHandle(NULL),
    handle(NULL) {}

  stream::stream(device_v *dHandle_, stream_t handle_) :
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
  //====================================
}
