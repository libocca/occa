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

#include "occa/kernel.hpp"
#include "occa/device.hpp"
#include "occa/memory.hpp"
#include "occa/uva.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  //---[ KernelArg ]--------------------
  kernelArg_t::kernelArg_t() {
    dHandle = NULL;
    mHandle = NULL;

    ::memset(&data, 0, sizeof(data));
    size = 0;
    info = kArgInfo::none;
  }

  kernelArg_t::kernelArg_t(const kernelArg_t &k) {
    *this = k;
  }

  kernelArg_t& kernelArg_t::operator = (const kernelArg_t &k) {
    dHandle = k.dHandle;
    mHandle = k.mHandle;

    ::memcpy(&data, &(k.data), sizeof(data));
    size = k.size;
    info = k.info;

    return *this;
  }

  kernelArg_t::~kernelArg_t() {}

  void* kernelArg_t::ptr() const {
    return ((info & kArgInfo::usePointer) ? data.void_ : (void*) &data);
  }

  kernelArg::kernelArg() {}
  kernelArg::~kernelArg() {}

  kernelArg::kernelArg(kernelArg_t &arg_) :
    arg(arg_) {}

  kernelArg::kernelArg(const kernelArg &k) :
    arg(k.arg),
    extraArgs(k.extraArgs) {}

  kernelArg& kernelArg::operator = (const kernelArg &k) {
    arg = k.arg;
    extraArgs = k.extraArgs;
    return *this;
  }

  void kernelArg::setupFrom(void *arg_,
                            bool lookAtUva, bool argIsUva) {

    setupFrom(arg_, sizeof(void*), lookAtUva, argIsUva);
  }

  void kernelArg::setupFrom(void *arg_, size_t bytes,
                            bool lookAtUva, bool argIsUva) {

    memory_v *mHandle = NULL;
    if (argIsUva) {
      mHandle = (memory_v*) arg_;
    } else if (lookAtUva) {
      ptrRangeMap_t::iterator it = uvaMap.find(arg_);
      if (it != uvaMap.end()) {
        mHandle = it->second;
      }
    }

    arg.info = kArgInfo::usePointer;
    arg.size = bytes;

    if (mHandle) {
      arg.mHandle = mHandle;
      arg.dHandle = mHandle->dHandle;

      arg.data.void_ = mHandle->handle;
    } else {
      arg.data.void_ = arg_;
    }
  }

  template <>
  kernelArg::kernelArg(const uint8_t &arg_) {
    arg.data.uint8_ = arg_; arg.size = sizeof(uint8_t);
  }

  template <>
  kernelArg::kernelArg(const uint16_t &arg_) {
    arg.data.uint16_ = arg_; arg.size = sizeof(uint16_t);
  }

  template <>
  kernelArg::kernelArg(const uint32_t &arg_) {
    arg.data.uint32_ = arg_; arg.size = sizeof(uint32_t);
  }

  template <>
  kernelArg::kernelArg(const uint64_t &arg_) {
    arg.data.uint64_ = arg_; arg.size = sizeof(uint64_t);
  }

  template <>
  kernelArg::kernelArg(const int8_t &arg_) {
    arg.data.int8_ = arg_; arg.size = sizeof(int8_t);
  }

  template <>
  kernelArg::kernelArg(const int16_t &arg_) {
    arg.data.int16_ = arg_; arg.size = sizeof(int16_t);
  }

  template <>
  kernelArg::kernelArg(const int32_t &arg_) {
    arg.data.int32_ = arg_; arg.size = sizeof(int32_t);
  }

  template <>
  kernelArg::kernelArg(const int64_t &arg_) {
    arg.data.int64_ = arg_; arg.size = sizeof(int64_t);
  }

  template <>
  kernelArg::kernelArg(const float &arg_) {
    arg.data.float_ = arg_; arg.size = sizeof(float);
  }

  template <>
  kernelArg::kernelArg(const double &arg_) {
    arg.data.double_ = arg_; arg.size = sizeof(double);
  }

  occa::device kernelArg::getDevice() const {
    return occa::device(arg.dHandle);
  }

  void kernelArg::setupForKernelCall(const bool isConst) const {
    occa::memory_v *mHandle = arg.mHandle;

    if (mHandle              &&
        mHandle->isManaged() &&
        mHandle->dHandle->hasSeparateMemorySpace()) {

      if (!mHandle->inDevice()) {
        mHandle->copyFrom(mHandle->uvaPtr, mHandle->size);
        mHandle->memInfo |= uvaFlag::inDevice;
      }
      if (!isConst && !mHandle->isStale()) {
        uvaStaleMemory.push_back(mHandle);
        mHandle->memInfo |= uvaFlag::isStale;
      }
    }
  }

  int kernelArg::argumentCount(const int kArgc, const kernelArg *kArgs) {
    int argc = kArgc;
    for (int i = 0; i < kArgc; ++i) {
      argc += kArgs[i].extraArgs.size();
    }
    return argc;
  }
  //====================================


  //---[ Kernel Properties ]------------
  std::string assembleHeader(const occa::properties &props) {
    const jsonArray_t &lines = props["headers"].array();
    const int lineCount = (int) lines.size();

    const jsonObject_t &defines = props["defines"].object();
    cJsonObjectIterator it = defines.begin();

    std::string header;
    while (it != defines.end()) {
      header += "#define ";
      header += ' ';
      header += it->first;
      header += ' ';
      // Avoid the quotes wrapping the string
      if (it->second.isString()) {
        header += it->second.string();
      } else {
        header += it->second.toString();
      }
      header += '\n';
      ++it;
    }
    for (int i = 0; i < lineCount; ++i) {
      header += lines[i].toString();
      header += '\n';
    }
    return header;
  }
  //====================================


  //---[ kernel_v ]---------------------
  kernel_v::kernel_v(const occa::properties &properties_) {
    dHandle = NULL;

    properties = properties_;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);
  }

  kernel_v::~kernel_v() {}

  void kernel_v::initFrom(const kernel_v &m) {
    dHandle = m.dHandle;

    name = m.name;
    properties = m.properties;

    metadata = m.metadata;

    dims = m.dims;
    inner = m.inner;
    outer = m.outer;

    nestedKernels = m.nestedKernels;
    arguments = m.arguments;
  }

  kernel* kernel_v::nestedKernelsPtr() {
    return &(nestedKernels[0]);
  }

  int kernel_v::nestedKernelCount() {
    return (int) nestedKernels.size();
  }

  kernelArg* kernel_v::argumentsPtr() {
    return &(arguments[0]);
  }

  int kernel_v::argumentCount() {
    return (int) arguments.size();
  }

  std::string kernel_v::binaryName(const std::string &filename) {
    return filename;
  }

  std::string kernel_v::getSourceFilename(const std::string &filename, hash_t &hash) {
    return io::hashDir(filename, hash) + kc::sourceFile;
  }

  std::string kernel_v::getBinaryFilename(const std::string &filename, hash_t &hash) {
    return io::hashDir(filename, hash) + binaryName(kc::binaryFile);
  }
  //====================================

  //---[ kernel ]-----------------------
  kernel::kernel() :
    kHandle(NULL) {}

  kernel::kernel(kernel_v *kHandle_) :
    kHandle(NULL) {
    setKHandle(kHandle_);
  }

  kernel::kernel(const kernel &k) :
    kHandle(NULL) {
    setKHandle(k.kHandle);
  }

  kernel& kernel::operator = (const kernel &k) {
    setKHandle(k.kHandle);
    return *this;
  }

  kernel::~kernel() {
    removeKHandleRef();
  }

  void kernel::setKHandle(kernel_v *kHandle_) {
    if (kHandle != kHandle_) {
      removeKHandleRef();
      kHandle = kHandle_;
      kHandle->addRef();
    }
  }

  void kernel::removeKHandleRef() {
    if (kHandle && !kHandle->removeRef()) {
      free();
      delete kHandle;
      kHandle = NULL;
    }
  }

  void kernel::dontUseRefs() {
    if (kHandle) {
      kHandle->dontUseRefs();
    }
  }

  bool kernel::isInitialized() {
    return (kHandle != NULL);
  }

  void* kernel::getHandle(const occa::properties &props) {
    return kHandle->getHandle(props);
  }

  kernel_v* kernel::getKHandle() {
    return kHandle;
  }

  occa::device kernel::getDevice() {
    return occa::device(kHandle->dHandle);
  }

  const std::string& kernel::mode() {
    return device(kHandle->dHandle).mode();
  }

  const std::string& kernel::name() {
    return kHandle->name;
  }

  const std::string& kernel::sourceFilename() {
    return kHandle->sourceFilename;
  }

  const std::string& kernel::binaryFilename() {
    return kHandle->binaryFilename;
  }

  void kernel::setWorkingDims(int dims, occa::dim inner, occa::dim outer) {
    for (int i = 0; i < dims; ++i) {
      inner[i] += (inner[i] ? 0 : 1);
      outer[i] += (outer[i] ? 0 : 1);
    }

    for (int i = dims; i < 3; ++i)
      inner[i] = outer[i] = 1;

    if (kHandle->nestedKernelCount()) {
      for (int k = 0; k < kHandle->nestedKernelCount(); ++k) {
        kHandle->nestedKernels[k].setWorkingDims(dims, inner, outer);
      }
    } else {
      kHandle->dims  = dims;
      kHandle->inner = inner;
      kHandle->outer = outer;
    }
  }

  int kernel::maxDims() {
    return kHandle->maxDims();
  }

  dim kernel::maxOuterDims() {
    return kHandle->maxOuterDims();
  }

  dim kernel::maxInnerDims() {
    return kHandle->maxInnerDims();
  }

  void kernel::addArgument(const int argPos, const kernelArg &arg) {
    if (kHandle->argumentCount() <= argPos) {
      OCCA_ERROR("Kernels can only have at most [" << OCCA_MAX_ARGS << "] arguments,"
                 << " [" << argPos << "] arguments were set",
                 argPos < OCCA_MAX_ARGS);

      kHandle->arguments.reserve(argPos + 1);
    }

    kHandle->arguments.insert(kHandle->arguments.begin() + argPos, arg);
  }

  void kernel::runFromArguments() const {
    const int argc = (int) kHandle->arguments.size();
    for (int i = 0; i < argc; ++i) {
      const bool argIsConst = kHandle->metadata.argIsConst(i);
      kHandle->arguments[i].setupForKernelCall(argIsConst);
    }

    // Add nestedKernels
    if (kHandle->nestedKernelCount()) {
      kHandle->arguments.insert(kHandle->arguments.begin(),
                                kHandle->nestedKernelsPtr());
    }

    kHandle->runFromArguments(kHandle->argumentCount(),
                              kHandle->argumentsPtr());

    // Remove nestedKernels
    if (kHandle->nestedKernelCount()) {
      kHandle->arguments.erase(kHandle->arguments.begin());
    }
  }

  void kernel::clearArgumentList() {
    kHandle->arguments.clear();
  }

#include "operators/definitions.cpp"

  void kernel::free() {
    if (kHandle == NULL) {
      return;
    }
    if (kHandle->nestedKernelCount()) {
      for (int k = 0; k < kHandle->nestedKernelCount(); ++k) {
        kHandle->nestedKernels[k].free();
      }
    }
    kHandle->free();
  }
  //====================================

  //---[ kernel builder ]---------------
  kernelBuilder::kernelBuilder() {}

  kernelBuilder::kernelBuilder(const kernelBuilder &k) :
    source_(k.source_),
    function_(k.function_),
    props_(k.props_),
    kernelMap(k.kernelMap),
    buildingFromFile(k.buildingFromFile) {}

  kernelBuilder& kernelBuilder::operator = (const kernelBuilder &k) {
    source_   = k.source_;
    function_ = k.function_;
    props_    = k.props_;
    kernelMap = k.kernelMap;
    buildingFromFile = k.buildingFromFile;
    return *this;
  }

  kernelBuilder kernelBuilder::fromFile(const std::string &filename,
                                        const std::string &function,
                                        const occa::properties &props) {
    kernelBuilder builder;
    builder.source_   = filename;
    builder.function_ = function;
    builder.props_    = props;
    builder.buildingFromFile = true;
    return builder;
  }

  kernelBuilder kernelBuilder::fromString(const std::string &content,
                                          const std::string &function,
                                          const occa::properties &props) {
    kernelBuilder builder;
    builder.source_   = content;
    builder.function_ = function;
    builder.props_    = props;
    builder.buildingFromFile = false;
    return builder;
  }

  bool kernelBuilder::isInitialized() {
    return (0 < function_.size());
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const hash_t &hash) {
    occa::kernel &k = kernelMap[hash];
    if (!k.isInitialized()) {
      if (buildingFromFile) {
        k = device.buildKernel(source_, function_, props_);
      } else {
        k = device.buildKernelFromString(source_, function_, props_);
      }
    }
    return k;
  }

  occa::kernel kernelBuilder::build(occa::device device) {
    return build(device,
                 hash(device));
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const occa::properties &props) {
    return build(device,
                 hash(device) ^
                 occa::hash(props_ + props));
  }

  occa::kernel kernelBuilder::build(const int id,
                                    occa::device device,
                                    const occa::properties &props) {
    return build(device,
                 hash(device) ^ id);
  }

  occa::kernel kernelBuilder::operator [] (occa::device device) {
    return build(device,
                 hash(device));
  }

  void kernelBuilder::free() {
    hashedKernelMapIterator it = kernelMap.begin();
    while (it != kernelMap.end()) {
      it->second.free();
      ++it;
    }
  }
  //====================================
}
