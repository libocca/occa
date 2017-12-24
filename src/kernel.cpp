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
  kernelArgData::kernelArgData() {
    dHandle = NULL;
    mHandle = NULL;

    ::memset(&data, 0, sizeof(data));
    size = 0;
    info = kArgInfo::none;
  }

  kernelArgData::kernelArgData(const kernelArgData &k) {
    *this = k;
  }

  kernelArgData& kernelArgData::operator = (const kernelArgData &k) {
    dHandle = k.dHandle;
    mHandle = k.mHandle;

    data = k.data;
    size = k.size;
    info = k.info;

    return *this;
  }

  kernelArgData::~kernelArgData() {}

  void* kernelArgData::ptr() const {
    return ((info & kArgInfo::usePointer) ? data.void_ : (void*) &data);
  }

  kernelArg::kernelArg() {}
  kernelArg::~kernelArg() {}

  kernelArg::kernelArg(kernelArgData &arg) {
    args.push_back(arg);
  }

  kernelArg::kernelArg(const kernelArg &k) :
    args(k.args) {}

  kernelArg& kernelArg::operator = (const kernelArg &k) {
    args = k.args;
    return *this;
  }

  kernelArg::kernelArg(const uint8_t arg) {
    kernelArgData kArg;
    kArg.data.uint8_ = arg; kArg.size = sizeof(uint8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint16_t arg) {
    kernelArgData kArg;
    kArg.data.uint16_ = arg; kArg.size = sizeof(uint16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint32_t arg) {
    kernelArgData kArg;
    kArg.data.uint32_ = arg; kArg.size = sizeof(uint32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint64_t arg) {
    kernelArgData kArg;
    kArg.data.uint64_ = arg; kArg.size = sizeof(uint64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int8_t arg) {
    kernelArgData kArg;
    kArg.data.int8_ = arg; kArg.size = sizeof(int8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int16_t arg) {
    kernelArgData kArg;
    kArg.data.int16_ = arg; kArg.size = sizeof(int16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int32_t arg) {
    kernelArgData kArg;
    kArg.data.int32_ = arg; kArg.size = sizeof(int32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int64_t arg) {
    kernelArgData kArg;
    kArg.data.int64_ = arg; kArg.size = sizeof(int64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const float arg) {
    kernelArgData kArg;
    kArg.data.float_ = arg; kArg.size = sizeof(float);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const double arg) {
    kernelArgData kArg;
    kArg.data.double_ = arg; kArg.size = sizeof(double);
    args.push_back(kArg);
  }

  void kernelArg::add(const kernelArg &arg) {
    const int newArgs = (int) arg.args.size();
    for (int i = 0; i < newArgs; ++i) {
      args.push_back(arg.args[i]);
    }
  }

  void kernelArg::add(void *arg,
                      bool lookAtUva, bool argIsUva) {
    add(arg, sizeof(void*), lookAtUva, argIsUva);
  }

  void kernelArg::add(void *arg, size_t bytes,
                      bool lookAtUva, bool argIsUva) {

    memory_v *mHandle = NULL;

    if (argIsUva) {
      mHandle = (memory_v*) arg;
    } else if (lookAtUva) {
      ptrRangeMap::iterator it = uvaMap.find(arg);
      if (it != uvaMap.end()) {
        mHandle = it->second;
      }
    }

    if (mHandle) {
      add(mHandle->makeKernelArg());
    } else {
      kernelArgData kArg;
      kArg.info       = kArgInfo::usePointer;
      kArg.size       = bytes;
      kArg.data.void_ = arg;
      args.push_back(kArg);
    }
  }

  void kernelArg::setupForKernelCall(const bool isConst) const {
    const int argCount = (int) args.size();
    for (int i = 0; i < argCount; ++i) {
      occa::memory_v *mHandle = args[i].mHandle;

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
  }

  int kernelArg::argumentCount(const int kArgc, const kernelArg *kArgs) {
    int argc = 0;
    for (int i = 0; i < kArgc; ++i) {
      argc += kArgs[i].args.size();
    }
    return argc;
  }
  //====================================


  //---[ Kernel Properties ]------------
  std::string assembleHeader(const occa::properties &props) {
    const jsonArray &lines = props["headers"].array();
    const int lineCount = (int) lines.size();

    const jsonObject &defines = props["defines"].object();
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

    inner = occa::dim();
    outer = occa::dim();
  }

  kernel_v::~kernel_v() {}

  // This should only be called in the very first reference
  void kernel_v::setDHandle(device_v *dHandle_) {
    dHandle = dHandle_;
    dHandle->addRef();
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

  std::string kernel_v::getLaunchSourceFilename(const std::string &filename,
                                                const hash_t &hash) {
    return io::hashDir(filename, hash) + kc::launchSourceFile;
  }

  std::string kernel_v::getLaunchBinaryFilename(const std::string &filename,
                                                const hash_t &hash) {
    return io::hashDir(filename, hash) + binaryName(kc::launchBinaryFile);
  }

  std::string kernel_v::getSourceFilename(const std::string &filename,
                                          const hash_t &hash) {
    return io::hashDir(filename, hash) + kc::sourceFile;
  }

  std::string kernel_v::getBinaryFilename(const std::string &filename,
                                          const hash_t &hash) {
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
      device::removeDHandleRefFrom(kHandle->dHandle);
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

  const std::string& kernel::mode() {
    return kHandle->dHandle->mode;
  }

  occa::properties& kernel::properties() {
    return kHandle->properties;
  }

  kernel_v* kernel::getKHandle() {
    return kHandle;
  }

  occa::device kernel::getDevice() {
    return occa::device(kHandle->dHandle);
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

  void kernel::setRunDims(occa::dim outer, occa::dim inner) {
    kHandle->inner = inner;
    kHandle->outer = outer;
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
    const bool hasNestedKernels = kHandle->nestedKernelCount();
    if (hasNestedKernels) {
      kHandle->arguments.insert(kHandle->arguments.begin(),
                                kHandle->nestedKernelsPtr());
    }

    kHandle->runFromArguments(kHandle->argumentCount(),
                              kHandle->argumentsPtr());

    // Remove nestedKernels
    if (hasNestedKernels) {
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

  //---[ kernelBuilder ]----------------
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

  occa::kernel kernelBuilder::build(occa::device device) {
    return build(device, hash(device), props_);
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const occa::properties &props) {
    occa::properties kernelProps = props_;
    kernelProps += props;
    return build(device,
                 hash(device) ^ hash(kernelProps),
                 kernelProps);
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const hash_t &hash) {
    return build(device, hash, props_);
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const hash_t &hash,
                                    const occa::properties &props) {
    occa::kernel &k = kernelMap[hash];
    if (!k.isInitialized()) {
      if (buildingFromFile) {
        k = device.buildKernel(source_, function_, props);
      } else {
        k = device.buildKernelFromString(source_, function_, props);
      }
    }
    return k;
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
