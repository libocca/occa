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

#include <occa/kernel.hpp>
#include <occa/device.hpp>
#include <occa/memory.hpp>
#include <occa/uva.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

namespace occa {
  //---[ KernelArg ]--------------------
  kernelArgData::kernelArgData() :
    modeDevice(NULL),
    modeMemory(NULL),
    size(0),
    info(kArgInfo::none) {
    ::memset(&data, 0, sizeof(data));
  }

  kernelArgData::kernelArgData(const kernelArgData &k) {
    *this = k;
  }

  kernelArgData& kernelArgData::operator = (const kernelArgData &k) {
    modeDevice = k.modeDevice;
    modeMemory = k.modeMemory;

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

  kernelArg::kernelArg(const kernelArgData &arg) {
    args.push_back(arg);
  }

  kernelArg::kernelArg(const kernelArg &k) :
    args(k.args) {}

  int kernelArg::size() {
    return (int) args.size();
  }

  kernelArgData& kernelArg::operator [] (const int index) {
    return args[index];
  }

  kernelArg& kernelArg::operator = (const kernelArg &k) {
    args = k.args;
    return *this;
  }

  kernelArg::kernelArg(const uint8_t arg) {
    kernelArgData kArg;
    kArg.data.uint8_ = arg;
    kArg.size        = sizeof(uint8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint16_t arg) {
    kernelArgData kArg;
    kArg.data.uint16_ = arg;
    kArg.size         = sizeof(uint16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint32_t arg) {
    kernelArgData kArg;
    kArg.data.uint32_ = arg;
    kArg.size         = sizeof(uint32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint64_t arg) {
    kernelArgData kArg;
    kArg.data.uint64_ = arg;
    kArg.size         = sizeof(uint64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int8_t arg) {
    kernelArgData kArg;
    kArg.data.int8_ = arg;
    kArg.size       = sizeof(int8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int16_t arg) {
    kernelArgData kArg;
    kArg.data.int16_ = arg;
    kArg.size        = sizeof(int16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int32_t arg) {
    kernelArgData kArg;
    kArg.data.int32_ = arg;
    kArg.size        = sizeof(int32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int64_t arg) {
    kernelArgData kArg;
    kArg.data.int64_ = arg;
    kArg.size        = sizeof(int64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const float arg) {
    kernelArgData kArg;
    kArg.data.float_ = arg;
    kArg.size         = sizeof(float);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const double arg) {
    kernelArgData kArg;
    kArg.data.double_ = arg;
    kArg.size         = sizeof(double);
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

    modeMemory_t *modeMemory = NULL;

    if (argIsUva) {
      modeMemory = (modeMemory_t*) arg;
    } else if (lookAtUva) {
      ptrRangeMap::iterator it = uvaMap.find(arg);
      if (it != uvaMap.end()) {
        modeMemory = it->second;
      }
    }

    if (modeMemory) {
      add(modeMemory->makeKernelArg());
    } else {
      kernelArgData kArg;
      kArg.data.void_ = arg;
      kArg.size       = bytes;
      kArg.info       = kArgInfo::usePointer;
      args.push_back(kArg);
    }
  }

  void kernelArg::setupForKernelCall(const bool isConst) const {
    const int argCount = (int) args.size();
    for (int i = 0; i < argCount; ++i) {
      occa::modeMemory_t *modeMemory = args[i].modeMemory;

      if (!modeMemory              ||
          !modeMemory->isManaged() ||
          !modeMemory->modeDevice->hasSeparateMemorySpace()) {
        continue;
      }
      if (!modeMemory->inDevice()) {
        modeMemory->copyFrom(modeMemory->uvaPtr, modeMemory->size);
        modeMemory->memInfo |= uvaFlag::inDevice;
      }
      if (!isConst && !modeMemory->isStale()) {
        uvaStaleMemory.push_back(modeMemory);
        modeMemory->memInfo |= uvaFlag::isStale;
      }
    }
  }

  int kernelArg::argumentCount(const std::vector<kernelArg> &arguments) {
    const int kArgCount = (int) arguments.size();
    int argc = 0;
    for (int i = 0; i < kArgCount; ++i) {
      argc += arguments[i].args.size();
    }
    return argc;
  }
  //====================================


  //---[ Kernel Properties ]------------
  // defines       : Object
  // includes      : Array
  // header        : Array
  // include_paths : Array
  std::string assembleHeader(const occa::properties &props) {
    std::string header;

    // Add defines
    const jsonObject &defines = props["defines"].object();
    jsonObject::const_iterator it = defines.begin();
    while (it != defines.end()) {
      header += "#define ";
      header += ' ';
      header += it->first;
      header += ' ';
      header += (std::string) it->second;
      header += '\n';
      ++it;
    }

    // Add includes
    const jsonArray &includes = props["includes"].array();
    const int includeCount = (int) includes.size();
    for (int i = 0; i < includeCount; ++i) {
      if (includes[i].isString()) {
        header += "#include \"";
        header += (std::string) includes[i];
        header += "\"\n";
      }
    }

    // Add header
    const jsonArray &lines = props["header"].array();
    const int lineCount = (int) lines.size();
    for (int i = 0; i < lineCount; ++i) {
      if (includes[i].isString()) {
        header += (std::string) lines[i];
        header += "\n";
      }
    }

    return header;
  }
  //====================================


  //---[ modeKernel_t ]---------------------
  modeKernel_t::modeKernel_t(modeDevice_t *modeDevice_,
                             const std::string &name_,
                             const std::string &sourceFilename_,
                             const occa::properties &properties_) :
    modeDevice(modeDevice_),
    name(name_),
    sourceFilename(sourceFilename_),
    properties(properties_) {
    modeDevice->addRef();
  }

  modeKernel_t::~modeKernel_t() {}

  kernelArg* modeKernel_t::argumentsPtr() {
    return &(arguments[0]);
  }

  int modeKernel_t::argumentCount() {
    return (int) arguments.size();
  }
  //====================================

  //---[ kernel ]-----------------------
  kernel::kernel() :
    modeKernel(NULL) {}

  kernel::kernel(modeKernel_t *modeKernel_) :
    modeKernel(NULL) {
    setModeKernel(modeKernel_);
  }

  kernel::kernel(const kernel &k) :
    modeKernel(NULL) {
    setModeKernel(k.modeKernel);
  }

  kernel& kernel::operator = (const kernel &k) {
    setModeKernel(k.modeKernel);
    return *this;
  }

  kernel& kernel::operator = (modeKernel_t *modeKernel_) {
    setModeKernel(modeKernel_);
    return *this;
  }

  kernel::~kernel() {
    removeRef();
  }

  void kernel::setModeKernel(modeKernel_t *modeKernel_) {
    if (modeKernel != modeKernel_) {
      removeRef();
      modeKernel = modeKernel_;
      if (modeKernel) {
        modeKernel->addRef();
      }
    }
  }

  void kernel::removeRef() {
    if (modeKernel && !modeKernel->removeRef()) {
      free();
      modeKernel = NULL;
    }
  }

  void kernel::dontUseRefs() {
    if (modeKernel) {
      modeKernel->dontUseRefs();
    }
  }

  bool kernel::isInitialized() {
    return (modeKernel != NULL);
  }

  const std::string& kernel::mode() const {
    static const std::string noMode = "No Mode";
    return (modeKernel
            ? modeKernel->modeDevice->mode
            : noMode);
  }

  const occa::properties& kernel::properties() const {
    static const occa::properties noProperties;
    return (modeKernel
            ? modeKernel->properties
            : noProperties);
  }

  modeKernel_t* kernel::getModeKernel() {
    return modeKernel;
  }

  occa::device kernel::getDevice() {
    return occa::device(modeKernel
                        ? modeKernel->modeDevice
                        : NULL);
  }

  bool kernel::operator == (const occa::kernel &other) const {
    return (modeKernel == other.modeKernel);
  }

  bool kernel::operator != (const occa::kernel &other) const {
    return (modeKernel != other.modeKernel);
  }

  const std::string& kernel::name() {
    static const std::string noName = "";
    return (modeKernel
            ? modeKernel->name
            : noName);
  }

  const std::string& kernel::sourceFilename() {
    static const std::string noSourceFilename = "";
    return (modeKernel
            ? modeKernel->sourceFilename
            : noSourceFilename);
  }

  const std::string& kernel::binaryFilename() {
    static const std::string noBinaryFilename = "";
    return (modeKernel
            ? modeKernel->binaryFilename
            : noBinaryFilename);
  }

  void kernel::setRunDims(occa::dim outerDims, occa::dim innerDims) {
    if (modeKernel) {
      modeKernel->innerDims = innerDims;
      modeKernel->outerDims = outerDims;
    }
  }

  int kernel::maxDims() {
    return (modeKernel
            ? modeKernel->maxDims()
            : -1);
  }

  dim kernel::maxOuterDims() {
    return (modeKernel
            ? modeKernel->maxOuterDims()
            : dim(-1, -1, -1));
  }

  dim kernel::maxInnerDims() {
    return (modeKernel
            ? modeKernel->maxInnerDims()
            : dim(-1, -1, -1));
  }

  void kernel::addArgument(const int argPos, const kernelArg &arg) {
    OCCA_ERROR("Kernel not initialized",
               modeKernel != NULL);

    if (modeKernel->argumentCount() <= argPos) {
      OCCA_ERROR("Kernels can only have at most [" << OCCA_MAX_ARGS << "] arguments,"
                 << " [" << argPos << "] arguments were set",
                 argPos < OCCA_MAX_ARGS);

      modeKernel->arguments.resize(argPos + 1);
    }

    modeKernel->arguments[argPos] = arg;
  }

  void kernel::run() const {
    OCCA_ERROR("Kernel not initialized",
               modeKernel != NULL);

    const int argc = (int) modeKernel->arguments.size();
    for (int i = 0; i < argc; ++i) {
      const bool argIsConst = modeKernel->metadata.argIsConst(i);
      modeKernel->arguments[i].setupForKernelCall(argIsConst);
    }

    modeKernel->run();
  }

  void kernel::clearArgumentList() {
    if (modeKernel) {
      modeKernel->arguments.clear();
    }
  }

#include "kernelOperators.cpp"

  void kernel::free() {
    if (modeKernel == NULL) {
      return;
    }
    modeDevice_t *modeDevice = modeKernel->modeDevice;
    // Remove kernel from cache map
    modeDevice->removeCachedKernel(modeKernel);
    modeDevice->removeRef();

    modeKernel->free();
    delete modeKernel;
    modeKernel = NULL;
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
