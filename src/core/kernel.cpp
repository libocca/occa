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

#include <occa/core/kernel.hpp>
#include <occa/core/device.hpp>
#include <occa/core/memory.hpp>
#include <occa/tools/uva.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

namespace occa {
  //---[ modeKernel_t ]-----------------
  modeKernel_t::modeKernel_t(modeDevice_t *modeDevice_,
                             const std::string &name_,
                             const std::string &sourceFilename_,
                             const occa::properties &properties_) :
    modeDevice(modeDevice_),
    name(name_),
    sourceFilename(sourceFilename_),
    properties(properties_) {
    modeDevice->addKernelRef(this);
  }

  modeKernel_t::~modeKernel_t() {
    // NULL all wrappers
    while (kernelRing.head) {
      kernel *mem = (kernel*) kernelRing.head;
      kernelRing.removeRef(mem);
      mem->modeKernel = NULL;
    }
    // Remove ref from device
    if (modeDevice) {
      modeDevice->removeKernelRef(this);
    }
  }

  void modeKernel_t::dontUseRefs() {
    kernelRing.dontUseRefs();
  }

  void modeKernel_t::addKernelRef(kernel *ker) {
    kernelRing.addRef(ker);
  }

  void modeKernel_t::removeKernelRef(kernel *ker) {
    kernelRing.removeRef(ker);
  }

  bool modeKernel_t::needsFree() const {
    return kernelRing.needsFree();
  }

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
    removeKernelRef();
  }

  void kernel::assertInitialized() const {
    OCCA_ERROR("Kernel not initialized or has been freed",
               modeKernel != NULL);
  }

  void kernel::setModeKernel(modeKernel_t *modeKernel_) {
    if (modeKernel != modeKernel_) {
      removeKernelRef();
      modeKernel = modeKernel_;
      if (modeKernel) {
        modeKernel->addKernelRef(this);
      }
    }
  }

  void kernel::removeKernelRef() {
    if (!modeKernel) {
      return;
    }
    modeKernel->removeKernelRef(this);
    if (modeKernel->modeKernel_t::needsFree()) {
      free();
    }
  }

  void kernel::dontUseRefs() {
    if (modeKernel) {
      modeKernel->modeKernel_t::dontUseRefs();
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

  modeKernel_t* kernel::getModeKernel() const {
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

  void kernel::pushArg(const kernelArg &arg) {
    assertInitialized();

    OCCA_ERROR("Kernels can have at most [" << OCCA_MAX_ARGS << "] arguments",
               (modeKernel->argumentCount() + 1) < OCCA_MAX_ARGS);

    modeKernel->arguments.push_back(arg);
  }

  void kernel::clearArgs() {
    if (modeKernel) {
      modeKernel->arguments.clear();
    }
  }

  void kernel::run() const {
    assertInitialized();

    const int argc = (int) modeKernel->arguments.size();
    for (int i = 0; i < argc; ++i) {
      const bool argIsConst = modeKernel->metadata.argIsConst(i);
      modeKernel->arguments[i].setupForKernelCall(argIsConst);
    }

    modeKernel->run();
  }

#include "kernelOperators.cpp"

  void kernel::free() {
    // ~modeKernel_t NULLs all wrappers
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


  //---[ Kernel Properties ]------------
  // defines       : Object
  // includes      : Array
  // header        : Array
  // include_paths : Array

  hash_t kernelHeaderHash(const occa::properties &props) {
    return (
      occa::hash(props["defines"])
      ^ props["includes"]
      ^ props["header"]
    );
  }

  std::string assembleKernelHeader(const occa::properties &props) {
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
}
