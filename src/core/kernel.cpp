#include <occa/core/device.hpp>
#include <occa/core/kernel.hpp>
#include <occa/core/memory.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/core/device.hpp>
#include <occa/internal/core/kernel.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/utils/uva.hpp>

namespace occa {
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

  hash_t kernel::hash() {
    return (modeKernel
            ? modeKernel->hash
            : hash_t());
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
            : dim(occa::UDIM_DEFAULT, occa::UDIM_DEFAULT, occa::UDIM_DEFAULT));
  }

  dim kernel::maxInnerDims() {
    return (modeKernel
            ? modeKernel->maxInnerDims()
            : dim(occa::UDIM_DEFAULT, occa::UDIM_DEFAULT, occa::UDIM_DEFAULT));
  }

  void kernel::pushArg(const kernelArg &arg) {
    assertInitialized();
    modeKernel->pushArgument(arg);
  }

  void kernel::clearArgs() {
    if (modeKernel) {
      modeKernel->arguments.clear();
    }
  }

  void kernel::run() const {
    assertInitialized();

    if (modeKernel->isNoop()) {
      return;
    }

    modeKernel->setupRun();
    modeKernel->run();
  }

  void kernel::run(std::initializer_list<kernelArg> args) const {
    kernel &self = *(const_cast<kernel*>(this));

    self.clearArgs();
    for (const kernelArg &arg : args) {
      self.pushArg(arg);
    }
    self.run();
  }

#include "kernelOperators.cpp_codegen"

  void kernel::free() {
    // ~modeKernel_t NULLs all wrappers
    delete modeKernel;
    modeKernel = NULL;
  }
  //====================================


  //---[ Kernel Properties ]------------
  // Properties:
  //   defines       : Object
  //   includes      : Array
  //   headers       : Array
  //   include_paths : Array

  hash_t kernelHeaderHash(const occa::properties &props) {
    return (
      occa::hash(props["defines"])
      ^ props["includes"]
      ^ props["headers"]
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
    const jsonArray &lines = props["headers"].array();
    const int lineCount = (int) lines.size();
    for (int i = 0; i < lineCount; ++i) {
      if (lines[i].isString()) {
        header += (std::string) lines[i];
        header += "\n";
      }
    }

    return header;
  }
  //====================================
}
