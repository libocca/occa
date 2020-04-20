#include <occa/core/device.hpp>
#include <occa/core/kernel.hpp>
#include <occa/core/memory.hpp>
#include <occa/io.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/transforms/builtins/finders.hpp>
#include <occa/tools/sys.hpp>
#include <occa/tools/uva.hpp>

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
      kernel *k = (kernel*) kernelRing.head;
      kernelRing.removeRef(k);
      k->modeKernel = NULL;
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

  void modeKernel_t::assertArgumentLimit() const {
    // Check argument limit
    OCCA_ERROR("(" << name << ") Kernels can have at most [" << OCCA_MAX_ARGS << "] arguments",
               ((int) arguments.size() + 1) < OCCA_MAX_ARGS);
  }

  void modeKernel_t::assertArgInDevice(const kernelArgData &arg) const {
    // Make sure the argument is from the same device as the kernel
    occa::modeDevice_t *argDevice = arg.getModeDevice();
    OCCA_ERROR("(" << name << ") Kernel argument was not created from the same device as the kernel",
               !argDevice || (argDevice == modeDevice));
  }

  void modeKernel_t::setArguments(kernelArg *args,
                                  const int count) {
    arguments.clear();
    arguments.reserve(count);
    for (int i = 0; i < count; ++i) {
      pushArgument(args[i]);
    }
  }

  void modeKernel_t::pushArgument(const kernelArg &arg) {
    const int argCount = (int) arg.size();
    for (int i = 0; i < argCount; ++i) {
      const kernelArgData &argi = arg[i];
      assertArgInDevice(argi);
      arguments.push_back(argi);
    }

    assertArgumentLimit();
  }

  void modeKernel_t::setupRun() {
    const int argc = (int) arguments.size();

    const bool validateTypes = (
      metadata.isInitialized()
      && properties.get("type_validation", true)
    );

    if (validateTypes) {
      const int metaArgc = (int) metadata.arguments.size();

      OCCA_ERROR("(" << name << ") Kernel expects ["
                 << metaArgc << "] argument"
                 << (metaArgc != 1 ? "s," : ",")
                 << " received ["
                 << argc << ']',
                 argc == metaArgc);

      // TODO: Get original arg #
      for (int i = 0; i < argc; ++i) {
        kernelArgData &arg = arguments[i];
        lang::argMetadata_t &argInfo = metadata.arguments[i];

        modeMemory_t *mem = arg.getModeMemory();
        const bool isNull = arg.isNull();
        const bool isPtr = mem || isNull;
        if (isPtr != argInfo.isPtr) {
          if (argInfo.isPtr) {
            OCCA_FORCE_ERROR("(" << name << ") Kernel expects an occa::memory for argument ["
                             << (i + 1) << "]");
          } else {
            OCCA_FORCE_ERROR("(" << name << ") Kernel expects a non-occa::memory type for argument ["
                             << (i + 1) << "]");
          }
        }

        if (!isPtr || isNull) {
          continue;
        }

        OCCA_ERROR("(" << name << ") Argument [" << (i + 1) << "] has wrong runtime type.\n"
                   << "Expected type: " << argInfo.dtype << '\n'
                   << "Received type: " << *(mem->dtype_) << '\n',
                   mem->dtype_->canBeCastedTo(argInfo.dtype));

        arg.setupForKernelCall(argInfo.isConst);
      }
      return;
    }

    // Non-OKL kernel setup
    // All memory arguments are expected to be non-const for UVA purposes
    for (int i = 0; i < argc; ++i) {
      kernelArgData &arg = arguments[i];
      modeMemory_t *mem = arg.getModeMemory();
      if (mem) {
        arg.setupForKernelCall(false);
      }
    }
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

    modeKernel->setupRun();
    modeKernel->run();
  }

#include "kernelOperators.cpp"

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
