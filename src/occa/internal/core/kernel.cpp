#include <occa/internal/core/device.hpp>
#include <occa/internal/core/kernel.hpp>
#include <occa/internal/core/memory.hpp>

namespace occa {
  modeKernel_t::modeKernel_t(modeDevice_t *modeDevice_,
                             const std::string &name_,
                             const std::string &sourceFilename_,
                             const occa::json &properties_) :
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

  void modeKernel_t::assertArgInDevice(const kernelArgData &arg,
                                       const int argIndex) const {
    // Make sure the argument is from the same device as the kernel
    occa::modeDevice_t *argDevice = arg.getModeDevice();
    OCCA_ERROR("(" << hash << ":" << name << ") Kernel argument ["
               << argIndex << "] was not created from the same device as the kernel\n"
               << "Kernel device: " << modeDevice->mode << "\n"
               << "Argument device: " << argDevice->mode << " \n",
               !argDevice || (argDevice->mode == modeDevice->mode));
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
      assertArgInDevice(argi, (int) arguments.size());
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

      OCCA_ERROR("(" << hash << ":" << name << ") Kernel expects ["
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
        const bool isNull = arg.value.isNull();
        const bool isPtr = mem || isNull;
        if (isPtr != argInfo.isPtr) {
          if (argInfo.isPtr) {
            OCCA_FORCE_ERROR("(" << hash << ":" << name << ") Kernel expects an occa::memory for argument ["
                             << (i + 1) << "]");
          } else {
            OCCA_FORCE_ERROR("(" << hash << ":" << name << ") Kernel expects a non-occa::memory type for argument ["
                             << (i + 1) << "]");
          }
        }

        if (!isPtr || isNull) {
          continue;
        }

        OCCA_ERROR("(" << hash << ":" << name << ") Argument [" << (i + 1) << "] has wrong runtime type.\n"
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

  bool modeKernel_t::isNoop() const {
    return (
      outerDims.isZero() && innerDims.isZero()
    );
  }
}
