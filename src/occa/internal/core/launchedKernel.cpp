#include <occa/internal/core/memory.hpp>
#include <occa/internal/core/launchedKernel.hpp>

namespace occa {
  launchedModeKernel_t::launchedModeKernel_t(modeDevice_t *modeDevice_,
                                             const std::string &name_,
                                             const std::string &sourceFilename_,
                                             const occa::json &properties_) :
    occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
    launcherKernel(NULL) {}

  launchedModeKernel_t::~launchedModeKernel_t() {
    if (!launcherKernel) {
      return;
    }

    delete launcherKernel;
    launcherKernel = NULL;

    const int kernelCount = (int) deviceKernels.size();
    for (int i = 0; i < kernelCount; ++i) {
      delete deviceKernels[i];
    }
    deviceKernels.clear();
  }

  const lang::kernelMetadata_t& launchedModeKernel_t::getMetadata() const {
    return deviceKernels[0]->metadata;
  }

  void launchedModeKernel_t::run() const {
    if (launcherKernel) {
      launcherRun();
    } else {
      deviceRun();
    }
  }

  void launchedModeKernel_t::launcherRun() const {
    // Add the kernel array as the first argument
    kernelArg launcherArgs(&(deviceKernels[0]));

    const int argCount = (int) arguments.size();
    for (int i = 0; i < argCount; ++i) {
      const kernelArgData &arg = arguments[i];
      if (arg.modeMemory) {
        launcherArgs.add((void*) arg.modeMemory);
      } else {
        launcherArgs.add(arg);
      }
    }
    launcherKernel->arguments = launcherArgs.args;


    int kernelCount = (int) deviceKernels.size();
    for (int i = 0; i < kernelCount; ++i) {
      deviceKernels[i]->arguments = arguments;
    }

    launcherKernel->run();
  }
}
