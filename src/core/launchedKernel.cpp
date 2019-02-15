#include <occa/core/launchedKernel.hpp>

namespace occa {
  launchedModeKernel_t::launchedModeKernel_t(modeDevice_t *modeDevice_,
                                             const std::string &name_,
                                             const std::string &sourceFilename_,
                                             const occa::properties &properties_) :
    occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
    launcherKernel(NULL) {}

  launchedModeKernel_t::~launchedModeKernel_t() {
    if (!launcherKernel) {
      return;
    }

    delete launcherKernel;
    launcherKernel = NULL;

    int kernelCount = (int) deviceKernels.size();
    for (int i = 0; i < kernelCount; ++i) {
      delete deviceKernels[i];
    }
    deviceKernels.clear();
  }

  void launchedModeKernel_t::run() const {
    if (launcherKernel) {
      launcherRun();
    } else {
      deviceRun();
    }
  }

  void launchedModeKernel_t::launcherRun() const {
    kernelArg arg(&(deviceKernels[0]));

    launcherKernel->arguments = arguments;
    launcherKernel->arguments.insert(
      launcherKernel->arguments.begin(),
      arg[0]
    );

    int kernelCount = (int) deviceKernels.size();
    for (int i = 0; i < kernelCount; ++i) {
      deviceKernels[i]->arguments = arguments;
    }

    launcherKernel->run();
  }
}
