#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/metal/kernel.hpp>
#include <occa/modes/metal/device.hpp>

namespace occa {
  namespace metal {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   metalDevice_t metalDevice_,
                   metalKernel_t metalKernel_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      metalDevice(metalDevice_),
      metalKernel(metalKernel_) {}

    kernel::~kernel() {
      metalKernel.free();
    }

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return metalDevice.getMaxOuterDims();
    }

    dim kernel::maxInnerDims() const {
      return metalDevice.getMaxInnerDims();
    }

    void kernel::deviceRun() const {
      metalKernel.clearArguments();

      // Set arguments
      const int args = (int) arguments.size();
      for (int i = 0; i < args; ++i) {
        metalKernel.addArgument(i, arguments[i]);
      }

      metalKernel.run(outerDims, innerDims);
    }
  }
}
