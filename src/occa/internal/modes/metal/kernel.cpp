#include <occa/core/base.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/metal/device.hpp>
#include <occa/internal/modes/metal/kernel.hpp>
#include <occa/internal/modes/metal/stream.hpp>

namespace occa {
  namespace metal {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::json &properties_) :
        occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   api::metal::device_t metalDevice_,
                   api::metal::function_t metalFunction_,
                   const occa::json &properties_) :
        occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
        metalDevice(metalDevice_),
        metalFunction(metalFunction_) {}

    kernel::~kernel() {
      metalFunction.free();
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
      metal::device &device = *((metal::device*) modeDevice);
      metal::stream &stream = (
        *((metal::stream*) (device.currentStream.getModeStream()))
      );
      api::metal::commandQueue_t &commandQueue = stream.metalCommandQueue;
      metalFunction.run(commandQueue,
                        outerDims,
                        innerDims,
                        arguments);
    }
  }
}
