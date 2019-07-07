#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/metal/kernel.hpp>
#include <occa/modes/metal/device.hpp>
#include <occa/modes/metal/stream.hpp>

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
                   api::metal::device_t metalDevice_,
                   api::metal::kernel_t metalKernel_,
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
      metal::device &device = *((metal::device*) modeDevice);
      metal::stream &stream = (
        *((metal::stream*) (device.currentStream.getModeStream()))
      );
      api::metal::commandQueue_t &commandQueue = stream.metalCommandQueue;
      metalKernel.run(commandQueue,
                      outerDims,
                      innerDims,
                      arguments);
    }
  }
}
