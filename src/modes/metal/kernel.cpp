#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

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
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      metalDevice(NULL),
      metalKernel(NULL) {}

    kernel::~kernel() {
    }

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return 3;
    }

    dim kernel::maxInnerDims() const {
      return 3;
    }

    void kernel::deviceRun() const {
      // Setup kernel dimensions
      occa::dim fullDims = (outerDims * innerDims);

      size_t fullDims_[3] = {
        fullDims.x, fullDims.y, fullDims.z
      };
      size_t innerDims_[3] = {
        innerDims.x, innerDims.y, innerDims.z
      };

      // Set arguments
      const int args = (int) arguments.size();
      for (int i = 0; i < args; ++i) {
        const kernelArgData &arg = arguments[i];
        // OCCA_OPENCL_ERROR("Kernel [" + name + "]"
        //                   << ": Setting Kernel Argument [" << (i + 1) << "]",
        //                   clSetKernelArg(clKernel, i, arg.size, arg.ptr()));
      }

      // OCCA_OPENCL_ERROR("Kernel [" + name + "]"
      //                   << " : Kernel Run",
      //                   clEnqueueNDRangeKernel(getCommandQueue(),
      //                                          clKernel,
      //                                          (cl_int) fullDims.dims,
      //                                          NULL,
      //                                          (size_t*) &fullDims_,
      //                                          (size_t*) &innerDims_,
      //                                          0, NULL, NULL));
    }
  }
}

#endif
