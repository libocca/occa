#include <occa/modes/cuda/kernel.hpp>
#include <occa/modes/cuda/device.hpp>
#include <occa/modes/cuda/utils.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace cuda {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      cuModule(NULL),
      cuFunction(NULL) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   CUmodule cuModule_,
                   CUfunction cuFunction_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      cuModule(cuModule_),
      cuFunction(cuFunction_) {}

    kernel::~kernel() {
      if (cuModule) {
        OCCA_CUDA_ERROR("Kernel (" + name + ") : Unloading Module",
                        cuModuleUnload(cuModule));
        cuModule = NULL;
      }
    }

    CUstream& kernel::getCuStream() const {
      return ((device*) modeDevice)->getCuStream();
    }

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(-1, -1, -1);
    }

    dim kernel::maxInnerDims() const {
      static dim maxInnerDims_(0);
      if (maxInnerDims_.x == 0) {
        int maxSize;
        OCCA_CUDA_ERROR("Kernel: Getting Maximum Inner-Dim Size",
                        cuFuncGetAttribute(&maxSize,
                                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           cuFunction));

        maxInnerDims_.x = maxSize;
      }
      return maxInnerDims_;
    }

    void kernel::deviceRun() const {
      const int args = (int) arguments.size();
      if (!args) {
        vArgs.resize(1);
      } else if ((int) vArgs.size() < args) {
        vArgs.resize(args);
      }

      // Set arguments
      for (int i = 0; i < args; ++i) {
        vArgs[i] = arguments[i].ptr();
        // Set a proper NULL pointer
        if (!vArgs[i]) {
          vArgs[i] = ((device*) modeDevice)->getNullPtr();
        }
      }

      OCCA_CUDA_ERROR("Launching Kernel",
                      cuLaunchKernel(cuFunction,
                                     outerDims.x, outerDims.y, outerDims.z,
                                     innerDims.x, innerDims.y, innerDims.z,
                                     0, getCuStream(),
                                     &(vArgs[0]), 0));
    }
  }
}
