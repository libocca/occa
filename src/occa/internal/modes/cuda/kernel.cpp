#include <occa/internal/modes/cuda/kernel.hpp>
#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/utils.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace cuda {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   CUmodule cuModule_,
                   const occa::json &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      cuModule(cuModule_),
      cuFunction(NULL) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   CUfunction cuFunction_,
                   const occa::json &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      cuModule(NULL),
      cuFunction(cuFunction_) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   CUmodule cuModule_,
                   CUfunction cuFunction_,
                   const occa::json &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      cuModule(cuModule_),
      cuFunction(cuFunction_) {}

    kernel::~kernel() {
      if (cuModule) {
        OCCA_CUDA_DESTRUCTOR_ERROR(
          "Kernel (" + name + ") : Unloading Module",
          cuModuleUnload(cuModule)
        );
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
      return dim(occa::UDIM_DEFAULT, occa::UDIM_DEFAULT, occa::UDIM_DEFAULT);
    }

    dim kernel::maxInnerDims() const {
      static dim maxInnerDims_(0);
      if (maxInnerDims_.x == 0) {
        int maxSize = 0;
        OCCA_CUDA_ERROR("Kernel: Getting Maximum Inner-Dim Size",
                        cuFuncGetAttribute(&maxSize,
                                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           cuFunction));

        maxInnerDims_.x = (udim_t) maxSize;
      }
      return maxInnerDims_;
    }

    void kernel::deviceRun() const {
      device *devicePtr = (device*) modeDevice;

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
          vArgs[i] = devicePtr->getNullPtr();
        }
      }

      devicePtr->setCudaContext();

      OCCA_CUDA_ERROR("Launching Kernel",
                      cuLaunchKernel(cuFunction,
                                     outerDims.x, outerDims.y, outerDims.z,
                                     innerDims.x, innerDims.y, innerDims.z,
                                     0, getCuStream(),
                                     &(vArgs[0]), 0));
    }
  }
}
