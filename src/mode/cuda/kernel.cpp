#include <occa/defines.hpp>

#if OCCA_CUDA_ENABLED

#include <occa/mode/cuda/kernel.hpp>
#include <occa/mode/cuda/device.hpp>
#include <occa/mode/cuda/utils.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace cuda {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      cuModule(NULL),
      cuFunction(NULL),
      launcherKernel(NULL) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   CUmodule cuModule_,
                   CUfunction cuFunction_,
                   const occa::properties &properties_) :
      occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      cuModule(cuModule_),
      cuFunction(cuFunction_),
      launcherKernel(NULL) {}

    kernel::~kernel() {
      if (!launcherKernel) {
        if (cuModule) {
          OCCA_CUDA_ERROR("Kernel (" + name + ") : Unloading Module",
                          cuModuleUnload(cuModule));
          cuModule = NULL;
        }
        return;
      }

      delete launcherKernel;
      launcherKernel = NULL;

      int kernelCount = (int) cuKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        delete cuKernels[i];
      }
      cuKernels.clear();
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

    void kernel::run() const {
      if (launcherKernel) {
        return launcherRun();
      }

      const int totalArgCount = kernelArg::argumentCount(arguments);
      if (!totalArgCount) {
        vArgs.resize(1);
      } else if ((int) vArgs.size() < totalArgCount) {
        vArgs.resize(totalArgCount);
      }

      const int kArgCount = (int) arguments.size();

      int argc = 0;
      for (int i = 0; i < kArgCount; ++i) {
        const kArgVector &iArgs = arguments[i].args;
        const int argCount = (int) iArgs.size();
        if (!argCount) {
          continue;
        }
        for (int ai = 0; ai < argCount; ++ai) {
          vArgs[argc++] = iArgs[ai].ptr();
        }
      }

      OCCA_CUDA_ERROR("Launching Kernel",
                      cuLaunchKernel(cuFunction,
                                     outerDims.x, outerDims.y, outerDims.z,
                                     innerDims.x, innerDims.y, innerDims.z,
                                     0, getCuStream(),
                                     &(vArgs[0]), 0));
    }

    void kernel::launcherRun() const {
      launcherKernel->arguments = arguments;
      launcherKernel->arguments.insert(
        launcherKernel->arguments.begin(),
        &(cuKernels[0])
      );

      int kernelCount = (int) cuKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        cuKernels[i]->arguments = arguments;
      }

      launcherKernel->run();
    }
  }
}

#endif
