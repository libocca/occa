/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include <occa/defines.hpp>

#if OCCA_CUDA_ENABLED

#include <occa/modes/cuda/kernel.hpp>
#include <occa/modes/cuda/device.hpp>
#include <occa/modes/cuda/utils.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/base.hpp>

namespace occa {
  namespace cuda {
    kernel::kernel(device_v *dHandle_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::kernel_v(dHandle_, name_, sourceFilename_, properties_),
      cuModule(NULL),
      cuFunction(NULL),
      launcherKernel(NULL) {}

    kernel::kernel(device_v *dHandle_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   CUmodule cuModule_,
                   CUfunction cuFunction_,
                   const occa::properties &properties_) :
      occa::kernel_v(dHandle_, name_, sourceFilename_, properties_),
      cuModule(cuModule_),
      cuFunction(cuFunction_),
      launcherKernel(NULL) {}

    kernel::~kernel() {}

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(-1, -1, -1);
    }

    dim kernel::maxInnerDims() const {
      static dim innerDims(0);
      if (innerDims.x == 0) {
        int maxSize;
        OCCA_CUDA_ERROR("Kernel: Getting Maximum Inner-Dim Size",
                        cuFuncGetAttribute(&maxSize,
                                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           cuFunction));

        innerDims.x = maxSize;
      }
      return innerDims;
    }

    void kernel::run() const {
      if (launcherKernel) {
        return launcherRun();
      }

      const int totalArgCount = kernelArg::argumentCount(arguments);
      if ((int) vArgs.size() < totalArgCount) {
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
                                     outer.x, outer.y, outer.z,
                                     inner.x, inner.y, inner.z,
                                     0, *((CUstream*) dHandle->currentStream),
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

    void kernel::free() {
      if (!launcherKernel) {
        if (cuModule) {
          OCCA_CUDA_ERROR("Kernel (" + name + ") : Unloading Module",
                          cuModuleUnload(cuModule));
          cuModule = NULL;
        }
        return;
      }

      launcherKernel->free();
      delete launcherKernel;
      launcherKernel = NULL;

      int kernelCount = (int) cuKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        cuKernels[i]->free();
        delete cuKernels[i];
      }
      cuKernels.clear();
    }
  }
}

#endif
