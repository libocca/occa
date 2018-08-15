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

#if OCCA_HIP_ENABLED

#include <occa/mode/hip/kernel.hpp>
#include <occa/mode/hip/device.hpp>
#include <occa/mode/hip/utils.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/base.hpp>

namespace occa {
  namespace hip {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      hipModule(NULL),
      hipFunction(NULL),
      launcherKernel(NULL) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   hipModule_t hipModule_,
                   hipFunction_t hipFunction_,
                   const occa::properties &properties_) :
      occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      hipModule(hipModule_),
      hipFunction(hipFunction_),
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
        int deviceID = properties["device_id"];
        hipDeviceProp_t props;
        OCCA_HIP_ERROR("Getting device properties",
                       hipGetDeviceProperties(&props, deviceID ));
        maxSize = props.maxThreadsPerBlock;

        innerDims.x = maxSize;
      }
      return innerDims;
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
      int rem = 0;
      for (int i = 0; i < kArgCount; ++i) {
        const kArgVector &iArgs = arguments[i].args;
        const int argCount = (int) iArgs.size();
        if (!argCount) {
          continue;
        }
        for (int ai = 0; ai < argCount; ++ai) {
          size_t Nbytes;
          if (rem+iArgs[ai].size<=sizeof(void*)) {
            Nbytes = iArgs[ai].size;
            rem = sizeof(void*) - rem - iArgs[ai].size;
          } else {
            Nbytes = sizeof(void*);
            argc+=rem;
            rem = 0;
          }

          memcpy((char*) vArgs.data() + argc,&(iArgs[ai].data.int64_), Nbytes);
          argc += Nbytes;
        }
      } 

      size_t size = vArgs.size()*sizeof(vArgs[0]);
      void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &(vArgs[0]), HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                        HIP_LAUNCH_PARAM_END};

      OCCA_HIP_ERROR("Launching Kernel",
                     hipModuleLaunchKernel(hipFunction,
                                           outerDims.x, outerDims.y, outerDims.z,
                                           innerDims.x, innerDims.y, innerDims.z,
                                           0, *((hipStream_t*) modeDevice->currentStream),
                                           NULL, (void**)&config));
    }

    void kernel::launcherRun() const {
      launcherKernel->arguments = arguments;
      launcherKernel->arguments.insert(
        launcherKernel->arguments.begin(),
        &(hipKernels[0])
      );

      int kernelCount = (int) hipKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        hipKernels[i]->arguments = arguments;
      }

      launcherKernel->run();
    }

    void kernel::free() {
      if (!launcherKernel) {
        if (hipModule) {
          OCCA_HIP_ERROR("Kernel (" + name + ") : Unloading Module",
                         hipModuleUnload(hipModule));
          hipModule = NULL;
        }
        return;
      }

      launcherKernel->free();
      delete launcherKernel;
      launcherKernel = NULL;

      int kernelCount = (int) hipKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        hipKernels[i]->free();
        delete hipKernels[i];
      }
      hipKernels.clear();
    }
  }
}

#endif
