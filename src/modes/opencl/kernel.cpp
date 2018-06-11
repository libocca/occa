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

#if OCCA_OPENCL_ENABLED

#include <occa/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/opencl/kernel.hpp>
#include <occa/modes/opencl/device.hpp>
#include <occa/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    kernel::kernel(device_v *dHandle_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::kernel_v(dHandle_, name_, sourceFilename_, properties_),
      clDevice(NULL),
      clKernel(NULL),
      launcherKernel(NULL) {}

    kernel::kernel(device_v *dHandle_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   cl_device_id clDevice_,
                   cl_kernel clKernel_,
                   const occa::properties &properties_) :
      occa::kernel_v(dHandle_, name_, sourceFilename_, properties_),
      clDevice(clDevice_),
      clKernel(clKernel_),
      launcherKernel(NULL) {}

    kernel::~kernel() {}

    int kernel::maxDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static cl_uint dims_ = 0;
      if (dims_ == 0) {
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                          0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                          bytes, &dims_, NULL));
      }
      return (int) dims_;
    }

    dim kernel::maxOuterDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim outerDims(0);
      if (outerDims.x == 0) {
        int dims_ = maxDims();
        size_t *od = new size_t[dims_];
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Outer Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                          0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Outer Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                          bytes, &od, NULL));
        for (int i = 0; i < dims_; ++i) {
          outerDims[i] = od[i];
        }
        delete [] od;
      }
      return outerDims;
    }

    dim kernel::maxInnerDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim innerDims(0);
      if (innerDims.x == 0) {
        size_t dims_;
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Inner Dims",
                          clGetKernelWorkGroupInfo(clKernel,
                                                   clDevice,
                                                   CL_KERNEL_WORK_GROUP_SIZE,
                                                   0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Inner Dims",
                          clGetKernelWorkGroupInfo(clKernel,
                                                   clDevice,
                                                   CL_KERNEL_WORK_GROUP_SIZE,
                                                   bytes, &dims_, NULL));
        innerDims.x = dims_;
      }
      return innerDims;
    }

    void kernel::run() const {
      if (launcherKernel) {
        return launcherRun();
      }

      // Setup kernel dimensions
      occa::dim fullOuter = (outer * inner);

      size_t fullOuter_[3] = {
        fullOuter.x, fullOuter.y, fullOuter.z
      };
      size_t inner_[3] = {
        inner.x, inner.y, inner.z
      };

      // Set arguments
      const int kArgCount = (int) arguments.size();

      int argc = 0;
      for (int i = 0; i < kArgCount; ++i) {
        const kArgVector &iArgs = arguments[i].args;
        const int argCount = (int) iArgs.size();
        if (!argCount) {
          continue;
        }

        for (int ai = 0; ai < argCount; ++ai) {
          const kernelArgData &kArg = iArgs[ai];
          OCCA_OPENCL_ERROR("Kernel [" + name + "]"
                            << ": Setting Kernel Argument [" << (i + 1) << "]",
                            clSetKernelArg(clKernel, argc++, kArg.size, kArg.ptr()));
        }
      }

      OCCA_OPENCL_ERROR("Kernel [" + name + "]"
                        << " : Kernel Run",
                        clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                               clKernel,
                                               (cl_int) fullOuter.dims,
                                               NULL,
                                               (size_t*) &fullOuter_,
                                               (size_t*) &inner_,
                                               0, NULL, NULL));
    }

    void kernel::launcherRun() const {
      launcherKernel->arguments = arguments;
      launcherKernel->arguments.insert(
        launcherKernel->arguments.begin(),
        &(clKernels[0])
      );

      int kernelCount = (int) clKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        clKernels[i]->arguments = arguments;
      }

      launcherKernel->run();
    }

    void kernel::free() {
      if (!launcherKernel) {
        if (clKernel) {
          OCCA_OPENCL_ERROR("Kernel [" + name + "]: Free",
                            clReleaseKernel(clKernel));
          clKernel = NULL;
        }
        return;
      }

      launcherKernel->free();
      delete launcherKernel;
      launcherKernel = NULL;

      int kernelCount = (int) clKernels.size();
      for (int i = 0; i < kernelCount; ++i) {
        clKernels[i]->free();
        delete clKernels[i];
      }
      clKernels.clear();
    }
  }
}

#endif
