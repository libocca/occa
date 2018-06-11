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
#  ifndef OCCA_OPENCL_KERNEL_HEADER
#  define OCCA_OPENCL_KERNEL_HEADER

#include <occa/kernel.hpp>
#include <occa/modes/opencl/headers.hpp>
#include <occa/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    class device;

    class kernel : public occa::kernel_v {
      friend class device;
      friend cl_kernel getCLKernel(occa::kernel kernel);

    private:
      cl_device_id clDevice;
      cl_kernel    clKernel;

      occa::kernel_v *launcherKernel;
      std::vector<kernel*> clKernels;

    public:
      kernel(device_v *dHandle_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::properties &properties_);

      kernel(device_v *dHandle_,
             const std::string &name_,
             const std::string &sourceFilename_,
             cl_device_id clDevice_,
             cl_kernel clKernel_,
             const occa::properties &properties_);

      ~kernel();

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void run() const;
      void launcherRun() const;

      void free();
    };
  }
}

#  endif
#endif
