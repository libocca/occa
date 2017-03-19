/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/defines.hpp"

#if OCCA_OPENCL_ENABLED

#include "occa/modes/opencl/kernel.hpp"
#include "occa/modes/opencl/device.hpp"
#include "occa/modes/opencl/utils.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/sys.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace opencl {
    kernel::kernel(const occa::properties &properties_) :
      occa::kernel_v(properties_) {}

    kernel::~kernel() {}

    info_t kernel::makeCLInfo() const {
      info_t info;
      info.clDeviceID = clDeviceID;
      info.clContext  = clContext;
      info.clProgram  = clProgram;
      info.clKernel   = clKernel;
      return info;
    }

    void* kernel::getHandle(const occa::properties &props) const {
      const std::string type = props["type"];

      if (type == "kernel") {
        return clKernel;
      }
      if (type == "program") {
        return clProgram;
      }
      return NULL;
    }

    void kernel::build(const std::string &filename,
                       const std::string &kernelName,
                       const occa::properties &props) {

      const occa::properties allProps = properties + props;
      name = kernelName;

      hash_t hash = occa::hashFile(filename);
      hash ^= allProps.hash();

      const std::string sourceFile = getSourceFilename(filename, hash);
      const std::string binaryFile = getBinaryFilename(filename, hash);
      bool foundBinary = true;

      const std::string hashTag = "opencl-kernel";
      if (!io::haveHash(hash, hashTag)) {
        io::waitForHash(hash, hashTag);
      } else if (sys::fileExists(binaryFile)) {
        io::releaseHash(hash, hashTag);
      } else {
        foundBinary = false;
      }

      if (foundBinary) {
        if (settings().get("verboseCompilation", true)) {
          std::cout << "Found cached binary of [" << io::shortname(filename)
                    << "] in [" << io::shortname(binaryFile) << "]\n";
        }
        return buildFromBinary(binaryFile, kernelName, props);
      }

      const std::string kernelDefines =
        io::cacheFile(env::OCCA_DIR + "/include/occa/modes/opencl/kernelDefines.hpp",
                      "openclKernelDefines.hpp");

      std::stringstream ss;
      ss << "#include \"" << kernelDefines << "\"\n"
         << assembleHeader(allProps);

      io::cacheFile(filename,
                    kc::sourceFile,
                    hash,
                    ss.str(),
                    allProps["footer"]);

      std::string cFunction = io::read(sourceFile);
      info_t clInfo = makeCLInfo();
      opencl::buildKernel(clInfo,
                          cFunction.c_str(), cFunction.size(),
                          kernelName,
                          allProps["compilerFlags"],
                          hash, sourceFile);
      clProgram = clInfo.clProgram;
      clKernel  = clInfo.clKernel;

      opencl::saveProgramBinary(clInfo, binaryFile, hash, hashTag);

      io::releaseHash(hash, hashTag);
    }

    void kernel::buildFromBinary(const std::string &filename,
                                 const std::string &kernelName,
                                 const occa::properties &props) {

      name = kernelName;

      std::string cFile = io::read(filename);
      info_t clInfo = makeCLInfo();
      opencl::buildKernelFromBinary(clInfo,
                                    (const unsigned char*) cFile.c_str(),
                                    cFile.size(),
                                    kernelName,
                                    ((opencl::device*) dHandle)->properties["compilerFlags"]);
      clProgram = clInfo.clProgram;
      clKernel  = clInfo.clKernel;
    }

    int kernel::maxDims() const {
      static cl_uint dims_ = 0;
      if (dims_ == 0) {
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Dims",
                          clGetDeviceInfo(clDeviceID,
                                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                          0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Dims",
                          clGetDeviceInfo(clDeviceID,
                                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                          bytes, &dims_, NULL));
      }
      return (int) dims_;
    }

    dim kernel::maxOuterDims() const {
      static occa::dim outerDims(0);
      if (outerDims.x == 0) {
        int dims_ = maxDims();
        size_t *od = new size_t[dims_];
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Outer Dims",
                          clGetDeviceInfo(clDeviceID,
                                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                          0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Outer Dims",
                          clGetDeviceInfo(clDeviceID,
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
      static occa::dim innerDims(0);
      if (innerDims.x == 0) {
        size_t dims_;
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Inner Dims",
                          clGetKernelWorkGroupInfo(clKernel,
                                                   clDeviceID,
                                                   CL_KERNEL_WORK_GROUP_SIZE,
                                                   0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Inner Dims",
                          clGetKernelWorkGroupInfo(clKernel,
                                                   clDeviceID,
                                                   CL_KERNEL_WORK_GROUP_SIZE,
                                                   bytes, &dims_, NULL));
        innerDims.x = dims_;
      }
      return innerDims;
    }

    void kernel::runFromArguments(const int kArgc, const kernelArg *kArgs) const {
      occa::dim fullOuter = outer*inner;

      size_t fullOuter_[3] = { fullOuter.x, fullOuter.y, fullOuter.z };
      size_t inner_[3] = { inner.x, inner.y, inner.z };

      int argc = 0;
      OCCA_OPENCL_ERROR("Kernel (" + metadata.name + ") : Setting Kernel Argument [0]",
                        clSetKernelArg(clKernel, argc++, sizeof(void*), NULL));

      for (int i = 0; i < kArgc; ++i) {
        const kernelArg_t &arg = kArgs[i].arg;
        const dim_t extraArgCount = kArgs[i].extraArgs.size();
        const kernelArg_t *extraArgs = extraArgCount ? &(kArgs[i].extraArgs[0]) : NULL;

        OCCA_OPENCL_ERROR("Kernel (" + metadata.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                          clSetKernelArg(clKernel, argc++, arg.size, arg.ptr()));

        for (int j = 0; j < extraArgCount; ++j) {
          OCCA_OPENCL_ERROR("Kernel (" + metadata.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                            clSetKernelArg(clKernel, argc++, extraArgs[j].size, extraArgs[j].ptr()));
        }
      }

      OCCA_OPENCL_ERROR("Kernel (" + metadata.name + ") : Kernel Run",
                        clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                               clKernel,
                                               (cl_int) dims,
                                               NULL,
                                               (size_t*) &fullOuter_,
                                               (size_t*) &inner_,
                                               0, NULL, NULL));
    }

    void kernel::free() {
      OCCA_OPENCL_ERROR("Kernel: Free",
                        clReleaseKernel(clKernel));
    }
  }
}

#endif
