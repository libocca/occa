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
#include <occa/lang/modes/opencl.hpp>

namespace occa {
  namespace opencl {
    kernel::kernel(const occa::properties &properties_) :
      occa::kernel_v(properties_) {}

    kernel::~kernel() {}

    info_t kernel::makeCLInfo() const {
      info_t info;
      info.clDevice  = clDevice;
      info.clContext = ((device*) dHandle)->clContext;
      info.clKernel  = clKernel;
      return info;
    }

    void kernel::build(const std::string &filename,
                       const std::string &kernelName,
                       const hash_t hash) {

      name = kernelName;
      sourceFilename = filename;
      binaryFilename = getBinaryFilename(filename, hash);
      bool foundBinary = true;

      const std::string hashTag = "opencl-kernel";
      if (io::haveHash(hash, hashTag)) {
        if (sys::fileExists(binaryFilename)) {
          io::releaseHash(hash, hashTag);
        } else {
          foundBinary = false;
        }
      }

      const bool verbose = properties.get("verbose", false);
      if (foundBinary) {
        if (verbose) {
           std::cout << "Loading cached ["
                     << kernelName
                     << "] from ["
                     << io::shortname(filename)
                     << "] in [" << io::shortname(binaryFilename) << "]\n";
        }
        return buildFromBinary(binaryFilename, kernelName);
      }

      // Cache raw origin
      std::string sourceFile = (
        io::cacheFile(sourceFilename,
                      kc::rawSourceFile,
                      hash,
                      assembleHeader(properties),
                      properties["footer"].string())
      );

      valid = true;
      if (properties.get("okl", true)) {
        const std::string outputFile = getSourceFilename(filename, hash);
        const std::string hostOutputFile = getLaunchSourceFilename(filename, hash);
        parseFile(sourceFile,
                  outputFile,
                  hostOutputFile,
                  properties["parser"]);
        if (!valid) {
          return;
        }
        sourceFile = outputFile;

        occa::kernel hostKernel = host().buildKernel(hostOutputFile,
                                                     kernelName,
                                                     "okl: false");
        // TODO 1.1: Store metadata in the build.json

        launcherKernel = hostKernel.getKHandle();
        launcherKernel->dontUseRefs();
        nestedKernels.push_back(this);
      }

      std::string cFunction = io::read(sourceFile);
      info_t clInfo = makeCLInfo();
      opencl::buildKernel(clInfo,
                          cFunction.c_str(), cFunction.size(),
                          kernelName,
                          properties["compilerFlags"],
                          hash,
                          sourceFile,
                          properties);
      clKernel = clInfo.clKernel;

      opencl::saveProgramBinary(clInfo,
                                binaryFilename,
                                hash,
                                hashTag);

      io::releaseHash(hash, hashTag);
    }

    void kernel::buildFromBinary(const std::string &filename,
                                 const std::string &kernelName) {

      name = kernelName;

      std::string cFile = io::read(filename);
      info_t clInfo = makeCLInfo();
      opencl::buildKernelFromBinary(clInfo,
                                    (const unsigned char*) cFile.c_str(),
                                    cFile.size(),
                                    kernelName,
                                    ((opencl::device*) dHandle)->properties["compilerFlags"]);
      clKernel = clInfo.clKernel;
    }

    void kernel::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const std::string &hostOutputFile,
                           const occa::properties &props) {
      lang::okl::openclParser parser(props);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      valid = parser.succeeded();
      if (!valid) {
        if (!props.get("silent", false)) {
          OCCA_FORCE_ERROR("Unable to transform OKL kernel");
        }
        return;
      }

      if (!sys::fileExists(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        const std::string hashTag = "parse-file";

        if (io::haveHash(hash, hashTag)) {
          parser.writeToFile(outputFile);
          io::releaseHash(hash, hashTag);
        }
      }

      if (!sys::fileExists(hostOutputFile)) {
        hash_t hash = occa::hash(hostOutputFile);
        const std::string hashTag = "parse-host-file";

        if (io::haveHash(hash, hashTag)) {
          parser.writeHostSourceToFile(hostOutputFile);
          io::releaseHash(hash, hashTag);
        }
      }

      setMetadata(parser);
    }

    int kernel::maxDims() const {
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

    void kernel::runFromArguments(const int kArgc, const kernelArg *kArgs) const {
      occa::dim fullOuter = outer*inner;

      size_t fullOuter_[3] = { fullOuter.x, fullOuter.y, fullOuter.z };
      size_t inner_[3] = { inner.x, inner.y, inner.z };

      int argc = 0;
      for (int i = 0; i < kArgc; ++i) {
        const int argCount = (int) kArgs[i].args.size();
        if (argCount) {
          const kernelArgData *kArgs_i = &(kArgs[i].args[0]);
          for (int j = 0; j < argCount; ++j) {
            const kernelArgData &kArg_j = kArgs_i[j];
            OCCA_OPENCL_ERROR("Kernel [" + name + "]"
                              << ": Setting Kernel Argument [" << (i + 1) << "]",
                              clSetKernelArg(clKernel, argc++, kArg_j.size, kArg_j.ptr()));
          }
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

    void kernel::free() {
      if (clKernel) {
        OCCA_OPENCL_ERROR("Kernel [" + name + "]: Free",
                          clReleaseKernel(clKernel));
        clKernel = NULL;
      }
    }
  }
}

#endif
