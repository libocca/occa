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

#if OCCA_HIP_ENABLED
#  ifndef OCCA_HIP_DEVICE_HEADER
#  define OCCA_HIP_DEVICE_HEADER

#include <occa/defines.hpp>
#include <occa/device.hpp>

#include "hip/hip_runtime_api.h"

namespace occa {
  namespace hip {
    class kernel;

    class device : public occa::device_v {
      friend class kernel;

    private:
      mutable hash_t hash_;

    public:
      int archMajorVersion, archMinorVersion;
      bool p2pEnabled;

      hipDevice_t  hipDevice;
      hipCtx_t hipContext;

      device(const occa::properties &properties_);
      virtual ~device();

      virtual void free();

      virtual void finish() const;

      virtual bool hasSeparateMemorySpace() const;

      virtual hash_t hash() const;

      //---[ Stream ]-------------------
      virtual stream_t createStream() const;
      virtual void freeStream(stream_t s) const;

      virtual streamTag tagStream() const;
      virtual void waitFor(streamTag tag) const;
      virtual double timeBetween(const streamTag &startTag,
                                 const streamTag &endTag) const;

      virtual stream_t wrapStream(void *handle_,
                                  const occa::properties &props) const;
      //================================

      //---[ Kernel ]-------------------
      bool parseFile(const std::string &filename,
                     const std::string &outputFile,
                     const std::string &hostOutputFile,
                     const occa::properties &kernelProps,
                     lang::kernelMetadataMap &hostMetadata,
                     lang::kernelMetadataMap &deviceMetadata);

      virtual kernel_v* buildKernel(const std::string &filename,
                                    const std::string &kernelName,
                                    const hash_t kernelHash,
                                    const occa::properties &props);

      void compileKernel(const std::string &hashDir,
                         const std::string &kernelName,
                         occa::properties &kernelProps,
                         io::lock_t &lock);

      kernel_v* buildOKLKernelFromBinary(const std::string &hashDir,
                                         const std::string &kernelName,
                                         lang::kernelMetadataMap &hostMetadata,
                                         lang::kernelMetadataMap &deviceMetadata,
                                         const occa::properties &kernelProps,
                                         const io::lock_t &lock = io::lock_t());

      kernel_v* buildLauncherKernel(const std::string &hashDir,
                                    const std::string &kernelName,
                                    lang::kernelMetadata &hostMetadata);

      virtual kernel_v* buildKernelFromBinary(const std::string &filename,
                                              const std::string &kernelName,
                                              const occa::properties &props);
      //================================

      //---[ Memory ]-------------------
      virtual memory_v* malloc(const udim_t bytes,
                               const void *src,
                               const occa::properties &props);

      virtual memory_v* mappedAlloc(const udim_t bytes,
                                    const void *src,
                                    const occa::properties &props);

      memory_v* managedAlloc(const udim_t bytes,
                             const void *src,
                             const occa::properties &props);

      virtual udim_t memorySize() const;
      //================================
    };
  }
}

#  endif
#endif
