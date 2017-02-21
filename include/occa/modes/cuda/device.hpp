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

#if OCCA_CUDA_ENABLED
#  ifndef OCCA_CUDA_DEVICE_HEADER
#  define OCCA_CUDA_DEVICE_HEADER

#include "occa/defines.hpp"
#include "occa/device.hpp"

#include <cuda.h>

namespace occa {
  namespace cuda {
    class device : public occa::device_v {
      hash_t hash_;

    public:
      int archMajorVersion, archMinorVersion;
      bool p2pEnabled;

      CUdevice  handle;
      CUcontext context;

      device(const occa::properties &properties_ = occa::properties());
      ~device();
      void free();

      void* getHandle(const occa::properties &props);

      void finish();

      bool hasSeparateMemorySpace();

      hash_t hash();

      //  |---[ Stream ]----------------
      stream_t createStream();
      void freeStream(stream_t s);

      streamTag tagStream();
      void waitFor(streamTag tag);
      double timeBetween(const streamTag &startTag, const streamTag &endTag);

      stream_t wrapStream(void *handle_, const occa::properties &props);
      //  |=============================

      //  |---[ Kernel ]----------------
      kernel_v* buildKernel(const std::string &filename,
                            const std::string &kernelName,
                            const occa::properties &props);

      kernel_v* buildKernelFromBinary(const std::string &filename,
                                      const std::string &kernelName,
                                      const occa::properties &props);
      //  |=============================

      //  |---[ Memory ]----------------
      memory_v* malloc(const udim_t bytes,
                       const void *src,
                       const occa::properties &props);

      memory_v* mappedAlloc(const udim_t bytes,
                            const void *src);

      memory_v* wrapMemory(void *handle_,
                           const udim_t bytes,
                           const occa::properties &props);

      udim_t memorySize();
      //  |=============================
    };
  }
}

#  endif
#endif
