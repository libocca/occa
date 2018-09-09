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
#  ifndef OCCA_MODES_HIP_UTILS_HEADER
#  define OCCA_MODES_HIP_UTILS_HEADER

#include <hip/hip_runtime_api.h>

#include <occa/core/device.hpp>

namespace occa {
  namespace hip {
    typedef int advice_t;

    bool init();

    int getDeviceCount();
    hipDevice_t getDevice(const int id);
    udim_t getDeviceMemorySize(hipDevice_t device);

    std::string getVersion();

    void enablePeerToPeer(hipCtx_t context);
    void checkPeerToPeer(hipDevice_t destDevice,
                         hipDevice_t srcDevice);

    void peerToPeerMemcpy(hipDevice_t destDevice,
                          hipCtx_t destContext,
                          hipDeviceptr_t destMemory,
                          hipDevice_t srcDevice,
                          hipCtx_t srcContext,
                          hipDeviceptr_t srcMemory,
                          const udim_t bytes,
                          hipStream_t usingStream);


    void asyncPeerToPeerMemcpy(hipDevice_t destDevice,
                               hipCtx_t destContext,
                               hipDeviceptr_t destMemory,
                               hipDevice_t srcDevice,
                               hipCtx_t srcContext,
                               hipDeviceptr_t srcMemory,
                               const udim_t bytes,
                               hipStream_t usingStream);

    void peerToPeerMemcpy(hipDevice_t destDevice,
                          hipCtx_t destContext,
                          hipDeviceptr_t destMemory,
                          hipDevice_t srcDevice,
                          hipCtx_t srcContext,
                          hipDeviceptr_t srcMemory,
                          const udim_t bytes,
                          hipStream_t usingStream,
                          const bool isAsync);

    void advise(occa::memory mem,
                advice_t advice,
                const dim_t bytes = -1);
    void advise(occa::memory mem,
                advice_t advice,
                occa::device device);
    void advise(occa::memory mem,
                advice_t advice,
                const dim_t bytes,
                occa::device device);

    void prefetch(occa::memory mem,
                  const dim_t bytes = -1);
    void prefetch(occa::memory mem,
                  occa::device device);
    void prefetch(occa::memory mem,
                  const dim_t bytes,
                  occa::device device);

    hipCtx_t getContext(occa::device device);

    void* getMappedPtr(occa::memory mem);

    occa::device wrapDevice(hipDevice_t device,
                            hipCtx_t context,
                            const occa::properties &props = occa::properties());

    occa::memory wrapMemory(occa::device device,
                            void *ptr,
                            const udim_t bytes,
                            const occa::properties &props = occa::properties());

    hipEvent_t& event(streamTag &tag);
    const hipEvent_t& event(const streamTag &tag);

    void warn(hipError_t errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message);

    void error(hipError_t errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message);

    std::string getErrorMessage(const hipError_t errorCode);
  }
}

#  endif
#endif
