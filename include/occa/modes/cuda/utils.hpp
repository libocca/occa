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

#if OCCA_CUDA_ENABLED
#  ifndef OCCA_CUDA_UTILS_HEADER
#  define OCCA_CUDA_UTILS_HEADER

#include <cuda.h>

#include "occa/device.hpp"

namespace occa {
  namespace cuda {
#if CUDA_VERSION >= 8000
    typedef CUmem_advise advice_t ;
#else
    typedef int advice_t ;
#endif

    void init();

    int getDeviceCount();
    CUdevice getDevice(const int id);
    udim_t getDeviceMemorySize(CUdevice device);

    std::string getVersion();

    void enablePeerToPeer(CUcontext context);
    void checkPeerToPeer(CUdevice destDevice,
                         CUdevice srcDevice);

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const udim_t bytes,
                          CUstream usingStream);


    void asyncPeerToPeerMemcpy(CUdevice destDevice,
                               CUcontext destContext,
                               CUdeviceptr destMemory,
                               CUdevice srcDevice,
                               CUcontext srcContext,
                               CUdeviceptr srcMemory,
                               const udim_t bytes,
                               CUstream usingStream);

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const udim_t bytes,
                          CUstream usingStream,
                          const bool isAsync);

    void advise(occa::memory mem, advice_t advice, const dim_t bytes = -1);
    void advise(occa::memory mem, advice_t advice, occa::device device);
    void advise(occa::memory mem, advice_t advice, const dim_t bytes, occa::device device);

    void prefetch(occa::memory mem, const dim_t bytes = -1);

    occa::device wrapDevice(CUdevice device,
                            CUcontext context,
                            const occa::properties &props);

    CUevent& event(streamTag &tag);
    const CUevent& event(const streamTag &tag);

    void warn(CUresult errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message);

    void error(CUresult errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message);

    std::string getErrorMessage(const CUresult errorCode);
  }
}

#  endif
#endif
