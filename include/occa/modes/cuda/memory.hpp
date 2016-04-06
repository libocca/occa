/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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
#  ifndef OCCA_CUDA_MEMORY_HEADER
#  define OCCA_CUDA_MEMORY_HEADER

#include "occa/defines.hpp"
#include "occa/memory.hpp"

#include <cuda.h>

namespace occa {
  namespace cuda {
    class memory : public occa::memory_v {
    public:
      memory();
      memory(const memory &m);
      memory& operator = (const memory &m);
      ~memory();

      void* getMemoryHandle();
      kernelArg makeKernelArg() const;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset,
                    const bool async);

      void copyFrom(const memory_v *src,
                    const udim_t bytes,
                    const udim_t destOffset,
                    const udim_t srcOffset,
                    const bool async);

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset,
                  const bool async);

      void copyTo(memory_v *dest,
                  const udim_t bytes,
                  const udim_t srcOffset,
                  const udim_t offset,
                  const bool async);

      void free();
      void detach();
    };
  };

    struct CUDADeviceData_t {
      CUdevice  device;
      CUcontext context;
      bool p2pEnabled;
    };

    struct CUDATextureData_t {
      CUarray array;
      CUsurfObject surface;
    };
    //==================================


    memory_t<CUDA>::memory_t();

    memory_t<CUDA>::memory_t(const memory_t &m);

    memory_t<CUDA>& memory_t<CUDA>::operator = (const memory_t &m);

    void* memory_t<CUDA>::getMemoryHandle();

    void* memory_t<CUDA>::getTextureHandle();

    void memory_t<CUDA>::copyFrom(const void *src,
                                  const udim_t bytes,
                                  const udim_t offset);

    void memory_t<CUDA>::copyFrom(const memory_v *src,
                                  const udim_t bytes,
                                  const udim_t destOffset,
                                  const udim_t srcOffset);

    void memory_t<CUDA>::copyTo(void *dest,
                                const udim_t bytes,
                                const udim_t offset);

    void memory_t<CUDA>::copyTo(memory_v *dest,
                                const udim_t bytes,
                                const udim_t destOffset,
                                const udim_t srcOffset);

    void memory_t<CUDA>::asyncCopyFrom(const void *src,
                                       const udim_t bytes,
                                       const udim_t offset);

    void memory_t<CUDA>::asyncCopyFrom(const memory_v *src,
                                       const udim_t bytes,
                                       const udim_t destOffset,
                                       const udim_t srcOffset);

    void memory_t<CUDA>::asyncCopyTo(void *dest,
                                     const udim_t bytes,
                                     const udim_t offset);

    void memory_t<CUDA>::asyncCopyTo(memory_v *dest,
                                     const udim_t bytes,
                                     const udim_t destOffset,
                                     const udim_t srcOffset);

    void memory_t<CUDA>::mappedFree();

    void memory_t<CUDA>::free();
  }
}

#  endif
#endif
