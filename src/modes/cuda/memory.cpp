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

#include "occa/defines.hpp"

#if OCCA_CUDA_ENABLED

#include "occa/CUDA.hpp"

namespace occa {
  namespace cuda {
    memory::memory() {
      strMode = "CUDA";

      memInfo = memFlag::none;

      handle    = NULL;
      mappedPtr = NULL;
      uvaPtr    = NULL;

      dHandle = NULL;
      size    = 0;

      textureInfo.arg = NULL;

      textureInfo.dim = 1;

      textureInfo.w  = textureInfo.h = textureInfo.d = 0;
    }

    memory::memory(const memory &m) {
      *this = m;
    }

    memory& memory::operator = (const memory &m) {
      memInfo = m.memInfo;

      handle    = m.handle;
      mappedPtr = m.mappedPtr;
      uvaPtr    = m.uvaPtr;

      dHandle = m.dHandle;
      size    = m.size;

      textureInfo.arg = m.textureInfo.arg;

      textureInfo.dim = m.textureInfo.dim;

      textureInfo.w = m.textureInfo.w;
      textureInfo.h = m.textureInfo.h;
      textureInfo.d = m.textureInfo.d;

      return *this;
    }

    memory::~memory() {}

    void* memory::getMemoryHandle() {
      return handle;
    }

    void* memory::getTextureHandle() {
      return (void*) ((CUDATextureData_t*) handle)->array;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset) {
      if (!isATexture()) {
        OCCA_CUDA_CHECK("Memory: Copy From",
                        cuMemcpyHtoD(*((CUdeviceptr*) handle) + offset, src, bytes) );
      } else {
        if (textureInfo.dim == 1) {
          OCCA_CUDA_CHECK("Texture Memory: Copy From",
                          cuMemcpyHtoA(((CUDATextureData_t*) handle)->array, offset, src, bytes) );
        } else {
          CUDA_MEMCPY2D info;

          info.srcXInBytes   = 0;
          info.srcY          = 0;
          info.srcMemoryType = CU_MEMORYTYPE_HOST;
          info.srcHost       = src;
          info.srcPitch      = 0;

          info.dstXInBytes   = offset;
          info.dstY          = 0;
          info.dstMemoryType = CU_MEMORYTYPE_ARRAY;
          info.dstArray      = ((CUDATextureData_t*) handle)->array;

          info.WidthInBytes = textureInfo.w * textureInfo.bytesInEntry;
          info.Height       = (bytes / info.WidthInBytes);

          cuMemcpy2D(&info);

          dHandle->finish();
        }
      }
    }

    void memory::copyFrom(const memory_v *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset) {
      void *dstPtr, *srcPtr;

      if (!isATexture()) {
        dstPtr = handle;
      } else {
        dstPtr = (void*) ((CUDATextureData_t*) handle)->array;
      }
      if ( !(src->isATexture()) ) {
        srcPtr = src->handle;
      } else {
        srcPtr = (void*) ((CUDATextureData_t*) src->handle)->array;
      }
      if (!isATexture()) {
        if (!src->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Copy From [Memory -> Memory]",
                          cuMemcpyDtoD(*((CUdeviceptr*) dstPtr) + destOffset,
                                       *((CUdeviceptr*) srcPtr) + srcOffset,
                                       bytes) );
        } else {
          OCCA_CUDA_CHECK("Memory: Copy From [Texture -> Memory]",
                          cuMemcpyAtoD(*((CUdeviceptr*) dstPtr) + destOffset,
                                       (CUarray) srcPtr         , srcOffset,
                                       bytes) );
        }
      } else {
        if (!src->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Copy From [Memory -> Texture]",
                          cuMemcpyDtoA((CUarray) dstPtr         , destOffset,
                                       *((CUdeviceptr*) srcPtr) + srcOffset,
                                       bytes) );
        } else {
          OCCA_CUDA_CHECK("Memory: Copy From [Texture -> Texture]",
                          cuMemcpyAtoA((CUarray) dstPtr, destOffset,
                                       (CUarray) srcPtr, srcOffset,
                                       bytes) );
        }
      }
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset) {
      if (!isATexture()) {
        OCCA_CUDA_CHECK("Memory: Copy To",
                        cuMemcpyDtoH(dest, *((CUdeviceptr*) handle) + offset, bytes) );
      } else {
        if (textureInfo.dim == 1) {
          OCCA_CUDA_CHECK("Texture Memory: Copy To",
                          cuMemcpyAtoH(dest, ((CUDATextureData_t*) handle)->array, offset, bytes) );
        } else {
          CUDA_MEMCPY2D info;

          info.srcXInBytes   = offset;
          info.srcY          = 0;
          info.srcMemoryType = CU_MEMORYTYPE_ARRAY;
          info.srcArray      = ((CUDATextureData_t*) handle)->array;

          info.dstXInBytes   = 0;
          info.dstY          = 0;
          info.dstMemoryType = CU_MEMORYTYPE_HOST;
          info.dstHost       = dest;
          info.dstPitch      = 0;

          info.WidthInBytes = textureInfo.w * textureInfo.bytesInEntry;
          info.Height       = (bytes / info.WidthInBytes);

          cuMemcpy2D(&info);

          dHandle->finish();
        }
      }
    }

    void memory::copyTo(memory_v *dest,
                        const udim_t bytes,
                        const udim_t destOffset,
                        const udim_t srcOffset) {
      void *dstPtr, *srcPtr;

      if (!isATexture()) {
        srcPtr = handle;
      } else {
        srcPtr = (void*) ((CUDATextureData_t*) handle)->array;
      }
      if ( !(dest->isATexture()) ) {
        dstPtr = dest->handle;
      } else {
        dstPtr = (void*) ((CUDATextureData_t*) dest->handle)->array;
      }
      if (!isATexture()) {
        if (!dest->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Copy To [Memory -> Memory]",
                          cuMemcpyDtoD(*((CUdeviceptr*) dstPtr) + destOffset,
                                       *((CUdeviceptr*) srcPtr) + srcOffset,
                                       bytes) );
        } else {
          OCCA_CUDA_CHECK("Memory: Copy To [Memory -> Texture]",
                          cuMemcpyDtoA((CUarray) dstPtr         , destOffset,
                                       *((CUdeviceptr*) srcPtr) + srcOffset,
                                       bytes) );
        }
      } else {
        if (dest->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Copy To [Texture -> Memory]",
                          cuMemcpyAtoD(*((CUdeviceptr*) dstPtr) + destOffset,
                                       (CUarray) srcPtr         , srcOffset,
                                       bytes) );
        } else {
          OCCA_CUDA_CHECK("Memory: Copy To [Texture -> Texture]",
                          cuMemcpyAtoA((CUarray) dstPtr, destOffset,
                                       (CUarray) srcPtr, srcOffset,
                                       bytes) );
        }
      }
    }

    void memory::asyncCopyFrom(const void *src,
                               const udim_t bytes,
                               const udim_t offset) {
      const CUstream &stream = *((CUstream*) dHandle->currentStream);

      if (!isATexture()) {
        OCCA_CUDA_CHECK("Memory: Asynchronous Copy From",
                        cuMemcpyHtoDAsync(*((CUdeviceptr*) handle) + offset, src, bytes, stream) );
      } else {
        OCCA_CUDA_CHECK("Texture Memory: Asynchronous Copy From",
                        cuMemcpyHtoAAsync(((CUDATextureData_t*) handle)->array, offset, src, bytes, stream) );
      }
    }

    void memory::asyncCopyFrom(const memory_v *src,
                               const udim_t bytes,
                               const udim_t destOffset,
                               const udim_t srcOffset) {
      const CUstream &stream = *((CUstream*) dHandle->currentStream);
      void *dstPtr, *srcPtr;

      if (!isATexture()) {
        dstPtr = handle;
      } else {
        dstPtr = (void*) ((CUDATextureData_t*) handle)->array;
      }
      if ( !(src->isATexture()) ) {
        srcPtr = src->handle;
      } else {
        srcPtr = (void*) ((CUDATextureData_t*) src->handle)->array;
      }
      if (!isATexture()) {
        if (!src->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy From [Memory -> Memory]",
                          cuMemcpyDtoDAsync(*((CUdeviceptr*) dstPtr) + destOffset,
                                            *((CUdeviceptr*) srcPtr) + srcOffset,
                                            bytes, stream) );
        } else {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy From [Texture -> Memory]",
                          cuMemcpyAtoD(*((CUdeviceptr*) dstPtr) + destOffset,
                                       (CUarray) srcPtr         , srcOffset,
                                       bytes) );
        }
      } else {
        if (src->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy From [Memory -> Texture]",
                          cuMemcpyDtoA((CUarray) dstPtr         , destOffset,
                                       *((CUdeviceptr*) srcPtr) + srcOffset,
                                       bytes) );
        } else {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy From [Texture -> Texture]",
                          cuMemcpyAtoA((CUarray) dstPtr, destOffset,
                                       (CUarray) srcPtr, srcOffset,
                                       bytes) );
        }
      }
    }

    void memory::asyncCopyTo(void *dest,
                             const udim_t bytes,
                             const udim_t offset) {
      const CUstream &stream = *((CUstream*) dHandle->currentStream);

      if (!isATexture()) {
        OCCA_CUDA_CHECK("Memory: Asynchronous Copy To",
                        cuMemcpyDtoHAsync(dest, *((CUdeviceptr*) handle) + offset, bytes, stream) );
      } else {
        OCCA_CUDA_CHECK("Texture Memory: Asynchronous Copy To",
                        cuMemcpyAtoHAsync(dest,((CUDATextureData_t*) handle)->array, offset, bytes, stream) );
      }
    }

    void memory::asyncCopyTo(memory_v *dest,
                             const udim_t bytes,
                             const udim_t destOffset,
                             const udim_t srcOffset) {
      const CUstream &stream = *((CUstream*) dHandle->currentStream);
      void *dstPtr, *srcPtr;

      if (!isATexture()) {
        srcPtr = handle;
      } else {
        srcPtr = (void*) ((CUDATextureData_t*) handle)->array;
      }
      if ( !(dest->isATexture()) ) {
        dstPtr = dest->handle;
      } else {
        dstPtr = (void*) ((CUDATextureData_t*) dest->handle)->array;
      }
      if (!isATexture()) {
        if (!dest->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy To [Memory -> Memory]",
                          cuMemcpyDtoDAsync(*((CUdeviceptr*) dstPtr) + destOffset,
                                            *((CUdeviceptr*) srcPtr) + srcOffset,
                                            bytes, stream) );
        } else {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy To [Memory -> Texture]",
                          cuMemcpyDtoA((CUarray) dstPtr         , destOffset,
                                       *((CUdeviceptr*) srcPtr) + srcOffset,
                                       bytes) );
        }
      } else {
        if (dest->isATexture()) {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy To [Texture -> Memory]",
                          cuMemcpyAtoD(*((CUdeviceptr*) dstPtr) + destOffset,
                                       (CUarray) srcPtr         , srcOffset,
                                       bytes) );
        } else {
          OCCA_CUDA_CHECK("Memory: Asynchronous Copy To [Texture -> Texture]",
                          cuMemcpyAtoA((CUarray) dstPtr, destOffset,
                                       (CUarray) srcPtr, srcOffset,
                                       bytes) );
        }
      }
    }

    void memory::mappedFree() {
      if (isMapped()) {
        OCCA_CUDA_CHECK("Device: mappedFree()",
                        cuMemFreeHost(mappedPtr));

        delete (CUdeviceptr*) handle;

        size = 0;
      }
    }

    void memory::free() {
      if (!isATexture()) {
        cuMemFree(*((CUdeviceptr*) handle));
        delete (CUdeviceptr*) handle;
      } else {
        CUarray &array        = ((CUDATextureData_t*) handle)->array;
        CUsurfObject &surface = ((CUDATextureData_t*) handle)->surface;

        cuArrayDestroy(array);
        cuSurfObjectDestroy(surface);

        delete (CUDATextureData_t*) handle;
        delete (CUaddress_mode*)    textureInfo.arg;
      }

      size = 0;
    }

    void memory::detach() {
      if (!isATexture()) {
        cuMemFree(*((CUdeviceptr*) handle));
        delete (CUdeviceptr*) handle;
      } else {
        delete (CUDATextureData_t*) handle;
        delete (CUaddress_mode*)    textureInfo.arg;
      }

      size = 0;
    }
  }
}

#endif
