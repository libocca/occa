#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/buffer.hpp>
#include <occa/internal/modes/cuda/memoryPool.hpp>
#include <occa/internal/modes/cuda/memory.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_) {}

    CUstream& memoryPool::getCuStream() const {
      return dynamic_cast<device*>(modeDevice)->getCuStream();
    }

    modeBuffer_t* memoryPool::makeBuffer() {
      return new cuda::buffer(modeDevice, 0, properties);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new cuda::memory(this, bytes, offset);
    }

    void memoryPool::setPtr(modeMemory_t* mem, modeBuffer_t* buf,
                            const dim_t offset) {

      cuda::memory* m = dynamic_cast<cuda::memory*>(mem);
      cuda::buffer* b = dynamic_cast<cuda::buffer*>(buf);

      m->offset = offset;
      if (b->isUnified || b->useHostPtr) {
        m->ptr = b->ptr + offset;
      }
      if (b->isUnified || !b->useHostPtr) {
        m->cuPtr = b->cuPtr + offset;
      }
    }

    void memoryPool::memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                            modeBuffer_t* src, const dim_t srcOffset,
                            const udim_t bytes) {

      cuda::buffer* dstBuf = dynamic_cast<cuda::buffer*>(dst);
      cuda::buffer* srcBuf = dynamic_cast<cuda::buffer*>(src);

      if (srcBuf->useHostPtr) {
        ::memcpy(dstBuf->ptr + dstOffset,
                 srcBuf->ptr + srcOffset,
                 bytes);
      } else {
        OCCA_CUDA_ERROR("Memory: Async Copy From",
                        cuMemcpyDtoDAsync(dstBuf->cuPtr + dstOffset,
                                          srcBuf->cuPtr + srcOffset,
                                          bytes,
                                          getCuStream()));
      }
    }
  }
}
