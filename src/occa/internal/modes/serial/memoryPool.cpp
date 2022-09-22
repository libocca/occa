#include <cstring>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/modes/serial/memoryPool.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace serial {
    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_) {}

    modeBuffer_t* memoryPool::makeBuffer() {
      return new serial::buffer(modeDevice, 0, properties);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new serial::memory(this, bytes, offset);
    }

    void memoryPool::setPtr(modeMemory_t* mem, modeBuffer_t* buf,
                            const dim_t offset) {
      mem->offset = offset;
      mem->ptr = buf->ptr + offset;
    }

    void memoryPool::memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                            modeBuffer_t* src, const dim_t srcOffset,
                            const udim_t bytes) {
      ::memcpy(dst->ptr + dstOffset,
               src->ptr + srcOffset,
               bytes);
    }
  }
}
