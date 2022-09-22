#include <occa/internal/modes/metal/device.hpp>
#include <occa/internal/modes/metal/buffer.hpp>
#include <occa/internal/modes/metal/memory.hpp>
#include <occa/internal/modes/metal/memoryPool.hpp>

namespace occa {
  namespace metal {
    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_) {}

    modeBuffer_t* memoryPool::makeBuffer() {
      return new metal::buffer(modeDevice, 0, properties);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new metal::memory(this, bytes, offset);
    }

    void memoryPool::setPtr(modeMemory_t* mem, modeBuffer_t* buf,
                            const dim_t offset) {

      metal::memory* m = dynamic_cast<metal::memory*>(mem);
      metal::buffer* b = dynamic_cast<metal::buffer*>(buf);

      m->offset = offset;
      m->metalBuffer = b->metalBuffer;
      m->ptr = (char*) b->metalBuffer.getPtr();
    }

    void memoryPool::memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                            modeBuffer_t* src, const dim_t srcOffset,
                            const udim_t bytes) {

      metal::buffer* dstBuf = dynamic_cast<metal::buffer*>(dst);
      metal::buffer* srcBuf = dynamic_cast<metal::buffer*>(src);

      const bool async = true;

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) modeDevice)->metalCommandQueue
      );
      metalCommandQueue.memcpy(dstBuf->metalBuffer,
                               dstOffset,
                               srcBuf->metalBuffer,
                               srcOffset,
                               bytes,
                               async);
    }
  }
}
