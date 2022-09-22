#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/buffer.hpp>
#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/modes/dpcpp/memoryPool.hpp>
#include <occa/internal/modes/dpcpp/stream.hpp>
#include <occa/internal/modes/dpcpp/streamTag.hpp>

namespace occa {
  namespace dpcpp {
    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_) {}

    modeBuffer_t* memoryPool::makeBuffer() {
      return new dpcpp::buffer(modeDevice, 0, properties);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new dpcpp::memory(this, bytes, offset);
    }

    void memoryPool::setPtr(modeMemory_t* mem, modeBuffer_t* buf,
                            const dim_t offset) {
      mem->offset = offset;
      mem->ptr = buf->ptr + offset;
    }

    void memoryPool::memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                            modeBuffer_t* src, const dim_t srcOffset,
                            const udim_t bytes) {

      occa::dpcpp::stream& q = getDpcppStream(modeDevice->currentStream);
      occa::dpcpp::streamTag e = q.memcpy(dst->ptr + dstOffset,
                                          src->ptr + srcOffset,
                                          bytes);
    }
  }
}
