#include <occa/internal/modes/metal/device.hpp>
#include <occa/internal/modes/metal/buffer.hpp>
#include <occa/internal/modes/metal/memory.hpp>
#include <occa/internal/modes/metal/memoryPool.hpp>

namespace occa {
  namespace metal {
    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_) {}

    memoryPool::~memoryPool() {
      free(metalBuffer, ptr);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new metal::memory(this, bytes, offset);
    }

    void memoryPool::malloc(api::metal::buffer_t &metalBuffer_,
                            char* &ptr_, const udim_t bytes) {
      metalBuffer_ = dynamic_cast<metal::device*>(modeDevice)
                      ->metalDevice.malloc(bytes, NULL);
    }

    void memoryPool::memcpy(api::metal::buffer_t &dstBuffer,
                            const udim_t dstOffset,
                            const api::metal::buffer_t &srcBuffer,
                            const udim_t srcOffset,
                            const udim_t bytes) {

      const bool async = true;

      api::metal::commandQueue_t &metalCommandQueue = (
        ((metal::device*) modeDevice)->metalCommandQueue
      );
      metalCommandQueue.memcpy(dstBuffer,
                               dstOffset,
                               srcBuffer,
                               srcOffset,
                               bytes,
                               async);
    }

    void memoryPool::free(api::metal::buffer_t &metalBuffer_,
                          char* &ptr_) {
      if (metalBuffer_.bufferObj) {
        metalBuffer_.free();
      }
      ptr_=nullptr;
    }

    void memoryPool::resize(const udim_t bytes) {

      OCCA_ERROR("Cannot resize memoryPool below current usage"
                 "(reserved: " << reserved << ", bytes: " << bytes << ")",
                 reserved<=bytes);

      if (reservations.size()==0) {
        free(metalBuffer, ptr);

        modeDevice->bytesAllocated -= size;

        malloc(metalBuffer, ptr, bytes);
        size=bytes;

        modeDevice->bytesAllocated += bytes;
        modeDevice->maxBytesAllocated = std::max(
          modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
        );
      } else {

        api::metal::buffer_t newMetalBuffer=0;
        char* newPtr=nullptr;
        malloc(newMetalBuffer, newPtr, bytes);

        modeDevice->bytesAllocated += bytes;
        modeDevice->maxBytesAllocated = std::max(
          modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
        );

        auto it = reservations.begin();
        memory* m = dynamic_cast<memory*>(*it);
        dim_t lo = m->offset;
        dim_t hi = lo + m->size;
        dim_t offset=0;
        m->offset=0;
        m->ptr = newPtr;
        m->metalBuffer = newMetalBuffer;
        m->bufferOffset = offset;
        do {
          it++;
          if (it==reservations.end()) {
            memcpy(newMetalBuffer, offset,
                   metalBuffer, lo, hi-lo);
          } else {
            m = dynamic_cast<memory*>(*it);
            const dim_t mlo = m->offset;
            const dim_t mhi = m->offset+m->size;
            if (mlo>hi) {
              memcpy(newMetalBuffer, offset,
                     metalBuffer, lo, hi-lo);

              offset+=hi-lo;
              lo=mlo;
              hi=mhi;
            } else {
              hi = std::max(hi, mhi);
            }
            m->offset -= lo-offset;
            m->ptr = newPtr + m->offset;
            m->metalBuffer = newMetalBuffer;
            m->bufferOffset = m->offset;
          }
        } while (it!=reservations.end());

        free(metalBuffer, ptr);
        modeDevice->bytesAllocated -= size;

        ptr = newPtr;
        metalBuffer = newMetalBuffer;
        size=bytes;
      }
    }

    void memoryPool::detach() {
      metalBuffer.bufferObj = NULL;
      size = 0;
      isWrapped = false;
    }
  }
}
