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

    memoryPool::~memoryPool() {
      free(ptr);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new serial::memory(this, bytes, offset);
    }

    void memoryPool::malloc(char* &ptr_, const udim_t bytes) {
      ptr_ = (char*) sys::malloc(bytes);
    }

    void memoryPool::memcpy(char* dst, const char* src,
                            const udim_t bytes) {
      ::memcpy(dst, src, bytes);
    }

    void memoryPool::free(char* &ptr_) {
      if (ptr_) {
        sys::free(ptr_);
      }
      ptr_=nullptr;
    }

    void memoryPool::resize(const udim_t bytes) {

      OCCA_ERROR("Cannot resize memoryPool below current usage"
                 "(reserved: " << reserved << ", bytes: " << bytes << ")",
                 reserved<=bytes);

      if (reservations.size()==0) {
        free(ptr);

        modeDevice->bytesAllocated -= size;

        malloc(ptr, bytes);
        size=bytes;

        modeDevice->bytesAllocated += bytes;
        modeDevice->maxBytesAllocated = std::max(
          modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
        );
      } else {
        char* newPtr=nullptr;
        malloc(newPtr, bytes);

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
        do {
          it++;
          if (it==reservations.end()) {
            memcpy(newPtr+offset, ptr+lo, hi-lo);
          } else {
            m = dynamic_cast<memory*>(*it);
            const dim_t mlo = m->offset;
            const dim_t mhi = m->offset+m->size;
            if (mlo>hi) {
              memcpy(newPtr+offset, ptr+lo, hi-lo);
              offset+=hi-lo;
              lo=mlo;
              hi=mhi;
            } else {
              hi = std::max(hi, mhi);
            }
            m->offset -= lo-offset;
            m->ptr = newPtr + m->offset;
          }
        } while (it!=reservations.end());

        free(ptr);
        modeDevice->bytesAllocated -= size;

        ptr = newPtr;
        size=bytes;
      }
    }

    void memoryPool::detach() {
      ptr = NULL;
      size=0;
    }
  }
}
