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

    memoryPool::~memoryPool() {
      free(ptr);
      size = 0;
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new dpcpp::memory(this, bytes, offset);
    }

    void memoryPool::malloc(char* &ptr_, const udim_t bytes) {
      dpcpp::device *device = reinterpret_cast<dpcpp::device*>(modeDevice);

      if (properties.get("host", false)) {
        ptr_ = static_cast<char *>(::sycl::malloc_host(bytes,
                                                       device->dpcppContext));
        OCCA_ERROR("DPCPP: malloc_host failed!", nullptr != ptr_);
      } else if (properties.get("unified", false)) {
        ptr_ = static_cast<char *>(::sycl::malloc_shared(bytes,
                                                         device->dpcppDevice,
                                                         device->dpcppContext));
        OCCA_ERROR("DPCPP: malloc_shared failed!", nullptr != ptr_);
      } else {
        ptr_ = static_cast<char *>(::sycl::malloc_device(bytes,
                                                         device->dpcppDevice,
                                                         device->dpcppContext));
        OCCA_ERROR("DPCPP: malloc_device failed!", nullptr != ptr_);
      }
    }

    void memoryPool::memcpy(char* dst, const char* src,
                            const udim_t bytes) {
      occa::dpcpp::stream& q = getDpcppStream(modeDevice->currentStream);
      occa::dpcpp::streamTag e = q.memcpy(dst, src, bytes);
    }

    void memoryPool::free(char* &ptr_) {
      if (ptr_) {
        auto& dpcpp_device = getDpcppDevice(modeDevice);
        OCCA_DPCPP_ERROR("Memory: Freeing SYCL alloc'd memory",
                         ::sycl::free(ptr_,dpcpp_device.dpcppContext));
      }
      ptr_ = nullptr;
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
        size = bytes;
      }
    }

    void memoryPool::detach() {
      ptr = nullptr;
      size = 0;
    }
  }
}
