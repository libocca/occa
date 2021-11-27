#include <occa/internal/modes/hip/device.hpp>
#include <occa/internal/modes/hip/buffer.hpp>
#include <occa/internal/modes/hip/memory.hpp>
#include <occa/internal/modes/hip/memoryPool.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    inline hipDeviceptr_t addHipPtrOffset(hipDeviceptr_t hipPtr, const udim_t offset) {
      return (hipDeviceptr_t) (((char*) hipPtr) + offset);
    }

    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_),
      hipPtr(reinterpret_cast<hipDeviceptr_t>(ptr)),
      useHostPtr(false) {

      if (properties.get("host", false)) {
        useHostPtr=true;
      }
    }

    memoryPool::~memoryPool() {
      free(hipPtr, ptr);
      useHostPtr = false;
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new hip::memory(this, bytes, offset);
    }

    hipStream_t& memoryPool::getHipStream() const {
      return dynamic_cast<device*>(modeDevice)->getHipStream();
    }

    void memoryPool::malloc(hipDeviceptr_t &hipPtr_, char* &ptr_,
                            const udim_t bytes) {

      if (useHostPtr) {

        OCCA_HIP_ERROR("Device: malloc host",
                       hipHostMalloc((void**) &(ptr_), bytes));
        OCCA_HIP_ERROR("Device: get device pointer from host",
                       hipHostGetDevicePointer((void**) &(hipPtr_),
                                               ptr_,
                                               0));

      } else  {

        OCCA_HIP_ERROR("Device: malloc",
                       hipMalloc((void**) &(hipPtr_), bytes));

      }
    }

    void memoryPool::memcpy(hipDeviceptr_t hipDst, char* dst,
                            const hipDeviceptr_t hipSrc, const char* src,
                            const udim_t bytes) {
      if (useHostPtr) {
        ::memcpy(dst, src, bytes);
      } else {
        OCCA_HIP_ERROR("Memory: Async Copy From",
                       hipMemcpyDtoDAsync(hipDst,
                                          hipSrc,
                                          bytes,
                                          getHipStream()));
      }
    }

    void memoryPool::free(hipDeviceptr_t &hipPtr_, char* &ptr_) {
      if (useHostPtr && ptr) {
        OCCA_HIP_ERROR("Device: hostFree()",
                         hipHostFree(ptr_));
      } else if (hipPtr_) {
        hipFree((void*) hipPtr_);
      }
      ptr_=nullptr;
      hipPtr_=0;
    }

    void memoryPool::resize(const udim_t bytes) {

      OCCA_ERROR("Cannot resize memoryPool below current usage"
                 "(reserved: " << reserved << ", bytes: " << bytes << ")",
                 reserved<=bytes);

      if (reservations.size()==0) {
        free(hipPtr, ptr);

        modeDevice->bytesAllocated -= size;

        malloc(hipPtr, ptr, bytes);
        size=bytes;

        modeDevice->bytesAllocated += bytes;
        modeDevice->maxBytesAllocated = std::max(
          modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
        );
      } else {

        hipDeviceptr_t newHipPtr=0;
        char* newPtr=nullptr;
        malloc(newHipPtr, newPtr, bytes);

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
        m->hipPtr = newHipPtr;
        do {
          it++;
          if (it==reservations.end()) {
            memcpy(addHipPtrOffset(newHipPtr, offset), newPtr+offset,
                   addHipPtrOffset(hipPtr, lo), ptr+lo, hi-lo);
          } else {
            m = dynamic_cast<memory*>(*it);
            const dim_t mlo = m->offset;
            const dim_t mhi = m->offset+m->size;
            if (mlo>hi) {
              memcpy(addHipPtrOffset(newHipPtr, offset), newPtr+offset,
                     addHipPtrOffset(hipPtr, lo), ptr+lo, hi-lo);

              offset+=hi-lo;
              lo=mlo;
              hi=mhi;
            } else {
              hi = std::max(hi, mhi);
            }
            m->offset -= lo-offset;
            m->ptr = newPtr + m->offset;
            m->hipPtr = addHipPtrOffset(newHipPtr, m->offset);
          }
        } while (it!=reservations.end());

        free(hipPtr, ptr);
        modeDevice->bytesAllocated -= size;

        ptr = newPtr;
        hipPtr = newHipPtr;
        size=bytes;
      }
    }

    void memoryPool::detach() {
      ptr = nullptr;
      hipPtr = NULL;
      size = 0;
      useHostPtr = false;
      isWrapped = false;
    }
  }
}
