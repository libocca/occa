#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/buffer.hpp>
#include <occa/internal/modes/cuda/memoryPool.hpp>
#include <occa/internal/modes/cuda/memory.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_),
      cuPtr(reinterpret_cast<CUdeviceptr>(ptr)),
      isUnified(false),
      useHostPtr(false) {

      if (properties.get("host", false)) {
        useHostPtr=true;
      } else if (properties.get("unified", false)) {
        isUnified = true;
      }
    }

    memoryPool::~memoryPool() {
      free(cuPtr, ptr);
      useHostPtr = false;
      isUnified = false;
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new cuda::memory(this, bytes, offset);
    }

    CUstream& memoryPool::getCuStream() const {
      return dynamic_cast<device*>(modeDevice)->getCuStream();
    }

    void memoryPool::malloc(CUdeviceptr &cuPtr_, char* &ptr_,
                            const udim_t bytes) {
      if (useHostPtr) {

        OCCA_CUDA_ERROR("Device: malloc host",
                        cuMemAllocHost((void**) &(ptr_), bytes));
        OCCA_CUDA_ERROR("Device: get device pointer from host",
                        cuMemHostGetDevicePointer(&(cuPtr_),
                                                  ptr_,
                                                  0));

      } else if (isUnified) {

#if CUDA_VERSION >= 8000
        const unsigned int flags = (properties.get("attached_host", false) ?
                                    CU_MEM_ATTACH_HOST : CU_MEM_ATTACH_GLOBAL);

        OCCA_CUDA_ERROR("Device: Unified alloc",
                        cuMemAllocManaged(&(cuPtr_),
                                          bytes,
                                          flags));
#else
        OCCA_FORCE_ERROR("CUDA version ["
                         << cuda::getVersion()
                         << "] does not support unified memory allocation");
#endif
        ptr_ = (char*) cuPtr_;

      } else {

        OCCA_CUDA_ERROR("Device: malloc",
                        cuMemAlloc(&(cuPtr_), bytes));

      }
    }

    void memoryPool::memcpy(CUdeviceptr cuDst, char* dst,
                            const CUdeviceptr cuSrc, const char* src,
                            const udim_t bytes) {
      if (useHostPtr) {
        ::memcpy(dst, src, bytes);
      } else {
        OCCA_CUDA_ERROR("Memory: Async Copy From",
                        cuMemcpyDtoDAsync(cuDst,
                                          cuSrc,
                                          bytes,
                                          getCuStream()));
      }
    }

    void memoryPool::free(CUdeviceptr &cuPtr_, char* &ptr_) {
      if (useHostPtr && ptr_) {
        OCCA_CUDA_DESTRUCTOR_ERROR(
          "Device: hostFree()",
          cuMemFreeHost(ptr_)
        );
      } else if (cuPtr_) {
        cuMemFree(cuPtr_);
      }
      ptr_=nullptr;
      cuPtr_=0;
    }

    void memoryPool::resize(const udim_t bytes) {

      OCCA_ERROR("Cannot resize memoryPool below current usage"
                 "(reserved: " << reserved << ", bytes: " << bytes << ")",
                 reserved<=bytes);

      if (reservations.size()==0) {
        free(cuPtr, ptr);

        modeDevice->bytesAllocated -= size;

        malloc(cuPtr, ptr, bytes);
        size=bytes;

        modeDevice->bytesAllocated += bytes;
        modeDevice->maxBytesAllocated = std::max(
          modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
        );
      } else {

        CUdeviceptr newCuPtr=0;
        char* newPtr=nullptr;
        malloc(newCuPtr, newPtr, bytes);

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
        m->cuPtr = newCuPtr;
        do {
          it++;
          if (it==reservations.end()) {
            memcpy(newCuPtr+offset, newPtr+offset,
                   cuPtr+lo, ptr+lo, hi-lo);
          } else {
            m = dynamic_cast<memory*>(*it);
            const dim_t mlo = m->offset;
            const dim_t mhi = m->offset+m->size;
            if (mlo>hi) {
              memcpy(newCuPtr+offset, newPtr+offset,
                     cuPtr+lo, ptr+lo, hi-lo);

              offset+=hi-lo;
              lo=mlo;
              hi=mhi;
            } else {
              hi = std::max(hi, mhi);
            }
            m->offset -= lo-offset;
            m->ptr = newPtr + m->offset;
            m->cuPtr = newCuPtr + m->offset;
          }
        } while (it!=reservations.end());

        free(cuPtr, ptr);
        modeDevice->bytesAllocated -= size;

        ptr = newPtr;
        cuPtr = newCuPtr;
        size=bytes;
      }
    }

    void memoryPool::detach() {
      ptr = nullptr;
      cuPtr = 0;
      size = 0;
      useHostPtr = false;
      isUnified = false;
      isWrapped = false;
    }
  }
}
