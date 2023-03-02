#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/buffer.hpp>
#include <occa/internal/modes/cuda/memory.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    memory::memory(buffer *b,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(b, size_, offset_) {
      isUnified = b->isUnified;
      useHostPtr = b->useHostPtr;
      if (isUnified || useHostPtr) {
        ptr = b->ptr + offset;
      }
      if (isUnified || !useHostPtr) {
        cuPtr = b->cuPtr + offset;
      }
    }

    memory::memory(memoryPool *memPool,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(memPool, size_, offset_) {
      cuda::buffer* b = dynamic_cast<cuda::buffer*>(memPool->buffer);
      isUnified = b->isUnified;
      useHostPtr = b->useHostPtr;
      if (isUnified || useHostPtr) {
        ptr = b->ptr + offset;
      }
      if (isUnified || !useHostPtr) {
        cuPtr = b->cuPtr + offset;
      }
    }

    memory::~memory() {
      cuPtr = 0;
      useHostPtr = false;
      isUnified = false;
    }

    CUstream& memory::getCuStream() const {
      return dynamic_cast<device*>(getModeDevice())->getCuStream();
    }

    void* memory::getKernelArgPtr() const {
      if (useHostPtr) {
        return (void*) &ptr;
      } else {
        return (void*) &cuPtr;
      }
    }

    void* memory::getPtr() const {
      if (useHostPtr) {
        return ptr;
      } else {
        return (void*) cuPtr;
      }
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset_,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      if (useHostPtr) {
        ::memcpy(ptr+offset_, src, bytes);
      } else {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyHtoD(cuPtr + offset_,
                                       src,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyHtoDAsync(cuPtr + offset_,
                                            src,
                                            bytes,
                                            getCuStream()));
        }
      }
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      if (useHostPtr && dynamic_cast<const memory*>(src)->useHostPtr) {
        // src: host, dest: host
        ::memcpy(ptr + destOffset, src->ptr + srcOffset, bytes);
      } else if (dynamic_cast<const memory*>(src)->useHostPtr) {
        // src: host, dest: device
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyHtoD(cuPtr + destOffset,
                                       src->ptr + srcOffset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyHtoDAsync(cuPtr + destOffset,
                                            src->ptr + srcOffset,
                                            bytes,
                                            getCuStream()));
        }
      } else if (useHostPtr) {
        // src: device, dest: host
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyDtoH(ptr + destOffset,
                                       ((memory*) src)->cuPtr + srcOffset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyDtoHAsync(ptr + destOffset,
                                            ((memory*) src)->cuPtr + srcOffset,
                                            bytes,
                                            getCuStream()));
        }
      } else {
        // src: device, dest: device
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyDtoD(cuPtr + destOffset,
                                       ((memory*) src)->cuPtr + srcOffset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyDtoDAsync(cuPtr + destOffset,
                                            ((memory*) src)->cuPtr + srcOffset,
                                            bytes,
                                            getCuStream()));
        }
      }
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset_,
                        const occa::json &props) const {
      const bool async = props.get("async", false);

      if (useHostPtr) {
        ::memcpy(dest, ptr+offset_, bytes);
      } else {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyDtoH(dest,
                                       cuPtr + offset_,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyDtoHAsync(dest,
                                            cuPtr + offset_,
                                            bytes,
                                            getCuStream()));
        }
      }
    }

    void* memory::unwrap() {
      return static_cast<void*>(&cuPtr);
    }
  }
}
