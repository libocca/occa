#include <occa/internal/modes/cuda/memory.hpp>
#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
        occa::modeMemory_t(modeDevice_, size_, properties_),
        cuPtr(reinterpret_cast<CUdeviceptr&>(ptr)),
        isUnified(false),
        useHostPtr(false) {}

    memory::~memory() {
      if (isOrigin) {
        if (useHostPtr) {
          OCCA_CUDA_DESTRUCTOR_ERROR(
            "Device: hostFree()",
            cuMemFreeHost(ptr)
          );
        } else if (cuPtr) {
          cuMemFree(cuPtr);
        }
      }
      ptr = nullptr;
      cuPtr = 0;
      size = 0;
      useHostPtr = false;
    }

    CUstream& memory::getCuStream() const {
      return ((device*) modeDevice)->getCuStream();
    }

    void* memory::getKernelArgPtr() const {
      return (void*) &cuPtr;
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      memory *m = new memory(modeDevice,
                             size - offset,
                             properties);
      m->cuPtr = cuPtr + offset;
      if (useHostPtr) {
        m->ptr = ptr + offset;
      }
      m->isUnified = isUnified;
      m->useHostPtr = useHostPtr;
      return m;
    }

    void* memory::getPtr() {
      if (useHostPtr) {
        return ptr;
      } else {
        return (void*) cuPtr;
      }
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      if (useHostPtr) {
        ::memcpy(ptr+offset, src, bytes);
      } else {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyHtoD(cuPtr + offset,
                                       src,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyHtoDAsync(cuPtr + offset,
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

      if (useHostPtr && ((memory*) src)->useHostPtr) {
        // src: host, dest: host
        ::memcpy(ptr + destOffset, src->ptr + srcOffset, bytes);
      } else if (((memory*) src)->useHostPtr) {
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
                        const udim_t offset,
                        const occa::json &props) const {
      const bool async = props.get("async", false);

      if (useHostPtr) {
        ::memcpy(dest, ptr+offset, bytes);
      } else {
        if (!async) {
          OCCA_CUDA_ERROR("Memory: Copy From",
                          cuMemcpyDtoH(dest,
                                       cuPtr + offset,
                                       bytes));
        } else {
          OCCA_CUDA_ERROR("Memory: Async Copy From",
                          cuMemcpyDtoHAsync(dest,
                                            cuPtr + offset,
                                            bytes,
                                            getCuStream()));
        }
      }
    }

    void memory::detach() {
      ptr = nullptr;
      cuPtr = 0;
      size = 0;
      useHostPtr = false;
    }
  }
}
