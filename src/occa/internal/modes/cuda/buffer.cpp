#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/buffer.hpp>
#include <occa/internal/modes/cuda/memory.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    buffer::buffer(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeBuffer_t(modeDevice_, size_, properties_),
        cuPtr(reinterpret_cast<CUdeviceptr>(ptr)),
        isUnified(false),
        useHostPtr(false) {}

    buffer::~buffer() {
      if (!isWrapped) {
        if (useHostPtr && ptr) {
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
      useHostPtr = false;
      isUnified = false;
    }

    void buffer::malloc(udim_t bytes) {

      if (properties.get("host", false)) {

        OCCA_CUDA_ERROR("Device: malloc host",
                        cuMemAllocHost((void**) &(ptr), bytes));
        OCCA_CUDA_ERROR("Device: get device pointer from host",
                        cuMemHostGetDevicePointer(&(cuPtr),
                                                  ptr,
                                                  0));
        useHostPtr=true;

      } else if (properties.get("unified", false)) {

#if CUDA_VERSION >= 8000
        const unsigned int flags = (properties.get("attached_host", false) ?
                                    CU_MEM_ATTACH_HOST : CU_MEM_ATTACH_GLOBAL);

        OCCA_CUDA_ERROR("Device: Unified alloc",
                        cuMemAllocManaged(&(cuPtr),
                                          bytes,
                                          flags));
#else
        OCCA_FORCE_ERROR("CUDA version ["
                         << cuda::getVersion()
                         << "] does not support unified memory allocation");
#endif
        ptr = (char*) cuPtr;
        isUnified = true;

      } else {

        OCCA_CUDA_ERROR("Device: malloc",
                        cuMemAlloc(&(cuPtr), bytes));

      }
      size = bytes;
    }

    void buffer::wrapMemory(const void *ptr_,
                            const udim_t bytes) {

      if (properties.get("host", false)) {
        ptr = (char*) const_cast<void*>(ptr_);
        useHostPtr=true;
      } else if (properties.get("unified", false)) {
        cuPtr = reinterpret_cast<CUdeviceptr>(const_cast<void*>(ptr_));
        ptr = (char*) const_cast<void*>(ptr_);
        isUnified = true;
      } else {
        cuPtr = reinterpret_cast<CUdeviceptr>(const_cast<void*>(ptr_));
      }
      size = bytes;
      isWrapped = true;
    }

    modeMemory_t* buffer::slice(const dim_t offset,
                                const udim_t bytes) {
      return new cuda::memory(this, bytes, offset);
    }

    void buffer::detach() {
      ptr = nullptr;
      cuPtr = 0;
      size = 0;
      useHostPtr = false;
      isUnified = false;
      isWrapped = false;
    }
  }
}
