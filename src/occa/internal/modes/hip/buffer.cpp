#include <occa/internal/modes/hip/device.hpp>
#include <occa/internal/modes/hip/buffer.hpp>
#include <occa/internal/modes/hip/memory.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    buffer::buffer(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeBuffer_t(modeDevice_, size_, properties_),
        hipPtr(reinterpret_cast<hipDeviceptr_t>(ptr)),
        useHostPtr(false) {}

    buffer::~buffer() {
      if (!isWrapped) {
        if (useHostPtr && ptr) {
          OCCA_HIP_ERROR("Device: hostFree()",
                           hipHostFree(ptr));
        } else if (hipPtr) {
          OCCA_HIP_ERROR("Device: free()",
                           hipFree((void*) hipPtr));
        }
      }
      ptr = nullptr;
      hipPtr = NULL;
      useHostPtr = false;
    }

    void buffer::malloc(udim_t bytes) {

      if (properties.get("host", false)) {

        OCCA_HIP_ERROR("Device: malloc host",
                       hipHostMalloc((void**) &(ptr), bytes));
        OCCA_HIP_ERROR("Device: get device pointer from host",
                       hipHostGetDevicePointer((void**) &(hipPtr),
                                               ptr,
                                               0));
        useHostPtr=true;

      } else  {

        OCCA_HIP_ERROR("Device: malloc",
                       hipMalloc((void**) &(hipPtr), bytes));

      }
      size = bytes;
    }

    void buffer::wrapMemory(const void *ptr_,
                            const udim_t bytes) {

      if (properties.get("host", false)) {
        ptr = (char*) const_cast<void*>(ptr_);
        useHostPtr=true;
      } else  {
        hipPtr = reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(ptr_));
      }
      size = bytes;
      isWrapped = true;
    }

    modeMemory_t* buffer::slice(const dim_t offset,
                                const udim_t bytes) {
      return new hip::memory(this, bytes, offset);
    }

    void buffer::detach() {
      ptr = nullptr;
      hipPtr = NULL;
      size = 0;
      useHostPtr = false;
      isWrapped = false;
    }
  }
}
