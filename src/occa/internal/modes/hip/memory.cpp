#include <occa/internal/modes/hip/memory.hpp>
#include <occa/internal/modes/hip/device.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    inline hipDeviceptr_t addHipPtrOffset(hipDeviceptr_t hipPtr, const udim_t offset) {
      return (hipDeviceptr_t) (((char*) hipPtr) + offset);
    }

    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeMemory_t(modeDevice_, size_, properties_),
      hipPtr(reinterpret_cast<hipDeviceptr_t&>(ptr)),
      useHostPtr(false) {}

    memory::~memory() {
      if (isOrigin) {
        if (useHostPtr) {
          OCCA_HIP_ERROR("Device: hostFree()",
                         hipHostFree(ptr));
        } else if (hipPtr) {
          hipFree((void*) hipPtr);
        }
      }
      ptr = nullptr;
      hipPtr = 0;
      size = 0;
      useHostPtr = false;
    }

    hipStream_t& memory::getHipStream() const {
      return ((device*) modeDevice)->getHipStream();
    }

    void* memory::getKernelArgPtr() const {
      return (void*) hipPtr;
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      memory *m = new memory(modeDevice,
                             size - offset,
                             properties);
      m->hipPtr = addHipPtrOffset(hipPtr, offset);
      if (useHostPtr) {
        m->ptr = ptr + offset;
      }
      m->useHostPtr = useHostPtr;
      return m;
    }

    void* memory::getPtr() {
      if (useHostPtr) {
        return ptr;
      } else {
        return static_cast<void*>(hipPtr);
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
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyHtoD(addHipPtrOffset(hipPtr, offset),
                                       const_cast<void*>(src),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyHtoDAsync(addHipPtrOffset(hipPtr, offset),
                                            const_cast<void*>(src),
                                            bytes,
                                            getHipStream()) );
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
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyHtoD(addHipPtrOffset(hipPtr, destOffset),
                                       src->ptr + srcOffset,
                                       bytes));
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyHtoDAsync(addHipPtrOffset(hipPtr, destOffset),
                                            src->ptr + srcOffset,
                                            bytes,
                                            getHipStream()));
        }
      } else if (useHostPtr) {
        // src: device, dest: host
        if (!async) {
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyDtoH(ptr + destOffset,
                                       addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                       bytes));
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyDtoHAsync(ptr + destOffset,
                                            addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                            bytes,
                                            getHipStream()));
        }
      } else {
        // src: device, dest: device
        if (!async) {
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyDtoD(addHipPtrOffset(hipPtr, destOffset),
                                       addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyDtoDAsync(addHipPtrOffset(hipPtr, destOffset),
                                            addHipPtrOffset(((memory*) src)->hipPtr, srcOffset),
                                            bytes,
                                            getHipStream()) );
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
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyDtoH(dest,
                                       addHipPtrOffset(hipPtr, offset),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyDtoHAsync(dest,
                                            addHipPtrOffset(hipPtr, offset),
                                            bytes,
                                            getHipStream()) );
        }
      }
    }

    void memory::detach() {
      ptr = 0;
      hipPtr = 0;
      size = 0;
      useHostPtr = false;
    }
  }
}
