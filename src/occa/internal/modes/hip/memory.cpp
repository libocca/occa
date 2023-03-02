#include <occa/internal/modes/hip/device.hpp>
#include <occa/internal/modes/hip/buffer.hpp>
#include <occa/internal/modes/hip/memory.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    inline hipDeviceptr_t addHipPtrOffset(hipDeviceptr_t hipPtr, const udim_t offset) {
      return (hipDeviceptr_t) (((char*) hipPtr) + offset);
    }

    memory::memory(buffer *b,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(b, size_, offset_) {
      useHostPtr = b->useHostPtr;
      if (useHostPtr) {
        ptr = b->ptr + offset;
      } else {
        hipPtr = addHipPtrOffset(b->hipPtr, offset);
      }
    }

    memory::memory(memoryPool *memPool,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(memPool, size_, offset_) {
      hip::buffer* b = dynamic_cast<hip::buffer*>(memPool->buffer);
      useHostPtr = b->useHostPtr;
      if (useHostPtr) {
        ptr = b->ptr + offset;
      } else {
        hipPtr = addHipPtrOffset(b->hipPtr, offset);
      }
    }

    memory::~memory() {
      hipPtr = NULL;
      useHostPtr = false;
    }

    hipStream_t& memory::getHipStream() const {
      return dynamic_cast<device*>(getModeDevice())->getHipStream();
    }

    void* memory::getKernelArgPtr() const {
      return getPtr();
    }

    void* memory::getPtr() const {
      if (useHostPtr) {
        return ptr;
      } else {
        return static_cast<void*>(hipPtr);
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
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyHtoD(addHipPtrOffset(hipPtr, offset_),
                                       const_cast<void*>(src),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyHtoDAsync(addHipPtrOffset(hipPtr, offset_),
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

      if (useHostPtr && dynamic_cast<const memory*>(src)->useHostPtr) {
        // src: host, dest: host
        ::memcpy(ptr + destOffset, src->ptr + srcOffset, bytes);
      } else if (dynamic_cast<const memory*>(src)->useHostPtr) {
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
                        const udim_t offset_,
                        const occa::json &props) const {
      const bool async = props.get("async", false);

      if (useHostPtr) {
        ::memcpy(dest, ptr+offset_, bytes);
      } else {
        if (!async) {
          OCCA_HIP_ERROR("Memory: Copy From",
                         hipMemcpyDtoH(dest,
                                       addHipPtrOffset(hipPtr, offset_),
                                       bytes) );
        } else {
          OCCA_HIP_ERROR("Memory: Async Copy From",
                         hipMemcpyDtoHAsync(dest,
                                            addHipPtrOffset(hipPtr, offset_),
                                            bytes,
                                            getHipStream()) );
        }
      }
    }

    void* memory::unwrap() {
      return static_cast<void*>(&hipPtr);
    }
  }
}
