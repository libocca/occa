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
      occa::modeMemoryPool_t(modeDevice_, properties_) {}

    hipStream_t& memoryPool::getHipStream() const {
      return dynamic_cast<device*>(modeDevice)->getHipStream();
    }

    modeBuffer_t* memoryPool::makeBuffer() {
      return new hip::buffer(modeDevice, 0, properties);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new hip::memory(this, bytes, offset);
    }

    void memoryPool::setPtr(modeMemory_t* mem, modeBuffer_t* buf,
                            const dim_t offset) {

      hip::memory* m = dynamic_cast<hip::memory*>(mem);
      hip::buffer* b = dynamic_cast<hip::buffer*>(buf);

      m->offset = offset;
      if (b->useHostPtr) {
        m->ptr = b->ptr + offset;
      } else {
        m->hipPtr = addHipPtrOffset(b->hipPtr, offset);
      }
    }

    void memoryPool::memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                            modeBuffer_t* src, const dim_t srcOffset,
                            const udim_t bytes) {

      hip::buffer* dstBuf = dynamic_cast<hip::buffer*>(dst);
      hip::buffer* srcBuf = dynamic_cast<hip::buffer*>(src);

      if (srcBuf->useHostPtr) {
        ::memcpy(dstBuf->ptr + dstOffset,
                 srcBuf->ptr + srcOffset,
                 bytes);
      } else {
        OCCA_HIP_ERROR("Memory: Async Copy From",
                       hipMemcpyDtoDAsync(addHipPtrOffset(dstBuf->hipPtr, dstOffset),
                                          addHipPtrOffset(srcBuf->hipPtr, srcOffset),
                                          bytes,
                                          getHipStream()));
      }
    }
  }
}
