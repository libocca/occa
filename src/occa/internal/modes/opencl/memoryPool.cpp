#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/buffer.hpp>
#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/modes/opencl/memoryPool.hpp>
#include <occa/internal/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    memoryPool::memoryPool(modeDevice_t *modeDevice_,
                           const occa::json &properties_) :
      occa::modeMemoryPool_t(modeDevice_, properties_) {}

    cl_command_queue& memoryPool::getCommandQueue() const {
      return dynamic_cast<device*>(modeDevice)->getCommandQueue();
    }

    modeBuffer_t* memoryPool::makeBuffer() {
      return new opencl::buffer(modeDevice, 0, properties);
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new opencl::memory(this, bytes, offset);
    }

    void memoryPool::setPtr(modeMemory_t* mem, modeBuffer_t* buf,
                            const dim_t offset) {

      opencl::memory* m = dynamic_cast<opencl::memory*>(mem);
      opencl::buffer* b = dynamic_cast<opencl::buffer*>(buf);

      if (offset==0 && m->size==b->size){
        m->clMem = b->clMem;
      } else {
        cl_buffer_region info;
        info.origin = offset;
        info.size   = size;

        cl_int error;
        m->clMem = clCreateSubBuffer(b->clMem,
                                     CL_MEM_READ_WRITE,
                                     CL_BUFFER_CREATE_TYPE_REGION,
                                     &info,
                                     &error);
        OCCA_OPENCL_ERROR("Device: clCreateSubBuffer", error);
      }
      if (m->useHostPtr) {
        m->ptr = b->ptr + offset;
      }
    }

    void memoryPool::memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                            modeBuffer_t* src, const dim_t srcOffset,
                            const udim_t bytes) {

      opencl::buffer* dstBuf = dynamic_cast<opencl::buffer*>(dst);
      opencl::buffer* srcBuf = dynamic_cast<opencl::buffer*>(src);

      const bool async = true;

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy From",
                        clEnqueueCopyBuffer(getCommandQueue(),
                                            srcBuf->clMem,
                                            dstBuf->clMem,
                                            srcOffset, dstOffset,
                                            bytes,
                                            0, NULL, NULL));
    }
  }
}
