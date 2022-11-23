#include <occa/internal/modes/opencl/buffer.hpp>
#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/utils.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace opencl {
    memory::memory(buffer *b,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(b, size_, offset_),
      useHostPtr(false) {
      useHostPtr = b->useHostPtr;

      if (offset==0 && size==b->size){
        clMem = b->clMem;
      } else {
        cl_buffer_region info;
        info.origin = offset;
        info.size   = size;

        cl_int error;
        clMem = clCreateSubBuffer(b->clMem,
                                  CL_MEM_READ_WRITE,
                                  CL_BUFFER_CREATE_TYPE_REGION,
                                  &info,
                                  &error);
        OCCA_OPENCL_ERROR("Device: clCreateSubBuffer", error);
      }
      if (useHostPtr) {
        ptr = b->ptr + offset;
      }
    }

    memory::memory(memoryPool *memPool,
                   udim_t size_, dim_t offset_) :
      occa::modeMemory_t(memPool, size_, offset_),
      useHostPtr(false) {
      opencl::buffer* b = dynamic_cast<opencl::buffer*>(memPool->buffer);
      useHostPtr = b->useHostPtr;

      if (offset==0 && size==b->size){
        clMem = b->clMem;
      } else {
        cl_buffer_region info;
        info.origin = offset;
        info.size   = size;

        cl_int error;
        clMem = clCreateSubBuffer(b->clMem,
                                  CL_MEM_READ_WRITE,
                                  CL_BUFFER_CREATE_TYPE_REGION,
                                  &info,
                                  &error);
        OCCA_OPENCL_ERROR("Device: clCreateSubBuffer", error);
      }
      if (useHostPtr) {
        ptr = b->ptr + offset;
      }
    }

    memory::~memory() {
      useHostPtr = false;
    }

    cl_command_queue& memory::getCommandQueue() const {
      return dynamic_cast<device*>(getModeDevice())->getCommandQueue();
    }

    void* memory::getKernelArgPtr() const {
      return (void*) &clMem;
    }

    void* memory::getPtr() const {
      if (useHostPtr) {
        return ptr;
      } else {
        return static_cast<void*>(clMem); // Dubious...
      }
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset_,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy From",
                        clEnqueueWriteBuffer(getCommandQueue(),
                                             clMem,
                                             async ? CL_FALSE : CL_TRUE,
                                             offset_, bytes, src,
                                             0, NULL, NULL));
    }

    void memory::copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy From",
                        clEnqueueCopyBuffer(getCommandQueue(),
                                            ((memory*) src)->clMem,
                                            clMem,
                                            srcOffset, destOffset,
                                            bytes,
                                            0, NULL, NULL));
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset_,
                        const occa::json &props) const {

      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy To",
                        clEnqueueReadBuffer(getCommandQueue(),
                                            clMem,
                                            async ? CL_FALSE : CL_TRUE,
                                            offset_, bytes, dest,
                                            0, NULL, NULL));
    }

    void* memory::unwrap() {
      return static_cast<void*>(&clMem);
    }
  }
}
