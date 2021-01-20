#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/utils.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace opencl {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
        occa::modeMemory_t(modeDevice_, size_, properties_),
        rootClMem(&clMem),
        rootOffset(0),
        useHostPtr(false) {}

    memory::~memory() {
      if (isOrigin) {
        // Free mapped-host pointer
        if (useHostPtr) {
          OCCA_OPENCL_ERROR("Mapped Free: clEnqueueUnmapMemObject",
                            clEnqueueUnmapMemObject(getCommandQueue(),
                                                    clMem,
                                                    ptr,
                                                    0, NULL, NULL));
        }
      }

      // Is the root cl_mem or the root cl_mem hasn't been freed yet
      if (size && (isOrigin || *rootClMem)) {
        OCCA_OPENCL_ERROR("Mapped Free: clReleaseMemObject",
                          clReleaseMemObject(clMem));
      }

      rootClMem = NULL;
      rootOffset = 0;

      ptr = nullptr;
      clMem = NULL;
      size = 0;
      useHostPtr = false;
    }

    cl_command_queue& memory::getCommandQueue() const {
      return ((device*) modeDevice)->getCommandQueue();
    }

    void* memory::getKernelArgPtr() const {
      return (void*) &clMem;
    }

    modeMemory_t* memory::addOffset(const dim_t offset) {
      opencl::memory *m = new opencl::memory(modeDevice,
                                             size - offset,
                                             properties);

      m->rootClMem = rootClMem;
      m->rootOffset = rootOffset + offset;

      cl_buffer_region info;
      info.origin = m->rootOffset;
      info.size   = m->size;

      cl_int error;
      m->clMem = clCreateSubBuffer(*rootClMem,
                                   CL_MEM_READ_WRITE,
                                   CL_BUFFER_CREATE_TYPE_REGION,
                                   &info,
                                   &error);

      if (useHostPtr) {
        m->ptr = ptr + offset;
      }

      OCCA_OPENCL_ERROR("Device: clCreateSubBuffer", error);
      return m;
    }

    void* memory::getPtr() {
      if (useHostPtr) {
        return ptr;
      } else {
        return static_cast<void*>(clMem); // Dubious...
      }
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::json &props) {
      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy From",
                        clEnqueueWriteBuffer(getCommandQueue(),
                                             clMem,
                                             async ? CL_FALSE : CL_TRUE,
                                             offset, bytes, src,
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
                        const udim_t offset,
                        const occa::json &props) const {

      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy To",
                        clEnqueueReadBuffer(getCommandQueue(),
                                            clMem,
                                            async ? CL_FALSE : CL_TRUE,
                                            offset, bytes, dest,
                                            0, NULL, NULL));
    }

    void memory::detach() {
      ptr = nullptr;
      size = 0;
      useHostPtr = false;
    }
  }
}
