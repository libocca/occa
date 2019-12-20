#include <occa/modes/opencl/memory.hpp>
#include <occa/modes/opencl/device.hpp>
#include <occa/modes/opencl/utils.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  namespace opencl {
    memory::memory(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::properties &properties_) :
        occa::modeMemory_t(modeDevice_, size_, properties_),
        rootOffset(0),
        mappedPtr(NULL) {}

    memory::~memory() {
      if (isOrigin) {
        if (mappedPtr) {
          OCCA_OPENCL_ERROR("Mapped Free: clEnqueueUnmapMemObject",
                            clEnqueueUnmapMemObject(getCommandQueue(),
                                                    clMem,
                                                    mappedPtr,
                                                    0, NULL, NULL));
        }
        if (size) {
          // Free mapped-host pointer
          OCCA_OPENCL_ERROR("Mapped Free: clReleaseMemObject",
                            clReleaseMemObject(clMem));
        }
      }

      rootClMem = NULL;
      rootOffset = 0;

      clMem = NULL;
      mappedPtr = NULL;
      size = 0;
    }

    cl_command_queue& memory::getCommandQueue() const {
      return ((device*) modeDevice)->getCommandQueue();
    }

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.modeMemory = const_cast<memory*>(this);
      arg.data.void_ = (void*) &clMem;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
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
      m->clMem = clCreateSubBuffer(rootClMem,
                                   CL_MEM_READ_WRITE,
                                   CL_BUFFER_CREATE_TYPE_REGION,
                                   &info,
                                   &error);

      OCCA_OPENCL_ERROR("Device: clCreateSubBuffer", error);
      return m;
    }

    void* memory::getPtr(const occa::properties &props) {
      if (props.get("mapped", false)) {
        return mappedPtr;
      }
      return ptr;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {
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
                          const occa::properties &props) {
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
                        const occa::properties &props) const {

      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy To",
                        clEnqueueReadBuffer(getCommandQueue(),
                                            clMem,
                                            async ? CL_FALSE : CL_TRUE,
                                            offset, bytes, dest,
                                            0, NULL, NULL));
    }

    void memory::detach() {
      size = 0;
    }
  }
}
