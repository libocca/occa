#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/buffer.hpp>
#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/modes/opencl/utils.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace opencl {
    buffer::buffer(modeDevice_t *modeDevice_,
                   udim_t size_,
                   const occa::json &properties_) :
      occa::modeBuffer_t(modeDevice_, size_, properties_),
      useHostPtr(false) {}

    buffer::~buffer() {
      // Free mapped-host pointer
      if (useHostPtr) {
        OCCA_OPENCL_ERROR("Mapped Free: clEnqueueUnmapMemObject",
                          clEnqueueUnmapMemObject(dynamic_cast<device*>(modeDevice)->getCommandQueue(),
                                                  clMem,
                                                  ptr,
                                                  0, NULL, NULL));
      }

      if (!isWrapped && size) {
        OCCA_OPENCL_ERROR("Mapped Free: clReleaseMemObject",
                          clReleaseMemObject(clMem));
      }

      ptr = nullptr;
      clMem = NULL;
      useHostPtr = false;
    }

    void buffer::malloc(udim_t bytes) {

      cl_int error;

      if (properties.get("host", false)) {
        clMem = clCreateBuffer(dynamic_cast<device*>(modeDevice)->clContext,
                               CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               bytes,
                               NULL, &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

        // Map memory to read/write
        ptr = (char*) clEnqueueMapBuffer(dynamic_cast<device*>(modeDevice)->getCommandQueue(),
                                        clMem,
                                        CL_TRUE,
                                        CL_MAP_READ | CL_MAP_WRITE,
                                        0, bytes,
                                        0, NULL, NULL,
                                        &error);
        OCCA_OPENCL_ERROR("Device: clEnqueueMapBuffer", error);

        useHostPtr = true;

      } else {
        clMem = clCreateBuffer(dynamic_cast<device*>(modeDevice)->clContext,
                               CL_MEM_READ_WRITE,
                               bytes, NULL, &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);
      }

      size = bytes;
    }

    void buffer::wrapMemory(const void *ptr_,
                            const udim_t bytes) {
      clMem = (cl_mem) const_cast<void*>(ptr_);
      size = bytes;
      isWrapped = true;
    }

    modeMemory_t* buffer::slice(const dim_t offset,
                                const udim_t bytes) {
      return new opencl::memory(this, bytes, offset);
    }

    void buffer::detach() {
      clMem = NULL;
      ptr = nullptr;
      size = 0;
      useHostPtr = false;
      isWrapped = false;
    }
  }
}
