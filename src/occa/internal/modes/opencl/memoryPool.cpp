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

    memoryPool::~memoryPool() {
      free(clMem, ptr);
      useHostPtr = false;
    }

    modeMemory_t* memoryPool::slice(const dim_t offset,
                                    const udim_t bytes) {
      return new opencl::memory(this, bytes, offset);
    }

    cl_command_queue& memoryPool::getCommandQueue() const {
      return dynamic_cast<device*>(modeDevice)->getCommandQueue();
    }

    void memoryPool::malloc(cl_mem &clMem_, char* &ptr_,
                            const udim_t bytes) {
      cl_int error;

      if (useHostPtr) {
        clMem_ = clCreateBuffer(dynamic_cast<device*>(modeDevice)->clContext,
                               CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               bytes,
                               NULL, &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

        // Map memory to read/write
        ptr_ = (char*) clEnqueueMapBuffer(dynamic_cast<device*>(modeDevice)->getCommandQueue(),
                                        clMem_,
                                        CL_TRUE,
                                        CL_MAP_READ | CL_MAP_WRITE,
                                        0, bytes,
                                        0, NULL, NULL,
                                        &error);
        OCCA_OPENCL_ERROR("Device: clEnqueueMapBuffer", error);

      } else {
        clMem_ = clCreateBuffer(dynamic_cast<device*>(modeDevice)->clContext,
                               CL_MEM_READ_WRITE,
                               bytes, NULL, &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);
      }
    }

    void memoryPool::memcpy(cl_mem &clDst,
                            const dim_t dstOffset,
                            const cl_mem &clSrc,
                            const dim_t srcOffset,
                            const udim_t bytes) {
      const bool async = true;

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy From",
                        clEnqueueCopyBuffer(getCommandQueue(),
                                            clSrc,
                                            clDst,
                                            srcOffset, dstOffset,
                                            bytes,
                                            0, NULL, NULL));
    }

    void memoryPool::free(cl_mem &clMem_, char* &ptr_) {
      if (useHostPtr) {
        OCCA_OPENCL_ERROR("Mapped Free: clEnqueueUnmapMemObject",
                          clEnqueueUnmapMemObject(getCommandQueue(),
                                                  clMem_,
                                                  ptr_,
                                                  0, NULL, NULL));
      }

      if (size) {
        OCCA_OPENCL_ERROR("Mapped Free: clReleaseMemObject",
                          clReleaseMemObject(clMem_));
      }
      ptr_ = nullptr;
      clMem_ = NULL;
    }

    void memoryPool::resize(const udim_t bytes) {

      OCCA_ERROR("Cannot resize memoryPool below current usage"
                 "(reserved: " << reserved << ", bytes: " << bytes << ")",
                 reserved<=bytes);

      if (reservations.size()==0) {
        free(clMem, ptr);

        modeDevice->bytesAllocated -= size;

        malloc(clMem, ptr, bytes);
        size=bytes;

        modeDevice->bytesAllocated += bytes;
        modeDevice->maxBytesAllocated = std::max(
          modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
        );
      } else {

        cl_mem newClMem=0;
        char* newPtr=nullptr;
        malloc(newClMem, newPtr, bytes);

        modeDevice->bytesAllocated += bytes;
        modeDevice->maxBytesAllocated = std::max(
          modeDevice->maxBytesAllocated, modeDevice->bytesAllocated
        );

        auto it = reservations.begin();
        memory* m = dynamic_cast<memory*>(*it);
        dim_t lo = m->offset;
        dim_t hi = lo + m->size;
        dim_t offset=0;
        m->offset=0;
        m->ptr = newPtr;
        m->clMem = newClMem;
        do {
          it++;
          if (it==reservations.end()) {
            memcpy(newClMem, offset,
                   clMem, lo, hi-lo);
          } else {
            m = dynamic_cast<memory*>(*it);
            const dim_t mlo = m->offset;
            const dim_t mhi = m->offset+m->size;
            if (mlo>hi) {
              memcpy(newClMem, offset,
                     clMem, lo, hi-lo);

              offset+=hi-lo;
              lo=mlo;
              hi=mhi;
            } else {
              hi = std::max(hi, mhi);
            }
            m->offset -= lo-offset;
            m->ptr = newPtr + m->offset;

            cl_buffer_region info;
            info.origin = m->offset;
            info.size   = m->size;

            cl_int error;
            m->clMem = clCreateSubBuffer(newClMem,
                                         CL_MEM_READ_WRITE,
                                         CL_BUFFER_CREATE_TYPE_REGION,
                                         &info,
                                         &error);
            OCCA_OPENCL_ERROR("Device: clCreateSubBuffer", error);
          }
        } while (it!=reservations.end());

        free(clMem, ptr);
        modeDevice->bytesAllocated -= size;

        ptr = newPtr;
        clMem = newClMem;
        size=bytes;
      }
    }

    void memoryPool::detach() {
      clMem = NULL;
      ptr = nullptr;
      size = 0;
      useHostPtr = false;
      isWrapped = false;
    }
  }
}
