#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_OPENCL_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_MEMORYPOOL_HEADER

#include <occa/internal/core/memoryPool.hpp>

namespace occa {
  namespace opencl {
    class memoryPool : public occa::modeMemoryPool_t {
     public:
      memoryPool(modeDevice_t *modeDevice_,
                 const occa::json &properties_ = occa::json());
      ~memoryPool();

      modeMemory_t* slice(const dim_t offset,
                          const udim_t bytes);

      void resize(const udim_t bytes);

      void detach();

     private:
      cl_command_queue& getCommandQueue() const;

      void malloc(cl_mem &clMem_, char* &ptr_, const udim_t bytes);

      void memcpy(cl_mem &clDst,
                  const dim_t dstOffset,
                  const cl_mem &clSrc,
                  const dim_t srcOffset,
                  const udim_t bytes);

      void free(cl_mem &clMem_, char* &ptr_);

     public:
      cl_mem clMem;
      bool useHostPtr;
    };
  }
}

#endif
