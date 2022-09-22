#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_CUDA_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_MODES_CUDA_MEMORYPOOL_HEADER

#include <occa/internal/core/memoryPool.hpp>

namespace occa {
  namespace cuda {
    class memoryPool : public occa::modeMemoryPool_t {
     public:
      memoryPool(modeDevice_t *modeDevice_,
                 const occa::json &properties_ = occa::json());

     private:
      CUstream& getCuStream() const;

      modeBuffer_t* makeBuffer();

      modeMemory_t* slice(const dim_t offset, const udim_t bytes);

      void setPtr(modeMemory_t* mem, modeBuffer_t* buf, const dim_t offset);

      void memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                  modeBuffer_t* src, const dim_t srcOffset,
                  const udim_t bytes);
    };
  }
}

#endif
