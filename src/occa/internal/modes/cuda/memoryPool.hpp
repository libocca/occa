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
      ~memoryPool();

      modeMemory_t* slice(const dim_t offset,
                          const udim_t bytes);

      void resize(const udim_t bytes);

      void detach();

     private:
      CUstream& getCuStream() const;

      void malloc(CUdeviceptr &cuPtr_, char* &ptr_, const udim_t bytes);

      void memcpy(CUdeviceptr cuDst, char* dst,
                  const CUdeviceptr cuSrc, const char* src,
                  const udim_t bytes);

      void free(CUdeviceptr &cuPtr_, char* &ptr_);

     public:
      CUdeviceptr cuPtr;
      bool isUnified;
      bool useHostPtr;
    };
  }
}

#endif
