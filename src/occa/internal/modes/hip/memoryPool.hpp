#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_HIP_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_MODES_HIP_MEMORYPOOL_HEADER

#include <occa/internal/core/memoryPool.hpp>

namespace occa {
  namespace hip {
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
      hipStream_t& getHipStream() const;

      void malloc(hipDeviceptr_t &hipPtr_, char* &ptr_, const udim_t bytes);

      void memcpy(hipDeviceptr_t hipDst, char* dst,
                  const hipDeviceptr_t hipSrc, const char* src,
                  const udim_t bytes);

      void free(hipDeviceptr_t &hipPtr_, char* &ptr_);

     public:
      hipDeviceptr_t hipPtr;
      bool useHostPtr;
    };
  }
}

#endif
