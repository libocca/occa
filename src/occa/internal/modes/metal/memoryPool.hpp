#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_METAL_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_MODES_METAL_MEMORYPOOL_HEADER

#include <occa/internal/core/memoryPool.hpp>

namespace occa {
  namespace metal {
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
      void malloc(api::metal::buffer_t &metalBuffer_,
                  char* &ptr_, const udim_t bytes);

      void memcpy(api::metal::buffer_t &dstBuffer,
                  const udim_t dstOffset,
                  const api::metal::buffer_t &srcBuffer,
                  const udim_t srcOffset,
                  const udim_t bytes);

      void free(api::metal::buffer_t &metalBuffer_,
                char* &ptr_);

     public:
      api::metal::buffer_t metalBuffer;
    };
  }
}

#endif
