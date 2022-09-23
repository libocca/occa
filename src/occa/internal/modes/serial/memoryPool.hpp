#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_SERIAL_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_MEMORYPOOL_HEADER

#include <occa/internal/core/memoryPool.hpp>

namespace occa {
  namespace serial {
    class memoryPool : public occa::modeMemoryPool_t {
     public:
      memoryPool(modeDevice_t *modeDevice_,
                 const occa::json &properties_ = occa::json());

     private:
      modeBuffer_t* makeBuffer() override;

      modeMemory_t* slice(const dim_t offset, const udim_t bytes) override;

      void setPtr(modeMemory_t* mem, modeBuffer_t* buf, const dim_t offset) override;

      void memcpy(modeBuffer_t* dst, const dim_t dstOffset,
                  modeBuffer_t* src, const dim_t srcOffset,
                  const udim_t bytes) override;
    };
  }
}

#endif
