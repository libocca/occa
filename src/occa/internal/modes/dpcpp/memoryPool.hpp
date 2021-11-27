#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_DPCPP_MEMORYPOOL_HEADER
#define OCCA_INTERNAL_MODES_DPCPP_MEMORYPOOL_HEADER

#include <occa/internal/core/memoryPool.hpp>

namespace occa {
  namespace dpcpp {
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
      void malloc(char* &ptr_, const udim_t bytes);

      void memcpy(char* dst, const char* src,
                  const udim_t bytes);

      void free(char* &ptr_);
    };
  }
}

#endif
