#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_CUDA_BUFFER_HEADER
#define OCCA_INTERNAL_MODES_CUDA_BUFFER_HEADER

#include <occa/internal/core/buffer.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/cuda/polyfill.hpp>

namespace occa {
  namespace cuda {
    class buffer : public occa::modeBuffer_t {
    public:
      buffer(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::json &properties_ = occa::json());
      ~buffer();

      void malloc(udim_t bytes);

      void wrapMemory(const void *ptr,
                      const udim_t bytes);

      modeMemory_t* slice(const dim_t offset,
                          const udim_t bytes);

      void detach();

    public:
      CUdeviceptr cuPtr;
      bool isUnified;
      bool useHostPtr;
    };
  }
}

#endif
