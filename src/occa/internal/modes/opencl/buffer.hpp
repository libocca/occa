#ifndef OCCA_INTERNAL_MODES_OPENCL_BUFFER_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_BUFFER_HEADER

#include <occa/internal/core/buffer.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/modes/opencl/polyfill.hpp>

namespace occa {
  namespace opencl {
    class buffer : public occa::modeBuffer_t {
      friend class opencl::memory;

    private:
      cl_mem clMem;
      bool useHostPtr;

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
    };
  }
}

#endif
