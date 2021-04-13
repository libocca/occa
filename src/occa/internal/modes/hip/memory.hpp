#ifndef OCCA_INTERNAL_MODES_HIP_MEMORY_HEADER
#define OCCA_INTERNAL_MODES_HIP_MEMORY_HEADER

#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/hip/polyfill.hpp>

namespace occa {
  namespace hip {
    class device;

    class memory : public occa::modeMemory_t {
    public:
      hipDeviceptr_t hipPtr;
      bool useHostPtr;

      memory(modeBuffer_t *modeBuffer_,
             udim_t size_, dim_t offset_);
      ~memory();

      hipStream_t& getHipStream() const;

      void* getKernelArgPtr() const;

      void* getPtr() const;

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset = 0,
                  const occa::json &props = occa::json()) const;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset = 0,
                    const occa::json &props = occa::json());

      void copyFrom(const modeMemory_t *src,
                    const udim_t bytes,
                    const udim_t destOffset = 0,
                    const udim_t srcOffset = 0,
                    const occa::json &props = occa::json());
    };
  }
}

#endif
