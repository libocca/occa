#ifndef OCCA_INTERNAL_MODES_METAL_MEMORY_HEADER
#define OCCA_INTERNAL_MODES_METAL_MEMORY_HEADER

#include <occa/internal/core/memory.hpp>
#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace metal {
    class memory : public occa::modeMemory_t {
    private:
      api::metal::buffer_t metalBuffer;
      udim_t bufferOffset;

    public:
      memory(modeBuffer_t *modeBuffer_,
             udim_t size_, dim_t offset_);
      ~memory();

      void* getKernelArgPtr() const;

      const api::metal::buffer_t& getMetalBuffer();

      void* getPtr() const;

      udim_t getOffset() const;

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
