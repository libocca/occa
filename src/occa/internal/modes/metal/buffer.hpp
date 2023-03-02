#ifndef OCCA_INTERNAL_MODES_METAL_BUFFER_HEADER
#define OCCA_INTERNAL_MODES_METAL_BUFFER_HEADER

#include <occa/internal/core/buffer.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace metal {
    class memory;
    class memoryPool;

    class buffer : public occa::modeBuffer_t {
      friend class metal::memory;
      friend class metal::memoryPool;

    public:
      buffer(modeDevice_t *modeDevice_,
             udim_t size_,
             const occa::json &properties_ = occa::json());
      virtual ~buffer();

      void malloc(udim_t bytes) override;

      void wrapMemory(const void *ptr,
                      const udim_t bytes);

      modeMemory_t* slice(const dim_t offset,
                          const udim_t bytes) override;

      void* getPtr();

      void detach() override;

    private:
      api::metal::buffer_t metalBuffer;
    };
  }
}

#endif
