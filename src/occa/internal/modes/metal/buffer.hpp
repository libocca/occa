#ifndef OCCA_INTERNAL_MODES_METAL_BUFFER_HEADER
#define OCCA_INTERNAL_MODES_METAL_BUFFER_HEADER

#include <occa/internal/core/buffer.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/api/metal.hpp>
#include <occa/internal/modes/metal/memory.hpp>

namespace occa {
  namespace metal {
    class buffer : public occa::modeBuffer_t {
    friend class metal::memory;

    private:
      api::metal::buffer_t metalBuffer;
      udim_t bufferOffset;

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

      void* getPtr();

      void detach();
    };
  }
}

#endif
