#ifndef OCCA_INTERNAL_MODES_SERIAL_BUFFER_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_BUFFER_HEADER

#include <occa/defines.hpp>
#include <occa/internal/core/buffer.hpp>
#include <occa/internal/core/memory.hpp>

namespace occa {
  namespace serial {
    class buffer : public occa::modeBuffer_t {
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

      void detach() override;
    };
  }
}

#endif
