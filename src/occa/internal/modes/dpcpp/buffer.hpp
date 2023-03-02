#ifndef OCCA_MODES_DPCPP_BUFFER_HEADER
#define OCCA_MODES_DPCPP_BUFFER_HEADER

#include <occa/internal/core/buffer.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>

namespace occa {
  namespace dpcpp {
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
  } // namespace dpcpp
} // namespace occa

#endif
