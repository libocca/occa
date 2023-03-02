#ifndef OCCA_INTERNAL_MODES_METAL_STREAM_HEADER
#define OCCA_INTERNAL_MODES_METAL_STREAM_HEADER

#include <occa/internal/core/stream.hpp>
#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace metal {
    class stream : public occa::modeStream_t {
    public:
      api::metal::commandQueue_t metalCommandQueue;

      bool isWrapped;

      stream(modeDevice_t *modeDevice_,
             const occa::json &properties_,
             api::metal::commandQueue_t metalCommandQueue_,
             bool isWrapped_=false);

      virtual ~stream();
      void finish() override;

      void* unwrap() override;
    };
  }
}

#endif
