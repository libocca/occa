#ifndef OCCA_MODES_METAL_STREAM_HEADER
#define OCCA_MODES_METAL_STREAM_HEADER

#include <occa/core/stream.hpp>
#include <occa/api/metal.hpp>

namespace occa {
  namespace metal {
    class stream : public occa::modeStream_t {
    public:
      api::metal::commandQueue_t metalCommandQueue;

      stream(modeDevice_t *modeDevice_,
             const occa::properties &properties_,
             api::metal::commandQueue_t metalCommandQueue_);

      virtual ~stream();
    };
  }
}

#endif
