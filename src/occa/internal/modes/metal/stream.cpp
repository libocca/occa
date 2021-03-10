#include <occa/internal/modes/metal/stream.hpp>

namespace occa {
  namespace metal {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   api::metal::commandQueue_t metalCommandQueue_) :
        modeStream_t(modeDevice_, properties_),
        metalCommandQueue(metalCommandQueue_) {}

    stream::~stream() {
      metalCommandQueue.free();
    }
  }
}
