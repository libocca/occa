#include <occa/internal/modes/metal/stream.hpp>

namespace occa {
  namespace metal {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   api::metal::commandQueue_t metalCommandQueue_,
                   bool isWrapped_) :
        modeStream_t(modeDevice_, properties_),
        metalCommandQueue(metalCommandQueue_),
        isWrapped(isWrapped_) {}

    stream::~stream() {
      if (!isWrapped) {
        metalCommandQueue.free();
      }
    }

    void stream::finish() {
      metalCommandQueue.finish();
    }

    void* stream::unwrap() {
      return static_cast<void*>(&metalCommandQueue);
    }
  }
}
