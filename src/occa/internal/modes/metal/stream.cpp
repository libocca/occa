#include <occa/internal/modes/metal/stream.hpp>
#include <occa/internal/modes/metal/streamTag.hpp>

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

    void stream::waitFor(occa::streamTag tag) {
      occa::metal::streamTag *metalTag = (
        dynamic_cast<occa::metal::streamTag*>(tag.getModeStreamTag())
      );
      metalCommandQueue.waitForEvent(metalTag->metalEvent);
    }

    void* stream::unwrap() {
      return static_cast<void*>(&metalCommandQueue);
    }
  }
}
