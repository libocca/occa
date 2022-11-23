#include <occa/internal/modes/metal/streamTag.hpp>

namespace occa {
  namespace metal {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         api::metal::event_t metalEvent_) :
        modeStreamTag_t(modeDevice_),
        metalEvent(metalEvent_),
        time(-1) {}

    streamTag::~streamTag() {
      metalEvent.free();
    }

    double streamTag::getTime() {
      return metalEvent.getTime();
    }

    void* streamTag::unwrap() {
      return static_cast<void*>(&metalEvent);
    }
  }
}
