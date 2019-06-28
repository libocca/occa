#include <occa/modes/metal/streamTag.hpp>
#include <occa/modes/metal/bridge.hpp>

namespace occa {
  namespace metal {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         metalEvent_t metalEvent_) :
      modeStreamTag_t(modeDevice_),
      metalEvent(metalEvent_),
      time(-1) {}

    streamTag::~streamTag() {
      metalEvent.free();
    }

    double streamTag::getTime() {
      return metalEvent.getTime();
    }
  }
}
