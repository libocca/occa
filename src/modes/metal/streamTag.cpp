#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/modes/metal/streamTag.hpp>
#include <occa/modes/metal/headers.hpp>

namespace occa {
  namespace metal {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         metalEvent_t metalEvent_) :
      modeStreamTag_t(modeDevice_),
      metalEvent(metalEvent_),
      time(-1) {}

    streamTag::~streamTag() {
      // TODO
    }

    double streamTag::getTime() {
      if (time < 0) {
        // TODO
      }
      return time;
    }
  }
}

#endif
