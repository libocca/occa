#include <occa/defines.hpp>

#include <occa/internal/modes/serial/streamTag.hpp>

namespace occa {
  namespace serial {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         double time_) :
      modeStreamTag_t(modeDevice_),
      time(time_) {}

    void* streamTag::unwrap() {
      OCCA_FORCE_ERROR("streamTag::unwrap is not defined for serial mode");
      return nullptr;
    }
  }
}
