#include <occa/internal/modes/serial/stream.hpp>
#include <occa/internal/modes/serial/streamTag.hpp>

namespace occa {
  namespace serial {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_) :
      modeStream_t(modeDevice_, properties_) {}

    stream::~stream() {}
    void stream::finish() {}
    void stream::waitFor(occa::streamTag tag) {}

    void* stream::unwrap() {
      OCCA_FORCE_ERROR("stream::unwrap is not defined for serial mode");
      return nullptr;
    }
  }
}
