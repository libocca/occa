#include <occa/internal/modes/serial/stream.hpp>

namespace occa {
  namespace serial {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_) :
      modeStream_t(modeDevice_, properties_) {}

    stream::~stream() {}
  }
}
