#include <occa/modes/serial/stream.hpp>

namespace occa {
  namespace serial {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::properties &properties_) :
      modeStream_t(modeDevice_, properties_) {}

    stream::~stream() {}
  }
}
