#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/modes/metal/stream.hpp>

namespace occa {
  namespace metal {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::properties &properties_,
                   metalCommandQueue_t metalCommandQueue_) :
      modeStream_t(modeDevice_, properties_),
      metalCommandQueue(metalCommandQueue_) {}

    stream::~stream() {
      // TODO
    }
  }
}

#endif
