#include <occa/modes/hip/stream.hpp>
#include <occa/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::properties &properties_,
                   hipStream_t hipStream_) :
      modeStream_t(modeDevice_, properties_),
      hipStream(hipStream_) {}

    stream::~stream() {
      OCCA_HIP_ERROR("Device: freeStream",
                      hipStreamDestroy(hipStream));
    }
  }
}
