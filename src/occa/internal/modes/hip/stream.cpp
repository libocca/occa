#include <occa/internal/modes/hip/stream.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   hipStream_t hipStream_) :
      modeStream_t(modeDevice_, properties_),
      hipStream(hipStream_) {}

    stream::~stream() {
      OCCA_HIP_ERROR("Device: freeStream",
                      hipStreamDestroy(hipStream));
    }
  }
}
