#include <occa/internal/modes/hip/stream.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   hipStream_t hipStream_,
                   bool isWrapped_) :
      modeStream_t(modeDevice_, properties_),
      hipStream(hipStream_),
      isWrapped(isWrapped_) {}

    stream::~stream() {
      if (!isWrapped) {
        OCCA_HIP_ERROR("Device: freeStream",
                        hipStreamDestroy(hipStream));
      }
    }

    void stream::finish() {
      OCCA_HIP_ERROR("Stream: Finish",
                     hipStreamSynchronize(hipStream));
    }

    void* stream::unwrap() {
      return static_cast<void*>(&hipStream);
    }
  }
}
