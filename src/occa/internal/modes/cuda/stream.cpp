#include <occa/internal/modes/cuda/stream.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   CUstream cuStream_, bool isWrapped_) :
      modeStream_t(modeDevice_, properties_),
      cuStream(cuStream_),
      isWrapped(isWrapped_) {}

    stream::~stream() {
      if (!isWrapped) {
        OCCA_CUDA_DESTRUCTOR_ERROR(
          "Device: freeStream",
          cuStreamDestroy(cuStream)
        );
      }
    }

    void stream::finish() {
      OCCA_CUDA_ERROR("Stream: Finish",
                      cuStreamSynchronize(cuStream));
    }

    void* stream::unwrap() {
      return static_cast<void*>(&cuStream);
    }
  }
}
