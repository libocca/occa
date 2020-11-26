#include <occa/modes/cuda/stream.hpp>
#include <occa/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::properties &properties_,
                   CUstream cuStream_) :
      modeStream_t(modeDevice_, properties_),
      cuStream(cuStream_) {}

    stream::~stream() {
      OCCA_CUDA_DESTRUCTOR_ERROR(
        "Device: freeStream",
        cuStreamDestroy(cuStream)
      );
    }
  }
}
