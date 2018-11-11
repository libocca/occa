#include <occa/defines.hpp>

#if OCCA_CUDA_ENABLED

#include <occa/mode/cuda/stream.hpp>
#include <occa/mode/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::properties &properties_,
                   CUstream cuStream_) :
      modeStream_t(modeDevice_, properties_),
      cuStream(cuStream_) {}

    stream::~stream() {
      OCCA_CUDA_ERROR("Device: freeStream",
                      cuStreamDestroy(cuStream));
    }
  }
}

#endif
