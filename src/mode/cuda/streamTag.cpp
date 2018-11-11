#include <occa/defines.hpp>

#if OCCA_CUDA_ENABLED

#include <occa/mode/cuda/streamTag.hpp>
#include <occa/mode/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         CUevent cuEvent_) :
      modeStreamTag_t(modeDevice_),
      cuEvent(cuEvent_) {}

    streamTag::~streamTag() {
      OCCA_CUDA_ERROR("streamTag: Freeing CUevent",
                      cuEventDestroy(cuEvent));
    }
  }
}

#endif
