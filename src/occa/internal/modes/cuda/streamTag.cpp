#include <occa/internal/modes/cuda/streamTag.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         CUevent cuEvent_) :
      modeStreamTag_t(modeDevice_),
      cuEvent(cuEvent_) {}

    streamTag::~streamTag() {
      OCCA_CUDA_DESTRUCTOR_ERROR(
        "streamTag: Freeing CUevent",
        cuEventDestroy(cuEvent)
      );
    }
  }
}
