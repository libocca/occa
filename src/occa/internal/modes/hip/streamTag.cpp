#include <occa/internal/modes/hip/streamTag.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         hipEvent_t hipEvent_) :
      modeStreamTag_t(modeDevice_),
      hipEvent(hipEvent_) {}

    streamTag::~streamTag() {
      OCCA_HIP_ERROR("streamTag: Freeing hipEvent_t",
                      hipEventDestroy(hipEvent));
    }

    void* streamTag::unwrap() {
      return static_cast<void*>(&hipEvent);
    }
  }
}
