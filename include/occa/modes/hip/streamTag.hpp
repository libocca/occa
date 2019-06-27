#ifndef OCCA_MODES_HIP_STREAMTAG_HEADER
#define OCCA_MODES_HIP_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/modes/hip/polyfill.hpp>

namespace occa {
  namespace hip {
    class streamTag : public occa::modeStreamTag_t {
    public:
      hipEvent_t hipEvent;

      streamTag(modeDevice_t *modeDevice_,
                hipEvent_t hipEvent_);

      virtual ~streamTag();
    };
  }
}

#endif
