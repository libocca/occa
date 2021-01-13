#ifndef OCCA_INTERNAL_MODES_METAL_STREAMTAG_HEADER
#define OCCA_INTERNAL_MODES_METAL_STREAMTAG_HEADER

#include <occa/internal/core/streamTag.hpp>
#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace metal {
    class streamTag : public occa::modeStreamTag_t {
    public:
      api::metal::event_t metalEvent;
      double time;

      streamTag(modeDevice_t *modeDevice_,
                api::metal::event_t metalEvent_);

      virtual ~streamTag();

      double getTime();
    };
  }
}

#endif
