#ifndef OCCA_MODES_METAL_STREAMTAG_HEADER
#define OCCA_MODES_METAL_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/api/metal.hpp>

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
