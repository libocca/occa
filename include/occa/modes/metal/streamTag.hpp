#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED
#  ifndef OCCA_MODES_METAL_STREAMTAG_HEADER
#  define OCCA_MODES_METAL_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/modes/metal/headers.hpp>

namespace occa {
  namespace metal {
    class streamTag : public occa::modeStreamTag_t {
    public:
      metalEvent_t metalEvent;
      double time;

      streamTag(modeDevice_t *modeDevice_,
                metalEvent_t metalEvent_);

      virtual ~streamTag();

      double getTime();
    };
  }
}

#  endif
#endif
