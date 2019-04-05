#include <occa/defines.hpp>

#ifndef OCCA_MODES_SERIAL_STREAMTAG_HEADER
#define OCCA_MODES_SERIAL_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>

namespace occa {
  namespace serial {
    class streamTag : public occa::modeStreamTag_t {
    public:
      double time;

      streamTag(modeDevice_t *modeDevice_,
                double time_);

      virtual ~streamTag();
    };
  }
}

#endif
