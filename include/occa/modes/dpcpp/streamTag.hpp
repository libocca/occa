#ifndef OCCA_MODES_DPCPP_STREAMTAG_HEADER
#define OCCA_MODES_DPCPP_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/modes/dpcpp/polyfill.hpp>

namespace occa {
  namespace dpcpp {
    class streamTag : public occa::modeStreamTag_t {
    public:
      cl_event clEvent;
      double time;

      streamTag(modeDevice_t *modeDevice_,
                cl_event clEvent_);

      virtual ~streamTag();

      double getTime();
    };
  }
}

#endif
