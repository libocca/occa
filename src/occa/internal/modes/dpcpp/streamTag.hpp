#ifndef OCCA_MODES_DPCPP_STREAMTAG_HEADER
#define OCCA_MODES_DPCPP_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/modes/dpcpp/polyfill.hpp>

namespace occa {
  namespace dpcpp {
    class streamTag : public occa::modeStreamTag_t {
    public:
    streamTag(modeDevice_t *modeDevice_);
      virtual ~streamTag();
      double getTime();
    };
  }
}

#endif
