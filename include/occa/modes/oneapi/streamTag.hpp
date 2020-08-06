#ifndef OCCA_MODES_OPENCL_STREAMTAG_HEADER
#define OCCA_MODES_OPENCL_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/modes/opencl/polyfill.hpp>

namespace occa {
  namespace opencl {
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
