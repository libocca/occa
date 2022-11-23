#ifndef OCCA_INTERNAL_MODES_OPENCL_STREAMTAG_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_STREAMTAG_HEADER

#include <occa/internal/core/streamTag.hpp>
#include <occa/internal/modes/opencl/polyfill.hpp>

namespace occa {
  namespace opencl {
    class streamTag : public occa::modeStreamTag_t {
    public:
      cl_event clEvent;
      double start_time;
      double end_time;

      streamTag(modeDevice_t *modeDevice_,
                cl_event clEvent_);

      virtual ~streamTag();

      void* unwrap() override;

      double startTime();
      double endTime();
    };
  }
}

#endif
