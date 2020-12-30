#include <occa/internal/modes/opencl/streamTag.hpp>
#include <occa/internal/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         cl_event clEvent_) :
      modeStreamTag_t(modeDevice_),
      clEvent(clEvent_),
      time(-1) {}

    streamTag::~streamTag() {
      OCCA_OPENCL_ERROR("streamTag: Freeing cl_event",
                        clReleaseEvent(clEvent));
    }

    double streamTag::getTime() {
      if (time < 0) {
        cl_ulong clTime;
        OCCA_OPENCL_ERROR("streamTag: Getting event profiling info",
                          clGetEventProfilingInfo(clEvent,
                                                  CL_PROFILING_COMMAND_END,
                                                  sizeof(cl_ulong),
                                                  &clTime, NULL));
        time = 1.0e-9 * clTime;
      }
      return time;
    }
  }
}
