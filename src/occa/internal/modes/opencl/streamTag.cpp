#include <occa/internal/modes/opencl/streamTag.hpp>
#include <occa/internal/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         cl_event clEvent_) :
      modeStreamTag_t(modeDevice_),
      clEvent(clEvent_),
      start_time(-1),
      end_time(-1) {}

    streamTag::~streamTag() {
      OCCA_OPENCL_ERROR("streamTag: Freeing cl_event",
                        clReleaseEvent(clEvent));
    }

    void* streamTag::unwrap() {
      return static_cast<void*>(&clEvent);
    }

    double streamTag::startTime() {
      if (start_time < 0) {
        cl_ulong clTime = 0;
        OCCA_OPENCL_ERROR("streamTag: Getting event profiling info",
                          clGetEventProfilingInfo(clEvent,
                                                  CL_PROFILING_COMMAND_START,
                                                  sizeof(cl_ulong),
                                                  &clTime, NULL));
        constexpr double nanoseconds{1.0e-9};
        start_time = nanoseconds * static_cast<double>(clTime);
      }
      return start_time;
    }

    double streamTag::endTime() {
      if (end_time < 0) {
        cl_ulong clTime = 0;
        OCCA_OPENCL_ERROR("streamTag: Getting event profiling info",
                          clGetEventProfilingInfo(clEvent,
                                                  CL_PROFILING_COMMAND_END,
                                                  sizeof(cl_ulong),
                                                  &clTime, NULL));
        constexpr double nanoseconds{1.0e-9};
        end_time = nanoseconds * static_cast<double>(clTime);
      }
      return end_time;
    }
  }
}
