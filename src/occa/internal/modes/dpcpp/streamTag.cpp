#include <occa/internal/modes/dpcpp/streamTag.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>

namespace occa
{
  namespace dpcpp
  {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         ::sycl::event dpcppEvent_)
        : modeStreamTag_t(modeDevice_),
          dpcppEvent{dpcppEvent_}
    {
    }

    void streamTag::waitFor()
    {
          OCCA_DPCPP_ERROR("streamTag: waitFor",
                           dpcppEvent.wait_and_throw())
    }

    double streamTag::submitTime()
    {
      // double submit_time;
      // OCCA_DPCPP_ERROR(
      //     "streamTag: startTime",
      //     submit_time = dpcppEvent.template get_profiling_info<sycl::info::event_profiling::command_submit>())
      // return submit_time;
      return 0.0;
    }

    double streamTag::startTime()
    {
      // double start_time;
      // OCCA_DPCPP_ERROR(
      //     "streamTag: startTime",
      //     start_time = dpcppEvent.template get_profiling_info<sycl::info::event_profiling::command_start>())
      // return start_time;
      return 0.0;
    }

    double streamTag::endTime()
    {
    //  double end_time;
    //   OCCA_DPCPP_ERROR(
    //       "streamTag: endTime",
    //       end_time = dpcppEvent.template get_profiling_info<sycl::info::event_profiling::command_end>())
    //   return end_time;
      return 0.0;
    }
  } // namespace dpcpp
} // namespace occa
