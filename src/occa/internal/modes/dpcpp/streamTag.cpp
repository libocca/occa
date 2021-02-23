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
      return getEventProfilingSubmit(dpcppEvent);
    }

    double streamTag::startTime()
    {
      return getEventProfilingStart(dpcppEvent);
    }

    double streamTag::endTime()
    {
      return getEventProfilingEnd(dpcppEvent);
    }
  } // namespace dpcpp
} // namespace occa
