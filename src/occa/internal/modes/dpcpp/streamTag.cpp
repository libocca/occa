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

    double streamTag::getTime()
    {
      return 0.0;
    }
  } // namespace dpcpp
} // namespace occa
