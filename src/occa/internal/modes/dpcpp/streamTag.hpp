#ifndef OCCA_MODES_DPCPP_STREAMTAG_HEADER
#define OCCA_MODES_DPCPP_STREAMTAG_HEADER

#include <occa/internal/core/streamTag.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>

namespace occa
{
  namespace dpcpp
  {
    class streamTag : public occa::modeStreamTag_t
    {
    public:
      ::sycl::event dpcppEvent;

      streamTag(modeDevice_t *modeDevice_,
                ::sycl::event dpcppEvent_);

      virtual ~streamTag() = default;

      void* unwrap() override;

      void waitFor();
      double submitTime();
      double startTime();
      double endTime();
    };
  } // namespace dpcpp
} // namespace occa

#endif
