#ifndef OCCA_MODES_DPCPP_STREAM_HEADER
#define OCCA_MODES_DPCPP_STREAM_HEADER

#include <occa/internal/core/stream.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>

namespace occa {
  namespace dpcpp {
    class streamTag;
    
    class stream : public occa::modeStream_t {
    public:
	    ::sycl::queue commandQueue;

      stream(modeDevice_t *modeDevice_,
             const occa::json &properties_,
             ::sycl::queue commandQueue_);

      virtual ~stream()=default;

      void finish() override;

      void* unwrap() override;

      occa::dpcpp::streamTag memcpy(void *dest, const void *src, occa::udim_t num_bytes);
    };
  }
}

#endif
