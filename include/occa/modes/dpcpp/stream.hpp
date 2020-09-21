#ifndef OCCA_MODES_DPCPP_STREAM_HEADER
#define OCCA_MODES_DPCPP_STREAM_HEADER

#include <occa/core/stream.hpp>
#include <occa/modes/dpcpp/polyfill.hpp>
#include <CL/sycl.hpp>

namespace occa {
  namespace dpcpp {
    class stream : public occa::modeStream_t {
    public:
	    ::sycl::queue* commandQueue;

      stream(modeDevice_t *modeDevice_,
             const occa::properties &properties_,
             ::sycl::queue* commandQueue_);

      virtual ~stream();
    };
  }
}

#endif
