#ifndef OCCA_MODES_HIP_STREAM_HEADER
#define OCCA_MODES_HIP_STREAM_HEADER

#include <occa/core/stream.hpp>
#include <occa/modes/hip/polyfill.hpp>

namespace occa {
  namespace hip {
    class stream : public occa::modeStream_t {
    public:
      hipStream_t hipStream;

      stream(modeDevice_t *modeDevice_,
             const occa::properties &properties_,
             hipStream_t hipStream_);

      virtual ~stream();
    };
  }
}

#endif
