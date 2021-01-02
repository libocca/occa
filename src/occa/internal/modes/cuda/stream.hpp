#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_CUDA_STREAM_HEADER
#define OCCA_INTERNAL_MODES_CUDA_STREAM_HEADER

#include <occa/internal/core/stream.hpp>
#include <occa/internal/modes/cuda/polyfill.hpp>

namespace occa {
  namespace cuda {
    class stream : public occa::modeStream_t {
    public:
      CUstream cuStream;

      stream(modeDevice_t *modeDevice_,
             const occa::json &properties_,
             CUstream cuStream_);

      virtual ~stream();
    };
  }
}

#endif
