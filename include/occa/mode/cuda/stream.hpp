#include <occa/defines.hpp>

#if OCCA_CUDA_ENABLED
#  ifndef OCCA_MODES_CUDA_STREAM_HEADER
#  define OCCA_MODES_CUDA_STREAM_HEADER

#include <cuda.h>

#include <occa/core/stream.hpp>

namespace occa {
  namespace cuda {
    class stream : public occa::modeStream_t {
    public:
      CUstream cuStream;

      stream(modeDevice_t *modeDevice_,
             const occa::properties &properties_,
             CUstream cuStream_);

      virtual ~stream();
    };
  }
}

#  endif
#endif
