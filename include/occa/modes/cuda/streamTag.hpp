#include <occa/defines.hpp>

#ifndef OCCA_MODES_CUDA_STREAMTAG_HEADER
#define OCCA_MODES_CUDA_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/modes/cuda/polyfill.hpp>

namespace occa {
  namespace cuda {
    class streamTag : public occa::modeStreamTag_t {
    public:
      CUevent cuEvent;

      streamTag(modeDevice_t *modeDevice_,
                CUevent cuEvent_);

      virtual ~streamTag();
    };
  }
}

#endif
