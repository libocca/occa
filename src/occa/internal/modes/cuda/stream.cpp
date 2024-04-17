#include <occa/internal/modes/cuda/stream.hpp>
#include <occa/internal/modes/cuda/streamTag.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   CUstream cuStream_, bool isWrapped_) :
      modeStream_t(modeDevice_, properties_),
      cuStream(cuStream_),
      isWrapped(isWrapped_) {}

    stream::~stream() {
      if (!isWrapped) {
        OCCA_CUDA_DESTRUCTOR_ERROR(
          "Device: freeStream",
          cuStreamDestroy(cuStream)
        );
      }
    }

    void stream::finish() {
      OCCA_CUDA_ERROR("Stream: Finish",
                      cuStreamSynchronize(cuStream));
    }

    void stream::waitFor(occa::streamTag tag) {
      occa::cuda::streamTag *cuTag = (
        dynamic_cast<occa::cuda::streamTag*>(tag.getModeStreamTag())
      );
      OCCA_CUDA_ERROR("Stream: waitFor",
                      cuStreamWaitEvent(cuStream, cuTag->cuEvent, 0));
    }

    void* stream::unwrap() {
      return static_cast<void*>(&cuStream);
    }
  }
}
