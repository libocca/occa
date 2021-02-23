#include <occa/internal/modes/dpcpp/stream.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>

namespace occa {
  namespace dpcpp {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   ::sycl::queue* commandQueue_) :
      modeStream_t(modeDevice_, properties_),
      commandQueue(commandQueue_) {}

    stream::~stream() {
     OCCA_DPCPP_ERROR("Stream: Freeing dpcpp queue",
                        free(commandQueue))
    }
  }
}
