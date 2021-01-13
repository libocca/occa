#include <occa/internal/modes/opencl/stream.hpp>
#include <occa/internal/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::json &properties_,
                   cl_command_queue commandQueue_) :
      modeStream_t(modeDevice_, properties_),
      commandQueue(commandQueue_) {}

    stream::~stream() {
      OCCA_OPENCL_ERROR("Device: Freeing cl_command_queue",
                        clReleaseCommandQueue(commandQueue));
    }
  }
}
