#include <occa/defines.hpp>

#if OCCA_OPENCL_ENABLED

#include <occa/mode/opencl/stream.hpp>
#include <occa/mode/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    stream::stream(modeDevice_t *modeDevice_,
                   const occa::properties &properties_,
                   cl_command_queue commandQueue_) :
      modeStream_t(modeDevice_, properties_),
      commandQueue(commandQueue_) {}

    stream::~stream() {
      OCCA_OPENCL_ERROR("Device: Freeing cl_command_queue",
                        clReleaseCommandQueue(commandQueue));
    }
  }
}

#endif
