#ifndef OCCA_MODES_OPENCL_STREAM_HEADER
#define OCCA_MODES_OPENCL_STREAM_HEADER

#include <occa/core/stream.hpp>
#include <occa/modes/opencl/polyfill.hpp>

namespace occa {
  namespace opencl {
    class stream : public occa::modeStream_t {
    public:
      cl_command_queue commandQueue;

      stream(modeDevice_t *modeDevice_,
             const occa::properties &properties_,
             cl_command_queue commandQueue_);

      virtual ~stream();
    };
  }
}

#endif
