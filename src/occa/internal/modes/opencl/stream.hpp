#ifndef OCCA_INTERNAL_MODES_OPENCL_STREAM_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_STREAM_HEADER

#include <occa/internal/core/stream.hpp>
#include <occa/internal/modes/opencl/polyfill.hpp>

namespace occa {
  namespace opencl {
    class stream : public occa::modeStream_t {
    public:
      cl_command_queue commandQueue;

      stream(modeDevice_t *modeDevice_,
             const occa::json &properties_,
             cl_command_queue commandQueue_);

      virtual ~stream();
    };
  }
}

#endif
