#ifndef OCCA_INTERNAL_MODES_SERIAL_STREAM_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_STREAM_HEADER

#include <occa/defines.hpp>
#include <occa/internal/core/stream.hpp>

namespace occa {
  namespace serial {
    class stream : public occa::modeStream_t {
    public:
      stream(modeDevice_t *modeDevice_,
             const occa::json &properties_);

      virtual ~stream();
    };
  }
}

#endif
