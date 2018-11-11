#ifndef OCCA_MODES_SERIAL_STREAM_HEADER
#define OCCA_MODES_SERIAL_STREAM_HEADER

#include <occa/defines.hpp>
#include <occa/core/stream.hpp>

namespace occa {
  namespace serial {
    class stream : public occa::modeStream_t {
    public:
      stream(modeDevice_t *modeDevice_,
             const occa::properties &properties_);

      virtual ~stream();
    };
  }
}

#endif
