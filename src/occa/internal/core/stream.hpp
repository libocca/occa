#ifndef OCCA_INTERNAL_CORE_STREAM_HEADER
#define OCCA_INTERNAL_CORE_STREAM_HEADER

#include <occa/core/stream.hpp>
#include <occa/types/properties.hpp>
#include <occa/internal/utils/gc.hpp>

namespace occa {
  class modeStream_t : public gc::ringEntry_t {
  public:
    occa::properties properties;

    gc::ring_t<stream> streamRing;

    modeDevice_t *modeDevice;

    modeStream_t(modeDevice_t *modeDevice_,
                 const occa::properties &properties_);

    void dontUseRefs();
    void addStreamRef(stream *s);
    void removeStreamRef(stream *s);
    bool needsFree() const;

    //---[ Virtual Methods ]------------
    virtual ~modeStream_t() = 0;
    //==================================
  };
}

#endif
