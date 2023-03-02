#ifndef OCCA_INTERNAL_CORE_STREAM_HEADER
#define OCCA_INTERNAL_CORE_STREAM_HEADER

#include <occa/core/stream.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/gc.hpp>

namespace occa {
  class modeStream_t : public gc::ringEntry_t {
   public:
    occa::json properties;

    gc::ring_t<stream> streamRing;

    modeDevice_t *modeDevice;

    modeStream_t(modeDevice_t *modeDevice_,
                 const occa::json &json_);
    virtual ~modeStream_t();

    void dontUseRefs();
    void addStreamRef(stream *s);
    void removeStreamRef(stream *s);
    bool needsFree() const;

    //---[ Virtual Methods ]------------
    virtual void finish() = 0;

    virtual void* unwrap() = 0;
    //==================================
  };
}

#endif
