#ifndef OCCA_INTERNAL_CORE_STREAMTAG_HEADER
#define OCCA_INTERNAL_CORE_STREAMTAG_HEADER

#include <occa/core/streamTag.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/gc.hpp>

namespace occa {
  class modeStreamTag_t : public gc::ringEntry_t {
   public:
    gc::ring_t<streamTag> streamTagRing;

    modeDevice_t *modeDevice;

    modeStreamTag_t(modeDevice_t *modeDevice_);

    void dontUseRefs();
    void addStreamTagRef(streamTag *s);
    void removeStreamTagRef(streamTag *s);
    bool needsFree() const;

    //---[ Virtual Methods ]------------
    virtual ~modeStreamTag_t();
    //==================================
  };
}

#endif
