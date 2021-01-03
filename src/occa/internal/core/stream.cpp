#include <occa/internal/core/device.hpp>
#include <occa/internal/core/stream.hpp>

namespace occa {
  modeStream_t::modeStream_t(modeDevice_t *modeDevice_,
                             const occa::json &properties_) :
    properties(properties_),
    modeDevice(modeDevice_) {
    modeDevice->addStreamRef(this);
  }

  modeStream_t::~modeStream_t() {
    // NULL all wrappers
    while (streamRing.head) {
      stream *mem = (stream*) streamRing.head;
      streamRing.removeRef(mem);
      mem->modeStream = NULL;
    }
    // Remove ref from device
    if (modeDevice) {
      modeDevice->removeStreamRef(this);
    }
  }

  void modeStream_t::dontUseRefs() {
    streamRing.dontUseRefs();
  }

  void modeStream_t::addStreamRef(stream *s) {
    streamRing.addRef(s);
  }

  void modeStream_t::removeStreamRef(stream *s) {
    streamRing.removeRef(s);
  }

  bool modeStream_t::needsFree() const {
    return streamRing.needsFree();
  }
}
