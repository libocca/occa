#include <occa/internal/core/device.hpp>
#include <occa/internal/core/streamTag.hpp>

namespace occa {
  modeStreamTag_t::modeStreamTag_t(modeDevice_t *modeDevice_) :
    modeDevice(modeDevice_) {
    modeDevice->addStreamTagRef(this);
  }

  modeStreamTag_t::~modeStreamTag_t() {
    // NULL all wrappers
    while (streamTagRing.head) {
      streamTag *mem = (streamTag*) streamTagRing.head;
      streamTagRing.removeRef(mem);
      mem->modeStreamTag = NULL;
    }
    // Remove ref from device
    if (modeDevice) {
      modeDevice->removeStreamTagRef(this);
    }
  }

  void modeStreamTag_t::dontUseRefs() {
    streamTagRing.dontUseRefs();
  }

  void modeStreamTag_t::addStreamTagRef(streamTag *s) {
    streamTagRing.addRef(s);
  }

  void modeStreamTag_t::removeStreamTagRef(streamTag *s) {
    streamTagRing.removeRef(s);
  }

  bool modeStreamTag_t::needsFree() const {
    return streamTagRing.needsFree();
  }
}
