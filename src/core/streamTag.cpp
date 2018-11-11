#include <occa/core/streamTag.hpp>
#include <occa/core/device.hpp>

namespace occa {
  //---[ modeStreamTag_t ]--------------
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

  //---[ streamTag ]-----------------------
  streamTag::streamTag() :
    modeStreamTag(NULL) {}

  streamTag::streamTag(modeStreamTag_t *modeStreamTag_) :
    modeStreamTag(NULL) {
    setModeStreamTag(modeStreamTag_);
  }

  streamTag::streamTag(const streamTag &m) :
    modeStreamTag(NULL) {
    setModeStreamTag(m.modeStreamTag);
  }

  streamTag& streamTag::operator = (const streamTag &m) {
    setModeStreamTag(m.modeStreamTag);
    return *this;
  }

  streamTag::~streamTag() {
    removeStreamTagRef();
  }

  void streamTag::setModeStreamTag(modeStreamTag_t *modeStreamTag_) {
    if (modeStreamTag != modeStreamTag_) {
      removeStreamTagRef();
      modeStreamTag = modeStreamTag_;
      if (modeStreamTag) {
        modeStreamTag->addStreamTagRef(this);
      }
    }
  }

  void streamTag::removeStreamTagRef() {
    if (!modeStreamTag) {
      return;
    }
    modeStreamTag->removeStreamTagRef(this);
    if (modeStreamTag->modeStreamTag_t::needsFree()) {
      free();
    }
  }

  void streamTag::dontUseRefs() {
    if (modeStreamTag) {
      modeStreamTag->modeStreamTag_t::dontUseRefs();
    }
  }

  bool streamTag::isInitialized() const {
    return (modeStreamTag != NULL);
  }

  modeStreamTag_t* streamTag::getModeStreamTag() const {
    return modeStreamTag;
  }

  modeDevice_t* streamTag::getModeDevice() const {
    return modeStreamTag->modeDevice;
  }

  occa::device streamTag::getDevice() const {
    return occa::device(modeStreamTag
                        ? modeStreamTag->modeDevice
                        : NULL);
  }

  void streamTag::wait() const {
    if (modeStreamTag) {
      modeStreamTag->modeDevice->waitFor(*this);
    }
  }

  bool streamTag::operator == (const occa::streamTag &other) const {
    return (modeStreamTag == other.modeStreamTag);
  }

  bool streamTag::operator != (const occa::streamTag &other) const {
    return (modeStreamTag != other.modeStreamTag);
  }

  void streamTag::free() {
    // ~modeStreamTag_t NULLs all wrappers
    delete modeStreamTag;
    modeStreamTag = NULL;
  }
}
