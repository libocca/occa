#include <occa/core/streamTag.hpp>
#include <occa/core/device.hpp>
#include <occa/internal/core/device.hpp>
#include <occa/internal/core/streamTag.hpp>

namespace occa {
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
