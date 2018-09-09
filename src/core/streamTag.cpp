/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

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
