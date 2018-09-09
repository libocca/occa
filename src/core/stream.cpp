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

#include <occa/core/stream.hpp>
#include <occa/core/device.hpp>

namespace occa {
  //---[ modeStream_t ]-----------------
  modeStream_t::modeStream_t(modeDevice_t *modeDevice_,
                             const occa::properties &properties_) :
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

  //---[ stream ]-----------------------
  stream::stream() :
    modeStream(NULL) {}

  stream::stream(modeStream_t *modeStream_) :
    modeStream(NULL) {
    setModeStream(modeStream_);
  }

  stream::stream(const stream &m) :
    modeStream(NULL) {
    setModeStream(m.modeStream);
  }

  stream& stream::operator = (const stream &m) {
    setModeStream(m.modeStream);
    return *this;
  }

  stream::~stream() {
    removeStreamRef();
  }

  void stream::setModeStream(modeStream_t *modeStream_) {
    if (modeStream != modeStream_) {
      removeStreamRef();
      modeStream = modeStream_;
      if (modeStream) {
        modeStream->addStreamRef(this);
      }
    }
  }

  void stream::removeStreamRef() {
    if (!modeStream) {
      return;
    }
    modeStream->removeStreamRef(this);
    if (modeStream->modeStream_t::needsFree()) {
      free();
    }
  }

  void stream::dontUseRefs() {
    if (modeStream) {
      modeStream->modeStream_t::dontUseRefs();
    }
  }

  bool stream::isInitialized() const {
    return (modeStream != NULL);
  }

  modeStream_t* stream::getModeStream() const {
    return modeStream;
  }

  modeDevice_t* stream::getModeDevice() const {
    return modeStream->modeDevice;
  }

  occa::device stream::getDevice() const {
    return occa::device(modeStream
                        ? modeStream->modeDevice
                        : NULL);
  }

  const std::string& stream::mode() const {
    static const std::string noMode = "No Mode";
    return (modeStream
            ? modeStream->modeDevice->mode
            : noMode);
  }

  const occa::properties& stream::properties() const {
    static const occa::properties noProperties;
    return (modeStream
            ? modeStream->properties
            : noProperties);
  }

  bool stream::operator == (const occa::stream &other) const {
    return (modeStream == other.modeStream);
  }

  bool stream::operator != (const occa::stream &other) const {
    return (modeStream != other.modeStream);
  }

  void stream::free() {
    // ~modeStream_t NULLs all wrappers
    delete modeStream;
    modeStream = NULL;
  }

  std::ostream& operator << (std::ostream &out,
                             const occa::stream &stream) {
    out << stream.properties();
    return out;
  }
}
