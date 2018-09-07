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

#ifndef OCCA_CORE_STREAMTAG_HEADER
#define OCCA_CORE_STREAMTAG_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/tools/gc.hpp>

namespace occa {
  class modeDevice_t; class device;
  class modeStreamTag_t; class streamTag;

  //---[ modeStreamTag_t ]---------------------
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
  //====================================

  //---[ streamTag ]-----------------------
  class streamTag : public gc::ringEntry_t {
    friend class occa::modeStreamTag_t;
    friend class occa::device;

  private:
    modeStreamTag_t *modeStreamTag;

  public:
    streamTag();
    streamTag(modeStreamTag_t *modeStreamTag_);

    streamTag(const streamTag &s);
    streamTag& operator = (const streamTag &m);
    ~streamTag();

  private:
    void setModeStreamTag(modeStreamTag_t *modeStreamTag_);
    void removeStreamTagRef();

  public:
    void dontUseRefs();

    bool isInitialized() const;

    modeStreamTag_t* getModeStreamTag() const;
    modeDevice_t* getModeDevice() const;

    occa::device getDevice() const;

    void wait() const;

    bool operator == (const occa::streamTag &other) const;
    bool operator != (const occa::streamTag &other) const;

    void free();
  };
  //====================================
}

#endif
