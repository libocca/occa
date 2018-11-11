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
