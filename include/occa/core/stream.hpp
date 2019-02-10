#ifndef OCCA_CORE_STREAM_HEADER
#define OCCA_CORE_STREAM_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/io/output.hpp>
#include <occa/tools/gc.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  class modeStream_t; class stream;
  class modeDevice_t; class device;

  //---[ modeStream_t ]---------------------
  class modeStream_t : public gc::ringEntry_t {
  public:
    occa::properties properties;

    gc::ring_t<stream> streamRing;

    modeDevice_t *modeDevice;

    modeStream_t(modeDevice_t *modeDevice_,
                 const occa::properties &properties_);

    void dontUseRefs();
    void addStreamRef(stream *s);
    void removeStreamRef(stream *s);
    bool needsFree() const;

    //---[ Virtual Methods ]------------
    virtual ~modeStream_t() = 0;
    //==================================
  };
  //====================================

  //---[ stream ]-----------------------
  class stream : public gc::ringEntry_t {
    friend class occa::modeStream_t;
    friend class occa::device;

  private:
    modeStream_t *modeStream;

  public:
    stream();
    stream(modeStream_t *modeStream_);

    stream(const stream &s);
    stream& operator = (const stream &m);
    ~stream();

  private:
    void setModeStream(modeStream_t *modeStream_);
    void removeStreamRef();

  public:
    void dontUseRefs();

    bool isInitialized() const;

    modeStream_t* getModeStream() const;
    modeDevice_t* getModeDevice() const;

    occa::device getDevice() const;

    const std::string& mode() const;
    const occa::properties& properties() const;

    bool operator == (const occa::stream &other) const;
    bool operator != (const occa::stream &other) const;

    void free();
  };
  //====================================

  std::ostream& operator << (std::ostream &out,
                           const occa::stream &stream);
}

#endif
