#ifndef OCCA_CORE_STREAM_HEADER
#define OCCA_CORE_STREAM_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/types.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeStream_t; class stream;
  class modeDevice_t; class device;

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
    const occa::json& properties() const;

    bool operator == (const occa::stream &other) const;
    bool operator != (const occa::stream &other) const;

    void free();
  };
  //====================================

  std::ostream& operator << (std::ostream &out,
                             const occa::stream &stream);
}

#endif
