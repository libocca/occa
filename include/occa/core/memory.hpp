#ifndef OCCA_CORE_MEMORY_HEADER
#define OCCA_CORE_MEMORY_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/dtype.hpp>
#include <occa/types.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelArg;

  typedef std::map<hash_t,occa::memory>   hashedMemoryMap;
  typedef hashedMemoryMap::iterator       hashedMemoryMapIterator;
  typedef hashedMemoryMap::const_iterator cHashedMemoryMapIterator;

  //---[ memory ]-----------------------
  class memory : public gc::ringEntry_t {
    friend class occa::modeMemory_t;
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    modeMemory_t *modeMemory;

  public:
    memory();
    memory(void *uvaPtr);
    memory(modeMemory_t *modeMemory_);

    memory(const memory &m);
    memory& operator = (const memory &m);
    ~memory();

  private:
    void assertInitialized() const;
    void setModeMemory(modeMemory_t *modeMemory_);
    void removeMemoryRef();

  public:
    void dontUseRefs();

    bool isInitialized() const;

    memory& swap(memory &m);

    template <class TM = void>
    TM* ptr();

    template <class TM = void>
    const TM* ptr() const;

    modeMemory_t* getModeMemory() const;
    modeDevice_t* getModeDevice() const;

    occa::device getDevice() const;

    operator kernelArg() const;

    const std::string& mode() const;
    const occa::json& properties() const;

    void setDtype(const dtype_t &dtype__);

    const dtype_t& dtype() const;

    udim_t size() const;
    udim_t length() const;

    template <class TM>
    udim_t length() const {
      return size() / sizeof(TM);
    }

    //---[ UVA ]------------------------
    bool isManaged() const;
    bool inDevice() const;
    bool isStale() const;

    void setupUva();
    void startManaging();
    void stopManaging();

    void syncToDevice(const dim_t bytes, const dim_t offset);
    void syncToHost(const dim_t bytes, const dim_t offset);

    bool uvaIsStale() const;
    void uvaMarkStale();
    void uvaMarkFresh();
    //==================================

    bool operator == (const occa::memory &other) const;
    bool operator != (const occa::memory &other) const;

    occa::memory operator + (const dim_t offset) const;
    occa::memory& operator += (const dim_t offset);

    occa::memory slice(const dim_t offset,
                       const dim_t count = -1) const;

    void copyFrom(const void *src,
                  const dim_t bytes = -1,
                  const dim_t offset = 0,
                  const occa::json &props = occa::json());

    void copyFrom(const memory src,
                  const dim_t bytes = -1,
                  const dim_t destOffset = 0,
                  const dim_t srcOffset = 0,
                  const occa::json &props = occa::json());

    void copyTo(void *dest,
                const dim_t bytes = -1,
                const dim_t offset = 0,
                const occa::json &props = occa::json()) const;

    void copyTo(const memory dest,
                const dim_t bytes = -1,
                const dim_t destOffset = 0,
                const dim_t srcOffset = 0,
                const occa::json &props = occa::json()) const;

    void copyFrom(const void *src,
                  const occa::json &props);

    void copyFrom(const memory src,
                  const occa::json &props);

    void copyTo(void *dest,
                const occa::json &props) const;

    void copyTo(const memory dest,
                const occa::json &props) const;

    occa::memory cast(const dtype_t &dtype_) const;

    occa::memory clone() const;

    void free();
    void detach();
    void deleteRefs(const bool freeMemory = false);
  };

  extern memory null;
  //====================================

  std::ostream& operator << (std::ostream &out,
                           const occa::memory &memory);

  template <>
  void* memory::ptr<void>();

  template <>
  const void* memory::ptr<void>() const;
}

#include "memory.tpp"

#endif
