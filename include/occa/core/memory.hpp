#ifndef OCCA_CORE_MEMORY_HEADER
#define OCCA_CORE_MEMORY_HEADER

#include <iostream>

#include <occa/defines.hpp>
#include <occa/dtype.hpp>
#include <occa/io/output.hpp>
#include <occa/tools/gc.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelArg;

  typedef std::map<hash_t,occa::memory>   hashedMemoryMap;
  typedef hashedMemoryMap::iterator       hashedMemoryMapIterator;
  typedef hashedMemoryMap::const_iterator cHashedMemoryMapIterator;

  namespace uvaFlag {
    static const int none      = 0;
    static const int isManaged = (1 << 0);
    static const int inDevice  = (1 << 1);
    static const int isStale   = (1 << 2);
  }

  //---[ modeMemory_t ]---------------------
  class modeMemory_t : public gc::ringEntry_t {
  public:
    int memInfo;
    occa::properties properties;

    gc::ring_t<memory> memoryRing;

    char *ptr;
    char *uvaPtr;

    occa::modeDevice_t *modeDevice;

    const dtype_t *dtype_;
    udim_t size;
    bool isOrigin;

    modeMemory_t(modeDevice_t *modeDevice_,
                 udim_t size_,
                 const occa::properties &properties_);

    void dontUseRefs();
    void addMemoryRef(memory *mem);
    void removeMemoryRef(memory *mem);
    bool needsFree() const;

    bool isManaged() const;
    bool inDevice() const;
    bool isStale() const;

    //---[ Virtual Methods ]------------
    virtual ~modeMemory_t() = 0;

    virtual kernelArg makeKernelArg() const = 0;

    virtual modeMemory_t* addOffset(const dim_t offset) = 0;

    virtual void* getPtr(const occa::properties &props);

    virtual void copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset = 0,
                        const occa::properties &props = occa::properties()) const = 0;

    virtual void copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset = 0,
                          const occa::properties &props = occa::properties()) = 0;

    virtual void copyFrom(const modeMemory_t *src,
                          const udim_t bytes,
                          const udim_t destOffset = 0,
                          const udim_t srcOffset = 0,
                          const occa::properties &props = occa::properties()) = 0;

    virtual void detach() = 0;
    //==================================

    //---[ Friend Functions ]-----------
    friend void memcpy(void *dest, void *src,
                       const dim_t bytes,
                       const occa::properties &props);

    friend void startManaging(void *ptr);
    friend void stopManaging(void *ptr);

    friend void syncToDevice(void *ptr, const dim_t bytes);
    friend void syncToHost(void *ptr, const dim_t bytes);

    friend void syncMemToDevice(occa::modeMemory_t *mem,
                                const dim_t bytes,
                                const dim_t offset);

    friend void syncMemToHost(occa::modeMemory_t *mem,
                              const dim_t bytes,
                              const dim_t offset);
  };
  //====================================


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

    void setDtype(const dtype_t &dtype__);

  public:
    void dontUseRefs();

    bool isInitialized() const;

    memory& swap(memory &m);

    template <class TM = void>
    TM* ptr();

    template <class TM = void>
    const TM* ptr() const;

    template <class TM = void>
    TM* ptr(const occa::properties &props);

    template <class TM = void>
    const TM* ptr(const occa::properties &props) const;

    modeMemory_t* getModeMemory() const;
    modeDevice_t* getModeDevice() const;

    occa::device getDevice() const;

    operator kernelArg() const;

    const std::string& mode() const;
    const occa::properties& properties() const;

    const dtype_t& dtype() const;

    udim_t size() const;
    udim_t length() const;

    template <class TM>
    udim_t size() const {
      return (modeMemory
              ? (modeMemory->size / sizeof(TM))
              : 0);
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
                       const dim_t bytes = -1) const;

    void copyFrom(const void *src,
                  const dim_t bytes = -1,
                  const dim_t offset = 0,
                  const occa::properties &props = occa::properties());

    void copyFrom(const memory src,
                  const dim_t bytes = -1,
                  const dim_t destOffset = 0,
                  const dim_t srcOffset = 0,
                  const occa::properties &props = occa::properties());

    void copyTo(void *dest,
                const dim_t bytes = -1,
                const dim_t offset = 0,
                const occa::properties &props = occa::properties()) const;

    void copyTo(const memory dest,
                const dim_t bytes = -1,
                const dim_t destOffset = 0,
                const dim_t srcOffset = 0,
                const occa::properties &props = occa::properties()) const;

    void copyFrom(const void *src,
                  const occa::properties &props);

    void copyFrom(const memory src,
                  const occa::properties &props);

    void copyTo(void *dest,
                const occa::properties &props) const;

    void copyTo(const memory dest,
                const occa::properties &props) const;

    occa::memory as(const dtype_t &dtype_) const;

    occa::memory clone() const;

    void free();
    void detach();
    void deleteRefs(const bool freeMemory = false);
  };

  extern memory null;
  //====================================

  std::ostream& operator << (std::ostream &out,
                           const occa::memory &memory);

  namespace cpu {
    occa::memory wrapMemory(occa::device dev,
                            void *ptr,
                            const udim_t bytes,
                            const occa::properties &props = occa::properties());
  }

  template <>
  void* memory::ptr<void>();

  template <>
  const void* memory::ptr<void>() const;

  template <>
  void* memory::ptr<void>(const occa::properties &props);

  template <>
  const void* memory::ptr<void>(const occa::properties &props) const;
}

#include "memory.tpp"

#endif
