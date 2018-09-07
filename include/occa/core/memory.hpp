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

#ifndef OCCA_CORE_MEMORY_HEADER
#define OCCA_CORE_MEMORY_HEADER

#include <iostream>

#include <occa/defines.hpp>
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

  public:
    void dontUseRefs();

    bool isInitialized() const;

    memory& swap(memory &m);

    void* ptr();
    const void* ptr() const;

    void* ptr(const occa::properties &props);
    const void* ptr(const occa::properties &props) const;

    modeMemory_t* getModeMemory() const;
    modeDevice_t* getModeDevice() const;

    occa::device getDevice() const;

    operator kernelArg() const;

    const std::string& mode() const;
    const occa::properties& properties() const;

    udim_t size() const;

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

    occa::memory clone() const;

    void free();
    void detach();
    void deleteRefs(const bool freeMemory = false);
  };
  //====================================

  std::ostream& operator << (std::ostream &out,
                             const occa::memory &memory);

  namespace cpu {
    occa::memory wrapMemory(occa::device dev,
                            void *ptr,
                            const udim_t bytes,
                            const occa::properties &props = occa::properties());
  }
}

#endif
