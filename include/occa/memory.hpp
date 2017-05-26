/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#ifndef OCCA_MEMORY_HEADER
#define OCCA_MEMORY_HEADER

#include <iostream>

#include "occa/defines.hpp"
#include "occa/tools/gc.hpp"
#include "occa/tools/properties.hpp"

namespace occa {
  class kernel_v; class kernel;
  class memory_v; class memory;
  class device_v; class device;
  class kernelArg;


  typedef std::map<hash_t,occa::memory>     hashedMemoryMap_t;
  typedef hashedMemoryMap_t::iterator       hashedMemoryMapIterator;
  typedef hashedMemoryMap_t::const_iterator cHashedMemoryMapIterator;

  namespace uvaFlag {
    static const int none         = 0;
    static const int isManaged    = (1 << 0);
    static const int inDevice     = (1 << 1);
    static const int isStale      = (1 << 2);
  }

  //---[ memory_v ]---------------------
  class memory_v : public withRefs {
  public:
    int memInfo;
    occa::properties properties;

    char *ptr;
    char *uvaPtr;
    occa::device_v *dHandle;

    udim_t size;

    memory_v(const occa::properties &properties_);

    bool isManaged() const;
    bool inDevice() const;
    bool isStale() const;

    //---[ Virtual Methods ]------------
    virtual ~memory_v() = 0;
    // Must be able to be called multiple times safely
    virtual void free() = 0;

    virtual kernelArg makeKernelArg() const = 0;

    virtual memory_v* addOffset(const dim_t offset, bool &needsFree) = 0;

    virtual void copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset = 0,
                        const occa::properties &props = occa::properties()) const = 0;

    virtual void copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset = 0,
                          const occa::properties &props = occa::properties()) = 0;

    virtual void copyFrom(const memory_v *src,
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

    friend void syncMemToDevice(occa::memory_v *mem,
                                const dim_t bytes,
                                const dim_t offset);

    friend void syncMemToHost(occa::memory_v *mem,
                              const dim_t bytes,
                              const dim_t offset);
  };
  //====================================

  //---[ memory ]-----------------------
  class memory {
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    memory_v *mHandle;

  public:
    memory();
    memory(void *uvaPtr);
    memory(memory_v *mHandle_);

    memory(const memory &m);
    memory& operator = (const memory &m);
    ~memory();

  private:
    void setMHandle(memory_v *mHandle_);
    void setDHandle(device_v *dHandle_);
    void removeMHandleRef();

  public:
    void dontUseRefs();

    bool isInitialized() const;

    memory& swap(memory &m);

    void* ptr();
    const void* ptr() const;

    memory_v* getMHandle() const;
    device_v* getDHandle() const;

    occa::device getDevice() const;

    operator kernelArg() const;

    const std::string& mode() const;

    udim_t size() const;

    template <class TM>
    udim_t size() const {
      if (mHandle == NULL) {
        return 0;
      }
      return (mHandle->size / sizeof(TM));
    }

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

    occa::memory operator + (const dim_t offset) const;
    occa::memory& operator += (const dim_t offset);

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

    void free();
    void detach();
    void deleteRefs(const bool freeMemory = false);
  };
  //====================================

  namespace cpu {
    occa::memory wrapMemory(void *ptr, const udim_t bytes);
  }
}

#endif
