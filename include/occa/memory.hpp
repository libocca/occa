/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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
#include "occa/tools/properties.hpp"

namespace occa {
  class kernel_v; class kernel;
  class memory_v; class memory;
  class device_v; class device;
  class kernelArg;

  namespace uvaFlag {
    static const int none         = 0;
    static const int isManaged    = (1 << 0);
    static const int inDevice     = (1 << 1);
    static const int isStale      = (1 << 2);
  }

  //---[ memory_v ]---------------------
  class memory_v {
  public:
    int memInfo;
    occa::properties properties;

    void *handle, *uvaPtr;
    occa::device_v *dHandle;

    udim_t size;

    memory_v(const occa::properties &properties_);

    void initFrom(const memory_v &m);

    bool isManaged() const;
    bool inDevice() const;
    bool isStale() const;

    void* uvaHandle();

    //---[ Virtual Methods ]------------
    virtual ~memory_v() = 0;
    virtual void free() = 0;

    virtual void* getHandle(const occa::properties &props) = 0;
    virtual kernelArg makeKernelArg() const = 0;

    virtual void copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset = 0,
                        const occa::properties &props = occa::properties()) = 0;

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
    friend void syncFromDevice(void *ptr, const dim_t bytes);

    friend void syncMemToDevice(occa::memory_v *mem,
                                const dim_t bytes,
                                const dim_t offset);

    friend void syncMemFromDevice(occa::memory_v *mem,
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

    bool isInitialized() const;

    memory& swap(memory &m);

    memory_v* getMHandle();
    device_v* getDHandle();

    occa::device getDevice() const;

    operator kernelArg() const;

    const std::string& mode() const;

    udim_t size() const;

    bool isManaged() const;
    bool inDevice() const;
    bool isStale() const;

    void* getHandle(const occa::properties &props = occa::properties());

    void setupUva();
    void startManaging();
    void stopManaging();

    void syncToDevice(const dim_t bytes, const dim_t offset);
    void syncFromDevice(const dim_t bytes, const dim_t offset);

    bool uvaIsStale();
    void uvaMarkStale();
    void uvaMarkFresh();

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
                const occa::properties &props = occa::properties());

    void copyTo(const memory dest,
                const dim_t bytes = -1,
                const dim_t destOffset = 0,
                const dim_t srcOffset = 0,
                const occa::properties &props = occa::properties());

    void free();
    void detach();
    void deleteRefs(const bool freeMemory = false);
  };
  //====================================
}

#endif
