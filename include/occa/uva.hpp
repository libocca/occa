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

#ifndef OCCA_UVA_HEADER
#define OCCA_UVA_HEADER

#include <iostream>
#include <vector>

#include "occa/defines.hpp"
#include "occa/types.hpp"

namespace occa {
  class device;
  class memory;
  class memory_v;

  class ptrRange {
  public:
    char *start, *end;

    ptrRange();
    ptrRange(void *ptr, const udim_t bytes = 0);
    ptrRange(const ptrRange &r);

    ptrRange& operator =  (const ptrRange &r);
    bool        operator == (const ptrRange &r) const;
    bool        operator != (const ptrRange &r) const;

    friend int operator < (const ptrRange &a, const ptrRange &b);
  };

  typedef std::map<ptrRange, occa::memory_v*> ptrRangeMap;
  typedef std::vector<occa::memory_v*>        memoryVector;

  extern ptrRangeMap uvaMap;
  extern memoryVector uvaStaleMemory;

  class uvaPtrInfo {
  private:
    occa::memory_v *mem;

  public:
    uvaPtrInfo();
    uvaPtrInfo(void *ptr);
    uvaPtrInfo(occa::memory_v *mem_);

    uvaPtrInfo(const uvaPtrInfo &upi);
    uvaPtrInfo& operator = (const uvaPtrInfo &upi);

    occa::device getDevice();
    occa::memory getMemory();
  };

  occa::memory_v* uvaToMemory(void *ptr);

  void startManaging(void *ptr);
  void stopManaging(void *ptr);

  void syncToDevice(void *ptr, const udim_t bytes = (udim_t) -1);
  void syncToHost(void *ptr, const udim_t bytes = (udim_t) -1);

  void syncMemToDevice(occa::memory_v *mem,
                       const udim_t bytes = (udim_t) -1,
                       const udim_t offset = 0);

  void syncMemToHost(occa::memory_v *mem,
                     const udim_t bytes = (udim_t) -1,
                     const udim_t offset = 0);

  bool needsSync(void *ptr);
  void sync(void *ptr);
  void dontSync(void *ptr);

  void removeFromStaleMap(void *ptr);
  void removeFromStaleMap(memory_v *mem);

  void setupMagicFor(void *ptr);

  void free(void *ptr);
}

#endif
