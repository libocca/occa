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

#include <map>

#include "occa/tools/misc.hpp"
#include "occa/tools/sys.hpp"
#include "occa/base.hpp"
#include "occa/uva.hpp"

namespace occa {
  ptrRangeMap uvaMap;
  memoryVector uvaStaleMemory;

  ptrRange::ptrRange() :
    start(NULL),
    end(NULL) {}

  ptrRange::ptrRange(void *ptr, const udim_t bytes) :
    start((char*) ptr),
    end(((char*) ptr) + bytes) {}

  ptrRange::ptrRange(const ptrRange &r) :
    start(r.start),
    end(r.end) {}

  ptrRange& ptrRange::operator = (const ptrRange &r) {
    start = r.start;
    end   = r.end;

    return *this;
  }

  bool ptrRange::operator == (const ptrRange &r) const {
    return ((start <= r.start) && (r.start < end));
  }

  bool ptrRange::operator != (const ptrRange &r) const {
    return ((r.start < start) || (end <= r.start));
  }

  int operator < (const ptrRange &a, const ptrRange &b) {
    return ((a != b) && (a.start < b.start));
  }

  uvaPtrInfo::uvaPtrInfo() :
    mem(NULL) {}

  uvaPtrInfo::uvaPtrInfo(void *ptr) {
    ptrRangeMap::iterator it = uvaMap.find(ptr);

    if (it != uvaMap.end()) {
      mem = (it->second);
    } else {
      mem = (occa::memory_v*) ptr; // Defaults to ptr being a memory_v
    }
  }

  uvaPtrInfo::uvaPtrInfo(occa::memory_v *mem_) :
    mem(mem_) {}

  uvaPtrInfo::uvaPtrInfo(const uvaPtrInfo &upi) :
    mem(upi.mem) {}

  uvaPtrInfo& uvaPtrInfo::operator = (const uvaPtrInfo &upi) {
    mem = upi.mem;
    return *this;
  }

  occa::device uvaPtrInfo::getDevice() {
    return occa::device(mem->dHandle);
  }

  occa::memory uvaPtrInfo::getMemory() {
    return occa::memory(mem);
  }

  occa::memory_v* uvaToMemory(void *ptr) {
    ptrRangeMap::iterator it = uvaMap.find(ptr);
    return (it == uvaMap.end()) ? NULL : it->second;
  }

  void startManaging(void *ptr) {
    occa::memory_v *mem = uvaToMemory(ptr);
    if (mem != NULL) {
      mem->memInfo |= uvaFlag::isManaged;
    }
  }

  void stopManaging(void *ptr) {
    occa::memory_v *mem = uvaToMemory(ptr);
    if (mem != NULL) {
      mem->memInfo &= ~uvaFlag::isManaged;
    }
  }

  void syncToDevice(void *ptr, const udim_t bytes) {
    occa::memory_v *mem = uvaToMemory(ptr);
    if (mem) {
      syncMemToDevice(mem, bytes, ptrDiff(mem->uvaPtr, ptr));
    }
  }

  void syncToHost(void *ptr, const udim_t bytes) {
    occa::memory_v *mem = uvaToMemory(ptr);
    if (mem) {
      syncMemToHost(mem, bytes, ptrDiff(mem->uvaPtr, ptr));
    }
  }

  void syncMemToDevice(occa::memory_v *mem,
                       const udim_t bytes,
                       const udim_t offset) {

    if (mem->dHandle->hasSeparateMemorySpace()) {
      occa::memory(mem).syncToDevice(bytes, offset);
    }
  }

  void syncMemToHost(occa::memory_v *mem,
                     const udim_t bytes,
                     const udim_t offset) {

    if (mem->dHandle->hasSeparateMemorySpace()) {
      occa::memory(mem).syncToHost(bytes, offset);
    }
  }

  bool needsSync(void *ptr) {
    occa::memory_v *mem = uvaToMemory(ptr);
    return (mem == NULL) ? false : mem->isStale();
  }

  void sync(void *ptr) {
    occa::memory_v *mem = uvaToMemory(ptr);

    if (mem != NULL) {
      if (mem->inDevice()) {
        syncMemToHost(mem);
      } else {
        syncMemToDevice(mem);
      }
    }
  }

  void dontSync(void *ptr) {
    removeFromStaleMap(ptr);
  }

  void removeFromStaleMap(void *ptr) {
    ptrRangeMap::iterator it = uvaMap.find(ptr);
    if (it == uvaMap.end()) {
      return;
    }

    memory m(it->second);
    if (!m.uvaIsStale()) {
      return;
    }

    removeFromStaleMap(m.getMHandle());
  }

  void removeFromStaleMap(memory_v *mem) {
    occa::memory m(mem);
    const size_t staleEntries = uvaStaleMemory.size();

    for (size_t i = 0; i < staleEntries; ++i) {
      if (uvaStaleMemory[i] == mem) {
        m.uvaMarkFresh();
        uvaStaleMemory.erase(uvaStaleMemory.begin() + i);
        break;
      }
    }
  }

  void free(void *ptr) {
    ptrRangeMap::iterator it = uvaMap.find(ptr);

    if ((it != uvaMap.end()) &&
        (((void*) it->first.start) != ((void*) it->second))) {
      occa::memory(it->second).free();
    } else {
      ::free(ptr);
    }
  }
}
