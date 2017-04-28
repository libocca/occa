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

#include "occa/modes/serial/memory.hpp"
#include "occa/tools/sys.hpp"
#include "occa/device.hpp"

namespace occa {
  namespace serial {
    memory::memory(const occa::properties &properties_) :
      occa::memory_v(properties_) {}

    memory::~memory() {}

    void* memory::getHandle(const occa::properties &props) const {
      return handle;
    }

    kernelArg memory::makeKernelArg() const {
      kernelArg_t arg;
      arg.data.void_ = handle;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;
      return kernelArg(arg);
    }

    memory_v* memory::addOffset(const dim_t offset, bool &needsFree) {
      memory *m = new memory(properties);
      m->handle = (((char*) handle) + offset);
      needsFree = false;
      return m;
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const {
      const void *srcPtr = ((char*) handle) + offset;

      ::memcpy(dest, srcPtr, bytes);
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {

      void *destPtr      = ((char*) handle) + offset;
      const void *srcPtr = src;

      ::memcpy(destPtr, srcPtr, bytes);
    }

    void memory::copyFrom(const memory_v *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {

      void *destPtr      = ((char*) handle)      + destOffset;
      const void *srcPtr = ((char*) src->handle) + srcOffset;

      ::memcpy(destPtr, srcPtr, bytes);
    }

    void memory::free() {
      if (handle) {
        sys::free(handle);
        handle = NULL;
        size = 0;
      }
    }

    void memory::detach() {
      handle = NULL;
      size = 0;
    }
  }
}
