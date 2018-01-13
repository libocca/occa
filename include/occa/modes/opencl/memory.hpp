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

#include "occa/defines.hpp"

#if OCCA_OPENCL_ENABLED
#  ifndef OCCA_OPENCL_MEMORY_HEADER
#  define OCCA_OPENCL_MEMORY_HEADER

#include "occa/memory.hpp"
#include "occa/modes/opencl/headers.hpp"

namespace occa {
  namespace opencl {
    class device;

    class memory : public occa::memory_v {
      friend class opencl::device;
      friend cl_mem getMem(occa::memory memory);
      friend occa::memory wrapMemory(occa::device device,
                                     cl_mem clMem,
                                     const udim_t bytes,
                                     const occa::properties &props);

    private:
      cl_mem clMem;
      void *mappedPtr;

    public:
      memory(const occa::properties &properties_ = occa::properties());
      ~memory();

      kernelArg makeKernelArg() const;

      memory_v* addOffset(const dim_t offset, bool &needsFree);

      void copyTo(void *dest,
                  const udim_t bytes,
                  const udim_t destOffset = 0,
                  const occa::properties &props = occa::properties()) const;

      void copyFrom(const void *src,
                    const udim_t bytes,
                    const udim_t offset = 0,
                    const occa::properties &props = occa::properties());

      void copyFrom(const memory_v *src,
                    const udim_t bytes,
                    const udim_t destOffset = 0,
                    const udim_t srcOffset = 0,
                    const occa::properties &props = occa::properties());

      void free();
      void detach();
    };
  }
}

#  endif
#endif
