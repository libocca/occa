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

#ifndef OCCA_BASE_HEADER
#define OCCA_BASE_HEADER

#include <iostream>
#include <vector>

#include <stdint.h>

#include "occa/defines.hpp"
#include "occa/types.hpp"

#if OCCA_SSE
#  include <xmmintrin.h>
#endif

#include "occa/device.hpp"
#include "occa/kernel.hpp"
#include "occa/memory.hpp"

namespace occa {
  //---[ Globals & Flags ]--------------
  properties& settings();
  //====================================

  //---[ Device Functions ]-------------
  device host();
  device& currentDevice();

  void setDevice(device d);
  void setDevice(const properties &props);

  properties& deviceProperties();

  void finish();

  void waitFor(streamTag tag);

  stream createStream();
  stream getStream();
  void setStream(stream s);
  stream wrapStream(void *handle_);

  streamTag tagStream();
  //====================================

  //---[ Kernel Functions ]-------------
  kernel buildKernel(const std::string &filename,
                     const std::string &functionName,
                     const properties &props = occa::properties());

  kernel buildKernelFromString(const std::string &content,
                               const std::string &functionName,
                               const properties &props = occa::properties());

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &functionName,
                               const properties &props = occa::properties());
  //====================================

  //---[ Memory Functions ]-------------
  occa::memory malloc(const dim_t bytes,
                      void *src = NULL,
                      const properties &props = occa::properties());

  void* uvaAlloc(const dim_t bytes,
                 void *src = NULL,
                 const properties &props = occa::properties());

  occa::memory wrapMemory(void *handle_,
                          const dim_t bytes,
                          const occa::properties &props = occa::properties());

  void memcpy(void *dest, const void *src,
              const dim_t bytes,
              const properties &props = properties());

  void memcpy(memory dest, const void *src,
              const dim_t bytes = -1,
              const dim_t offset = 0,
              const properties &props = properties());

  void memcpy(void *dest, memory src,
              const dim_t bytes = -1,
              const dim_t offset = 0,
              const properties &props = properties());

  void memcpy(memory dest, memory src,
              const dim_t bytes = -1,
              const dim_t destOffset = 0,
              const dim_t srcOffset = 0,
              const properties &props = properties());
  //====================================

  //---[ Free Functions ]---------------
  void free(device d);
  void free(stream s);
  void free(kernel k);
  void free(memory m);
  //====================================

  void printModeInfo();
}

#endif
