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

#include "occa/defines.hpp"

#if OCCA_OPENCL_ENABLED

#include "occa/modes/opencl/memory.hpp"
#include "occa/modes/opencl/device.hpp"
#include "occa/modes/opencl/utils.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  namespace opencl {
    memory::memory(const occa::properties &properties_) :
      occa::memory_v(properties_),
      mappedPtr(NULL) {}

    memory::~memory() {}

    kernelArg memory::makeKernelArg() const {
      kernelArgData arg;

      arg.dHandle = dHandle;
      arg.mHandle = const_cast<memory*>(this);

      arg.data.void_ = (void*) &clMem;
      arg.size       = sizeof(void*);
      arg.info       = kArgInfo::usePointer;

      return kernelArg(arg);
    }

    memory_v* memory::addOffset(const dim_t offset, bool &needsFree) {
      opencl::device &dev = *((opencl::device*) dHandle);
      opencl::memory *m = new opencl::memory();
      m->dHandle = &dev;
      m->size    = size - offset;

      cl_buffer_region info;
      info.origin = offset;
      info.size   = m->size;

      cl_int error;
      m->clMem = clCreateSubBuffer(clMem,
                                   CL_MEM_READ_WRITE,
                                   CL_BUFFER_CREATE_TYPE_REGION,
                                   &info,
                                   &error);

      OCCA_OPENCL_ERROR("Device: clCreateSubBuffer", error);
      needsFree = true;
      return m;
    }

    void memory::copyFrom(const void *src,
                          const udim_t bytes,
                          const udim_t offset,
                          const occa::properties &props) {
      cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);
      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy From",
                        clEnqueueWriteBuffer(stream, clMem,
                                             async ? CL_FALSE : CL_TRUE,
                                             offset, bytes, src,
                                             0, NULL, NULL));
    }

    void memory::copyFrom(const memory_v *src,
                          const udim_t bytes,
                          const udim_t destOffset,
                          const udim_t srcOffset,
                          const occa::properties &props) {
      cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);
      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy From",
                        clEnqueueCopyBuffer(stream,
                                            ((memory*) src)->clMem,
                                            clMem,
                                            srcOffset, destOffset,
                                            bytes,
                                            0, NULL, NULL));
    }

    void memory::copyTo(void *dest,
                        const udim_t bytes,
                        const udim_t offset,
                        const occa::properties &props) const {

      const cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);
      const bool async = props.get("async", false);

      OCCA_OPENCL_ERROR("Memory: " << (async ? "Async " : "") << "Copy To",
                        clEnqueueReadBuffer(stream, clMem,
                                            async ? CL_FALSE : CL_TRUE,
                                            offset, bytes, dest,
                                            0, NULL, NULL));
    }

    void memory::free() {
      if (mappedPtr) {
        cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

        OCCA_OPENCL_ERROR("Mapped Free: clEnqueueUnmapMemObject",
                          clEnqueueUnmapMemObject(stream,
                                                  clMem,
                                                  mappedPtr,
                                                  0, NULL, NULL));
      }
      if (size) {
        // Free mapped-host pointer
        OCCA_OPENCL_ERROR("Mapped Free: clReleaseMemObject",
                          clReleaseMemObject(clMem));
        size = 0;
      }
    }

    void memory::detach() {
      size = 0;
    }
  }
}

#endif
