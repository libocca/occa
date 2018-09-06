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

#include <occa/defines.hpp>

#if OCCA_OPENCL_ENABLED

#include <occa/mode/opencl/streamTag.hpp>
#include <occa/mode/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    streamTag::streamTag(modeDevice_t *modeDevice_,
                         cl_event clEvent_) :
      modeStreamTag_t(modeDevice_),
      clEvent(clEvent_),
      time(-1) {}

    streamTag::~streamTag() {
      OCCA_OPENCL_ERROR("streamTag: Freeing cl_event",
                        clReleaseEvent(clEvent));
    }

    double streamTag::getTime() {
      if (time < 0) {
        cl_ulong clTime;
        OCCA_OPENCL_ERROR("streamTag: Getting event profiling info",
                          clGetEventProfilingInfo(clEvent,
                                                  CL_PROFILING_COMMAND_END,
                                                  sizeof(cl_ulong),
                                                  &clTime, NULL));
        time = 1.0e-9 * clTime;
      }
      return time;
    }
  }
}

#endif
