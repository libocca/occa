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

#ifndef OCCA_THREADS_DEVICE_HEADER
#define OCCA_THREADS_DEVICE_HEADER

#include "occa/defines.hpp"
#include "occa/modes/serial/device.hpp"
#include "occa/modes/threads/headers.hpp"
#include "occa/modes/threads/utils.hpp"
#include "occa/device.hpp"

namespace occa {
  namespace threads {
    class device : public serial::device {
    public:
      int coreCount;

      int threads;
      schedule_t schedule;

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      pthread_t tid[50];
#else
      DWORD tid[50];
#endif

      std::queue<job_t> jobs[50];

      mutex jobMutex;

      device(const occa::properties &properties_ = occa::properties());
      ~device();

      void finish() const;

      //---[ Stream ]-------------------
      streamTag tagStream() const;
      void waitFor(streamTag tag) const;
      double timeBetween(const streamTag &startTag, const streamTag &endTag) const;
      //================================

      //---[ Custom ]-------------------
      void addJob(job_t &job);
      //================================
    };
  }
}

#endif
