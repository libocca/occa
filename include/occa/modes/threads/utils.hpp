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

#ifndef OCCA_THREADS_UTILS_HEADER
#define OCCA_THREADS_UTILS_HEADER

#include <iostream>
#include <queue>

#include "occa/defines.hpp"
#include "occa/modes/threads/headers.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  namespace threads {
    //---[ Types ]----------------------
    enum schedule_t {
      compact, scatter, manual
    };

    std::string toString(schedule_t s);

    class job_t {
    public:
      int rank, count;
      schedule_t schedule;

      handleFunction_t handle;

      int dims;
      occa::dim inner, outer;

      std::vector<void*> args;

      job_t();
      job_t(const job_t &k);
      job_t& operator = (const job_t &k);
    };

    struct workerData_t {
      int rank, count;
      int pinnedCore;

      std::queue<job_t> *jobs;

      mutex_t *jobMutex;
    };
    //==================================

    //---[ Functions ]------------------
    void* limbo(void *args);
    void run(job_t &job);
    //==================================
  }
}

#endif
