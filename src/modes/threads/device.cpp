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

#include "occa/modes/threads/device.hpp"
#include "occa/modes/threads/kernel.hpp"
#include "occa/modes/serial/memory.hpp"
#include "occa/modes/threads/utils.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace threads {
    device::device(const occa::properties &properties_) :
      serial::device(properties_) {
      coreCount = sys::getCoreCount();

      std::vector<int> pinnedCores;

      threads  = properties.get("threads", coreCount);

      if (properties.get<std::string>("schedule", "compact") == "compact") {
        schedule = compact;
      } else {
        schedule = scatter;
      }

      if (properties.has("pinnedCores")) {
        pinnedCores = properties.getList<int>("pinnedCores");

        if (pinnedCores.size() != (size_t) threads) {
          threads = (int) pinnedCores.size();
          std::cout << "[Threads]: Mismatch between thread count and pinned cores\n"
                    << "           Setting threads to " << threads << '\n';
        }

        for (size_t i = 0; i < pinnedCores.size(); ++i)
          if (pinnedCores[i] < 0) {
            const int newPC = ((pinnedCores[i] % coreCount) + coreCount);

            std::cout << "Trying to pin thread on core ["
                      << pinnedCores[i] << "], changing it to ["
                      << newPC << "]\n";

            pinnedCores[i] = newPC;
          } else if (coreCount <= pinnedCores[i]) {
            const int newPC = (pinnedCores[i] % coreCount);

            std::cout << "Trying to pin thread on core ["
                      << pinnedCores[i] << "], changing it to ["
                      << newPC << "]\n";

            pinnedCores[i] = newPC;
          }

        schedule = manual;
      }

      for (int t = 0; t < threads; ++t) {
        workerData_t *args = new workerData_t;

        args->rank  = t;
        args->count = threads;

        // [-] Need to know number of sockets
        if (schedule & compact) {
          args->pinnedCore = (t % coreCount);
        } else if (schedule & scatter) {
          args->pinnedCore = (t % coreCount);
        } else {
          args->pinnedCore = pinnedCores[t];
        }

        args->jobs = &(jobs[t]);
        args->jobMutex = &(jobMutex);

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
        pthread_create(&tid[t], NULL, threads::limbo, args);
#else
        CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) threads::limbo, args, 0, &tid[t]);
#endif
      }
    }

    device::~device() {}

    void device::finish() {
      bool done = false;
      while (!done) {
        done = true;
        for (int t = 0; t < threads; ++t) {
          if (jobs[t].size()) {
            done = false;
            break;
          }
        }
        OCCA_LFENCE;
      }
    }

    //---[ Stream ]---------------------
    streamTag device::tagStream() {
      streamTag ret;
      ret.tagTime = sys::currentTime();
      return ret;
    }

    void device::waitFor(streamTag tag) {
      finish(); // [-] Not done
    }

    double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
      return (endTag.tagTime - startTag.tagTime);
    }
    //==================================

    //---[ Custom ]---------------------
    void device::addJob(job_t &job) {
      jobMutex.lock();
      jobs[job.rank].push(job);
      jobMutex.unlock();
    }
    //==================================
  }
}
