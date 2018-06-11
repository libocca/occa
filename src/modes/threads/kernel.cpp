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
#if 0

#include <occa/modes/threads/kernel.hpp>
#include <occa/modes/threads/device.hpp>
#include <occa/modes/threads/utils.hpp>
#include <occa/base.hpp>

namespace occa {
  namespace threads {
    kernel::kernel(device_v *dHandle_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      serial::kernel(dHandle_, name_, sourceFilename_, properties_) {

      threads = properties.get("threads", sys::getCoreCount());
    }

    kernel::~kernel() {}

    void kernel::run() const {
      job_t job;

      job.count  = threads;
      job.handle = handle;
      job.inner  = inner;
      job.outer  = outer;

      const int argc = kernelArg::argumentCount(kArgc, kArgs);
      for (int i = 0; i < argc; ++i) {
        const int argCount = (int) kArgs[i].args.size();
        if (argCount) {
          const kernelArgData *kArgs_i = &(kArgs[i].args[0]);
          for (int j = 0; j < argCount; ++j) {
            job.args.push_back(kArgs_i[j].ptr());
          }
        }
      }

      for (int t = 0; t < threads; ++t) {
        job.rank = t;
        ((device*) dHandle)->addJob(job);
      }
    }
  }
}

#endif
