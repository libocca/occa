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

#include <occa/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/lang/modes/serial.hpp>

namespace occa {
  namespace serial {
    kernel::kernel(device_v *dHandle_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::kernel_v(dHandle_, name_, sourceFilename_, properties_) {
      dlHandle = NULL;
      function = NULL;
    }

    kernel::~kernel() {}

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(-1,-1,-1);
    }

    dim kernel::maxInnerDims() const {
      return dim(-1,-1,-1);
    }

    void kernel::run() const {
      const int totalArgCount = kernelArg::argumentCount(arguments);
      if ((int) vArgs.size() < totalArgCount) {
        vArgs.resize(totalArgCount);
      }

      const int kArgCount = (int) arguments.size();

      int argc = 0;
      for (int i = 0; i < kArgCount; ++i) {
        const kArgVector &iArgs = arguments[i].args;
        const int argCount = (int) iArgs.size();
        if (!argCount) {
          continue;
        }
        for (int ai = 0; ai < argCount; ++ai) {
          vArgs[argc++] = iArgs[ai].ptr();
        }
      }

      sys::runFunction(function, argc, &(vArgs[0]));
    }

    void kernel::free() {
      if (dlHandle) {
        sys::dlclose(dlHandle);
        dlHandle = NULL;
      }
    }
  }
}
