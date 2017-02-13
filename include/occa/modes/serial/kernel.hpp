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

#ifndef OCCA_SERIAL_KERNEL_HEADER
#define OCCA_SERIAL_KERNEL_HEADER

#include "occa/defines.hpp"
#include "occa/kernel.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  namespace serial {
    class kernel : public occa::kernel_v {
    protected:
      void *dlHandle;
      handleFunction_t handle;

      void *vArgs[2*OCCA_MAX_ARGS];

    public:
      kernel(const occa::properties &properties_ = occa::properties());
      ~kernel();

      void* getHandle(const occa::properties &props);

      std::string binaryName(const std::string &filename);

      void build(const std::string &filename,
                 const std::string &functionName,
                 const occa::properties &props);

      void buildFromBinary(const std::string &filename,
                           const std::string &functionName,
                           const occa::properties &props);

      int maxDims();
      dim maxOuterDims();
      dim maxInnerDims();

      void runFromArguments(const int kArgc, const kernelArg *kArgs);

      void free();
    };
  }
}
#endif
