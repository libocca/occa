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

#if OCCA_CUDA_ENABLED
#  ifndef OCCA_CUDA_KERNEL_HEADER
#  define OCCA_CUDA_KERNEL_HEADER

#include <cuda.h>

#include "occa/kernel.hpp"

namespace occa {
  namespace cuda {
    class device;

    class kernel : public occa::kernel_v {
      friend class device;

    private:
      CUmodule   cuModule;
      CUfunction cuFunction;

    public:
      kernel(const occa::properties &properties_ = occa::properties());
      ~kernel();

      void build(const std::string &filename,
                 const std::string &kernelName,
                 const hash_t hash,
                 const occa::properties &props);

      void buildFromBinary(const std::string &filename,
                           const std::string &kernelName,
                           const occa::properties &props);

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void runFromArguments(const int kArgc, const kernelArg *kArgs) const;

      void free();
    };
  }
}

#  endif
#endif
