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

#ifndef OCCA_ARRAY_KERNELS_HEADER
#define OCCA_ARRAY_KERNELS_HEADER

#include <cmath>

#include "occa/defines.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace linalg {
    std::string getKernelFile();

    template <class TM>
    inline std::map<int, TM*>& hostBufferMap() {
      static std::map<int, TM*> bufferMap;
      return bufferMap;
    }

    template <class TM>
    inline hashedMemoryMap_t& deviceBufferMap() {
      static hashedMemoryMap_t bufferMap;
      return bufferMap;
    }

    template <class TM>
    TM *hostReductionBuffer(const int size);

    template <class TM>
    occa::memory deviceReductionBuffer(occa::device device,
                                       const int size);

    template <class VTYPE, class RETTYPE>
    RETTYPE* reduce(occa::memory vec,
                    occa::kernelBuilder &builder);

    template <class VTYPE, class RETTYPE>
    RETTYPE l1Norm(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE l2Norm(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE lpNorm(const float p, occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE lInfNorm(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE max(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE min(occa::memory vec);

    template <class VTYPE, class RETTYPE>
    RETTYPE dot(occa::memory vec);
  }
}

#include "occa/array/linalg.tpp"

#endif
