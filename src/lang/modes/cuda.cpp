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
#include <occa/lang/modes/cuda.hpp>

/*
//---[ Loop Info ]--------------------------------
#define occaOuterDim2 gridDim.z
#define occaOuterId2  blockIdx.z

#define occaOuterDim1 gridDim.y
#define occaOuterId1  blockIdx.y

#define occaOuterDim0 gridDim.x
#define occaOuterId0  blockIdx.x
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerDim2 blockDim.z
#define occaInnerId2  threadIdx.z

#define occaInnerDim1 blockDim.y
#define occaInnerId1  threadIdx.y

#define occaInnerDim0 blockDim.x
#define occaInnerId0  threadIdx.x
//================================================


//---[ Standard Functions ]-----------------------
@barrier("local")  __syncthreads()
@barrier("global") __syncthreads()
//================================================


//---[ Attributes ]-------------------------------
#define occaShared   __shared__
#define occaRestrict __restrict__
#define occaConstant __constant__
//================================================


//---[ Kernel Info ]------------------------------
#define occaKernel         extern "C" __global__
#define occaFunction       __device__
#define occaDeviceFunction __device__
//================================================
 */

namespace occa {
  namespace lang {
    namespace okl {
      cudaParser::cudaParser() {
      }

      void cudaParser::afterParsing() {
        updateConstToConstant();
        addOccaFors();
        setupKernelArgs();
        setupLaunchKernel();
      }

      void cudaParser::updateConstToConstant() {
      }

      void cudaParser::addOccaFors() {
      }

      void cudaParser::setupKernelArgs() {
      }

      void cudaParser::setupLaunchKernel() {
      }
    }
  }
}
