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

#ifndef OCCA_SERIAL_DEFINES_HEADER
#define OCCA_SERIAL_DEFINES_HEADER

#include "occa/defines.hpp"
#include "occa/defines/cpuMode.hpp"

//---[ Defines ]----------------------------------
#define OCCA_USING_SERIAL   1
#define OCCA_USING_OPENMP   0
#define OCCA_USING_OPENCL   0
#define OCCA_USING_CUDA     0
#define OCCA_USING_PTHREADS 0
//================================================


//---[ Atomics ]----------------------------------
#define occaAtomicAdd(ptr, update) *(ptr) += (update)
#define occaAtomicSub(ptr, update) *(ptr) -= (update)

#define occaAtomicInc(ptr) ((*(ptr))++)
#define occaAtomicDec(ptr) ((*(ptr))--)

#define occaAtomicAnd(ptr, update) *(ptr) &= update
#define occaAtomicOr(ptr, update)  *(ptr) |= update
#define occaAtomicXor(ptr, update) *(ptr) ^= update

template <class TM>
TM occaAtomicSwap(TM *ptr, const TM &update) {
  const TM old = *ptr;
  *ptr = update;

  return old;
}

template <class TM>
TM occaAtomicMin(TM *ptr, const TM &update) {
  const TM old = *ptr;
  *ptr = ((old < update) ? old : update);

  return old;
}

template <class TM>
TM occaAtomicMax(TM *ptr, const TM &update) {
  const TM old = *ptr;
  *ptr = ((old < update) ? update : old);

  return old;
}

template <class TM>
TM occaAtomicCAS(TM *ptr, const int comp, const TM &update) {
  TM old = *ptr;
  if (comp) {
    *ptr = update;
  }
  return old;
}

#define occaAtomicAdd64  occaAtomicAdd
#define occaAtomicSub64  occaAtomicSub
#define occaAtomicSwap64 occaAtomicSwap
#define occaAtomicInc64  occaAtomicInc
#define occaAtomicDec64  occaAtomicDec
//================================================


//---[ Misc ]-------------------------------------
#define occaParallelFor2
#define occaParallelFor1
#define occaParallelFor0
#define occaParallelFor
//================================================

#endif
