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
#include <iostream>

#include "occa.hpp"
#include "occa/modes/cuda/utils.hpp"

#include "cuda_runtime_api.h"

int main(int argc, char **argv) {
  int entries = 5;

  //---[ Init CUDA ]------------------
  int cuDeviceID;
  cudaStream_t cuStream;
  void *cu_a, *cu_b, *cu_ab;

  // Default: cuStream = 0
  cudaStreamCreate(&cuStream);

  cudaMalloc(&cu_a , entries*sizeof(float));
  cudaMalloc(&cu_b , entries*sizeof(float));
  cudaMalloc(&cu_ab, entries*sizeof(float));

  //  ---[ Get CUDA Info ]----
  CUdevice cuDevice;
  CUcontext cuContext;

  cuDeviceGet(&cuDevice, cuDeviceID);
  cuCtxGetCurrent(&cuContext);
  //  ========================
  //====================================

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  occa::device device = occa::cuda::wrapDevice(cuDevice, cuContext);

  occa::stream stream = device.wrapStream(&cuStream);
  device.setStream(stream);

  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  o_a  = occa::cuda::wrapMemory(device, cu_a , entries*sizeof(float));
  o_b  = occa::cuda::wrapMemory(device, cu_b , entries*sizeof(float));
  o_ab = occa::cuda::wrapMemory(device, cu_ab, entries*sizeof(float));

  addVectors = device.buildKernel("addVectors.okl",
                                  "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for (int i = 0; i < 5; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
