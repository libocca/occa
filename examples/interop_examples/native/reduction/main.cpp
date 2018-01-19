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

int main(int argc, char **argv) {
  occa::setDevice("mode: 'CUDA', deviceID: 0");
  // occa::setDevice("mode: 'OpenCL', platformID: 0, deviceID: 0");

  occa::kernel reduction;

  // Choosing something not divisible by 256
  int entries = 10000;
  int block   = 256;
  int blocks  = (entries + block - 1)/block;

  float *vec      = new float[entries];
  float *blockSum = new float[blocks];
  occa::memory o_vec, o_blockSum;

  float sum = 0;

  // Initialize device memory
  for (int i = 0; i < entries; ++i) {
    vec[i] = 1;
    sum   += vec[i];
  }

  for (int i = 0; i < blocks; ++i) {
    blockSum[i] = 0;
  }

  // Allocate memory on the device
  o_vec      = occa::malloc(entries * sizeof(float));
  o_blockSum = occa::malloc(blocks  * sizeof(float));

  // Pass value of 'block' at kernel compile-time
  occa::properties reductionProps;
  reductionProps["OKL"] = false; // Disable OKL parsing
  reductionProps["defines/block"] = block;

  const std::string kernelFile = (
    (occa::getDevice().mode() == "CUDA")
    ? "reduction.cu"
    : "reduction.cl"
  );

  reduction = occa::buildKernel(kernelFile,
                                "reduction",
                                reductionProps);

  // Host -> Device
  o_vec.copyFrom(vec);

  // Set kernel launch dimensions before launching
  reduction.setRunDims(blocks, block);
  reduction(entries, o_vec, o_blockSum);

  // Host <- Device
  o_blockSum.copyTo(blockSum);

  // Finalize the reduction in the host
  for (int i = 1; i < blocks; ++i) {
    blockSum[0] += blockSum[i];
  }

  // Validate
  if (blockSum[0] != sum) {
    std::cout << "sum      = " << sum << '\n'
              << "blockSum = " << blockSum[0] << '\n';

    std::cout << "Reduction failed\n";
    throw 1;
  }
  else {
    std::cout << "Reduction = " << blockSum[0] << '\n';
  }

  // Free host memory
  delete [] vec;
  delete [] blockSum;

  // Device memory is automatically freed

  return 0;
}
