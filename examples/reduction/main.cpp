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
  int entries = 10000; // Not divisible by 256
  int p_Nred = 256;
  int reducedEntries = (entries + p_Nred - 1)/p_Nred;

  float *a    = new float[entries];
  float *aRed = new float[reducedEntries];

  float trueRed = 0;

  for (int i = 0; i < entries; ++i) {
    a[i]     = 1;
    trueRed += a[i];
  }

  for (int i = 0; i < reducedEntries; ++i)
    aRed[i] = 0;

  occa::device device;
  occa::kernel reduction;
  occa::memory o_a, o_aRed;

  device.setup("mode: 'Serial'");

  o_a  = device.malloc(entries*sizeof(float));
  o_aRed = device.malloc(reducedEntries*sizeof(float));

  occa::properties kernelProps;
  kernelProps["defines/p_Nred"] = p_Nred;

#if 1
  reduction = device.buildKernel("reduction.okl",
                                 "reduction",
                                 kernelProps);
#else
  kernelProps["OKL"] = false;
  reduction = device.buildKernel("reduction.cu",
                                 "reduction",
                                 kernelProps);

  size_t dims     = 1;
  occa::dim inner(p_Nred);
  occa::dim outer((entries + p_Nred - 1) / p_Nred);

  reduction.setRunDims(dims, inner, outer);
#endif

  o_a.copyFrom(a);

  reduction(entries, o_a, o_aRed);

  o_aRed.copyTo(aRed);

  for (int i = 1; i < reducedEntries; ++i)
    aRed[0] += aRed[i];

  if (aRed[0] != trueRed) {
    std::cout << "aRed[0] = " << aRed[0] << '\n'
              << "trueRed = " << trueRed << '\n';

    std::cout << "Reduction failed\n";
    throw 1;
  }
  else
    std::cout << "Reduction(a) = " << aRed[0] << '\n';

  delete [] a;
  delete [] aRed;

  reduction.free();
  o_a.free();
  o_aRed.free();

  device.free();

  return 0;
}
