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
  // occa::setDevice("mode: 'OpenCL', platformID: 0, deviceID: 1");

  int entries = 8;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  occa::stream streamA, streamB;

  streamA = occa::getStream();
  streamB = occa::createStream();

  o_a  = occa::malloc(entries*sizeof(float));
  o_b  = occa::malloc(entries*sizeof(float));
  o_ab = occa::malloc(entries*sizeof(float));

  addVectors = occa::buildKernel("addVectors.okl",
                                  "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  occa::setStream(streamA);
  addVectors(entries, o_a, o_b, o_ab);

  occa::setStream(streamB);
  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for (int i = 0; i < entries; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  delete [] a;
  delete [] b;
  delete [] ab;
}
