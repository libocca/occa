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

#include <occa.hpp>

int main(int argc, char **argv) {
  occa::printModeInfo();
  /*
    Try running with OCCA_VERBOSE=1 or set
    verbose at run-time with:
    occa::settings()["kernel/verbose"] = true;
  */

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  device.setup("mode: 'Serial'");

  // device.setup("mode     : 'OpenMP', "
  //              "schedule : 'compact', "
  //              "chunk    : 10");

  // device.setup("mode        : 'OpenCL', "
  //              "platform_id : 0, "
  //              "device_id   : 1");

  // device.setup("mode      : 'CUDA', "
  //              "device_id : 0");

  // device.setup("mode      : 'HIP', "
  //              "device_id : 0");

  // device.setup("mode         : 'Threads', "
  //              "threads      : 4, "
  //              "schedule     : 'compact', "
  //              "pinned_cores : [0, 0, 1, 1]");
  //========================================================

  // Allocate memory on the device
  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  // Compile the kernel at run-time
  addVectors = device.buildKernel("addVectors.okl",
                                  "addVectors");

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);

  // Copy result to the host
  o_ab.copyTo(ab);

  // Assert values
  for (int i = 0; i < 5; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i])) {
      throw 1;
    }
  }

  // Free host memory
  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
