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
#include "stdlib.h"
#include "stdio.h"

#include "occa.h"

int main(int argc, char **argv) {
  occaPrintModeInfo();

  /*
    Try running with OCCA_VERBOSE=1 or set
    verbose at run-time with:
    occaPropertiesSet(occaSettings(),
                      "kernel/verbose,
                      occaBool(1));
  */

  int entries = 5;
  int i;

  float *a  = (float*) malloc(entries*sizeof(float));
  float *b  = (float*) malloc(entries*sizeof(float));
  float *ab = (float*) malloc(entries*sizeof(float));

  for (i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occaDevice device;
  occaKernel addVectors;
  occaMemory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  const char *deviceInfo = "mode: 'Serial'";

  // const char *deviceInfo = ("mode     : 'OpenMP', "
  //                           "schedule : 'compact', "
  //                           "chunk    : 10");

  // const char *deviceInfo = ("mode       : 'OpenCL', "
  //                           "platformID : 0, "
  //                           "deviceID   : 1");

  // const char *deviceInfo = ("mode     : 'CUDA', "
  //                           "deviceID : 0");

  // const char *deviceInfo = ("mode        : 'Threads', "
  //                           "threadCount : 4, "
  //                           "schedule    : 'compact', "
  //                           "pinnedCores : [0, 0, 1, 1]");

  device = occaCreateDevice(occaString(deviceInfo));
  //========================================================

  // Allocate memory on the device
  o_a  = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);
  o_b  = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);
  o_ab = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);

  // Setup properties that can be passed to the kernel
  occaProperties props = occaCreateProperties();
  occaPropertiesSet(props, "defines/TILE_SIZE", occaInt(10));

  // Compile the kernel at run-time
  addVectors = occaDeviceBuildKernel(device,
                                     "addVectors.okl",
                                     "addVectors",
                                     props);

  // Copy memory to the device
  occaCopyPtrToMem(o_a, a, entries*sizeof(float), 0, occaDefault);
  occaCopyPtrToMem(o_b, b, occaAllBytes         , 0, occaDefault);

  // Launch device kernel
  occaKernelRun(addVectors,
                occaInt(entries), o_a, o_b, o_ab);

  // Copy result to the host
  occaCopyMemToPtr(ab, o_ab, occaAllBytes, 0, occaDefault);

  // Assert values
  for (i = 0; i < 5; ++i) {
    printf("%d = %f\n", i, ab[i]);
  }
  for (i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i]))
      exit(1);
  }

  // Free host memory
  free(a);
  free(b);
  free(ab);

  // Free device memory and occa objects
  occaFree(props);
  occaFree(addVectors);
  occaFree(o_a);
  occaFree(o_b);
  occaFree(o_ab);
  occaFree(device);
}
