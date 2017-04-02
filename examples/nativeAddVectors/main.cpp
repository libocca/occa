#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv) {
  occa::printModeInfo();

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
  // device.setup("mode: 'CUDA', deviceID: 0");
  // device.setup("mode: 'OpenCL', platformID : 0, deviceID: 1");
  //========================================================

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  // Native Serial kernel
  addVectors = device.buildKernel("addVectors.cpp",
                                  "addVectors",
                                  "OKL: false");
  // Native CUDA kernel
  // addVectors = device.buildKernel("addVectors.cu",
  //                                 "addVectors",
  //                                 "OKL: false");
  // Native OpenCL kernel
  // addVectors = device.buildKernel("addVectors.cl",
  //                                 "addVectors",
  //                                 "OKL: false");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Needed for CUDA and OpenCL kernels
  // addVectors.setRunDims((entries + 15) / 16, 16);
  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for (int i = 0; i < 5; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i]))
      throw 1;
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
