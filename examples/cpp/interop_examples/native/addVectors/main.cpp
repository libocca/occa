#include <iostream>

#include <occa.hpp>

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
  // device.setup("mode: 'CUDA', device_id: 0");
  // device.setup("mode: 'OpenCL', platform_id : 0, device_id: 1");
  //========================================================

  o_a  = device.malloc(entries, occa::dtype::float_);
  o_b  = device.malloc(entries, occa::dtype::float_);
  o_ab = device.malloc(entries, occa::dtype::float_);

  // Native Serial kernel
  addVectors = device.buildKernel("addVectors.cpp",
                                  "addVectors",
                                  "okl: false");
  // Native CUDA kernel
  // addVectors = device.buildKernel("addVectors.cu",
  //                                 "addVectors",
  //                                 "okl: false");
  // Native OpenCL kernel
  // addVectors = device.buildKernel("addVectors.cl",
  //                                 "addVectors",
  //                                 "okl: false");

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
