#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv) {
  int entries = 8;

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

  occa::stream streamA, streamB;

  // device.setup("mode: 'CUDA', deviceID: 0");
  device.setup("mode: 'OpenCL', platformID: 0, deviceID: 1");

  streamA = device.getStream();
  streamB = device.createStream();

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  addVectors = device.buildKernel("addVectors.okl",
                                  "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  device.setStream(streamA);
  addVectors(entries, o_a, o_b, o_ab);

  device.setStream(streamB);
  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for (int i = 0; i < entries; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  delete [] a;
  delete [] b;
  delete [] ab;

  addVectors.free();
  o_a.free();
  o_b.free();
  o_ab.free();
  device.free();
}
