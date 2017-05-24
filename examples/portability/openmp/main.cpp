#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv) {
  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  device.setup("mode: 'OpenMP'");

  o_a  = occa::cpu::wrapMemory(a , entries*sizeof(float));
  o_b  = occa::cpu::wrapMemory(b , entries*sizeof(float));
  o_ab = occa::cpu::wrapMemory(ab, entries*sizeof(float));

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
