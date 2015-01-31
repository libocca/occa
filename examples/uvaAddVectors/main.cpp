#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  occa::enableUVA();

  int entries = 5;

  occa::device device;
  device.setup("mode = OpenMP, schedule = compact, chunk = 10");

  float *a  = (float*) device.managedUvaAlloc(entries * sizeof(float));
  float *b  = (float*) device.managedUvaAlloc(entries * sizeof(float));
  float *ab = (float*) device.managedUvaAlloc(entries * sizeof(float));

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::kernel addVectors = device.buildKernelFromSource("addVectors.okl",
                                                         "addVectors");

  addVectors(entries, a, b, ab);

  device.finish();

  for(int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  for(int i = 0; i < entries; ++i){
    if(ab[i] != (a[i] + b[i]))
      throw 1;
  }

  occa::free(a);
  occa::free(b);
  occa::free(ab);

  addVectors.free();
  device.free();

  return 0;
}
