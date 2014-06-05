#include <iostream>

#include "occa.hpp"

occa::device device;
occa::kernel addVectors;
occa::memory o_a, o_b, o_ab;

int entries;

float *a, *b, *ab;

void freeStuff();

int main(int argc, char **argv){
  int entries = 5;

  a  = new float[entries];
  b  = new float[entries];
  ab = new float[entries];

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  std::string mode = "OpenMP";
  int platformID = 0;
  int deviceID   = 0;

  device.setup(mode, platformID, deviceID);

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  addVectors = device.buildKernelFromSource("addVectors.occa",
                                            "addVectors");

  int dims = 1;
  int itemsPerGroup(2);
  int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  addVectors.setWorkingDims(dims, itemsPerGroup, groups);

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for(int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  freeStuff();

  return 0;
}

void freeStuff(){
  delete [] a;
  delete [] b;
  delete [] ab;

  device.free();

  addVectors.free();

  o_a.free();
  o_b.free();
  o_ab.free();
}
