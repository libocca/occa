#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  occa::printAvailableDevices();

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  device.setup("mode = Serial");
  // device.setup("mode = OpenMP  , schedule = compact, chunk = 10");
  // device.setup("mode = OpenCL  , platformID = 0, deviceID = 1");
  // device.setup("mode = CUDA    , deviceID = 0");
  // device.setup("mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]");
  // device.setup("mode = COI     , deviceID = 0");
  //========================================================

  //---[ Device setup with Python-like arguments ]----------
  //    device.setup("OpenMP",
  //                 occa::schedule = "compact",
  //                 occa::chunk    = 10);
  //
  //    device.setup("OpenCL",
  //                 occa::platformID = 0,
  //                 occa::deviceID   = 0);
  //
  //    device.setup("CUDA",
  //                 occa::deviceID = 0);
  //
  //    device.setup("Pthreads",
  //                 occa::threadCount = 4,
  //                 occa::schedule    = "compact",
  //                 occa::pinnedCores = "[0, 0, 1, 1]");
  //
  //    device.setup("COI",
  //                 occa::deviceID = 0);
  //========================================================

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  // OKL: OCCA Kernel Language
  addVectors = device.buildKernelFromSource("addVectors.okl",
                                            "addVectors");

  // OFL: OCCA Fortran Language
  // addVectors = device.buildKernelFromSource("addVectors.ofl",
  //                                           "addVectors");

  //---[ Don't need to set these up when using OKL/OFL ]----
  // int dims = 1;
  // int itemsPerGroup(16);
  // int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  // addVectors.setWorkingDims(dims, itemsPerGroup, groups);
  //========================================================

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for(int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  for(int i = 0; i < entries; ++i){
    if(ab[i] != (a[i] + b[i]))
      throw 1;
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  addVectors.free();
  o_a.free();
  o_b.free();
  o_ab.free();
  device.free();

  return 0;
}
