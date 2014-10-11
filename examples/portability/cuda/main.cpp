#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  int entries = 5;

  //---[ Init CUDA ]------------------
  int cuDeviceID;
  cudaStream_t cuStream;
  void *cu_a, *cu_b, *cu_ab;

  // Default: cuStream = 0
  cudaStreamCreate(&customStream);

  cudaMalloc(&cu_a , entries*sizeof(float));
  cudaMalloc(&cu_b , entries*sizeof(float));
  cudaMalloc(&cu_ab, entries*sizeof(float));

  //  ---[ Get CUDA Info ]----
  CUdevice cuDevice;
  CUcontext cuContext;

  cuDeviceGet(&cuDevice, cuDeviceID);
  cuCtxGetCurrent(&cuContext);
  //  ========================
  //====================================

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  occa::device device = occa::cuda::wrapDevice(cuDevice, cuContext);

  occa::stream stream = device.wrapStream(&cuStream);
  device.setStream(stream);

  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  o_a  = device.wrapMemory(&cu_a , entries*sizeof(float));
  o_b  = device.wrapMemory(&cu_b , entries*sizeof(float));
  o_ab = device.wrapMemory(&cu_ab, entries*sizeof(float));

  addVectors = device.buildKernelFromSource("addVectors.occa",
                                            "addVectors");

  int dims = 1;
  int itemsPerGroup(2);
  int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  addVectors.setWorkingDims(dims, itemsPerGroup, groups);

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  occa::initTimer(device);

  occa::tic("addVectors");

  addVectors(entries, o_a, o_b, o_ab);

  double elapsedTime = occa::toc("addVectors", addVectors);

  o_ab.copyTo(ab);

  std::cout << "Elapsed time = " << elapsedTime << " s" << std::endl;

  occa::printTimer();

  for(int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  addVectors.free();
  o_a.free();
  o_b.free();
  o_ab.free();

  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
