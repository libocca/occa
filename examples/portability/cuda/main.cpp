#include <iostream>

#include "occa.hpp"
#include "occa/modes/cuda/utils.hpp"

#include "cuda_runtime_api.h"

int main(int argc, char **argv) {
  int entries = 5;

  //---[ Init CUDA ]------------------
  int cuDeviceID;
  cudaStream_t cuStream;
  void *cu_a, *cu_b, *cu_ab;

  // Default: cuStream = 0
  cudaStreamCreate(&cuStream);

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

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  o_a  = occa::cuda::wrapMemory(device, cu_a , entries*sizeof(float));
  o_b  = occa::cuda::wrapMemory(device, cu_b , entries*sizeof(float));
  o_ab = occa::cuda::wrapMemory(device, cu_ab, entries*sizeof(float));

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
