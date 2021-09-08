#include <iostream>

#include <occa.hpp>
#include <cuda_runtime_api.h>

int main(int argc, char **argv) {
  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  //---[ CUDA ]-------------------------
  float *cu_a, *cu_b, *cu_ab;

  cudaMalloc((void**) &cu_a , entries * sizeof(float));
  cudaMalloc((void**) &cu_b , entries * sizeof(float));
  cudaMalloc((void**) &cu_ab, entries * sizeof(float));


  cudaMemcpy(cu_a, a, entries * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cu_b, b, entries * sizeof(float), cudaMemcpyHostToDevice);
  //====================================


  //---[ OCCA ]-------------------------
  occa::setDevice({
    {"mode", "CUDA"},
    {"device_id", 0}
  });

  occa::memory o_a  = occa::wrapMemory<float>(cu_a , entries);
  occa::memory o_b  = occa::wrapMemory<float>(cu_b , entries);
  occa::memory o_ab = occa::wrapMemory<float>(cu_ab, entries);

  occa::kernel addVectors = (
    occa::buildKernel("addVectors.okl",
                       "addVectors")
  );

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);
  //====================================

  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
