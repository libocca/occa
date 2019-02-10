#include <iostream>

#include <occa.hpp>

int main(int argc, char **argv) {
  occa::setDevice("mode: 'CUDA', device_id: 0");
  // occa::setDevice("mode: 'OpenCL', platform_id: 0, device_id: 0");

  occa::kernel reduction;

  // Choosing something not divisible by 256
  int entries = 10000;
  int block   = 256;
  int blocks  = (entries + block - 1)/block;

  float *vec      = new float[entries];
  float *blockSum = new float[blocks];
  occa::memory o_vec, o_blockSum;

  float sum = 0;

  // Initialize device memory
  for (int i = 0; i < entries; ++i) {
    vec[i] = 1;
    sum   += vec[i];
  }

  for (int i = 0; i < blocks; ++i) {
    blockSum[i] = 0;
  }

  // Allocate memory on the device
  o_vec      = occa::malloc(entries, occa::dtype::float_);
  o_blockSum = occa::malloc(blocks, occa::dtype::float_);

  // Pass value of 'block' at kernel compile-time
  occa::properties reductionProps;
  reductionProps["okl"] = false; // Disable OKL parsing
  reductionProps["defines/block"] = block;

  const std::string kernelFile = (
    (occa::getDevice().mode() == "CUDA")
    ? "reduction.cu"
    : "reduction.cl"
  );

  reduction = occa::buildKernel(kernelFile,
                                "reduction",
                                reductionProps);

  // Host -> Device
  o_vec.copyFrom(vec);

  // Set kernel launch dimensions before launching
  reduction.setRunDims(blocks, block);
  reduction(entries, o_vec, o_blockSum);

  // Host <- Device
  o_blockSum.copyTo(blockSum);

  // Finalize the reduction in the host
  for (int i = 1; i < blocks; ++i) {
    blockSum[0] += blockSum[i];
  }

  // Validate
  if (blockSum[0] != sum) {
    std::cout << "sum      = " << sum << '\n'
              << "blockSum = " << blockSum[0] << '\n';

    std::cout << "Reduction failed\n";
    throw 1;
  }
  else {
    std::cout << "Reduction = " << blockSum[0] << '\n';
  }

  // Free host memory
  delete [] vec;
  delete [] blockSum;

  // Device memory is automatically freed

  return 0;
}
