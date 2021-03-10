#include <iostream>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  occa::setDevice((std::string) args["options/device"]);

  // Choosing something not divisible by 256
  int entries = 10000;
  int block   = 256;
  int blocks  = (entries + block - 1)/block;

  float *vec      = new float[entries];
  float *blockSum = new float[blocks];
  float atomicSum = 0;

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
  occa::memory o_vec       = occa::malloc<float>(entries);
  occa::memory o_blockSum  = occa::malloc<float>(blocks);
  occa::memory o_atomicSum = occa::malloc<float>(1, &atomicSum);

  // Pass value of 'block' at kernel compile-time
  occa::json reductionProps({
    {"defines/block", block},
  });

  occa::kernel reductionWithSharedMemory = (
    occa::buildKernel("reduction.okl",
                      "reductionWithSharedMemory",
                      reductionProps)
  );

  occa::kernel reductionWithAtomics = (
    occa::buildKernel("reduction.okl",
                      "reductionWithAtomics",
                      reductionProps)
  );

  // Host -> Device
  o_vec.copyFrom(vec);

  reductionWithSharedMemory(entries, o_vec, o_blockSum);
  reductionWithAtomics(entries, o_vec, o_atomicSum);

  // Host <- Device
  o_blockSum.copyTo(blockSum);
  o_atomicSum.copyTo(&atomicSum);

  // Finalize the reduction in the host
  for (int i = 1; i < blocks; ++i) {
    blockSum[0] += blockSum[i];
  }

  // Validate
  bool hasError = false;
  if (!occa::areBitwiseEqual(blockSum[0], sum)) {
    std::cout << "sum      = " << sum << '\n'
              << "blockSum = " << blockSum[0] << '\n';

    std::cout << "(Shared) Reduction failed\n";
    hasError = true;
  }
  else {
    std::cout << "(Shared) Reduction = " << blockSum[0] << '\n';
  }

  if (!occa::areBitwiseEqual(atomicSum, sum)) {
    std::cout << "sum       = " << sum << '\n'
              << "atomicSum = " << atomicSum << '\n';

    std::cout << "(Atomic) Reduction failed\n";
    hasError = true;
  }
  else {
    std::cout << "(Atomic) Reduction = " << blockSum[0] << '\n';
  }

  if (hasError) {
    throw 1;
  }

  // Free host memory
  delete [] vec;
  delete [] blockSum;

  // Device memory is automatically freed

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example of a reduction kernel which sums a vector in parallel"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'Serial'}\")")
      .withArg()
      .withDefaultValue("{mode: 'Serial'}")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
