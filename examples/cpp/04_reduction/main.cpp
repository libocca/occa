#include <iostream>

#include <occa.hpp>
#include <occa/types/fp.hpp>

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  // occa::setDevice("mode: 'Serial'");
  // occa::setDevice("mode: 'OpenMP'");
  // occa::setDevice("mode: 'CUDA'  , device_id: 0");
  // occa::setDevice("mode: 'OpenCL', platform_id: 0, device_id: 0");
  // occa::setDevice("mode: 'Metal', device_id: 0");
  occa::setDevice((std::string) args["options/device"]);

  // Choosing something not divisible by 256
  int entries = 10000;
  int block   = 256;
  int blocks  = (entries + block - 1)/block;

  float *vec      = new float[entries];
  float *blockSum = new float[blocks];

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
  occa::memory o_vec      = occa::malloc<float>(entries);
  occa::memory o_blockSum = occa::malloc<float>(blocks);

  // Pass value of 'block' at kernel compile-time
  occa::properties reductionProps;
  reductionProps["defines/block"] = block;

  occa::kernel reduction = occa::buildKernel("reduction.okl",
                                             "reduction",
                                             reductionProps);

  // Host -> Device
  o_vec.copyFrom(vec);

  reduction(entries, o_vec, o_blockSum);

  // Host <- Device
  o_blockSum.copyTo(blockSum);

  // Finalize the reduction in the host
  for (int i = 1; i < blocks; ++i) {
    blockSum[0] += blockSum[i];
  }

  // Validate
  if (!occa::areBitwiseEqual(blockSum[0], sum)) {
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

occa::json parseArgs(int argc, const char **argv) {
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example of a reduction kernel which sums a vector in parallel"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"mode: 'Serial'\")")
      .withArg()
      .withDefaultValue("mode: 'Serial'")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
