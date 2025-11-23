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

  int entries = 32 * 32;

  float *h_data = new float[entries];
  float h_result = 0;
  float ref_result = 0;

  for (int i = 0; i < entries; ++i) {
    h_data[i] = i;
  }

  // Setup the platform and device IDs
  occa::device device({{"mode", "CUDA"},
                       {"device_id", (int)args["options/device-id"]}});

  // Allocate memory on the device
  occa::memory d_data = device.malloc<float>(entries);
  occa::memory d_result = device.malloc<float>(1);

  // Compile a regular CUDA kernel at run-time
  occa::json kernelProps({{"okl/enabled", false}, {"sharedMemBytes", 32 * 4}});

  occa::kernel reduce =
      device.buildKernel("sum_reductiom_dynamic_shm.cu", "reduce", kernelProps);

  // Copy memory to the device
  d_data.copyFrom(h_data);
  d_result.copyFrom(&h_result);

  // Set the kernel dimensions
  reduce.setRunDims(32, 32);

  // Launch device kernel
  reduce(d_data, d_result);

  // Copy result to the host
  d_result.copyTo(&h_result);

  // Calculate reference
  for (int i = 0; i < entries; ++i) {
    ref_result += h_data[i];
  }

  // Assert values
  printf("Ref result: %f, GPU result: %f\n", ref_result, h_result);

  // Free host memory
  delete[] h_data;

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
      .withDescription(
          "Example of using a regular CUDA kernel instead of an OCCA kernel")
      .addOption(
          occa::cli::option('d', "device-id", "OpenCL device ID (default: 0)")
              .withArg()
              .withDefaultValue(0))
      .addOption(
          occa::cli::option('v', "verbose", "Compile kernels in verbose mode"));

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
