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

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  // Setup the platform and device IDs
  occa::device device({
    {"mode", "OpenCL"},
    {"platform_id", (int) args["options/platform-id"]},
    {"device_id", (int) args["options/device-id"]}
  });

  // Allocate memory on the device
  occa::memory o_a = device.malloc<float>(entries);
  occa::memory o_b = device.malloc<float>(entries);
  occa::memory o_ab = device.malloc<float>(entries);

  // Compile a regular OpenCL kernel at run-time
  occa::json kernelProps({
    {"okl/enabled", false}
  });

  occa::kernel addVectors = device.buildKernel("addVectors.cl",
                                               "addVectors",
                                               kernelProps);

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Set the kernel dimensions
  //   setRunDims(
  //     occa::dim(groupsX, groupsY = 1, groupsZ = 1), <- @outer dims in OKL
  //     occa::dim(itemsX, itemsY = 1, itemsZ = 1)     <- @inner dims in OKL
  //   )
  addVectors.setRunDims((entries + 15) / 16, 16);

  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);

  // Copy result to the host
  o_ab.copyTo(ab);

  // Assert values
  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

  // Free host memory
  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example of using a regular OpenCL kernel instead of an OCCA kernel"
    )
    .addOption(
      occa::cli::option('p', "platform-id",
                        "OpenCL platform ID (default: 0)")
      .withArg()
      .withDefaultValue(0)
    )
    .addOption(
      occa::cli::option('d', "device-id",
                        "OpenCL device ID (default: 0)")
      .withArg()
      .withDefaultValue(0)
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
