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

  int entries = 16;

  int *a  = new int[entries];
  int *b  = new int[entries];
  int *ab = new int[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1-i;
    ab[i] = 0;
  }

  // Setup the platform and device IDs
  occa::properties deviceProps;
  deviceProps["mode"] = "dpcpp";
  deviceProps["platform_id"] = (int) args["options/platform-id"];
  deviceProps["device_id"] = (int) args["options/device-id"];

  occa::device device(deviceProps);
  // Allocate memory on the device
  occa::memory o_a = device.malloc<int>(entries);
  occa::memory o_b = device.malloc<int>(entries);
  occa::memory o_ab = device.malloc<int>(entries);

  // Compile a regular DPCPP kernel at run-time
  occa::properties kernelProps;
  kernelProps["okl/enabled"] = false;
 
  occa::kernel addVectors = device.buildKernel("addVectors.cpp",
                                               "addVectors",
                                               kernelProps);
 
  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);
  o_ab.copyFrom(ab);

  addVectors.setRunDims(entries/4,4);
  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);
  // Copy result to the host
  device.finish();
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
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example of using a regular SYCL/DPC++ kernel instead of an OCCA kernel"
    )
    .addOption(
      occa::cli::option('p', "platform-id",
                        "DPC++ platform ID (default: 0)")
      .withArg()
      .withDefaultValue(0)
    )
    .addOption(
      occa::cli::option('d', "device-id",
                        "DPC++ device ID (default: 0)")
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
