#include <iostream>

#include <occa.hpp>
#include <occa/types/fp.hpp>
#include <CL/sycl.hpp>

occa::json parseArgs(int argc, const char **argv);


int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 5;

  int *a  = new int[entries];
  int *b  = new int[entries];
  int *ab = new int[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = i;
    ab[i] = 0;
  }

  // Setup the platform and device IDs
  occa::properties deviceProps;
  deviceProps["mode"] = "dpcpp";
  deviceProps["platform_id"] = (int) args["options/platform-id"];
  deviceProps["device_id"] = (int) args["options/device-id"];
//  occa::device device(deviceProps);

  occa::device device(deviceProps);
  // Allocate memory on the device
  occa::memory o_a = device.malloc<int>(entries);
  occa::memory o_b = device.malloc<int>(entries);
  occa::memory o_ab = device.malloc<int>(entries);

  // Compile a regular OpenCL kernel at run-time
  occa::properties kernelProps;
  kernelProps["okl/enabled"] = false;
  kernelProps["compiler"] = "dpcpp";
  kernelProps["compiler_linker_flags"] = "-shared -fPIC";
 
  occa::kernel addVectors = device.buildKernel("addVectors.cpp",
                                               "addVectors_it",
                                               kernelProps);
 
 // occa::kernel addVectors(&device, "addvector", kernelProps, addVector_it);
  //occa::kernel addVectors;//only for testing dpcpp is found by occa

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);
  o_ab.copyFrom(ab);

  // Set the kernel dimensions
  //   setRunDims(
  //     occa::dim(groupsX, groupsY = 1, groupsZ = 1), <- @outer dims in OKL
  //     occa::dim(itemsX, itemsY = 1, itemsZ = 1)     <- @inner dims in OKL
  //   )
  addVectors.setRunDims(entries, (entries+8)/8);
  // Launch device kernel
  addVectors(o_a, o_b, o_ab);

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
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
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
