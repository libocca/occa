#include <iostream>

#include <occa.hpp>
#include <occa/types/fp.hpp>

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = i;
    ab[i] = 0;
  }

  occa::device device;
  occa::memory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  device.setup((std::string) args["options/device"]);

  //device.setup("mode: 'Serial'");
  // device.setup("mode: 'dpcpp'");

   //device.setup("mode     : 'OpenMP',"
   //             "schedule : 'compact',"
   //             "chunk    : 10");

  // device.setup("mode      : 'CUDA',"
  //              "device_id : 0");

  device.setup("mode      : 'dpcpp',"
		  "platform_id : 3,"
                "device_id : 0");


  // device.setup("mode        : 'OpenCL',"
  //              "platform_id : 0,"
  //              "device_id   : 1");

  // device.setup("mode        : 'Metal',"
  //              "device_id   : 1");
  

//  occa::properties deviceProps;
//  deviceProps["mode"] = "dpcpp";
//  deviceProps["platform_id"] = (int) args["options/platform-id"];
//  deviceProps["device_id"] = (int) args["options/device-id"];

//  occa::device device(deviceProps);

  // Compile a regular DPCPP kernel at run-time
  occa::properties kernelProps;
  kernelProps["okl/enabled"] = true;
  kernelProps["compiler"] = "dpcpp";
  kernelProps["compiler_linker_flags"] = "-shared -fPIC";

//========================================================

  // Allocate memory on the device
  o_a = device.malloc<float>(entries);
  o_b = device.malloc<float>(entries);

  // We can also allocate memory without a dtype
  // WARNING: This will disable runtime type checking
  o_ab = device.malloc(entries * sizeof(float));

  // Compile the kernel at run-time
  occa::kernel addVectors = device.buildKernel("addVectors.okl",
                                  "addVectors", kernelProps);
  addVectors.setRunDims(entries, 4);

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

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
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example adding two vectors"
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
