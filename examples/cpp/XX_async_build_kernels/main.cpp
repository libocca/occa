#include <iostream>
#include <future>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);

occa::kernel myBuildKernel(
  std::string file_name,
  std::string kernel_name,
  const occa::device& d
) {
  return d.buildKernel(file_name,kernel_name);
}

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 12;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::device device;
  occa::memory o_a, o_b, o_ab;

  //---[ Device Setup ]-------------------------------------
  device.setup((std::string) args["options/device"]);

  // Compile the kernel early, (potentially) using a separate thread.
  std::string file_name = "addVectors.okl";
  std::string kernel_name = "addVectors";
  std::future<occa::kernel> future_kernel = std::async(
    std::launch::async,
    &myBuildKernel,
    file_name,
    kernel_name,
    device
  );
  //========================================================

  // Allocate memory on the device
  o_a = device.malloc<float>(entries);
  o_b = device.malloc<float>(entries);

  // We can also allocate memory without a dtype
  // WARNING: This will disable runtime type checking
  o_ab = device.malloc(entries * sizeof(float));

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Get the device kernel just before we need it;
  occa::kernel addVectors = future_kernel.get();
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
      "Example adding two vectors"
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
