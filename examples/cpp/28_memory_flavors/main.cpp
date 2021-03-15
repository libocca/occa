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

  constexpr int entries{12};

  //---[ Device Setup ]-------------------------------------
  occa::device device;
  device.setup((std::string) args["options/device"]);

  //========================================================

  // Allocate USM memory: accessible by the host and the device
  occa::json input_properties;
  input_properties["unified"] = true;
  occa::memory o_a = device.malloc<float>(entries,input_properties);
  occa::memory o_b = device.malloc<float>(entries,input_properties);

  // Allocate device accessible memory on host
  occa::json output_properties;
  output_properties["host"] = true;
  occa::memory o_ab = device.malloc<float>(entries,output_properties);

  // Set input values
  auto* a = o_a.ptr<float>();
  auto* b = o_b.ptr<float>();

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = i;
  }
  // No need to copy to device since USM is migrated automatically

  // Compile the kernel at run-time
  occa::kernel addVectors = device.buildKernel("addVectors.okl","addVectors");


  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);
  device.finish();

  // No need to copy to host
  auto* ab = o_ab.ptr<float>();

  // Assert values
  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

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
