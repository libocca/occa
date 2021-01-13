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

  int entries = 5;

  // umalloc: [U]nified [M]emory [Alloc]ation
  // Allocate host memory that auto-syncs with the device
  //   between before kernel calls and device::finish()
  //   if needed.
  float *a  = occa::umalloc<float>(entries);
  float *b  = occa::umalloc<float>(entries);
  float *ab = occa::umalloc<float>(entries);

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::kernel addVectors = occa::buildKernel("addVectors.okl",
                                              "addVectors");

  // Arrays a, b, and ab are now resident
  //   on [device]
  addVectors(entries, a, b, ab);

  // b is not const in the kernel, so we can use
  //   dontSync(b) to manually force b to not sync
  occa::dontSync(b);

  // Finish work queued up in [device],
  //   synchronizing a, b, and ab and
  //   making it safe to use them again
  occa::finish();

  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

  // Memory is automatically freed

  return 0;
}
occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example using unified memory, where host and device data is mirrored and synced"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'Serial'}\")")
      .withArg()
      .withDefaultValue("{mode: 'CUDA', device_id: 0}")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
