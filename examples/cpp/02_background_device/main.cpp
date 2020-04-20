#include <iostream>

#include <occa.hpp>
#include <occa/types/fp.hpp>

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  // Other useful functions:
  //   occa::setDevice("mode: 'OpenMP'")
  //   occa::device = occa::getDevice();
  // Options:
  //   occa::setDevice("mode: 'Serial'");
  //   occa::setDevice("mode: 'OpenMP'");
  //   occa::setDevice("mode: 'CUDA'  , device_id: 0");
  //   occa::setDevice("mode: 'OpenCL', platform_id: 0, device_id: 0");
  //   occa::setDevice("mode: 'Metal', device_id: 0");
  //
  // The default device uses "mode: 'Serial'"
  occa::setDevice((std::string) args["options/device"]);

  int entries = 5;

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

  occa::free(a);
  occa::free(b);
  occa::free(ab);

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example showing how to use background devices, allowing passing of the device implicitly"
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
