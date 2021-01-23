#include <iostream>

#include <occa.hpp>
#include <occa/experimental.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  // OCCA keeps 1 device in the background at a time
  // Rather than keeping the device object, we can rely on the background device
  occa::setDevice((std::string) args["options/device"]);

  int entries = 10;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  // Uses the background device
  occa::memory o_a  = occa::malloc<float>(entries, a);
  occa::memory o_b  = occa::malloc<float>(entries, b);
  occa::memory o_ab = occa::malloc<float>(entries);

  occa::scope scope({
    // Capture variables
    {"a", o_a},
    {"b", o_b},
    {"ab", o_ab}
  }, {
    // Props for the kernel are passed here
    // For example, if we wanted to define a variable at compile-time:
    // {"defines/MY_VALUE", value}
  });

  // JIT-compile a kernel given the for-loop definitions:
  //
  // for (int index = 0; index < entries; ++index; @tile(16, @outer, @inner) {
  //   <lambda>
  // }
  //
  // We support 1D, 2D, and 3D loops which depend on how many arguments .tile(), .outer(), and .inner() take.
  // Based on the loop types, outerIndex and/or innerIndex will be of types:
  //   1 argument -> int
  //   2 argument -> int2 (outerIndex.x, outerIndex.y)
  //   3 argument -> int3 (outerIndex.x, outerIndex.y, outerIndex.z)
  //
  // For finer granularity, this could have been called using:
  //
  //    .outer(entries / 2)
  //    .inner(2)
  //
  // which is useful when outer and inner dimensions don't match
  occa::forLoop()
    .tile({entries, 16})
    .run(OCCA_FUNCTION(scope, [=](const int i) -> void {
      ab[i] = a[i] + b[i];
    }));

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
      "Example using occa::forLoop for inline kernels"
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
