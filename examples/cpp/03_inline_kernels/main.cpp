#include <iostream>

#include <occa.hpp>

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  occa::setDevice((std::string) args["options/device"]);

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::memory o_a  = occa::malloc(entries, occa::dtype::float_, a);
  occa::memory o_b  = occa::malloc(entries, occa::dtype::float_, b);
  occa::memory o_ab = occa::malloc(entries, occa::dtype::float_);

  occa::properties props;
  props["defines/TILE_SIZE"] = 16;

  // Restrictions on using inlined kernels:
  //   - Memory allocations must include a dtype
  // Temporary restrictions for using inlined kernels:
  //   - Cannot use unified memory from occa::umalloc
  OCCA_INLINED_KERNEL(
    (entries, o_a, o_b, o_ab),
    props,
    (
      for (int i = 0; i < entries; ++i; @tile(TILE_SIZE, @outer, @inner)) {
        o_ab[i] = o_a[i] + o_b[i];
      }
    )
  );

  // Copy result to the host
  o_ab.copyTo(ab);

  for (int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i]))
      throw 1;
  }

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
