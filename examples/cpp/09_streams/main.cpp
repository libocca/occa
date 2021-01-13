#include <iostream>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 8;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  occa::stream streamA, streamB;

  streamA = occa::getStream();
  streamB = occa::createStream();

  o_a  = occa::malloc<float>(entries);
  o_b  = occa::malloc<float>(entries);
  o_ab = occa::malloc<float>(entries);

  addVectors = occa::buildKernel("addVectors.okl",
                                 "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  occa::setStream(streamA);
  addVectors(entries, o_a, o_b, o_ab);

  occa::setStream(streamB);
  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for (int i = 0; i < entries; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  delete [] a;
  delete [] b;
  delete [] ab;
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example showing the use of multiple device streams"
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
