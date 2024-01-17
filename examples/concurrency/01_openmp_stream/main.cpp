#include <iostream>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
//======================================

/* Description:
 * Runs kernels on multiple "streams" for OpenMP mode
 * which delivers the same concept as in other modes
 * through internal implementation using OpenMP threads
 * (i.e. multi-threaded operations with synchronization).
 * Note:
 * - this example requires a way to validate the correctness
 *   of the process.
 * - this WIP which may have additional updates.
 */

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  occa::setDevice(occa::json::parse(args["options/device"]));

  const int n_streams = 8;
  int entries = 1<<20;
  int block = 64;
  int group = 1;

  // Prepare data array and stream
  occa::memory o_x[n_streams];
  occa::stream streams[n_streams];

  occa::json streamProps({
    {"nonblocking", true},
  });

  occa::json kernelProps({
    {"defines/block", block},
    {"defines/group", group},
    {"serial/include_std", true},
  });
  occa::kernel kernel = occa::buildKernel("powerOfPi.okl",
                                          "powerOfPi",
                                          kernelProps);

  // memory allocation & stream creation
  for (auto i = 0; i < n_streams; i++) {
    o_x[i]  = occa::malloc<float>(entries);
    streams[i] = occa::createStream(streamProps);
  }

  // launch kernels on OpenMP threads "streams"
  for (auto i = 0; i < n_streams; i++) {
    occa::setStream(streams[i]);
    kernel(o_x[i], entries);
  }

  // synchronzie streams
  for (auto i = 0; i < n_streams; i++)
    streams[i].finish();
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example showing the use of OpenMP streams"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'OpenMP'}\")")
      .withArg()
      .withDefaultValue("{mode: 'OpenMP'}")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
