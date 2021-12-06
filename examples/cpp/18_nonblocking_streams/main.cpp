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

  occa::setDevice(occa::json::parse(args["options/device"]));

  const int n_streams = 8;
  int entries = 1<<20;
  int block = 64;
  int group = 1;

  occa::memory o_x[n_streams];
  occa::memory o_x_d = occa::malloc<float>(1);

  occa::json kernelProps({
    {"defines/block", block},
    {"defines/group", group},
  });
  occa::kernel powerOfPi = occa::buildKernel("powerOfPi.okl",
                                             "powerOfPi",
                                             kernelProps);

  occa::stream streams[n_streams];
  occa::json streamProps({
    {"nonblocking", true},
  });
  occa::stream default_stream = occa::getStream();

  for (auto i = 0; i < n_streams; i++) {
      streams[i] = occa::createStream(streamProps);

      o_x[i]  = occa::malloc<float>(entries);

      occa::setStream(streams[i]);

      powerOfPi(o_x[i], entries);

      occa::setStream(default_stream);

      powerOfPi(o_x_d, 1);
  }
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example showing the use of multiple device streams"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'CUDA', device_id: 0}\")")
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
