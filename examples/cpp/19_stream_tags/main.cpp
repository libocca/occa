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

  int entries = 1<<20;
  int block = 64;
  int group = 1;

  float *a = new float[entries];
  for (int i = 0; i < entries; i++)
    a[i] = 0.f;

  occa::memory o_a = occa::malloc<float>(entries);
  o_a.copyFrom(a);

  occa::json kernelProps({
    {"defines/block", block},
    {"defines/group", group},
    {"serial/include_std", true},
  });
  occa::kernel powerOfPi2 = occa::buildKernel("powerOfPi2.okl",
                                             "powerOfPi2",
                                             kernelProps);
  occa::json streamProps({
    {"nonblocking", true},
  });
  occa::stream stream_a = occa::createStream(streamProps);
  occa::stream stream_b = occa::createStream(streamProps);

  occa::setStream(stream_a);
  powerOfPi2(o_a, entries);
  occa::streamTag tag_a = occa::tagStream();

  // set stream_b to wait for the job(s) to be finished in stream_a
  occa::streamWait(stream_b, tag_a);
  
  occa::setStream(stream_b);
  powerOfPi2(o_a, entries);
  occa::streamTag tag_b = occa::tagStream();

  // set the device to wait for stream_b to finish
  occa::waitFor(tag_b);

  o_a.copyTo(a);

  const float tol = 1e-3;
  for (auto i = 0; i < entries; i++) {
    if (fabs(a[i] - 3.14159) > tol) {
      std::cerr << "Invalid output value: " << a[i] << " in " << i << std::endl;
      return -1;
    }
  }
  return 0;
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
