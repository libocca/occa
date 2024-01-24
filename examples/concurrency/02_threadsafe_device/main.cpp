#include <iostream>
#include <thread>
#include <atomic>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
//======================================

/* Description:
 * Thread-safety on device properties (or other OCCA properties)
 * can be provided by implementing the structure as a fully thread-safe object
 * or with APIs enabling/disabling limited access to the object.
 * Note:
 * - this example does not check the correctness of the process,
 *   which can be validated through a profiling.
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
  occa::memory o_x_d = occa::malloc<float>(1);
  occa::stream streams[n_streams];
  occa::kernel kernels[n_streams];
  std::atomic<int> id_counter = 0;

  occa::json streamProps({
    {"nonblocking", true},
  });

  // memory allocation & stream creation & kernels build
  for (auto i = 0; i < n_streams; i++) {
    streams[i] = occa::createStream(streamProps);
    o_x[i]  = occa::malloc<float>(entries);

    auto kernel_name = "power_of_pi_" + std::to_string(i);
    occa::json kernelProps({
      {"defines/block", block},
      {"defines/group", group},
      {"defines/KERNEL_NAME", kernel_name.c_str()},
      {"serial/include_std", true},
    });
    kernels[i] = occa::buildKernel("powerOfPiMult.okl",
                                   kernel_name,
                                   kernelProps);
  }

  auto launch = [&]() {
    const auto kern_id(id_counter++);

    // start thread-safe block to make sure the kernel launches
    // on the desired stream
    // - perhaps one way to implement this is to store operations
    // in thread local queues after lock() and execute them
    // when unlock() called? - there might be an easier way
    //occa::device::lock();

    occa::setStream(streams[kern_id]);
    kernels[kern_id](o_x[kern_id], entries);

    // end thread-safe zone
    //occa::device::unlock();

    streams[kern_id].finish();
  };

  std::thread threads[n_streams];
  for (auto i = 0; i < n_streams; i++)
    threads[i] = std::thread(launch);

  for (auto i = 0; i < n_streams; i++)
    threads[i].join();
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
