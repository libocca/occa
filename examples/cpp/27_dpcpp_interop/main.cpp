#include <iostream>

#include <occa.hpp>
#include <CL/sycl.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);


int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 16;

  int *a  = new int[entries];
  int *b  = new int[entries];
  int *ab = new int[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = i;
    ab[i] = 0;
  }

  int platform_id = (int) args["options/platform-id"];
  int device_id = (int) args["options/device-id"];

  // Setup the platform and device IDs
  occa::json deviceProps;
  deviceProps["mode"] = "dpcpp";
  deviceProps["platform_id"] = platform_id;
  deviceProps["device_id"] = device_id;

  occa::device device(deviceProps);

  // Allocate dpcpp memory on the device
  auto dpcpp_device = sycl::platform::get_platforms()[platform_id].get_devices()[device_id];
  sycl::context dpcpp_context(dpcpp_device);

  int *dpcpp_a = sycl::malloc_device<int>(entries, dpcpp_device, dpcpp_context);
  int *dpcpp_b = sycl::malloc_device<int>(entries,dpcpp_device,dpcpp_context);
  int *dpcpp_ab = sycl::malloc_device<int>(entries,dpcpp_device,dpcpp_context);
  
  // Wrap dpcpp memory in oc
  // occa::setDevice(device);
  device.dontUseRefs();
  occa::memory o_a = device.wrapMemory<int>(dpcpp_a, entries);
  occa::memory o_b = device.wrapMemory<int>(dpcpp_b, entries);
  occa::memory o_ab = device.wrapMemory<int>(dpcpp_ab, entries);
  
  occa::kernel addVectors = device.buildKernel("addVectors.okl", "addVectors");

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);
  o_ab.copyFrom(ab);

  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);
  // Copy result to the host
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

  // @todo: Do we need to free dpcpp memory? Or does OCCA handle this?
  ::sycl::free(dpcpp_a, dpcpp_context);
  ::sycl::free(dpcpp_b, dpcpp_context);
  ::sycl::free(dpcpp_ab, dpcpp_context);

  // Free host memory
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
      "Example of using a regular SYCL/DPC++ kernel instead of an OCCA kernel"
    )
    .addOption(
      occa::cli::option('p', "platform-id",
                        "DPC++ platform ID (default: 0)")
      .withArg()
      .withDefaultValue(0)
    )
    .addOption(
      occa::cli::option('d', "device-id",
                        "DPC++ device ID (default: 0)")
      .withArg()
      .withDefaultValue(0)
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
