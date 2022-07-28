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

  // The default stream
  occa::stream streamA = occa::getStream(); 
  
  // Another, new stream
  occa::stream streamB = occa::createStream(); 

  occa::memory o_a  = occa::malloc<float>(entries);
  occa::memory o_b  = occa::malloc<float>(entries);
  occa::memory o_ab = occa::malloc<float>(entries);

  occa::kernel addVectors = occa::buildKernel("addVectors.okl",
                                 "addVectors");

  // Pass this property to make copies non-blocking on the host.
  occa::json async_copy({{"async", true}});

  // These copies will be submitted to the current
  // stream, which is streamA--the default stream.
  o_a.copyFrom(a,async_copy);
  o_b.copyFrom(b,async_copy);
  
  // Waits the copies in streamA to complete
  streamA.finish(); 
  
  // **IMPORTANT**
  // Operating on overlaping memory regions simultaneously
  // from different streams, without appropriate 
  // synchronization, leads to undefined behavior. 
  
  // Create *non-overlapping* memory slices for use by 
  // kernels launched in separate streams.
  occa::memory o_a1 = o_a.slice(0); // First half of a
  occa::memory o_a2 = o_a.slice(entries/2); // Second half of a

  occa::memory o_b1 = o_b.slice(0);
  occa::memory o_b2 = o_b.slice(entries/2);

  occa::memory o_ab1 = o_ab.slice(0);
  occa::memory o_ab2 = o_ab.slice(entries/2);

  occa::setStream(streamA);
  // This kernel launch is submitted to streamA.
  // It operates on the first half of each vector.
  addVectors(entries/2, o_a1, o_b1, o_ab1);

  occa::setStream(streamB);
  // This kernel launch is submitted to streamB.
  // It operates on the second half of each vector.
  addVectors(entries/2, o_a2, o_b2, o_ab2);

  // The copy below will be submitted to streamB; 
  // however, we need to wait for the kernel
  // submitted to streamA to finish since the 
  // entire vector is copied.
  streamA.finish();
  o_ab.copyTo(ab,async_copy);

  // Wait for streamB to finish
  streamB.finish();

  // Verify the results
  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

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
