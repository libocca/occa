#include <iostream>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 12;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 + i;
    ab[i] = 0;
  }

  //---[ Device Setup ]-------------------------------------
  occa::device device( (std::string) args["options/device"]);


  // Allocate memory on the device
  occa::memory o_a = device.malloc<float>(entries);
  occa::memory o_b = device.malloc<float>(entries);

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Compile the kernel at run-time
  occa::kernel addVectors = device.buildKernel("addVectors.okl","addVectors");

  //---[ Memory Pool Setup ]-------------------------------------
  occa::experimental::memoryPool memPool = device.createMemoryPool();

  std::cout << "Creation           - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

  // Make a reservation from memory pool
  // Memory pool is a single allocation, and consists of just o_ab
  /*
      |====o_ab====|
  */
  occa::memory o_ab = memPool.reserve<float>(entries);

  std::cout << "First reservation  - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);

  // Copy result to the host
  o_ab.copyTo(ab);

  // Check values
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

  // Unallocated occa::memory
  occa::memory mem;

  // New scope
  {
    /*Slicing o_ab will not trigger reallocation or
      increase memoryPool's reservation size*/
    /*
      |====o_ab====|
      |ab_h1||ab_h2|
    */
    occa::memory o_ab_half1 = o_ab.slice(0, entries/2);
    occa::memory o_ab_half2 = o_ab.slice(entries/2);

    std::cout << "Slice (no resize)  - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

    // Copy result to the host
    o_ab_half1.copyTo(ab);
    o_ab_half2.copyTo(ab+entries/2);

    // Check values
    for (int i = 0; i < entries; ++i) {
      if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
        throw 1;
      }
    }


    // Trigger a resize by requesting a new reservation:
    /*
      |====o_ab====||==tempMem===|
      |ab_h1||ab_h2|
    */
    occa::memory tempMem = memPool.reserve<float>(entries);

    std::cout << "New reservation    - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

    // o_ab still has its data after resize
    o_ab.copyTo(ab);

    // Check values
    for (int i = 0; i < entries; ++i) {
      if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
        throw 1;
      }
    }


    // Trigger another resize by reserving the outer-scope mem
    /*
      |====o_ab====||==tempMem===||====mem=====|
      |ab_h1||ab_h2|
    */
    mem = memPool.reserve<float>(entries);

    std::cout << "Second reservation - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;
  }


  // tempMem leaves scope and is destroyed. This leaves a 'hole' in the memory pool
  /*
    |====o_ab====||------------||====mem=====|
  */
  std::cout << "Release tempMem    - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

  // Request a new reservation, should fit in the hole and not trigger resize:
  /*
    |====o_ab====||==tempMem===||====mem=====|
  */
  occa::memory tempMem = memPool.reserve<float>(entries);

  std::cout << "Re-reserve         - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

  // Freeing doesnt change the mempool size, only the reserved size
  /*
    |====o_ab====||------------||====mem=====|
  */
  tempMem.free();

  // Requesting a new reservation that *doesn't* fit in the hole triggers a resize.
  // The mempool is defragmented on resizing
  /*
    |====o_ab====||====mem=====||========tempMem=========|
  */
  tempMem = memPool.reserve<float>(2*entries);

  // Note the mempool size increases by only entries*sizeof(float),
  //  despite the new 2*entries float reservation!
  std::cout << "Reserve and resize - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

  // o_ab still has its data after resize
  o_ab.copyTo(ab);

  // Check values
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

  // Finally, free a reserved memory then "re-size to fit"
  /*
    |====o_ab====||------------||========tempMem=========|
  */
  mem.free();
  /*
    |====o_ab====||========tempMem=========|
  */
  memPool.shrinkToFit();

  std::cout << "Resize to fit      - Memory pool: Size = " << memPool.size()
                   << ", Reserved = " << memPool.reserved() << std::endl;

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
      "Example using memory pool"
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
