#include <iostream>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv);

struct myFloat {
  float value;
};

struct myFloat2 {
  float x, y;
};

struct myFloat4 {
  float values[4];
};

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 8;

  myFloat *a   = new myFloat[entries];
  myFloat2 *b  = new myFloat2[entries / 2];
  myFloat4 *ab = new myFloat4[entries / 4];

  for (int i = 0; i < (entries / 4); ++i) {
    for (int j = 4*i; j < (4*i + 4); ++j) {
      a[j].value = j;
    }
    for (int j = 2*i; j < (2*i + 2); ++j) {
      b[j].x = 2*j;
      b[j].y = 2*j + 1;
    }
    for (int j = 0; j < 4; ++j) {
      ab[i].values[j] = 0;
    }
  }

  occa::setDevice((std::string) args["options/device"]);

  // Explanation for dtypes
  // - occa::memory can have a dtype to enable runtime type checking
  // - dtype_t used in typed memory allocations must be 'global'
  // - Global dtype_t objects are treated as singletons and assumed
  //     to exist while the memory objects are still alive
  // NOTE:
  // - Don't deallocate used dtype_t
  // - Don't use local dtype_t objects

  // Basic dtype
  // NOTE: We're using local dtype_t since their only use is in this file
  occa::dtype_t myFloatDtype("myFloat", sizeof(float));
  myFloatDtype.registerType();

  // Struct dtype
  occa::dtype_t myFloat2Dtype;
  myFloat2Dtype.registerType();
  myFloat2Dtype.addField("x", occa::dtype::float_);
  myFloat2Dtype.addField("y", occa::dtype::float_);

  // Tuple dtype
  occa::dtype_t myFloat4Dtype = occa::dtype_t::tuple(occa::dtype::float_, 4);
  myFloat4Dtype.registerType();

  // Allocate memory on the device
  occa::memory o_a  = occa::malloc(entries    , myFloatDtype);
  occa::memory o_b  = occa::malloc(entries / 2, myFloat2Dtype);
  occa::memory o_ab = occa::malloc(entries / 4, myFloat4Dtype);

  // Compile the kernel at run-time
  occa::kernel addVectors = occa::buildKernel("addVectors.okl",
                                              "addVectors");

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Launch device kernel
  addVectors(entries,
             o_a.cast(occa::dtype::float_),
             o_b,
             o_ab);

  // Copy result to the host
  o_ab.copyTo(ab);

  // Assert values
  for (int i = 0; i < (entries / 4); ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << '(' << i << ',' << j << ") : " << ab[i].values[j] << '\n';
    }
  }
  for (int i = 0; i < entries; ++i) {
    float a_i  = a[i].value;
    float b_i  = (i % 2) ? b[i / 2].y : b[i / 2].x;
    float ab_i = ab[i / 4].values[i % 4];
    if (!occa::areBitwiseEqual(ab_i, a_i + b_i)) {
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
      "Example with custom dtypes, showcasing runtime type checking"
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
