#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv) {
  int entries = 5;

  // [U]nified [V]irtual [A]dressing is
  //   disabled by default
  // occa::enableUVAByDefault();
  occa::device device("mode       : 'OpenCL', "
                      "platformID : 0, "
                      "deviceID   : 0, "
                      "uva        : true");

  // Allocate [uva] arrays that will
  //   automatically synchronize between
  //   the process and [device]
  float *a  = (float*) device.uvaAlloc(entries * sizeof(float));
  float *b  = (float*) device.uvaAlloc(entries * sizeof(float));
  float *ab = (float*) device.uvaAlloc(entries * sizeof(float));

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::kernel addVectors = device.buildKernel("addVectors.okl",
                                               "addVectors");

  // Arrays a, b, and ab are now resident
  //   on [device]
  addVectors(entries, a, b, ab);

  // b is not const in the kernel, so we can use
  //   dontSync(b) to manually force b to not sync
  occa::dontSync(b);

  // Finish work queued up in [device],
  //   synchronizing a, b, and ab and
  //   making it safe to use them again
  device.finish();

  for (int i = 0; i < 5; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i])) {
      throw 1;
    }
  }

  occa::free(a);
  occa::free(b);
  occa::free(ab);

  addVectors.free();
  device.free();

  return 0;
}
