#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv) {
  int entries = 5;

  // Other useful functions:
  //   occa::setDevice("mode: 'OpenMP'")
  //   occa::device device("mode       : 'OpenCL', "
  //                       "platformID : 0, "
  //                       "deviceID   : 0");
  //   occa::device = occa::getDevice();

  // Use the default device (mode = Serial)
  float *a  = (float*) occa::umalloc(entries * sizeof(float));
  float *b  = (float*) occa::umalloc(entries * sizeof(float));
  float *ab = (float*) occa::umalloc(entries * sizeof(float));

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::kernel addVectors = occa::buildKernel("addVectors.okl",
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
  occa::finish();

  for (int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i]))
      throw 1;
  }

  occa::free(a);
  occa::free(b);
  occa::free(ab);

  occa::free(addVectors);

  // The user can also free the device
  occa::free(occa::getDevice());


  return 0;
}
