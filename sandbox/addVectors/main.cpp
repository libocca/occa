#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  int entries = 16;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  // occa::availableDevices<occa::OpenCL>();

  std::string mode = "OpenMP";
  int platformID = 0;
  int deviceID   = 0;

  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  device.setup(mode, platformID, deviceID);

  occa::setDevice(0);
  const int devID = occa::getDevice();

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  addVectors = device.buildKernelFromSource("addVectors.okl",
                                            "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  occa::initTimer(device);

  occa::tic("addVectors");

  addVectors(entries, o_a, o_b, o_ab);

  double elapsedTime = occa::toc("addVectors", addVectors);

  o_ab.copyTo(ab);

  std::cout << "Elapsed time = " << elapsedTime << " s" << std::endl;

  occa::printTimer();

  for(int i = 0; i < entries; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  delete [] a;
  delete [] b;
  delete [] ab;

  addVectors.free();
  o_a.free();
  o_b.free();
  o_ab.free();
  device.free();

  return 0;
}
