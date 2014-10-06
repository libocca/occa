#include <iostream>

#include "occa.hpp"

void createLibrary(){
  {
    occa::device d;
    d.setup("OpenMP");
    d.cacheKernelInLibrary("addVectors.occa", "addVectors");
  }

#if OCCA_OPENCL_ENABLED
  {
    occa::device d;
    d.setup("OpenCL", 0, 0);
    d.cacheKernelInLibrary("addVectors.occa", "addVectors");
  }
#endif

#if OCCA_CUDA_ENABLED
  {
    occa::device d;
    d.setup("CUDA", 0);
    d.cacheKernelInLibrary("addVectors.occa", "addVectors");
  }
#endif

  occa::library::save("testLib");
}

void loadFromLibrary(){
  occa::library::load("testLib");
}

int main(int argc, char **argv){
  createLibrary();
  loadFromLibrary();

  int entries = 5;

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

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  addVectors = occa::library::loadKernel(device, "addVectors");

  int dims = 1;
  int itemsPerGroup(2);
  int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  addVectors.setWorkingDims(dims, itemsPerGroup, groups);

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  occa::initTimer(device);

  occa::tic("addVectors");

  addVectors(entries, o_a, o_b, o_ab);

  double elapsedTime = occa::toc("addVectors", addVectors);

  o_ab.copyTo(ab);

  std::cout<<"Elapsed time = " << elapsedTime << " s" << std::endl;

  occa::printTimer();

  for(int i = 0; i < 5; ++i)
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
