#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  int entries = 1024;
  int p_Nred = 256;
  int reducedEntries = entries/p_Nred;

  float *a    = new float[entries];
  float *aRed = new float[reducedEntries];

  for(int i = 0; i < entries; ++i){
    a[i]  = 1;
  }

  for(int i = 0; i < reducedEntries; ++i){
    aRed[i]  = 0;
  }

  // occa::availableDevices<occa::OpenCL>();

  std::string mode = "OpenMP";
  int platformID = 0;
  int deviceID   = 2;

  occa::device device;
  occa::kernel reduction;
  occa::memory o_a, o_aRed;

  occa::kernelInfo reductionInfo;

  device.setup(mode, platformID, deviceID);

  o_a  = device.malloc(entries*sizeof(float));
  o_aRed = device.malloc(reducedEntries*sizeof(float));

  reductionInfo.addDefine("p_Nred", p_Nred);

  reduction = device.buildKernelFromSource("reduction.okl",
                                           "reduction",
                                           reductionInfo);

  o_a.copyFrom(a);

  occa::initTimer(device);

  occa::tic("reduction");

  reduction(entries, o_a, o_aRed);

  double elapsedTime = occa::toc("reduction", reduction);

  o_aRed.copyTo(aRed);

  std::cout << "Elapsed time = " << elapsedTime << " s" << std::endl;

  occa::printTimer();

  for(int i = 0; i < reducedEntries; ++i)
    std::cout << i << ": " << aRed[i] << '\n';

  for(int i = 0; i < reducedEntries; ++i){
    float red = 0;

    for(int j = 0; j < p_Nred; ++j)
      red += a[i];

    if(aRed[i] != red)
      throw 1;
  }

  delete [] a;
  delete [] aRed;

  reduction.free();
  o_a.free();
  o_aRed.free();

  device.free();

  return 0;
}
