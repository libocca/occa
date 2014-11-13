#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  int entries = 1024;
  int p_Nred = 256;
  int reducedEntries = entries/p_Nred;

  float *a    = new float[entries];
  float *reda = new float[reducedEntries];

  for(int i = 0; i < entries; ++i){
    a[i]  = 1;
  }
  for(int i = 0; i < reducedEntries; ++i){
    reda[i]  = 0;
  }

  // occa::availableDevices<occa::OpenCL>();

  std::string mode = "OpenCL";
  int platformID = 0;
  int deviceID   = 2;

  occa::device device;
  occa::kernel reduction;
  occa::memory o_a, o_reda;

  occa::kernelInfo reductionInfo;

  device.setup(mode, platformID, deviceID);

  o_a  = device.malloc(entries*sizeof(float));
  o_reda = device.malloc(reducedEntries*sizeof(float));

  reductionInfo.addDefine("p_Nred"  , p_Nred);

  reduction = device.buildKernelFromSource("reduction.cu",
					   "reduction", 
					   reductionInfo);
  
  o_a.copyFrom(a);
  
  occa::initTimer(device);
  
  occa::tic("reduction");
  
  reduction(entries, o_a, o_reda);
  
  double elapsedTime = occa::toc("reduction", reduction);
  
  o_reda.copyTo(reda);
  
  std::cout << "Elapsed time = " << elapsedTime << " s" << std::endl;
  
  occa::printTimer();
  
  for(int i = 0; i < reducedEntries; ++i)
    std::cout << i << ": " << reda[i] << '\n';

  delete [] a;
  delete [] reda;
  
  reduction.free();
  o_a.free();
  o_reda.free();
  
  device.free();
  
  return 0;
}
